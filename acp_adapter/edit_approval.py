"""Pre-execution ACP edit approval helpers.

This module is intentionally isolated from the generic tool registry.  ACP binds
an edit approval requester in a ContextVar for the duration of one ACP agent run;
CLI, gateway, and other sessions leave it unset and therefore bypass this guard.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import tempfile
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import ContextVar, Token
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EditProposal:
    """A proposed single-file edit that can be shown to an ACP client."""

    tool_name: str
    path: str
    old_text: str | None
    new_text: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class DeniedEdit:
    """A denied ACP edit path that later write-capable tools must not mutate."""

    path: str
    resolved_path: str
    tool_name: str


@dataclass(frozen=True)
class DeniedEditReattempt:
    """A write-capable tool call that appears to retry a denied edit path."""

    tool_name: str
    path: str
    resolved_path: str
    denied_tool_name: str
    arguments: dict[str, Any]


EditApprovalRequest = EditProposal | DeniedEditReattempt
EditApprovalRequester = Callable[[EditApprovalRequest], bool]


_EDIT_APPROVAL_REQUESTER: ContextVar[EditApprovalRequester | None] = ContextVar(
    "ACP_EDIT_APPROVAL_REQUESTER",
    default=None,
)
_DENIED_EDITS: ContextVar[tuple[DeniedEdit, ...]] = ContextVar(
    "ACP_DENIED_EDITS",
    default=(),
)
_PERMISSION_REQUEST_IDS = count(1)


SENSITIVE_AUTO_APPROVE_NAMES = {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519"}
AUTO_APPROVE_ASK = "ask"
AUTO_APPROVE_WORKSPACE = "workspace_session"
AUTO_APPROVE_SESSION = "session"
DENIED_EDIT_BYPASS_TOOLS = {"terminal", "execute_code"}
_DENIED_EDIT_WRITE_PATTERNS = (
    re.compile(r"\bwrite_(?:text|bytes)\s*\("),
    re.compile(r"\bopen\s*\([^)]*,\s*['\"][^'\"]*[wax+][^'\"]*['\"]", re.DOTALL),
    re.compile(r"(?:^|[\s;&|])(?:cat|echo|printf)\b[^\n]*(?:>|>>)", re.DOTALL),
    re.compile(r"(?:^|[^<])>{1,2}\s*\S+"),
    re.compile(r"\b(?:tee|truncate|touch|mv|cp|rm|install)\b"),
    re.compile(r"\b(?:sed|perl)\b[^\n]*\s-i(?:\s|$)", re.DOTALL),
)
_SHELL_REDIRECT_OPERATOR = re.compile(r"^(?:\d*|&)?>{1,2}$")
_SHELL_REDIRECT_ATTACHED = re.compile(r"^(?:\d*|&)?>{1,2}(.+)$")


def set_edit_approval_requester(requester: EditApprovalRequester | None) -> Token:
    """Bind an ACP edit approval requester for the current context."""

    _DENIED_EDITS.set(())
    return _EDIT_APPROVAL_REQUESTER.set(requester)


def reset_edit_approval_requester(token: Token) -> None:
    """Restore a previous edit approval requester binding."""

    _DENIED_EDITS.set(())
    _EDIT_APPROVAL_REQUESTER.reset(token)


def clear_edit_approval_requester() -> None:
    """Clear the current requester; primarily used by tests."""

    _DENIED_EDITS.set(())
    _EDIT_APPROVAL_REQUESTER.set(None)


def get_edit_approval_requester() -> EditApprovalRequester | None:
    return _EDIT_APPROVAL_REQUESTER.get()


def _read_text_if_exists(path: str) -> str | None:
    p = Path(path).expanduser()
    if not p.exists():
        return None
    if not p.is_file():
        raise OSError(f"Cannot edit non-file path: {path}")
    return p.read_text(encoding="utf-8", errors="replace")


def _proposal_for_write_file(arguments: dict[str, Any]) -> EditProposal:
    path = str(arguments.get("path") or "")
    if not path:
        raise ValueError("path required")
    content = arguments.get("content")
    if content is None:
        raise ValueError("content required")
    return EditProposal(
        tool_name="write_file",
        path=path,
        old_text=_read_text_if_exists(path),
        new_text=str(content),
        arguments=dict(arguments),
    )


def _proposal_for_patch_replace(arguments: dict[str, Any]) -> EditProposal:
    path = str(arguments.get("path") or "")
    if not path:
        raise ValueError("path required")
    old_string = arguments.get("old_string")
    new_string = arguments.get("new_string")
    if old_string is None or new_string is None:
        raise ValueError("old_string and new_string required")

    old_text = _read_text_if_exists(path)
    if old_text is None:
        raise ValueError(f"Failed to read file: {path}")

    from tools.fuzzy_match import fuzzy_find_and_replace

    new_text, match_count, _strategy, error = fuzzy_find_and_replace(
        old_text,
        str(old_string),
        str(new_string),
        bool(arguments.get("replace_all", False)),
    )
    if error or match_count == 0:
        raise ValueError(error or f"Could not find match for old_string in {path}")

    return EditProposal(
        tool_name="patch",
        path=path,
        old_text=old_text,
        new_text=new_text,
        arguments=dict(arguments),
    )


def build_edit_proposal(tool_name: str, arguments: dict[str, Any]) -> EditProposal | None:
    """Return an edit proposal for supported file mutation calls."""

    if tool_name == "write_file":
        return _proposal_for_write_file(arguments)
    if tool_name == "patch" and arguments.get("mode", "replace") == "replace":
        return _proposal_for_patch_replace(arguments)
    return None


def _is_sensitive_auto_approve_path(path: str) -> bool:
    parts = Path(path).expanduser().parts
    lowered = {part.lower() for part in parts}
    if ".git" in lowered or ".ssh" in lowered:
        return True
    return Path(path).name.lower() in SENSITIVE_AUTO_APPROVE_NAMES


def should_auto_approve_edit(proposal: EditProposal, policy: str, cwd: str | None = None) -> bool:
    """Return whether an ACP edit proposal may bypass the prompt for this session.

    This is intentionally session-scoped and conservative: sensitive paths still
    ask even under autonomous policies.
    """

    policy = str(policy or AUTO_APPROVE_ASK).strip()
    if policy == AUTO_APPROVE_ASK or _is_sensitive_auto_approve_path(proposal.path):
        return False
    path = Path(proposal.path).expanduser().resolve(strict=False)
    if policy == AUTO_APPROVE_SESSION:
        return True
    if policy == AUTO_APPROVE_WORKSPACE:
        # `/tmp` is the POSIX path but tempfile.gettempdir() is the real one on
        # every platform: `/private/tmp` on macOS (because `/tmp` is a symlink
        # and Path.resolve() follows it) and the per-user Temp dir on Windows.
        tmp_root = Path(tempfile.gettempdir()).resolve(strict=False)
        try:
            path.relative_to(tmp_root)
            return True
        except ValueError:
            pass
        if cwd:
            root = Path(cwd).expanduser().resolve(strict=False)
            try:
                path.relative_to(root)
                return True
            except ValueError:
                return False
    return False


def _resolved_edit_path(path: str) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _record_denied_edit(proposal: EditProposal) -> None:
    denied = DeniedEdit(
        path=proposal.path,
        resolved_path=_resolved_edit_path(proposal.path),
        tool_name=proposal.tool_name,
    )
    existing = tuple(item for item in _DENIED_EDITS.get() if item.resolved_path != denied.resolved_path)
    _DENIED_EDITS.set((*existing, denied))


def _forget_denied_edit(proposal: EditProposal) -> None:
    resolved_path = _resolved_edit_path(proposal.path)
    _DENIED_EDITS.set(tuple(item for item in _DENIED_EDITS.get() if item.resolved_path != resolved_path))


def _write_capable_tool_text(tool_name: str, arguments: dict[str, Any]) -> str | None:
    if tool_name == "terminal":
        command = arguments.get("command")
        return str(command) if command is not None else None
    if tool_name == "execute_code":
        code = arguments.get("code")
        return str(code) if code is not None else None
    return None


def _path_matches_denied_edit(candidate: str, denied: DeniedEdit) -> bool:
    if not candidate:
        return False
    if candidate in {denied.path, denied.resolved_path}:
        return True
    try:
        return _resolved_edit_path(candidate) == denied.resolved_path
    except (OSError, RuntimeError):
        return False


def _mentions_denied_path(text: str, denied: DeniedEdit) -> bool:
    path_candidates = {denied.path, denied.resolved_path}
    path_candidates.discard("")
    return any(candidate in text for candidate in path_candidates)


def _shell_redirection_targets(command: str) -> tuple[str, ...]:
    """Return obvious shell redirection targets after quote expansion.

    This is not a shell parser; it covers simple redirect forms so shell-quoted
    denied paths (for example paths containing apostrophes) still require fresh
    ACP approval instead of bypassing the literal string containment check.
    """

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return ()

    targets: list[str] = []
    for index, token in enumerate(tokens):
        if _SHELL_REDIRECT_OPERATOR.match(token):
            if index + 1 < len(tokens):
                targets.append(tokens[index + 1])
            continue
        attached = _SHELL_REDIRECT_ATTACHED.match(token)
        if attached:
            targets.append(attached.group(1))
    return tuple(targets)


def _terminal_mentions_denied_path(command: str, denied: DeniedEdit) -> bool:
    if _mentions_denied_path(command, denied):
        return True
    return any(_path_matches_denied_edit(target, denied) for target in _shell_redirection_targets(command))


def _has_write_intent(text: str) -> bool:
    return any(pattern.search(text) for pattern in _DENIED_EDIT_WRITE_PATTERNS)


def _denied_edit_reattempt(tool_name: str, arguments: dict[str, Any]) -> DeniedEditReattempt | None:
    """Return a fresh approval request if a tool appears to retry a denied edit."""

    if tool_name not in DENIED_EDIT_BYPASS_TOOLS:
        return None
    text = _write_capable_tool_text(tool_name, arguments)
    if not text or not _has_write_intent(text):
        return None
    for denied in _DENIED_EDITS.get():
        mentions_denied_path = (
            _terminal_mentions_denied_path(text, denied)
            if tool_name == "terminal"
            else _mentions_denied_path(text, denied)
        )
        if mentions_denied_path:
            return DeniedEditReattempt(
                tool_name=tool_name,
                path=denied.path,
                resolved_path=denied.resolved_path,
                denied_tool_name=denied.tool_name,
                arguments=dict(arguments),
            )
    return None


def maybe_require_edit_approval(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Run ACP edit approval if bound.

    Returns a JSON tool-error string when the edit must be blocked, otherwise
    ``None`` so dispatch can continue.  Requester exceptions deny by default.
    """

    requester = get_edit_approval_requester()
    if requester is None:
        return None

    try:
        proposal = build_edit_proposal(tool_name, arguments)
    except Exception as exc:
        logger.warning("Could not build ACP edit approval proposal for %s: %s", tool_name, exc)
        return json.dumps({"error": f"Edit approval denied: could not prepare diff ({exc})"}, ensure_ascii=False)

    if proposal is None:
        reattempt = _denied_edit_reattempt(tool_name, arguments)
        if reattempt is not None:
            try:
                approved = bool(requester(reattempt))
            except Exception as exc:
                logger.warning("ACP alternate write approval requester failed: %s", exc)
                approved = False
            if approved:
                return None
            return json.dumps(
                {
                    "error": (
                        "Alternate write approval denied by ACP client; "
                        f"tool was not run for previously denied path: {reattempt.path}"
                    )
                },
                ensure_ascii=False,
            )
        return None

    try:
        approved = bool(requester(proposal))
    except Exception as exc:
        logger.warning("ACP edit approval requester failed: %s", exc)
        approved = False

    if approved:
        _forget_denied_edit(proposal)
        return None
    _record_denied_edit(proposal)
    return json.dumps({"error": "Edit approval denied by ACP client; file was not modified."}, ensure_ascii=False)


def build_acp_edit_tool_call(proposal: EditProposal):
    """Build the ToolCallUpdate payload for ACP request_permission."""

    import acp

    tool_call_id = f"edit-approval-{next(_PERMISSION_REQUEST_IDS)}"
    return acp.update_tool_call(
        tool_call_id,
        title=f"Approve edit: {proposal.path}",
        kind="edit",
        status="pending",
        content=[
            acp.tool_diff_content(
                path=proposal.path,
                old_text=proposal.old_text,
                new_text=proposal.new_text,
            )
        ],
        raw_input={"tool": proposal.tool_name, "arguments": proposal.arguments},
    )


def build_acp_write_reattempt_tool_call(proposal: DeniedEditReattempt):
    """Build the ToolCallUpdate payload for same-path write reattempt approval."""

    import acp

    tool_call_id = f"edit-reattempt-{next(_PERMISSION_REQUEST_IDS)}"
    text = (
        f"A previous ACP edit request for `{proposal.path}` was denied. "
        f"The `{proposal.tool_name}` tool call appears able to modify the same path. "
        "Run it only with fresh approval."
    )
    return acp.update_tool_call(
        tool_call_id,
        title=f"Approve previously denied write: {proposal.path}",
        kind="execute",
        status="pending",
        content=[acp.tool_content(acp.text_block(text))],
        raw_input={
            "tool": proposal.tool_name,
            "arguments": proposal.arguments,
            "denied_path": proposal.path,
            "denied_tool": proposal.denied_tool_name,
        },
    )


def make_acp_edit_approval_requester(
    request_permission_fn: Callable,
    loop: asyncio.AbstractEventLoop,
    session_id: str,
    timeout: float = 60.0,
    auto_approve_getter: Callable[[], tuple[str, str | None]] | None = None,
) -> EditApprovalRequester:
    """Return a sync requester that bridges edit proposals to ACP permissions."""

    def _requester(proposal: EditApprovalRequest) -> bool:
        from acp.schema import PermissionOption
        from agent.async_utils import safe_schedule_threadsafe

        if isinstance(proposal, EditProposal) and auto_approve_getter is not None:
            try:
                policy, cwd = auto_approve_getter()
                if should_auto_approve_edit(proposal, policy, cwd):
                    logger.info("Auto-approved ACP edit under policy %s: %s", policy, proposal.path)
                    return True
            except Exception:
                logger.debug("ACP edit auto-approval policy check failed", exc_info=True)

        options = [
            PermissionOption(option_id="allow_once", kind="allow_once", name="Allow edit"),
            PermissionOption(option_id="deny", kind="reject_once", name="Deny"),
        ]
        if isinstance(proposal, EditProposal):
            tool_call = build_acp_edit_tool_call(proposal)
        else:
            tool_call = build_acp_write_reattempt_tool_call(proposal)
        coro = request_permission_fn(
            session_id=session_id,
            tool_call=tool_call,
            options=options,
        )
        future = safe_schedule_threadsafe(
            coro,
            loop,
            logger=logger,
            log_message="Edit approval request: failed to schedule on loop",
        )
        if future is None:
            return False
        try:
            response = future.result(timeout=timeout)
        except (FutureTimeout, Exception) as exc:
            future.cancel()
            logger.warning("Edit approval request timed out or failed: %s", exc)
            return False
        outcome = getattr(response, "outcome", None)
        return (
            getattr(outcome, "outcome", None) == "selected"
            and getattr(outcome, "option_id", None) == "allow_once"
        )

    return _requester
