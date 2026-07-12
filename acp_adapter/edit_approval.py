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


EditApprovalRequester = Callable[[EditProposal], bool]

_EDIT_APPROVAL_REQUESTER: ContextVar[EditApprovalRequester | None] = ContextVar(
    "ACP_EDIT_APPROVAL_REQUESTER",
    default=None,
)
_PERMISSION_REQUEST_IDS = count(1)


SENSITIVE_AUTO_APPROVE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
    # Shell startup / login files: writing these runs arbitrary code at the
    # next shell or login.
    ".bashrc",
    ".bash_profile",
    ".bash_login",
    ".profile",
    ".zshrc",
    ".zshenv",
    ".zprofile",
    ".zlogin",
    # Tool config files that can execute commands or redirect trust.
    ".gitconfig",
    ".npmrc",
    ".pypirc",
    ".netrc",
    ".envrc",
}
# Path components that indicate credentials, VCS internals, or persistence
# locations. Any edit whose path contains one of these must always prompt.
SENSITIVE_AUTO_APPROVE_DIR_PARTS = {".git", ".ssh", "autostart", "systemd"}
AUTO_APPROVE_ASK = "ask"
AUTO_APPROVE_WORKSPACE = "workspace_session"
AUTO_APPROVE_SESSION = "session"


def set_edit_approval_requester(requester: EditApprovalRequester | None) -> Token:
    """Bind an ACP edit approval requester for the current context."""

    return _EDIT_APPROVAL_REQUESTER.set(requester)


def reset_edit_approval_requester(token: Token) -> None:
    """Restore a previous edit approval requester binding."""

    _EDIT_APPROVAL_REQUESTER.reset(token)


def clear_edit_approval_requester() -> None:
    """Clear the current requester; primarily used by tests."""

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


def _extract_v4a_patch_paths(patch_body: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(
        r'^\*\*\*\s+(?:Update|Add|Delete)\s+File:\s*(.+)$',
        patch_body,
        re.MULTILINE,
    ):
        path = match.group(1).strip()
        if path:
            paths.append(path)
    for match in re.finditer(
        r'^\*\*\*\s+Move\s+File:\s*(.+?)\s*->\s*(.+)$',
        patch_body,
        re.MULTILINE,
    ):
        src = match.group(1).strip()
        dst = match.group(2).strip()
        if src:
            paths.append(src)
        if dst:
            paths.append(dst)
    return paths


def _proposal_for_patch_v4a(arguments: dict[str, Any]) -> EditProposal:
    patch_body = arguments.get("patch")
    if not isinstance(patch_body, str) or not patch_body:
        raise ValueError("patch content required")

    paths = _extract_v4a_patch_paths(patch_body)
    if not paths:
        raise ValueError("no file paths found in V4A patch")

    proposal_path = paths[0] if len(paths) == 1 else ", ".join(paths)
    old_text = _read_text_if_exists(paths[0]) if len(paths) == 1 else None
    return EditProposal(
        tool_name="patch",
        path=proposal_path,
        old_text=old_text,
        # ACP only supports a single diff payload here.  Surface the exact V4A
        # patch content before execution so patch-mode calls are permissioned
        # and denied patches cannot mutate.
        new_text=patch_body,
        arguments=dict(arguments),
    )


def build_edit_proposal(tool_name: str, arguments: dict[str, Any]) -> EditProposal | None:
    """Return an edit proposal for supported file mutation calls."""

    if tool_name == "write_file":
        return _proposal_for_write_file(arguments)
    if tool_name == "patch":
        mode = arguments.get("mode", "replace")
        if mode == "replace":
            return _proposal_for_patch_replace(arguments)
        if mode == "patch":
            return _proposal_for_patch_v4a(arguments)
    return None


def _is_sensitive_auto_approve_path(path: str) -> bool:
    expanded = Path(path).expanduser()
    lowered = {part.lower() for part in expanded.parts}
    if lowered & SENSITIVE_AUTO_APPROVE_DIR_PARTS:
        return True
    if expanded.name.lower() in SENSITIVE_AUTO_APPROVE_NAMES:
        return True
    # Any dotfile directly in the user's home directory is treated as sensitive.
    # This catches shell and tool rc/config files (arbitrary code execution or
    # trust redirection) that are not individually enumerated above.
    try:
        resolved = expanded.resolve(strict=False)
        if resolved.name.startswith(".") and resolved.parent == Path.home().resolve(strict=False):
            return True
    except (OSError, RuntimeError):
        pass
    return False


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
        return None

    try:
        approved = bool(requester(proposal))
    except Exception as exc:
        logger.warning("ACP edit approval requester failed: %s", exc)
        approved = False

    if approved:
        return None
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


def make_acp_edit_approval_requester(
    request_permission_fn: Callable,
    loop: asyncio.AbstractEventLoop,
    session_id: str,
    timeout: float = 60.0,
    auto_approve_getter: Callable[[], tuple[str, str | None]] | None = None,
) -> EditApprovalRequester:
    """Return a sync requester that bridges edit proposals to ACP permissions."""

    def _requester(proposal: EditProposal) -> bool:
        from acp.schema import PermissionOption
        from agent.async_utils import safe_schedule_threadsafe

        if auto_approve_getter is not None:
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
        tool_call = build_acp_edit_tool_call(proposal)
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
