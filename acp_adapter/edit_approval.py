"""Pre-execution ACP edit approval helpers.

Intentionally isolated from the generic tool registry: ACP binds an edit
approval requester in a ContextVar for the duration of one ACP agent run; CLI,
gateway, and other sessions leave it unset and therefore bypass this guard.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
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

_EDIT_APPROVAL_REQUESTER: ContextVar[EditApprovalRequester | None] = ContextVar("ACP_EDIT_APPROVAL_REQUESTER", default=None)
_PERMISSION_REQUEST_IDS = count(1)

SENSITIVE_AUTO_APPROVE_NAMES = {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519"}
AUTO_APPROVE_ASK = "ask"
AUTO_APPROVE_WORKSPACE = "workspace_session"
AUTO_APPROVE_SESSION = "session"

_V4A_FILE_RE = re.compile(r'^\*\*\*\s+(?:Update|Add|Delete)\s+File:\s*(.+)$', re.MULTILINE)
_V4A_MOVE_RE = re.compile(r'^\*\*\*\s+Move\s+File:\s*(.+?)\s*->\s*(.+)$', re.MULTILINE)


def set_edit_approval_requester(requester: EditApprovalRequester | None) -> Token:
    """Bind an ACP edit approval requester for the current context."""
    return _EDIT_APPROVAL_REQUESTER.set(requester)


def reset_edit_approval_requester(token: Token) -> None:
    """Restore a previous edit approval requester binding."""
    _EDIT_APPROVAL_REQUESTER.reset(token)


def _read_text_if_exists(path: str) -> str | None:
    p = Path(path).expanduser()
    if p.is_file():
        return p.read_text(encoding="utf-8", errors="replace")
    if p.exists():
        raise OSError(f"Cannot edit non-file path: {path}")
    return None


def _required_path(arguments: dict[str, Any]) -> str:
    path = str(arguments.get("path") or "")
    if not path:
        raise ValueError("path required")
    return path


def _proposal_for_write_file(arguments: dict[str, Any]) -> EditProposal:
    path = _required_path(arguments)
    content = arguments.get("content")
    if content is None:
        raise ValueError("content required")
    return EditProposal("write_file", path, _read_text_if_exists(path), str(content), dict(arguments))


def _proposal_for_patch_replace(arguments: dict[str, Any]) -> EditProposal:
    path = _required_path(arguments)
    old_string, new_string = arguments.get("old_string"), arguments.get("new_string")
    if old_string is None or new_string is None:
        raise ValueError("old_string and new_string required")
    old_text = _read_text_if_exists(path)
    if old_text is None:
        raise ValueError(f"Failed to read file: {path}")

    from tools.fuzzy_match import fuzzy_find_and_replace

    new_text, match_count, _strategy, error = fuzzy_find_and_replace(
        old_text, str(old_string), str(new_string), bool(arguments.get("replace_all", False)))
    if error or match_count == 0:
        raise ValueError(error or f"Could not find match for old_string in {path}")
    return EditProposal("patch", path, old_text, new_text, dict(arguments))


def _extract_v4a_patch_paths(patch_body: str) -> list[str]:
    paths = [m.group(1).strip() for m in _V4A_FILE_RE.finditer(patch_body)]
    for match in _V4A_MOVE_RE.finditer(patch_body):
        paths.extend(match.group(i).strip() for i in (1, 2))
    return [p for p in paths if p]


def _proposal_for_patch_v4a(arguments: dict[str, Any]) -> EditProposal:
    patch_body = arguments.get("patch")
    if not isinstance(patch_body, str) or not patch_body:
        raise ValueError("patch content required")
    paths = _extract_v4a_patch_paths(patch_body)
    if not paths:
        raise ValueError("no file paths found in V4A patch")
    single = len(paths) == 1
    # ACP only supports a single diff payload: surface the exact V4A patch as new_text so
    # patch-mode calls are permissioned and denied patches cannot mutate.
    return EditProposal(
        "patch", paths[0] if single else ", ".join(paths),
        _read_text_if_exists(paths[0]) if single else None, patch_body, dict(arguments),
    )


# (tool_name, patch mode or None) -> proposal builder.
_PROPOSAL_BUILDERS = {
    ("write_file", None): _proposal_for_write_file, ("patch", "replace"): _proposal_for_patch_replace,
    ("patch", "patch"): _proposal_for_patch_v4a,
}


def build_edit_proposal(tool_name: str, arguments: dict[str, Any]) -> EditProposal | None:
    """Return an edit proposal for supported file mutation calls."""
    mode = arguments.get("mode", "replace") if tool_name == "patch" else None
    builder = _PROPOSAL_BUILDERS.get((tool_name, mode))
    return builder(arguments) if builder else None


def _is_sensitive_auto_approve_path(path: str) -> bool:
    lowered = {part.lower() for part in Path(path).expanduser().parts}
    return bool(lowered & {".git", ".ssh"}) or Path(path).name.lower() in SENSITIVE_AUTO_APPROVE_NAMES


def should_auto_approve_edit(proposal: EditProposal, policy: str, cwd: str | None = None) -> bool:
    """Return whether an ACP edit proposal may bypass the prompt for this session.

    Session-scoped and conservative: sensitive paths still ask under autonomous policies."""
    policy = str(policy or AUTO_APPROVE_ASK).strip()
    if policy == AUTO_APPROVE_ASK or _is_sensitive_auto_approve_path(proposal.path):
        return False
    path = Path(proposal.path).expanduser().resolve(strict=False)
    if policy == AUTO_APPROVE_SESSION:
        return True
    if policy == AUTO_APPROVE_WORKSPACE:
        # tempfile.gettempdir() is the real temp root on every platform
        # (``/private/tmp`` on macOS since resolve() follows the symlink).
        return path.is_relative_to(Path(tempfile.gettempdir()).resolve(strict=False)) or (
            bool(cwd) and path.is_relative_to(Path(cwd).expanduser().resolve(strict=False)))
    return False


def _denied(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def maybe_require_edit_approval(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Run ACP edit approval if bound.

    Returns a JSON tool-error string when the edit must be blocked, otherwise
    ``None`` so dispatch can continue.  Requester exceptions deny by default."""
    requester = _EDIT_APPROVAL_REQUESTER.get()
    if requester is None:
        return None
    try:
        proposal = build_edit_proposal(tool_name, arguments)
    except Exception as exc:
        logger.warning("Could not build ACP edit approval proposal for %s: %s", tool_name, exc)
        return _denied(f"Edit approval denied: could not prepare diff ({exc})")
    if proposal is None:
        return None
    try:
        approved = bool(requester(proposal))
    except Exception as exc:
        logger.warning("ACP edit approval requester failed: %s", exc)
        approved = False
    return None if approved else _denied("Edit approval denied by ACP client; file was not modified.")


def build_acp_edit_tool_call(proposal: EditProposal):
    """Build the ToolCallUpdate payload for ACP request_permission."""
    import acp

    return acp.update_tool_call(
        f"edit-approval-{next(_PERMISSION_REQUEST_IDS)}", title=f"Approve edit: {proposal.path}", kind="edit",
        status="pending",
        content=[acp.tool_diff_content(path=proposal.path, old_text=proposal.old_text, new_text=proposal.new_text)],
        raw_input={"tool": proposal.tool_name, "arguments": proposal.arguments},
    )


def make_acp_edit_approval_requester(
    request_permission_fn: Callable, loop: asyncio.AbstractEventLoop, session_id: str,
    timeout: float = 60.0, auto_approve_getter: Callable[[], tuple[str, str | None]] | None = None,
) -> EditApprovalRequester:
    """Return a sync requester that bridges edit proposals to ACP permissions."""

    def _requester(proposal: EditProposal) -> bool:
        from acp.schema import PermissionOption
        from acp_adapter.permissions import await_permission

        if auto_approve_getter is not None:
            try:
                policy, cwd = auto_approve_getter()
                if should_auto_approve_edit(proposal, policy, cwd):
                    logger.info("Auto-approved ACP edit under policy %s: %s", policy, proposal.path)
                    return True
            except Exception:
                logger.debug("ACP edit auto-approval policy check failed", exc_info=True)

        response, _timed_out = await_permission(
            request_permission_fn, loop, session_id, tool_call=build_acp_edit_tool_call(proposal),
            options=[PermissionOption(option_id="allow_once", kind="allow_once", name="Allow edit"),
                     PermissionOption(option_id="deny", kind="reject_once", name="Deny")],
            timeout=timeout, what="Edit approval request",
        )
        outcome = getattr(response, "outcome", None)
        return getattr(outcome, "outcome", None) == "selected" and getattr(outcome, "option_id", None) == "allow_once"

    return _requester


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from concurrent.futures import TimeoutError as FutureTimeout  # noqa: F401,E402

def clear_edit_approval_requester() -> None:
    """Clear the current requester; primarily used by tests."""

    _EDIT_APPROVAL_REQUESTER.set(None)

def get_edit_approval_requester() -> EditApprovalRequester | None:
    return _EDIT_APPROVAL_REQUESTER.get()
# ---- END PLUGIN-COMPAT ----
