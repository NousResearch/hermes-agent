"""Single source of truth for the agent working directory.

`TERMINAL_CWD` is the runtime carrier for the configured working directory (`terminal.cwd`
is bridged to it once at gateway/cron startup; the local CLI leaves it unset and relies on
the launch dir). Reading it in one place keeps the system prompt, tool surfaces, and
context-file discovery agreeing on where the agent lives. Multi-session gateways can pin a
logical cwd via `_SESSION_CWD`.
"""

import logging
import os
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_UNSET: Any = object()

_SESSION_CWD: ContextVar = ContextVar("HERMES_SESSION_CWD", default=_UNSET)

# The package/source root (<root>/agent/runtime_cwd.py). A backend launched from or
# self-spawned into this tree (desktop default) must never let an os.getcwd() fallback
# inject this repo's contributor AGENTS.md as project context.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _is_install_tree(p: Path) -> bool:
    """True only when ``p`` IS the package root or sits inside it — ancestors
    (a home dir containing the checkout) are legitimate workspaces."""
    try:
        p = p.resolve()
    except Exception:
        return False
    return p == _PACKAGE_ROOT or _PACKAGE_ROOT in p.parents


def set_session_cwd(cwd: str | None) -> Token:
    """Pin the logical cwd for the current context."""
    return _SESSION_CWD.set((cwd or "").strip())


def clear_session_cwd() -> None:
    _SESSION_CWD.set("")


def reset_session_cwd(token: Token) -> None:
    """Restore the logical cwd that was active before the matching ``set_session_cwd``."""
    _SESSION_CWD.reset(token)


def scope_terminal_cwd() -> str:
    """Scope-aware TERMINAL_CWD value (may be empty) — every cwd consumer reads through this.

    Under gateway multiplexing the per-turn terminal scope carries the active profile's cwd;
    the process-global env var may hold another profile's. Only an ImportError falls back: an
    active refusal scope must raise, not silently resolve the launch profile's cwd.
    """
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        return os.environ.get("TERMINAL_CWD", "")
    return terminal_env("TERMINAL_CWD", "")


def _existing_dir(raw: str, label: str) -> Path | None:
    p = Path(raw).expanduser()
    if p.is_dir():
        return p
    logger.warning("%s does not exist: %s", label, raw)
    return None


def _resolve_configured_cwd(*, override_is_final: bool) -> Path | None:
    """Session override, then TERMINAL_CWD; each validated as a real directory.

    ``override_is_final``: a set-but-missing session override yields None
    instead of falling through to TERMINAL_CWD.
    """
    override = _SESSION_CWD.get()
    override = "" if override is _UNSET else str(override).strip()
    if override:
        p = _existing_dir(override, "configured working directory")
        if p is not None or override_is_final:
            return p
    raw = scope_terminal_cwd().strip()
    return _existing_dir(raw, "TERMINAL_CWD") if raw else None


def resolve_agent_cwd() -> Path:
    """Configured cwd, else the launch dir (os.getcwd()'s OSError on a deleted cwd deliberately propagates)."""
    return _resolve_configured_cwd(override_is_final=False) or Path(os.getcwd())


# Markers that make an ancestor directory a *project* for memory scoping (issue #33638).
# Deliberately narrower than coding_context._PROJECT_MARKERS: those answer "is this a code
# workspace" for the posture prompt, while a memory scope is something a user opts into by
# tagging entries, and `.hermes-memory.md` is the marker the memory docs hand them for a
# non-repo workspace. Same discovery shape (bounded walk, $HOME/temp excluded).
_PROJECT_SCOPE_MARKERS = (".git", "AGENTS.md", ".hermes-memory.md")


def resolve_project_scope(cwd: Optional[Path] = None) -> str:
    """Project identifier for the agent's working directory ("" = not in a project).

    Returns the basename of the nearest ancestor holding a ``_PROJECT_SCOPE_MARKERS``
    entry, for the ``memory.project_scoping`` filter (issue #33638). Callers pass the cwd
    they pinned the result against; ``None`` resolves the configured cwd itself (an
    unconfigured one falls back to the launch dir). Fails open to ``""`` — an
    unconfigured cwd (a gateway serving many cwds), ``$HOME``, the shared temp root, and
    the Hermes install tree are all "no project", which leaves the memory block
    unfiltered rather than scoping it to something arbitrary.

    This is a filesystem probe with no cache; ``agent/system_prompt.py::
    pinned_project_scope`` owns the per-session pin that keeps the block byte-stable.
    """
    import tempfile

    if cwd is None:
        cwd = resolve_context_cwd() or Path(os.getcwd())
    # A backend self-spawned into the Hermes tree (desktop default) must not scope every
    # session that didn't configure a cwd to the checkout itself.
    if _is_install_tree(cwd):
        return ""
    current = cwd.resolve()

    try:
        home = Path.home().resolve()
    except (OSError, RuntimeError):
        home = None
    try:
        temp_root = Path(tempfile.gettempdir()).resolve()
    except Exception:
        temp_root = None

    for depth, parent in enumerate([current, *current.parents]):
        if depth > 6:
            break
        # $HOME and the shared temp root end the walk: their parents are never a project,
        # and a .git sitting above home would otherwise scope every unconfigured session.
        if parent == home or (temp_root is not None and parent == temp_root):
            break
        for marker in _PROJECT_SCOPE_MARKERS:
            try:
                if (parent / marker).exists():
                    return parent.name
            except PermissionError:
                # Unreadable ancestor: skip its markers, keep walking up.
                continue
    return ""


def resolve_context_cwd() -> Path | None:
    """Configured cwd for context-file discovery, or None (build_context_files_prompt then falls back to the
    launch dir). An existing configured path is honored verbatim — including the Hermes source tree, a
    legitimate workspace when developing Hermes; fallback-directory policy lives in the caller."""
    return _resolve_configured_cwd(override_is_final=True)
