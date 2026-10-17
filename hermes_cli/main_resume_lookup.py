"""Session MRU lookups for the CLI launch paths: ``hermes -c`` / ``--resume latest``.

Split out of ``hermes_cli/main.py`` (which is over its file cap and may only shrink). The
workspace-scoping and cross-surface rules live here so the launch path in main stays thin.

Both surfaces that can resume are represented:

- the **CLI family** (``cli``, ``oneshot``) — a finite ``hermes -z`` / ``chat -q`` run is CLI
  history, so ``-c`` chains on it (#112550);
- the **UI transports** (``desktop``, ``tui``) — the Desktop app and TUI write their transcripts
  into the same ``state.db``, so the CLI can continue the conversation the user just had there.

Names that still live in ``main`` (``_session_db``, ``_resolve_workspace_key``) are resolved
through the module at call time to avoid an import cycle and to keep the launch path patchable.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

from agent.session_source import CLI_FAMILY_SOURCES, UI_TRANSPORT_SOURCES

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.main")


def _main():
    """The ``hermes_cli.main`` module, imported lazily (it imports this one)."""
    import hermes_cli.main as main

    return main


def _resolve_last_session(
    source: Union[str, Sequence[str]] = "cli", *, workspace_only: bool = False
) -> Optional[str]:
    """Look up the most recently-used session ID for a source (or several).

    Scoped to the current workspace first (git repo root, else cwd) so
    ``hermes -c`` from repo A continues repo A's last session rather than the
    global MRU. Falls back to the unscoped MRU when no session matches the
    current workspace, preserving the old behaviour for fresh directories.

    ``workspace_only`` skips that global fallback. ``_latest_session_id`` uses it so the UI
    transports can never win a bare ``-c`` with a conversation from an unrelated workspace — a
    Desktop session in *another* project must not outrank this workspace's CLI history.
    """
    # A finite `hermes -z`/`chat -q` run is CLI history too: `hermes -z … --resume latest` chains on it.
    if source == "cli":
        source = sorted(CLI_FAMILY_SOURCES)
    main = _main()
    with main._session_db() as db:
        ws_key = main._resolve_workspace_key()
        if ws_key:
            sessions = db.search_sessions(source=source, limit=1, workspace_key=ws_key)
            if sessions:
                return sessions[0]["id"]
        if workspace_only:
            return None
        # Fallback: global MRU for this source.
        sessions = db.search_sessions(source=source, limit=1)
        return sessions[0]["id"] if sessions else None
    return None


def _session_started_at(session_id: str) -> float:
    """``started_at`` for one session row, or 0.0 when the row is unreadable.

    0.0 sorts such a row last rather than raising inside a resume lookup, which must never
    fail the launch it is serving.
    """
    main = _main()
    with main._session_db() as db:
        try:
            row = db.get_session(session_id) or {}
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.debug("started_at unreadable for %s: %s", session_id, exc)
            return 0.0
        try:
            return float(row.get("started_at") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _latest_session_id(use_tui: bool, *, resolve=None) -> Optional[str]:
    """MRU session for the active interface; the interactive CLI also reads the UI transports' MRU.

    A bare ``hermes -c`` (or ``--resume latest``) is the user asking to continue *the* conversation
    they were just having, and the Desktop app and TUI write their transcripts into the same
    workspace of the same ``state.db``. Scoping the lookup to the ``cli`` family alone therefore
    skipped a newer Desktop conversation the user could see in every picker. So the CLI compares its
    own MRU against the UI transports' MRU — still workspace-scoped on both sides, so another
    project's Desktop chat can never win — and takes whichever is newer.

    The comparison is by ``started_at`` and not by presence: preferring the UI MRU whenever one
    existed would send ``-c`` to an older Desktop chat and break ordinary CLI resume.

    TUI keeps its own order (``tui`` first, then the CLI family) but deliberately does not adopt the
    Desktop MRU: the TUI and Desktop are two live surfaces of the same transport family, and a TUI
    launch grabbing the Desktop's conversation would hijack the other window.

    ``resolve`` is the lookup to call, defaulting to this module's. ``hermes_cli.main`` passes its
    own wrapper so patching ``main._resolve_last_session`` (the launch tests' seam) still works.
    """
    resolve = resolve or _resolve_last_session
    if use_tui:
        return resolve(source="tui") or resolve(source="cli")

    candidates = [
        resolve(source="cli", workspace_only=True),
        resolve(source=sorted(UI_TRANSPORT_SOURCES), workspace_only=True),
    ]
    ranked = [sid for sid in candidates if sid]
    if ranked:
        return max(ranked, key=_session_started_at)
    # Nothing in this workspace: the old global CLI-family fallback, so a fresh directory still
    # behaves as it always did. The UI transports never reach this branch — a Desktop chat from an
    # unrelated project must not be dragged into this workspace.
    return resolve(source="cli")
