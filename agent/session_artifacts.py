"""Resolve the optional, session-scoped deliverables directory."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_session_artifacts_dir(session_id: str | None, parent_session_id: str | None = None) -> str:
    """Return an inherited or configured artifacts directory without creating it.

    ``sessions.artifacts_dir`` is an opt-in root.  Each root conversation owns
    ``<root>/<session-id>/artifacts``; children preserve the parent value through
    the session environment bridge rather than making a second directory.
    """
    if parent_session_id:
        try:
            from gateway.session_context import get_session_env
            inherited = get_session_env("HERMES_SESSION_ARTIFACTS_DIR", "").strip()
        except Exception:
            inherited = os.environ.get("HERMES_SESSION_ARTIFACTS_DIR", "").strip()
        if inherited:
            return inherited
    if not session_id:
        return ""
    try:
        from hermes_cli.config import load_config_readonly
        sessions = (load_config_readonly() or {}).get("sessions") or {}
        root = sessions.get("artifacts_dir") if isinstance(sessions, dict) else None
    except Exception:
        return ""
    if not isinstance(root, str) or not root.strip():
        return ""
    return str(Path(os.path.expandvars(root)).expanduser() / str(session_id) / "artifacts")
