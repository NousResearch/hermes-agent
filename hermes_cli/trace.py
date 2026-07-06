"""``hermes trace`` CLI subcommand — upload a session transcript to Hugging Face.

Thin CLI wrapper over :mod:`agent.trace_upload`. The heavy lifting (session
load, Claude Code JSONL conversion, HF upload) lives there so the command
handler stays small and deterministic.
"""

from __future__ import annotations

import sys


def _latest_session_id() -> str | None:
    """Return the most recently active session id, or None.

    Used when the user runs ``hermes trace upload`` without an explicit id
    (the common case from the shell — they mean "the last thing I did").
    """
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        sessions = db.list_sessions_rich(limit=1, order_by_last_active=True)
        if sessions:
            return sessions[0].get("id")
    except Exception:
        pass
    return None


def run_trace(args) -> None:
    """Dispatch ``hermes trace <subcommand>``."""
    sub = getattr(args, "trace_command", None)
    if sub in ("upload", "up"):
        _run_upload(args)
        return
    # No/unknown subcommand → brief usage.
    print("Usage: hermes trace upload [SESSION_ID] [--public] [--no-redact]")
    print("Upload a session transcript to your private Hugging Face traces dataset,")
    print("viewable in the HF Agent Trace Viewer (https://huggingface.co/docs/hub/agent-traces).")


def _run_upload(args) -> None:
    from hermes_cli.env_loader import load_hermes_dotenv
    from hermes_cli.config import get_env_path, get_project_root

    # Load .env so HF_TOKEN is visible even outside a running session.
    env_path = get_env_path()
    try:
        load_hermes_dotenv(
            hermes_home=env_path.parent,
            project_env=get_project_root() / ".env",
        )
    except Exception:
        pass

    from agent.trace_upload import upload_session_trace

    session_id = getattr(args, "session_id", None) or _latest_session_id()
    if not session_id:
        print("No session found to upload. Pass a SESSION_ID explicitly.", file=sys.stderr)
        sys.exit(1)

    status = upload_session_trace(
        session_id,
        cwd="",
        redact=not getattr(args, "no_redact", False),
        private=not getattr(args, "public", False),
    )
    print(status)
