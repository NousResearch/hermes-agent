"""hosting-digest plugin — manually trigger hosting cost digest reports.

Registers ``/hosting-digest`` as a gateway slash command.  The command runs the
same script used by the scheduled Vultr cost digest so operators can request an
on-demand infrastructure cost snapshot without waiting for the hourly cron.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


DEFAULT_TIMEOUT_SECONDS = 60


def _default_script_path() -> Path:
    return Path.home() / ".hermes" / "scripts" / "vultr_cost_digest.py"


def _script_path() -> Path:
    configured = os.getenv("HOSTING_DIGEST_SCRIPT", "").strip()
    return Path(configured).expanduser() if configured else _default_script_path()


def _handle_slash(raw_args: str) -> str:
    argv = raw_args.strip().split()
    if argv and argv[0] in {"help", "-h", "--help"}:
        return (
            "/hosting-digest — run the hosting cost digest now\n\n"
            "This runs the same digest script used by the scheduled hosting cost cron."
        )

    script = _script_path()
    if not script.exists():
        return f"hosting-digest script not found: {script}"

    env = os.environ.copy()
    env.setdefault("HERMES_HOME", str(Path.home() / ".hermes"))

    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            text=True,
            capture_output=True,
            timeout=DEFAULT_TIMEOUT_SECONDS,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"hosting-digest timed out after {DEFAULT_TIMEOUT_SECONDS}s."
    except Exception as exc:  # pragma: no cover - defensive fallback
        return f"hosting-digest failed to start: {exc}"

    output = (proc.stdout or proc.stderr or "").strip()
    if proc.returncode != 0:
        detail = output or f"exit code {proc.returncode}"
        return f"hosting-digest failed: {detail}"
    return output or "hosting-digest completed with no output."


def register(ctx) -> None:
    ctx.register_command(
        "hosting-digest",
        handler=_handle_slash,
        description="Run the hosting infrastructure cost digest now.",
    )
