"""Isolated process entrypoint for side-effecting cron agent execution."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _touch_activity_pulse() -> None:
    """Refresh the metadata-only parent supervision pulse when configured."""
    pulse = os.getenv("HERMES_CRON_WORKER_PULSE", "").strip()
    if not pulse:
        return
    path = Path(pulse)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _read_request() -> dict[str, Any]:
    """Read one JSON request from the anonymous stdin pipe."""
    payload = json.loads(sys.stdin.read())
    if not isinstance(payload, dict):
        raise ValueError("cron worker request must be an object")
    return payload


def main() -> int:
    """Run one in-process worker and emit a nonce-bound pipe response."""
    request = _read_request()
    home = Path(str(request["home"])).expanduser().resolve()
    job = request["job"]
    nonce = str(request["nonce"])
    if not isinstance(job, dict) or not nonce:
        raise ValueError("cron worker request is incomplete")

    # Parent also sets this before process start; assign explicitly so direct
    # module invocation cannot accidentally bind state to another profile.
    os.environ["HERMES_HOME"] = str(home)
    _touch_activity_pulse()

    from agent.secret_scope import (
        build_profile_secret_scope,
        reset_secret_scope,
        set_secret_scope,
    )
    from cron.jobs import use_cron_store
    from cron.scheduler import _CRON_WORKER_RESULT_PREFIX, run_job

    scope_token = set_secret_scope(build_profile_secret_scope(home))
    try:
        with use_cron_store(home):
            result = run_job(job)
    finally:
        reset_secret_scope(scope_token)
    response = json.dumps({"result": list(result)}, ensure_ascii=False)
    sys.stdout.write(f"{_CRON_WORKER_RESULT_PREFIX}{nonce}:{response}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
