"""Frozen compat surface for releases that finish `hermes update` via the post-swap hand-off.

Releases from 2026-09-16 (94ced1a2b2) lazily import ``hermes_cli.update_handoff``
from the NEW tree after the checkout swap. Like every other retired updater
hook, this module keeps that import working by routing into the historical
takeover (``hermes_cli._old_updater`` → ``_update_takeover.py``) instead of
re-executing ``hermes update --post-swap``; the pulled tree is never imported
into the pre-pull interpreter.

Guarded by tests/compat/old_updater_surface.json — do not remove a public name
without regenerating the frozen surface and proving no shipped release loads it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Set on the post-swap child: the receipt header says "continued", the lock is the parent's.
POST_SWAP_ENV = "HERMES_UPDATE_POST_SWAP"


def is_post_swap_child() -> bool:
    return os.environ.get(POST_SWAP_ENV) == "1"


def write_handoff(payload: dict[str, Any]) -> Path:
    """Persist the post-swap payload under HERMES_HOME; returns its path."""
    from hermes_constants import get_hermes_home

    directory = get_hermes_home() / "logs" / "update_receipts"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"post_swap_{os.getpid()}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def read_handoff(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"post-swap hand-off {path} is not a JSON object")
    return payload


def post_swap_python() -> Path:
    """Interpreter for the child: the project venv's python when this process runs from (or
    under) the project's Windows console shim — the shim can never be re-executed on Windows
    because it holds itself open for the whole process lifetime (#88838, #89599) — else the
    running interpreter."""
    if sys.platform != "win32":
        return Path(sys.executable)
    from hermes_cli._launchers import _is_windows
    from hermes_constants import project_venv_dir
    from pm.environments import venv_python

    venv_dir = project_venv_dir(Path(__file__).resolve().parents[1])
    if venv_dir is not None and _is_windows():
        candidate = venv_python(venv_dir, windows=True)
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def post_swap_command(handoff_path: Path, argv_tail: list[str]) -> list[str]:
    """``python -m hermes_cli.main update <original flags> --post-swap <file>``.

    Historical command shape; kept so a printed manual-continuation line reads
    exactly as older releases expect. The takeover child this tree actually
    spawns (see :func:`continue_update_in_fresh_interpreter`) ignores the flag.
    """
    return [str(post_swap_python()), "-m", "hermes_cli.main", "update", *argv_tail,
            "--post-swap", str(handoff_path)]


def post_swap_child_env() -> dict[str, str]:
    """Environment for the child. ``HERMES_UPDATE_REEXEC`` marks it as already off the Windows
    shim (no second re-exec at the sync boundary). The lock hand-off pid is only claimed when
    nobody upstream (Tauri/Electron updater) already named theirs."""
    from hermes_cli.update_lock import HANDOFF_PID_ENV

    env = {**os.environ, POST_SWAP_ENV: "1", "HERMES_UPDATE_REEXEC": "1"}
    env.setdefault(HANDOFF_PID_ENV, str(os.getpid()))
    return env


def _takeover_request(payload: dict[str, Any], argv_tail: list[str] | None) -> dict[str, Any]:
    """Map a historical post-swap payload onto the takeover request schema.

    Payloads written before the takeover seam (e.g. release 0.21.3) name the
    desktop flag and the Windows resume token differently and carry no
    assume_yes at all; `update_finish.py` reads the new names with bare
    subscriptions, so an unmapped payload KeyErrors mid-completion with the
    checkout already swapped. New-schema payloads pass through unchanged.
    """
    request = dict(payload)
    request.setdefault("desktop", bool(payload.get("had_desktop_app_before_update", False)))
    request.setdefault("windows_resume", payload.get("windows_gateway_resume"))
    request.setdefault("assume_yes", "--yes" in (argv_tail or []))
    request.setdefault("restart_update", False)
    return request


def continue_update_in_fresh_interpreter(payload: dict[str, Any], *, argv_tail: list[str] | None = None) -> int | None:
    """Run the post-swap tail in a child interpreter on the pulled code.

    The old caller's payload carries the same fields the completion request
    needs (receipt, plan, windows_resume, gateway_mode, ...); ``argv_tail`` is
    accepted and ignored because the takeover tail does not re-parse update
    flags. Returns the child's exit code, or ``None`` when no child could be
    started (the caller then owns the failure bookkeeping). Ctrl-C semantics
    mirror the historical hand-off: wait for the child's cleanup instead of
    subprocess.run's kill-on-interrupt.
    """
    from hermes_cli._old_updater import _run_child

    handoff_path = write_handoff(payload)
    request = _takeover_request(payload, argv_tail)
    cmd = post_swap_command(handoff_path, argv_tail or [])
    print(f"→ Post-swap hand-off: completing the update in a fresh interpreter ({handoff_path})")
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        code, _completed = _run_child(request)
        return int(code)
    except OSError as exc:
        print(f"  ⚠ Could not start the post-update interpreter: {exc}")
        print("  The code update is applied. Finish it with:")
        print(f"    {subprocess.list2cmdline(cmd)}")
        return None
