"""Post-swap hand-off: finish ``hermes update`` in an interpreter born on the pulled code.

``hermes update`` starts in an interpreter that imported the PRE-pull tree. Once ``git merge``
(or the ZIP swap) has replaced the checkout, every later phase — dependency sync, Node/web/
Desktop builds, maintenance, fleet restart, verification, receipt — used to keep running in
that stale process and lazily import NEW source into an OLD ``sys.modules`` graph. Any rename
between the two commits then surfaced as an ``ImportError``/``AttributeError`` inside the
updater itself, after the code swap had already succeeded (#87134, #112465, #112558, #112604
and their siblings). Module purges and targeted reloads only ever moved the crash to the next
unpurged module.

The permanent shape: the pre-pull process stops at the swap, writes everything the tail needs
into a hand-off file and re-executes ``hermes update --post-swap <file>`` under the venv
interpreter. The child imports exclusively from the pulled tree, resumes the open receipt and
owns the rest of the run; the parent relays its exit code. Nothing in the updater runs pulled
code inside a pre-pull interpreter any more, so there is no stale-symbol class left to isolate.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from hermes_cli.update_cmd_common import _best_effort

logger = logging.getLogger("hermes_cli.update_cmd")

# Set on the post-swap child: the receipt header says "continued", the lock is the parent's.
POST_SWAP_ENV = "HERMES_UPDATE_POST_SWAP"


def _json_default(value: Any):
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_handoff(payload: dict[str, Any]) -> Path:
    """Persist the post-swap payload under HERMES_HOME; returns its path."""
    from hermes_constants import get_hermes_home

    directory = get_hermes_home() / "logs" / "update_receipts"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"post_swap_{os.getpid()}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def read_handoff(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"post-swap hand-off {path} is not a JSON object")
    return payload


def is_post_swap_child() -> bool:
    return os.environ.get(POST_SWAP_ENV) == "1"


def _running_from_windows_shim() -> bool:
    from hermes_cli.main_install_repair import _windows_shim_in_process_chain

    return _windows_shim_in_process_chain() is not None


def post_swap_python() -> Path:
    """Interpreter for the child: the project venv's python (the console shim can never be
    re-executed on Windows — it holds itself open), else the running interpreter."""
    from hermes_cli.main_install_repair import _windows_shim_in_process_chain

    shim = _windows_shim_in_process_chain()
    if shim is not None:
        from hermes_constants import venv_python_path

        candidate = venv_python_path(shim.parent.parent, windows=True)
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def post_swap_command(handoff_path: Path, argv_tail: list[str]) -> list[str]:
    """``python -m hermes_cli.main update <original flags> --post-swap <file>``."""
    return [str(post_swap_python()), "-m", "hermes_cli.main", "update", *argv_tail, "--post-swap", str(handoff_path)]


def post_swap_child_env() -> dict[str, str]:
    """Environment for the child. ``HERMES_UPDATE_REEXEC`` marks it as already off the Windows
    shim (no second re-exec at the sync boundary; ``cmd_update`` hard-exits it when its receipt
    is durable instead of waiting on a leftover non-daemon thread). The lock hand-off pid is
    only claimed when nobody upstream (Tauri/Electron updater) already named theirs."""
    from hermes_cli.main_install_repair import _UPDATE_REEXEC_ENV
    from hermes_cli.update_lock import HANDOFF_PID_ENV

    env = {**os.environ, POST_SWAP_ENV: "1", _UPDATE_REEXEC_ENV: "1"}
    env.setdefault(HANDOFF_PID_ENV, str(os.getpid()))
    return env


def _print_manual_continuation(cmd: list[str], exc: OSError) -> None:
    logger.warning("Post-swap hand-off could not start: %s", exc)
    print(f"  ⚠ Could not start the post-update interpreter: {exc}")
    print("  The code update is applied. Finish it with:")
    print(f"    {subprocess.list2cmdline(cmd)}")


def continue_update_in_fresh_interpreter(payload: dict[str, Any], *, argv_tail: list[str]) -> int | None:
    """Run the post-swap tail in a child interpreter on the pulled code.

    Returns the child's exit code, or ``None`` when no child could be started (the caller then
    owns the failure bookkeeping). The parent has already detached from the receipt (the child
    resumes it) and only relays the exit code. On Windows, when this process runs from
    ``hermes.exe``, the child cannot be awaited: the shim is one of the files the dependency
    sync must replace and it stays open for as long as this process lives (#88838, #89599).
    That case spawns detached, prints where the run continues and returns 0 — the child prints
    its own result and ``--gateway`` writes the true exit code to ``.update_exit_code``.

    Ctrl-C reaches parent and child together; the child owns the receipt and the Windows
    gateway resume, so the parent keeps waiting for it instead of ``subprocess.run``'s
    kill-on-interrupt, which would cut it off mid-cleanup.
    """
    handoff_path = write_handoff(payload)
    cmd = post_swap_command(handoff_path, argv_tail)
    env = post_swap_child_env()
    logger.debug("Post-swap hand-off → %s", subprocess.list2cmdline(cmd))
    sys.stdout.flush()
    sys.stderr.flush()

    if _running_from_windows_shim():
        try:
            subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL)
        except OSError as exc:
            _print_manual_continuation(cmd, exc)
            return None
        print("→ Windows: hermes.exe cannot replace itself while it runs; the update")
        print("  continues under the venv Python. The code update is already applied and")
        print("  this shell returns right away; the install finishes below.")
        return 0

    try:
        child = subprocess.Popen(cmd, env=env, stdin=sys.stdin)
    except OSError as exc:
        _print_manual_continuation(cmd, exc)
        return None
    try:
        return int(child.wait())
    except KeyboardInterrupt:
        print("\n  Interrupted — waiting for the update child to finish its cleanup...")
        try:
            return int(child.wait(timeout=60))
        except subprocess.TimeoutExpired:
            child.terminate()
            return 130
        except KeyboardInterrupt:
            child.terminate()
            return 130
