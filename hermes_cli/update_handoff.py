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
from copy import deepcopy
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from hermes_cli.update_cmd_common import _best_effort

logger = logging.getLogger("hermes_cli.update_cmd")

# Set on the post-swap child: the receipt header says "continued", the lock is the parent's.
POST_SWAP_ENV = "HERMES_UPDATE_POST_SWAP"
# Set on a child spawned DETACHED off the Windows console shim: the pid of the process that
# still holds ``hermes.exe`` open (the launcher, or the interpreter it ran). The child waits
# for it before it touches the venv —
# the shim quarantine is a single rename with sub-second retries, and the reporter's runs
# (#101600) reached it while the parent was still alive (relaunching gateways, or just
# tearing down), so the rename failed and the whole install was deferred.
SHIM_PARENT_PID_ENV = "HERMES_UPDATE_SHIM_PARENT_PID"
# Legacy re-exec (``_reexec_dependency_sync_off_windows_shim``, no hand-off file): the Windows
# pause token travels here so the child resumes exactly the fleet the parent stopped instead
# of re-running pause discovery — which found the parent's freshly relaunched gateway before
# its pid file existed and force-killed it as "unmapped" (#101600).
GATEWAY_RESUME_ENV = "HERMES_UPDATE_GATEWAY_RESUME"

SHIM_PARENT_EXIT_TIMEOUT_SECONDS = 30.0


def _json_default(value: Any):
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _validated_pinned_payload(payload: dict[str, Any], *, intent: dict[str, Any] | None = None) -> dict[str, Any]:
    """Copy and validate the T4 identity before it can cross the process boundary."""
    if not isinstance(payload, dict):
        raise TypeError("post-swap hand-off must be an object")
    body = deepcopy(payload)
    candidate = intent if intent is not None else body.get("pinned_intent")
    if candidate is None:
        return body
    from hermes_cli.update_target import validate_update_intent

    frozen = validate_update_intent(candidate)
    receipt = body.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError("pinned handoff receipt is required")
    receipt_intent = receipt.get("update_intent")
    if validate_update_intent(receipt_intent) != frozen:
        raise ValueError("handoff receipt intent mismatch")
    receipt["update_intent"] = deepcopy(frozen)
    if "target_intent" in body and body["target_intent"] != frozen:
        raise ValueError("handoff target intent mismatch")
    redundant = {
        "requested_sha": frozen["target"],
        "install_id": frozen["install_id"],
        "correlation_id": frozen["correlation_id"],
        "prior_sha": frozen["prior_sha"],
    }
    for key, expected in redundant.items():
        if key in body and body[key] != expected:
            raise ValueError(f"handoff {key} mismatch")
    if body.get("branch") != frozen["branch"] or body.get("pre_pull_sha") != frozen["prior_sha"]:
        raise ValueError("handoff operational intent mismatch")
    body["target_intent"] = deepcopy(frozen)
    body["pinned_intent"] = deepcopy(frozen)
    return body


def write_handoff(payload: dict[str, Any], *, intent: dict[str, Any] | None = None) -> Path:
    """Persist a detached copy of the post-swap payload under HERMES_HOME."""
    from hermes_constants import get_hermes_home

    body = _validated_pinned_payload(payload, intent=intent)
    directory = get_hermes_home() / "logs" / "update_receipts"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"post_swap_{os.getpid()}.json"
    if body.get("pinned_intent") is not None:
        body["_handoff_ack_path"] = str(path.with_suffix(".ack"))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(body, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)
    return path


def read_handoff(path: str | Path) -> dict[str, Any]:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"post-swap hand-off {path} is not a JSON object")
    return _validated_pinned_payload(body)


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


def detached_shim_child_env(env: dict[str, str], gateway_resume: dict | None = None) -> dict[str, str]:
    """Env for a child that outlives this shim-run process: names the process holding the shim
    open (the ``hermes.exe`` launcher above us when psutil sees it, else this interpreter) and,
    for the legacy re-exec (no hand-off file), carries the Windows pause token."""
    from hermes_cli.main_install_repair import _windows_shim_holder_pid

    env = {**env, SHIM_PARENT_PID_ENV: str(_windows_shim_holder_pid())}
    if gateway_resume is not None:
        env[GATEWAY_RESUME_ENV] = json.dumps(gateway_resume, default=_json_default)
    return env


def adopt_handed_off_gateway_resume() -> dict | None:
    """The pause token a shim-run parent handed to this legacy re-exec child, or ``None``.
    Consumed on read so nothing this run spawns (relaunched gateways) inherits it."""
    raw = os.environ.pop(GATEWAY_RESUME_ENV, None)
    if not raw:
        return None
    try:
        token = json.loads(raw)
    except ValueError:
        logger.warning("Ignoring malformed %s from the update hand-off", GATEWAY_RESUME_ENV)
        return None
    return token if isinstance(token, dict) else None


def wait_for_shim_parent_exit(timeout: float = SHIM_PARENT_EXIT_TIMEOUT_SECONDS) -> bool:
    """Block until the shim-run parent named in :data:`SHIM_PARENT_PID_ENV` has exited.

    Returns True when it is gone (or none was named), False when it outlived ``timeout``; the
    caller proceeds either way — the strict shim quarantine refuses if it still holds the exe.
    A pid younger than this process is a recycled pid, never our parent. Consumed on read.
    """
    raw = os.environ.pop(SHIM_PARENT_PID_ENV, "")
    try:
        pid = int(raw.strip())
    except ValueError:
        return True
    if pid <= 0 or pid == os.getpid():
        return True
    import psutil

    try:
        parent = psutil.Process(pid)  # pins (pid, create_time): a recycled pid reads as gone
        if parent.create_time() > psutil.Process().create_time():
            return True
        deadline = time.monotonic() + timeout
        while parent.is_running() and parent.status() != psutil.STATUS_ZOMBIE:
            if time.monotonic() >= deadline:
                print(f"  ⚠ The hermes.exe process that started this update (PID {pid}) is still "
                      f"running after {int(timeout)}s; continuing.")
                return False
            time.sleep(0.2)
    except psutil.Error:
        pass
    return True


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
    # Keep the path in the in-memory envelope so the parent can distinguish a
    # child that imported and consumed the receipt from one that never imported.
    payload["_handoff_path"] = str(handoff_path)
    payload["_handoff_ack_path"] = str(handoff_path.with_suffix(".ack"))
    payload["_handoff_detached"] = False
    cmd = post_swap_command(handoff_path, argv_tail)
    env = post_swap_child_env()
    logger.debug("Post-swap hand-off → %s", subprocess.list2cmdline(cmd))
    sys.stdout.flush()
    sys.stderr.flush()

    if _running_from_windows_shim():
        try:
            subprocess.Popen(cmd, env=detached_shim_child_env(env), stdin=subprocess.DEVNULL)
        except OSError as exc:
            _print_manual_continuation(cmd, exc)
            return None
        print("→ Windows: hermes.exe cannot replace itself while it runs; the update")
        print("  continues under the venv Python. The code update is already applied and")
        print("  this shell returns right away; the install finishes below.")
        payload["_handoff_detached"] = True
        return 0

    try:
        child = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL)
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
