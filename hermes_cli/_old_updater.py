"""The newly imported escape hatch from an already-running historical updater.

This module must remain stdlib-only: the parent still has the old interpreter,
modules and native extensions. Only the child imports the updated application.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, NoReturn


_result: int | None = None


def _historical_context() -> tuple[dict, list[dict], Any]:
    """Carry data already held by old frames, without importing their modules.

    Older callers did not pass a continuation object. Read only the historical
    updater's known local names; do not serialize frames or arbitrary globals.
    """
    names = ("had_desktop_app_before_update", "pre_update_snapshot_id",
             "pre_update_version", "gateway_mode", "assume_yes", "_pre_update_plan")
    found = {}
    resumes = []
    restart_update = None
    frame = sys._getframe(1)
    try:
        while frame is not None:
            # Capture hooks can be imported BEFORE the old updater has pulled.
            # A declared-but-unassigned version in its innermost known frame
            # proves early entry; a None value is still a post-capture value.
            if (restart_update is None
                    and frame.f_globals.get("__name__") in ("hermes_cli.update_cmd", "hermes_cli.main")
                    and frame.f_code.co_name in ("_cmd_update_impl", "cmd_update")
                    and "pre_update_version" in frame.f_code.co_varnames):
                restart_update = "pre_update_version" not in frame.f_locals
            for name in names:
                if name not in found and name in frame.f_locals:
                    found[name] = frame.f_locals[name]
            token = frame.f_locals.get("_windows_gateway_resume")
            if isinstance(token, dict) and not any(token is item for item in resumes):
                resumes.append(token)
            frame = frame.f_back
    finally:
        del frame
    receipt_module = sys.modules.get("hermes_cli.update_receipt")
    receipt_slot = vars(receipt_module).get("_current") if receipt_module else None
    get_current = getattr(receipt_slot, "get", None)
    current = get_current() if get_current is not None else receipt_slot
    receipt = getattr(current, "data", None)
    plan = found.get("_pre_update_plan")
    if dataclasses.is_dataclass(plan) and not isinstance(plan, type):
        plan = dataclasses.asdict(plan)
    elif not isinstance(plan, dict):
        plan = receipt.get("plan") if isinstance(receipt, dict) else None
    return {
        "restart_update": restart_update is True,
        "desktop": bool(found.get("had_desktop_app_before_update", False)),
        "pre_update_snapshot_id": found.get("pre_update_snapshot_id"),
        "pre_update_version": found.get("pre_update_version"),
        "gateway_mode": bool(found.get("gateway_mode", "--gateway" in sys.argv)),
        "assume_yes": bool(found.get("assume_yes", "--yes" in sys.argv)),
        "windows_resume": resumes[0] if resumes else None,
        "plan": plan,
        "receipt": receipt if isinstance(receipt, dict) else None,
    }, resumes, receipt_slot


def _run_child(request: dict) -> tuple[int, dict]:
    """One JSON exchange; the old process never imports the updated graph."""
    root = Path(__file__).resolve().parents[1]
    home = os.environ.get("HERMES_HOME")
    constants = sys.modules.get("hermes_constants")
    override = vars(constants).get("_HERMES_HOME_OVERRIDE") if constants else None
    if override is not None:
        value = override.get()
        if isinstance(value, (str, Path)) and value:
            home = str(value)
    request.update(root=str(root), home=home, argv=list(sys.argv))
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTHON", "UV_")) and key != "VIRTUAL_ENV"}
    if home:
        env["HERMES_HOME"] = home
    env.setdefault("HERMES_UPDATE_HANDOFF_PID", str(os.getpid()))
    with tempfile.TemporaryDirectory(prefix="hermes-update-takeover-") as directory:
        context = Path(directory) / "request.json"
        result = Path(directory) / "result.json"
        context.write_text(json.dumps(request), encoding="utf-8")
        child = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-X", "utf8", str(root / "hermes_cli/_update_takeover.py"),
             str(context), str(result)], cwd=root, env=env,
        )
        completed = json.loads(result.read_text(encoding="utf-8")) if result.is_file() else {}
        if not isinstance(completed, dict):
            raise ValueError("invalid update takeover acknowledgement")
        return child.returncode if child.returncode >= 0 else 1, completed


def stop_for_relaunch() -> NoReturn:
    """Finish in a fresh child, then exit rather than resume an old fallback.

    The historical name is retained. A finally/atexit path can call another
    shim while unwinding; it must receive the same result, not start again.
    """
    global _result
    if _result is not None:
        raise SystemExit(_result)
    _result = 1
    try:
        request, resumes, receipt_slot = _historical_context()
        print("Completing the update with the new Hermes updater…", file=sys.stderr, flush=True)
        _result, completed = _run_child(request)
        if completed.get("resume_handled"):
            for token in resumes:
                token["resume_needed"] = False
        if completed.get("receipt_handled"):
            if hasattr(receipt_slot, "set"):
                receipt_slot.set(None)
            elif receipt_slot is not None:
                vars(sys.modules["hermes_cli.update_receipt"])["_current"] = None
    except (OSError, ValueError, TypeError) as exc:
        _result = 1
        print(f"Update takeover failed: {exc}", file=sys.stderr, flush=True)
    raise SystemExit(_result)


def relaunch_stopped_serves(token: dict) -> None:
    """A historical atexit token is separate work, not another whole update."""
    if not token.get("pending"):
        return
    code = 1
    try:
        code, completed = _run_child({"stopped_serves": token})
        if completed.get("serves_handled") is True:
            token["pending"] = False
    except (OSError, ValueError, TypeError) as exc:
        print(f"Stopped serve recovery failed: {exc}", file=sys.stderr, flush=True)
    if code or token.get("pending"):
        print("Stopped serve recovery did not complete; restart the affected "
              "serve/dashboard backends manually.", file=sys.stderr, flush=True)
