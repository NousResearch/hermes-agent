"""Fresh-checkout source update completion and its stdlib-only parent transport.

Imported before a swap; executed by path from the selected tree afterward. The
parent never imports application helpers from the replacement checkout.
"""

from __future__ import annotations

import codecs
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)


def _exit_status(code: int) -> int:
    return code if code >= 0 else 128 - code


def _failed_result(request: dict, result_path: Path, code: int) -> int:
    code = _exit_status(code) or 1
    _write_json(result_path, {
        "schema": 1, "update_id": request["receipt"]["update_id"], "exit_code": code,
        "receipt": None, "windows_resume": None, "pm_receipt": request.get("pm_receipt"),
    })
    return code


def run_completion(request: dict) -> dict:
    """Wait for new code; zero exit without a correlated terminal result fails closed."""
    root = Path(request["source"])
    env = dict(os.environ, HERMES_HOME=request["home"], PYTHONUNBUFFERED="1")
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="hermes-completion-") as directory:
        request_path = Path(directory) / "request.json"
        result_path = Path(directory) / "result.json"
        request = {**request, "stdout_isatty": sys.stdout.isatty()}
        request["bytecode_cache"] = str(Path(directory) / "bytecode")
        _write_json(request_path, request)
        command = [sys.executable, "-I", "-S", "-u", "-X", f"pycache_prefix={request['bytecode_cache']}",
                   str(root / "hermes_cli/update_completion.py"),
                   str(request_path), str(result_path)]
        proc = subprocess.Popen(
            command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            **({"start_new_session": True} if os.name == "posix" else
               {"creationflags": subprocess.CREATE_NO_WINDOW}))
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while True:
                chunk = proc.stdout.read1(8192)
                sys.stdout.write(decoder.decode(chunk, final=not chunk))
                sys.stdout.flush()
                if not chunk:
                    break
            code = proc.wait()
        except BaseException as exc:
            # This group/retained process handle belongs exclusively to us.
            # Try to stop descendants before releasing the command's update lock.
            try:
                try:
                    if os.name == "posix":
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)  # windows-footgun: ok — os.name == "posix"; own isolated group
                        except ProcessLookupError:
                            pass
                    else:
                        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, timeout=10, check=True,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
                finally:
                    # A failed tree kill must not bypass retained-handle cleanup.
                    try:
                        proc.kill()
                    finally:
                        proc.wait(timeout=10)
            except BaseException as cleanup_error:
                exc.add_note("Completion cleanup failed; child processes may still be running.")
                raise exc from cleanup_error
            raise
        finally:
            proc.stdout.close()
        code = _exit_status(code)
        try:
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
            if result["schema"] != 1 or result["update_id"] != request["receipt"]["update_id"]:
                raise ValueError("completion response identity mismatch")
            if result["exit_code"] != code:
                raise ValueError("completion response disagrees with process exit")
            receipt = result.get("receipt")
            if receipt is not None and (
                receipt.get("update_id") != request["receipt"]["update_id"]
                or not receipt.get("finished_at")
                or (code == 0) != (receipt.get("outcome") == "success")
            ):
                raise ValueError("completion receipt does not attest this outcome")
            if code == 0 and receipt is None:
                raise ValueError("completion did not publish a terminal receipt")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"✗ Source update completion did not finish: {exc}")
            return {"exit_code": code or 1, "receipt": None, "windows_resume": None}
        return result


def _resume_receipt(data: dict) -> None:
    from hermes_cli import update_receipt

    # Hydrate the existing run, not a new receipt with a new identity/pre-update probe.
    receipt = object.__new__(update_receipt.UpdateReceipt)
    receipt.data = data
    receipt.correlation_id = data["update_id"]
    receipt.current_token = update_receipt._current.set(receipt)


def _read_terminal_receipt(request: dict) -> dict | None:
    directory = Path(request["home"]) / "logs/update_receipts"
    # Never latest.json: another profile/context may have finalized more recently.
    for path in directory.glob(f"update_*_{request['receipt']['update_id']}.json"):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if data.get("update_id") == request["receipt"]["update_id"] and data.get("finished_at"):
            return data
    return None


def _prepare(request: dict, request_path: Path, result_path: Path) -> int:
    import pm
    from pm import receipt
    from pm.environments import activation_environment, project_python

    root = Path(request["source"])
    update_id = request["receipt"]["update_id"]
    with receipt.worker_context(update_id):
        try:
            pm.sync_venv(explicit=True, project_root=root)
        finally:
            request["pm_receipt"] = receipt.last_for_update(update_id)
            _write_json(request_path, request)
    command = [str(project_python(root)),
               "-I", "-S", "-u", "-X", f"pycache_prefix={request['bytecode_cache']}",
               str(root / "hermes_cli/update_completion.py"),
               str(request_path), str(result_path), "--prepared"]
    # A second interpreter is mandatory: PM may have selected a different Python
    # and dependency graph. No application maintenance runs in this bootstrap.
    code = _exit_status(subprocess.call(command, cwd=root, env=activation_environment(root)))
    if not result_path.exists():
        return _failed_result(request, result_path, code)
    return code


def _complete_selected(request: dict) -> None:
    from hermes_cli import main, update_cmd, update_cmd_config
    from hermes_cli.source_completion import complete_source_checkout
    from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan

    root = Path(request["source"])
    main.PROJECT_ROOT = root
    update_cmd_config._LAST_SIBLING_SNAPSHOTS = request["sibling_snapshots"]
    plan_data = request["plan"]
    plan = None if plan_data is None else UpdatePlan(**{
        **plan_data, "runtimes": [RuntimeRecord(**row) for row in plan_data.get("runtimes", [])]})
    update_cmd._sweep_bytecode_after_update(request["branch"])
    # Launchers, products and post-build maintenance live in one place so an
    # install and an update cannot end in different states.
    complete = complete_source_checkout(
        root, desktop=request["desktop"], assume_yes=request["assume_yes"],
        gateway_mode=request["gateway_mode"], pre_update_snapshot_id=request["snapshot_id"],
        pre_update_version=request["pre_update_version"],
        completion_message=request.get("completion_message"),
        announce=None if request.get("completion_message") else "\n✓ Code updated!")
    # systemctl's KillMode=mixed fallback can kill this whole cgroup. Publish the
    # gateway watcher's status BEFORE that operation, and demote on later failure.
    if request["gateway_mode"]:
        update_cmd._write_gateway_update_exit_code(complete)
    if request.get("no_gateway_restart", False):
        from hermes_cli.update_receipt import record_skip

        record_skip("gateway_restart", "--no-gateway-restart: deferred, marker kept")
        print("→ Gateway restart deferred (--no-gateway-restart); restart gateways separately.")
        if not complete:
            raise SystemExit(1)
        return
    restart = update_cmd._restart_gateway_fleet_after_update(plan, request["gateway_mode"])
    update_cmd._resume_windows_gateways_and_merge_outcome(restart, request["windows_resume"], request["gateway_mode"])
    update_cmd._verify_fleet_after_update(
        restart, _pre_update_plan=plan, _windows_gateway_resume=request["windows_resume"], update_complete=complete)


class _ForwardedOutput:
    """The parent's pipe preserves its terminal's prompt policy and log mirror."""

    def __init__(self, stream, isatty: bool):
        self.stream, self.terminal = stream, isatty

    def isatty(self):
        return self.terminal

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _finish(request: dict, result_path: Path) -> int:
    from hermes_cli import update_receipt
    from pm.receipt import accept_worker_receipt

    _resume_receipt(request["receipt"])
    accept_worker_receipt(request.get("pm_receipt"), request["receipt"]["update_id"])
    code, reason = 0, "source update completion"
    try:
        _complete_selected(request)
    except SystemExit as exc:
        code = _exit_status(exc.code) if isinstance(exc.code, int) else 1
        reason = f"completion exited {code}"
    except BaseException as exc:
        code = _exit_status(exc.returncode) if isinstance(exc, subprocess.CalledProcessError) else 1
        reason = f"{type(exc).__name__}: {exc}"
        print(f"✗ Source update completion failed: {reason}")
    finally:
        if code and request["gateway_mode"]:
            from hermes_cli.update_cmd import _write_gateway_update_exit_code
            _write_gateway_update_exit_code(False)
        # The new interpreter owns recovery too. The original parent's atexit
        # token is updated from the response; it acts only if this process dies.
        try:
            from hermes_cli.update_cmd import _resume_windows_gateways_after_update
            _resume_windows_gateways_after_update(request["windows_resume"])
        except Exception as exc:
            code, reason = 1, f"Windows gateway recovery failed: {exc}"
            print(f"✗ {reason}")
        update_receipt.finalize_pending_update_receipt(code, reason)
        terminal_receipt = _read_terminal_receipt(request)
        if not terminal_receipt:
            code = code or 1
        _write_json(result_path, {
            "schema": 1, "update_id": request["receipt"]["update_id"], "exit_code": code,
            "receipt": terminal_receipt, "windows_resume": request["windows_resume"],
        })
    return code


def main() -> int:
    request_path, result_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    if request["schema"] != 1:
        raise ValueError("unsupported source completion request")
    root = Path(__file__).resolve().parents[1]
    if root != Path(request["source"]).resolve():
        raise ValueError("completion checkout does not match request")
    # -I deliberately ignores inherited PYTHONPATH during PM preparation.
    sys.path.insert(0, str(root))
    sys.stdout = _ForwardedOutput(sys.stdout, request.get("stdout_isatty", False))
    if "--prepared" in sys.argv[3:]:
        # Claim the selected generation's lease and process its .pth files only
        # after PM selection, before importing any application dependencies.
        from pm.environments import activate_dependencies
        activate_dependencies(root)
        return _finish(request, result_path)
    try:
        return _prepare(request, request_path, result_path)
    except BaseException as exc:
        # PM failed before application dependencies were ready. Leave the parent
        # receipt and paused-gateway obligation intact for boundary recovery.
        print(f"✗ Source update preparation failed: {exc}")
        code = exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1
        return _failed_result(request, result_path, code)


if __name__ == "__main__":
    raise SystemExit(main())
