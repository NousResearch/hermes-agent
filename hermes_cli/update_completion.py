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


def _write_bootstrap_result(request: dict, result_path: Path, code: int, receipt: dict | None) -> int:
    """The bootstrap child's answer; windows_resume stays the parent's token (and its atexit net)."""
    _write_json(result_path, {
        "schema": 1, "update_id": request["receipt"]["update_id"], "exit_code": code,
        "receipt": receipt, "windows_resume": None, "pm_receipt": request.get("pm_receipt"),
    })
    return code


def run_completion(request: dict) -> dict:
    """Wait for new code; zero exit without a correlated terminal result fails closed."""
    root = Path(request["source"])
    # The child below runs in a new session without a controlling terminal, so
    # this is the last point where sudo can ask for a password. Run the new
    # tree's pre-install as its own process (this parent stays stdlib-only); a
    # tree without it, or any failure, just leaves the in-lock repair to report.
    if sys.platform.startswith("linux"):
        subprocess.run([sys.executable, "-I", "-S", "-B", "-c",
                        "import sys; sys.path.insert(0, sys.argv[1]); "
                        "from pm.libatomic import install_before_lock; install_before_lock()", str(root)],
                       cwd=root, stdout=None, stderr=subprocess.DEVNULL, check=False)
    env = dict(os.environ, HERMES_HOME=request["home"], PYTHONUNBUFFERED="1")
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="hermes-completion-") as directory:
        request_path = Path(directory) / "request.json"
        result_path = Path(directory) / "result.json"
        request = {**request, "stdout_isatty": sys.stdout.isatty()}
        request["bytecode_cache"] = str(Path(directory) / "bytecode")
        _write_json(request_path, request)
        command = [sys.executable, "-I", "-S", "-u", "-X", "utf8", "-X", f"pycache_prefix={request['bytecode_cache']}",
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
    from hermes_cli import update_receipt

    # The ROOT home's receipts (the run's own file), never latest.json: another
    # profile/context may have finalized more recently.
    update_id = request["receipt"]["update_id"]
    try:
        for path in update_receipt._receipt_dir().glob(f"update_*_{update_id}.json"):
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if data.get("update_id") == update_id and data.get("finished_at"):
                return data
    except (OSError, ValueError):
        pass
    # The store refused the terminal write after the commit point (loudly): this process still
    # holds the correlated terminal record it finalized, so the exit status never turns into 1.
    return update_receipt.finalized_receipt(update_id)


def _running_record(request: dict) -> dict | None:
    from hermes_cli.update_receipt import _receipt_dir

    for path in _receipt_dir().glob(f"update_*_{request['receipt']['update_id']}.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if data.get("update_id") == request["receipt"]["update_id"] and data.get("outcome") == "running":
            data.pop("writer_pid", None)
            return data
    return None


def _owed_user_action(request: dict) -> str | None:
    """The unsettled-autostash notice (#122557) when this run parked the user's local changes.

    ``_complete_source_update`` hands it over as the completion message, and a completion message
    that is not a ``✓`` line is never a success (``_print_verified_update_completion``).
    """
    message = request.get("completion_message") or ""
    return message if message and not message.startswith("✓") else None


def _record_owed_user_action(request: dict) -> bool:
    """Land the parked local changes on the receipt; True when nothing is owed to the user."""
    from hermes_cli.update_receipt import record_user_action

    notice = _owed_user_action(request)
    if notice:
        record_user_action("local_changes", notice)
    return notice is None


def _publish_gateway_success(request: dict) -> None:
    """The gateway /update watcher's status: the code is committed (same as _complete_selected).

    Guarded like ``update_cmd_fleet._write_gateway_update_exit_code`` (which this stdlib-only
    process cannot import): an unwritable home must not turn the committed run into an error
    before its result is written (review P1). The watcher then falls back to its own timeout.
    """
    if not request.get("gateway_mode"):
        return
    try:
        (Path(request["home"]) / ".update_exit_code").write_text("0", encoding="utf-8")
    except OSError as exc:
        print(f"  ⚠ Could not publish the gateway's update status: {exc}")


#: The one whole line the Desktop hand-off scripts match (scripts/desktop-update/posix.sh and
#: windows.ps1); ``source_build.build_update_products`` prints the same prefix when the build fails.
DESKTOP_BUILD_OWED = "Desktop app build owed:"


def _report_desktop_build_owed(request: dict, record: dict | None, reason: str) -> None:
    """Name the owed Desktop build when this run never got the build to its own verdict (F6, review P4).

    The builder prints the line only when the build itself fails; a run that never reached it
    (dependencies owed, the tail refused by the lock, a completion process lost) would otherwise end
    as a plain "Update complete." in the hand-off. A ``build`` stage on the run's record means the
    build ran and printed this line itself when the Desktop app failed.
    """
    if not request.get("desktop"):
        return
    if any(isinstance(mark, dict) and mark.get("name") == "build" for mark in (record or {}).get("stages") or ()):
        return
    print(f"  {DESKTOP_BUILD_OWED} {reason}", flush=True)


def _settle_after_commit(request: dict, result_path: Path, step: str, reason: str) -> int:
    """The tree already moved, so a failure here is owed work, never a failed update (C3, A6).

    Runs in the bootstrap interpreter (stdlib + the new tree) when dependency preparation failed or
    the prepared child died without a result. The tail obligation stays armed, so the next launch
    syncs the dependencies and finishes the tail; the run's receipt is a success that names the
    follow-up, so nothing reports "still on the previous version" while the tree is new.
    """
    from hermes_cli import update_receipt
    from hermes_cli.venv_sync import arm_completion

    try:
        arm_completion(Path(request["source"]))
    except OSError as exc:  # the next launch still sees stale dependencies and syncs them
        print(f"  ⚠ Could not record the owed source-update tail: {exc}")
    receipt = _read_terminal_receipt(request)  # a prepared child that finalized, then died
    if receipt is None:
        # A prepared child that died mid-tail persisted stages the parent's snapshot lacks.
        record = _running_record(request) or request["receipt"]
        _report_desktop_build_owed(request, record, f"{step} did not finish")
        _resume_receipt(record)
        update_receipt.record_stage("deps" if step == "dependencies" else "build", "failed")
        update_receipt.record_followup(step, reason, retry="dependencies not installed yet — the next launch retries"
                                       if step == "dependencies" else "the next launch or `hermes update` retries it")
        if not _record_owed_user_action(request):
            print(_owed_user_action(request))  # the completion child that would print it never ran
        update_receipt.finalize_pending_update_receipt(0, f"{step} owed after the code was updated")
        receipt = _read_terminal_receipt(request)
    code = 0 if receipt is not None and receipt.get("outcome") == "success" else 1
    if code == 0:
        _publish_gateway_success(request)
    return _write_bootstrap_result(request, result_path, code, receipt)


def _prepare(request: dict, request_path: Path, result_path: Path) -> int:
    import pm
    from pm import receipt
    from pm.client import ensure_tools_for_sync
    from pm.environments import activation_environment, project_python

    root = Path(request["source"])
    update_id = request["receipt"]["update_id"]
    from hermes_cli.venv_sync import (
        arm_completion, collect_superseded_generations, refuse_foreign_owned_venv,
    )

    refuse_foreign_owned_venv(root)
    arm_completion(root)
    from hermes_cli.gitlock import convert_treeless_checkout_first
    convert_treeless_checkout_first(root)
    with receipt.worker_context(update_id):
        try:
            # This file runs from the new tree, so its lockfile carries the new
            # pins; tools (incl. bumped uv/python) land before the sync uses them.
            ensure_tools_for_sync()
            # An update never fails because of a plugin: misfits are disabled and reported.
            pm.sync_venv(explicit=True, project_root=root, evict_incompatible_plugins=True)
            collect_superseded_generations(root)
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
        return _settle_after_commit(request, result_path, "completion",
                                    f"the completion process exited {code} without a result")
    return code


def _complete_selected(request: dict) -> bool:
    """Everything after the commit point. No step failure fails the update (contract C3).

    Each step is independent; a failed one prints ``⚠``, lands on the receipt as a follow-up
    and keeps its own obligation armed (``source-completion-pending`` for the tail, the fleet
    restart obligation for gateways), so the next launch or ``hermes update`` retries it.
    Returns False only when the user's local changes were left parked in the stash: nothing
    retries that, so the run is ``partial`` and exits 1 (#122557), never "Update complete".
    """
    from hermes_cli import main, update_cmd, update_cmd_config
    from hermes_cli.source_completion import complete_source_checkout
    from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan
    from hermes_cli.update_receipt import (
        TAIL_FOLLOWUPS, record_build_stage, record_followup, record_skip, record_stage)

    root = Path(request["source"])
    main.PROJECT_ROOT = root
    complete = _record_owed_user_action(request)
    update_cmd_config._LAST_SIBLING_SNAPSHOTS = request["sibling_snapshots"]
    plan_data = request["plan"]
    plan = None if plan_data is None else UpdatePlan(**{
        **plan_data, "runtimes": [RuntimeRecord(**row) for row in plan_data.get("runtimes", [])]})
    update_cmd._sweep_bytecode_after_update(request["branch"])
    # Launchers, products and post-build maintenance live in one place so an
    # install and an update cannot end in different states.
    followups: list[tuple[str, str]] = []
    # Exit status and runtime safety are separate facts (R8): an unsafe SQLite runtime keeps the
    # committed update at exit 0 (reported as the ``sqlite_runtime`` follow-up), but fleet
    # verification still needs the real verdict so it never auto-migrates the gateway topology
    # on a runtime that can corrupt sessions. Unknown (the tail raised) is not proven safe.
    runtime_safe = False
    try:
        runtime_safe = complete_source_checkout(
            root, desktop=request["desktop"], assume_yes=request["assume_yes"],
            gateway_mode=request["gateway_mode"], pre_update_snapshot_id=request["snapshot_id"],
            pre_update_version=request["pre_update_version"],
            completion_message=request.get("completion_message"),
            announce=None if request.get("completion_message") else "\n✓ Code updated!",
            followups=followups)
    except (Exception, SystemExit) as exc:  # health: allow BLE001 -- e.g. the shared update lock refused the tail
        reason = str(exc) or type(exc).__name__
        # The tail's steps catch their own failures, so a raise here means the build never ran.
        _report_desktop_build_owed(request, None, "the source-update tail did not run")
        record_followup("completion", reason)
        followups.append(("completion", reason))
    tail_owed = any(step in TAIL_FOLLOWUPS for step, _ in followups)
    record_build_stage(followups)
    if not tail_owed:
        from hermes_cli.venv_sync import clear_completion
        clear_completion(root)
    # systemctl's KillMode=mixed fallback can kill this whole cgroup. Publish the
    # gateway watcher's status BEFORE that operation: the code is committed, so it is 0 unless
    # the user's own changes are still parked.
    if request["gateway_mode"]:
        update_cmd._write_gateway_update_exit_code(complete)
    if request.get("no_gateway_restart", False):
        record_skip("gateway_restart", "--no-gateway-restart: deferred, marker kept")
        record_stage("restart", "skipped")
        print("→ Gateway restart deferred (--no-gateway-restart); restart gateways separately.")
        return complete
    skip = update_cmd._fleet_restart_skip_reason(plan)
    if skip and update_cmd._pending_fleet_restart_needed():
        # A host already stamped "restarted" for this SHA whose fleet is still off the checkout
        # (a stale sibling, a failed resume) gets the restart again instead of a dead end.
        print(f"  → Gateway restart not skipped ({skip}): gateways are still off the checkout code.")
        skip = None
    if skip:
        record_skip("gateway_restart", skip)
        record_stage("restart", "skipped")
        print(f"  ✓ Gateway restart skipped: {skip}.")
        return complete
    restart = update_cmd._restart_gateway_fleet_after_update(plan, request["gateway_mode"])
    record_stage("restart", "failed" if getattr(restart, "incomplete", False) else "success")
    update_cmd._resume_windows_gateways_and_merge_outcome(restart, request["windows_resume"], request["gateway_mode"])
    update_cmd._verify_fleet_after_update(
        restart, _pre_update_plan=plan, _windows_gateway_resume=request["windows_resume"],
        update_complete=bool(runtime_safe) and complete)
    return complete


class _ForwardedOutput:
    """The parent's pipe preserves its terminal's prompt policy and log mirror."""

    def __init__(self, stream, isatty: bool):
        self.stream, self.terminal = stream, isatty

    def isatty(self):
        return self.terminal

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _report_unbuilt_desktop(request: dict) -> None:
    """The tail raised out of ``_complete_selected``: before the build stage, the build never ran."""
    from hermes_cli import update_receipt

    current = update_receipt._current.get()
    if current is not None:  # a finalized run got past the build (verification finalizes it)
        _report_desktop_build_owed(request, current.data, "the source-update tail did not finish")


def _finish(request: dict, result_path: Path) -> int:
    from hermes_cli import update_receipt
    from pm.receipt import accept_worker_receipt

    _resume_receipt(request["receipt"])
    accept_worker_receipt(request.get("pm_receipt"), request["receipt"]["update_id"])
    update_receipt.record_stage("deps", "success")  # only a completed PM preparation reaches --prepared
    code, reason = 0, "source update completion"
    try:
        if not _complete_selected(request):
            code, reason = 1, "local changes left in the stash; re-apply them by hand"
    except KeyboardInterrupt:
        # An operator interrupt is not a step failure, and the code already moved: the run is
        # ``interrupted`` (never "failed"), and the armed obligations finish the tail.
        code, reason = 130, "KeyboardInterrupt: interrupted after the code was updated"
        update_receipt.finalize_interrupted_update_receipt(reason, exit_code=code)
    except SystemExit as exc:
        if exc.code not in (0, None):
            _report_unbuilt_desktop(request)
            update_receipt.record_followup("completion", f"completion exited {exc.code}")
    except BaseException as exc:  # noqa: BLE001 — after the commit point nothing fails the update
        _report_unbuilt_desktop(request)
        update_receipt.record_followup("completion", f"{type(exc).__name__}: {exc}")
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
            step_reason = f"Windows gateway recovery failed: {exc}"
            update_receipt.record_followup("windows_resume", step_reason)
            if update_receipt._current.get() is None:  # verification already finalized this run
                update_receipt.amend_terminal_followup(request["receipt"]["update_id"], "windows_resume", step_reason)
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
    return _bootstrap(request, request_path, result_path)


def _bootstrap(request: dict, request_path: Path, result_path: Path) -> int:
    """Dependency preparation in the parent's interpreter; its failure is a follow-up (A6)."""
    try:
        return _prepare(request, request_path, result_path)
    except KeyboardInterrupt:
        _resume_receipt(request["receipt"])
        from hermes_cli import update_receipt
        update_receipt.finalize_interrupted_update_receipt(
            "KeyboardInterrupt: interrupted while installing dependencies", exit_code=130)
        return _write_bootstrap_result(request, result_path, 130, _read_terminal_receipt(request))
    except (Exception, SystemExit) as exc:  # health: allow BLE001 -- after the commit point nothing fails the update
        # The code is committed but its dependencies are not (A6): an owed follow-up, exit 0.
        # Paused Windows gateways stay the parent's to resume (it has the dependencies).
        return _settle_after_commit(request, result_path, "dependencies", str(exc) or type(exc).__name__)


if __name__ == "__main__":
    raise SystemExit(main())
