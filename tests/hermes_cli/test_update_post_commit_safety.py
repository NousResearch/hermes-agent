"""Post-commit completion keeps exit status, runtime safety and owed debt as separate facts.

Each test runs the real completion (or receipt rotation) in a separate process against a
disposable home and a real git checkout. The only seams are machine boundaries: the selected
interpreter's linked SQLite (a shim interpreter that answers the SQLite probe and execs the real
Python for everything else), and the gateway fleet restart / dashboard / multiplex migration
service calls, which are replaced by recorders so the test can never touch a host's services.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _child_env(**values: str) -> dict[str, str]:
    """A fresh env for a child of THIS interpreter: same dependency path, no hermes state."""
    env = {key: value for key, value in os.environ.items() if not key.startswith(("HERMES_", "PYTEST_"))}
    # The test runner may provide dependencies through sys.path rather than site-packages.
    env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    env.update(values)
    return env

_CHILD = textwrap.dedent('''
    import copy, json, os, subprocess, sys
    from pathlib import Path
    root_checkout, cell, mode = sys.argv[1:4]
    sys.path.insert(0, root_checkout)
    cell = Path(cell)
    source = cell / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\\nname="completion-probe"\\nversion="0.0.0"\\n')
    git = ["git", "-c", "user.email=probe@example.invalid", "-c", "user.name=Probe", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run([*git, "-C", str(source), "commit", "-qm", "fixture"], check=True)
    if mode.startswith("sqlite"):
        # The selected interpreter is the shim: its SQLite probe is the runtime-safety fact.
        sys.executable = os.environ["PROBE_SHIM"]
    import hermes_yaml as yaml
    from hermes_cli.config import DEFAULT_CONFIG
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "_config_version": DEFAULT_CONFIG["_config_version"] - 1,
        "telemetry": {"shared_metrics": {"enabled": False}},
        "updates": {"refresh_cua_driver": False}}))
    # Service boundaries: record instead of touching the host's gateways/dashboards.
    from hermes_cli import gateway_migrate, update_cmd, update_cmd_fleet as fleet
    from hermes_cli import update_cmd_fleet_verify as verify, update_cmd_maint as maint
    entered = cell / "migration-entered"
    gateway_migrate.maybe_auto_migrate_after_update = lambda: entered.write_text("entered")
    real_restart = update_cmd._restart_gateway_fleet_after_update
    update_cmd._restart_gateway_fleet_after_update = lambda plan, gateway_mode: fleet._GatewayRestartOutcome(
        incomplete=False, phase_errors=[], pre_restart_gateway_pids=[], restarted_services=[],
        failed_or_stale_units=[], relaunched_profiles=[], externally_supervised_profiles=[], killed_pids=set())
    update_cmd._fleet_restart_skip_reason = lambda plan: None
    update_cmd._surviving_pre_update_serve_runtimes = lambda plan: []
    maint._refresh_dashboard_after_update = lambda **kwargs: None
    verify._print_legacy_units_warning = lambda: None
    verify._collect_fleet_snapshot = lambda *args: []
    windows_resume = None
    marker = home / ".update_exit_code"
    if mode.startswith("ipc-"):
        # Barrier: verification starts after the restart phase returned. A gateway /update reader
        # polling the marker can consume it from here on, before the run ever finalizes.
        verify._print_legacy_units_warning = lambda: (cell / "marker-at-verify").write_text(
            marker.read_text() if marker.exists() else "absent")
    if mode in ("ipc-failed-unit", "ipc-abort"):
        # The real restart phase; only the host's service manager and process table are replaced.
        from hermes_cli import gateway as gateway_module
        update_cmd._restart_gateway_fleet_after_update = real_restart
        fleet._restart_manual_gateways = lambda out, budget: None
        fleet._force_kill_stuck_gateways = lambda pids: None
        if mode == "ipc-failed-unit":
            gateway_module.find_gateway_pids = lambda **kwargs: []
            fleet._restart_systemd_gateway_units = lambda restarted, failed, *rest: failed.append(
                "hermes-gateway-probe.service")
        else:
            def unavailable(*args, **kwargs):
                raise RuntimeError("systemctl unavailable")
            gateway_module.find_gateway_pids = unavailable
            fleet._restart_systemd_gateway_units = unavailable
            update_cmd._recover_gateway_restart_after_abort = lambda plan, **kwargs: {}
    if mode == "ipc-windows":
        # The Windows service manager refuses the paused gateway's restart.
        from hermes_cli import main as hermes_main
        def scm(token):
            if token and token.get("resume_needed"):
                raise RuntimeError("Could not restart Windows gateway service(s): HermesGatewayProbe")
        hermes_main._resume_windows_gateways_after_update = scm
        update_cmd._resume_windows_gateways_after_update = scm
        windows_resume = {"resume_needed": True, "services": ["HermesGatewayProbe"],
                          "service_profiles": {"HermesGatewayProbe": "probe"}}
    if mode == "profile-sync":
        def profile_sync():
            raise SystemExit("profile skill sync exited 3")
        maint._sync_profiles_after_update = profile_sync
    from hermes_cli import update_receipt
    from hermes_cli.venv_sync import arm_completion, completion_pending_path
    arm_completion(source)
    update_receipt.begin_update_receipt()
    receipt = copy.deepcopy(update_receipt._current.get().data)
    # The completion process never inherits the parent's open receipt context.
    update_receipt._current.reset(update_receipt._current.get().current_token)
    request = {"schema": 1, "source": str(source), "home": str(home), "branch": "main", "desktop": False,
               "assume_yes": True, "gateway_mode": mode.startswith("ipc-"),
               "no_gateway_restart": mode == "config-ro",
               "pre_update_version": None, "snapshot_id": None, "sibling_snapshots": {}, "plan": None,
               "receipt": receipt, "windows_resume": windows_resume}
    if mode == "config-ro":
        home.chmod(0o555)  # a late filesystem failure: the profile home refuses the config write
    if mode.startswith("receipts-blocked"):
        # The receipt store fails after the commit point: its directory is now an ordinary file.
        import shutil
        shutil.rmtree(update_receipt._receipt_dir())
        update_receipt._receipt_dir().write_text("not a directory")
    from hermes_cli.update_completion import _finish, _settle_after_commit
    try:
        if mode == "receipts-blocked-settle":
            code = _settle_after_commit(request, cell / "result.json", "dependencies", "dependency sync refused")
        else:
            code = _finish(request, cell / "result.json")
    finally:
        home.chmod(0o755)
    latest = update_receipt.read_latest_receipt() or {}
    result = json.loads((cell / "result.json").read_text()) if (cell / "result.json").exists() else None
    (cell / "summary.json").write_text(json.dumps({
        "code": code, "outcome": latest.get("outcome"), "update_id": receipt["update_id"], "result": result,
        "followups": [row["step"] for row in latest.get("followups", [])],
        "stages": {mark["name"]: mark["outcome"] for mark in latest.get("stages", [])},
        "config_version": yaml.safe_load((home / "config.yaml").read_text())["_config_version"],
        "latest_config_version": DEFAULT_CONFIG["_config_version"],
        "pending": completion_pending_path(source).is_file(),
        "stamp": (source / "install-stamp.json").is_file(),
        "marker_at_verify": (cell / "marker-at-verify").read_text() if (cell / "marker-at-verify").exists() else None,
        "marker_final": marker.read_text() if marker.exists() else None,
        "migration_entered": entered.is_file()}))
''')


def _run_completion(tmp_path: Path, mode: str, *, sqlite: str | None = None) -> dict:
    cell = tmp_path / "cell"
    root_home = cell / "home" / ".hermes"
    # A named profile: receipts stay in the (writable) root home while the profile home refuses.
    profile = root_home / "profiles" / "probe"
    for name in ("logs/update_receipts", "skills", "cache", "profiles/probe"):
        (root_home / name).mkdir(parents=True, exist_ok=True)
    env = _child_env(HOME=str(cell / "home"), HERMES_HOME=str(profile), HERMES_DISABLE_LAZY_INSTALLS="1",
                     HERMES_RUNTIME_DIR=str(tmp_path / "store"), PYTHONDONTWRITEBYTECODE="1",
                     XDG_CONFIG_HOME=str(cell / "home/.config"), XDG_CACHE_HOME=str(cell / "home/.cache"),
                     XDG_STATE_HOME=str(cell / "home/.local/state"),
                     HERMES_GATEWAY_LOCK_DIR=str(cell / "home/.local/state/hermes/gateway-locks"))
    if sqlite is not None:
        payload = json.dumps({"base_prefix": sys.base_prefix, "executable": sys.executable,
                              "python_version": list(sys.version_info[:3]),
                              "sqlite_version": [int(p) for p in sqlite.split(".")],
                              "sqlite_version_string": sqlite, "sqlite_source_id": "fixture"})
        shim = tmp_path / "python-shim"
        script = (f"#!/bin/sh\ncase \"$3\" in *sqlite_source_id*) printf '%s\\n' '{payload}'; exit 0;; esac\n"
                  f"exec '{sys.executable}' \"$@\"\n")
        shim.write_text(script, encoding="utf-8")
        shim.chmod(0o755)
        env["PROBE_SHIM"] = str(shim)
    child = tmp_path / "child.py"
    child.write_text(_CHILD, encoding="utf-8")
    done = subprocess.run([sys.executable, "-B", str(child), str(ROOT), str(cell), mode],
                          env=env, cwd=tmp_path, capture_output=True, text=True, timeout=240)
    assert done.returncode == 0, done.stdout + done.stderr
    summary = json.loads((cell / "summary.json").read_text(encoding="utf-8-sig"))
    summary["output"] = done.stdout + done.stderr
    return summary


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("sqlite, safe", [("3.46.1", False), ("3.53.1", True)])
def test_unsafe_sqlite_runtime_keeps_exit_zero_but_vetoes_topology_migration(tmp_path, sqlite, safe):
    summary = _run_completion(tmp_path, f"sqlite-{sqlite}", sqlite=sqlite)
    # Committed code: success and exit 0 either way.
    assert summary["code"] == 0, summary["output"]
    assert summary["outcome"] == "success"
    if safe:
        assert summary["followups"] == []
        assert summary["migration_entered"] is True
    else:
        # The unsafe runtime is a visible, owed step — and it must still veto the migration.
        assert f"SQLite ({sqlite})" in summary["output"]
        assert summary["followups"] == ["sqlite_runtime"]
        assert summary["migration_entered"] is False


@pytest.mark.platforms("posix")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory modes")
def test_config_format_write_failure_is_owed_not_reported_complete(tmp_path):
    summary = _run_completion(tmp_path, "config-ro")
    assert summary["code"] == 0, summary["output"]
    assert summary["config_version"] == summary["latest_config_version"] - 1
    # The receipt names the debt, the tail stays pending and nothing stamps the tree as done.
    assert summary["followups"] == ["config_migration"]
    assert summary["pending"] is True
    assert summary["stamp"] is False
    # C3: the receipt names what actually failed. The build succeeded; only the config is owed.
    assert summary["stages"].get("build") == "success", summary["stages"]


@pytest.mark.platforms("posix")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory modes")
def test_profile_sync_followup_keeps_the_tail_owed(tmp_path):
    summary = _run_completion(tmp_path, "profile-sync")
    assert summary["code"] == 0, summary["output"]
    assert summary["outcome"] == "success"
    assert summary["followups"] == ["profile_sync"]
    # A profile sync that did not finish is tail work: retried by the next launch, never stamped done.
    assert summary["pending"] is True
    assert summary["stamp"] is False


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("path", ["finish", "settle"])
def test_receipt_store_failure_after_commit_keeps_exit_zero_and_a_correlated_receipt(tmp_path, path):
    """The receipt store is written after the commit point: losing it is loud, never exit 1."""
    summary = _run_completion(tmp_path, f"receipts-blocked-{path}")
    assert "Update receipt not written" in summary["output"]
    assert summary["code"] == 0, summary["output"]
    result = summary["result"]
    assert result["exit_code"] == 0
    # run_completion's correlation gate: this run's id, terminal, and its outcome agrees with exit 0.
    receipt = result["receipt"]
    assert receipt is not None, summary["output"]
    assert receipt["update_id"] == summary["update_id"]
    assert receipt["finished_at"]
    assert receipt["outcome"] == "success"
    # Retry obligations are unchanged: owed dependencies stay armed, a finished tail is cleared.
    if path == "settle":
        assert [row["step"] for row in receipt["followups"]] == ["dependencies"]
    assert summary["pending"] is (path == "settle")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("branch", ["failed-unit", "abort", "windows"])
def test_restart_failure_is_owed_and_never_reported_as_a_failed_update(tmp_path, branch):
    """The gateway /update reader hears the committed success; the restart is a follow-up."""
    summary = _run_completion(tmp_path, f"ipc-{branch}")
    assert summary["code"] == 0, summary["output"]
    assert summary["outcome"] == "success"
    assert "gateway_restart" in summary["followups"], summary["output"]
    # What a reader sees between the failed restart and the run's finalize, and at the end.
    assert summary["marker_at_verify"] == "0", summary["output"]
    assert summary["marker_final"] == "0"


# The NEW tree's completion child after a committed update whose Windows resume the service
# manager refused (what _finish answers): the run is a success with a windows_resume follow-up,
# and the token it hands back still owes the restart.
_REFUSED_RESUME_COMPLETION = textwrap.dedent('''
    import json, os, sys
    from pathlib import Path
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    receipts = Path(os.environ["PROBE_RECEIPTS"])
    receipt = dict(request["receipt"], outcome="success", exit_code=0, finished_at="now", followups=[
        {"step": "windows_resume", "reason": "Windows gateway recovery failed: refused"}])
    payload = json.dumps(receipt)
    (receipts / ("update_probe_" + receipt["update_id"] + ".json")).write_text(payload)
    (receipts / "latest.json").write_text(payload)
    (Path(request["home"]) / ".update_exit_code").write_text("0")
    Path(sys.argv[2]).write_text(json.dumps({
        "schema": 1, "update_id": receipt["update_id"], "exit_code": 0, "receipt": receipt,
        "windows_resume": dict(request["windows_resume"], resume_needed=True)}))
''')

_ZIP_PARENT = textwrap.dedent('''
    import json, os, sys
    from pathlib import Path
    from types import SimpleNamespace
    root_checkout, cell = sys.argv[1:3]
    sys.path.insert(0, root_checkout)
    cell = Path(cell)
    calls = cell / "parent-resume-calls.jsonl"
    from hermes_cli import main as hermes_main, update_cmd, update_inventory, update_receipt
    hermes_main.PROJECT_ROOT = cell / "source"
    receipts = update_receipt._receipt_dir()
    receipts.mkdir(parents=True, exist_ok=True)
    os.environ["PROBE_RECEIPTS"] = str(receipts)
    # Machine boundaries only: the host's process inventory, backups, terminal/hangup handling,
    # the Windows service manager, and the ZIP download + swap (a no-.git install).
    def no_inventory():
        raise RuntimeError("the host's runtime inventory is not part of this probe")
    update_inventory.collect_runtime_inventory = no_inventory
    hermes_main._update_preflight_handled = lambda args: False
    hermes_main._install_hangup_protection = lambda **kwargs: None
    hermes_main._finalize_update_output = lambda state: None
    hermes_main._run_pre_update_backup = lambda args: None
    token = {"resume_needed": True, "profiles": {}, "unmapped": [], "services": ["HermesGatewayProbe"],
             "service_profiles": {"HermesGatewayProbe": "probe"}}
    hermes_main._pause_windows_gateways_for_update = lambda: token
    def scm(resume_token):
        with calls.open("a") as handle:
            handle.write(json.dumps(bool(resume_token and resume_token.get("resume_needed"))) + "\\n")
        if resume_token and resume_token.get("resume_needed"):
            raise RuntimeError("Could not restart Windows gateway service(s): HermesGatewayProbe")
    hermes_main._resume_windows_gateways_after_update = scm
    update_cmd._prepare_git_command = lambda: (True, None, False)
    def via_zip(args, *, had_desktop_app_before_update, target_sha=None, completion_request=None, **kwargs):
        completion_request["expected_sha"] = target_sha
        completion_request["apply_mode"] = "zip"
        update_cmd._complete_source_update(completion_request)
    update_cmd._update_via_zip = via_zip
    class Args(SimpleNamespace):
        def __getattr__(self, name):
            return None
    hermes_main.cmd_update(Args(gateway=True, branch="main", yes=True))
''')


@pytest.mark.platforms("posix")
def test_zip_update_never_replays_a_refused_resume_after_the_commit_point(tmp_path):
    """The child already tried the resume and recorded the follow-up: the parent's finally and its
    atexit net must not run it again, and a restart error must not become exit 1 / "/update failed"."""
    cell = tmp_path / "cell"
    home = cell / "home" / ".hermes"
    (home / "logs/update_receipts").mkdir(parents=True)
    completion = cell / "source" / "hermes_cli" / "update_completion.py"
    completion.parent.mkdir(parents=True)
    completion.write_text(_REFUSED_RESUME_COMPLETION, encoding="utf-8")
    env = _child_env(HOME=str(cell / "home"), HERMES_HOME=str(home), HERMES_DISABLE_LAZY_INSTALLS="1",
                     HERMES_RUNTIME_DIR=str(tmp_path / "store"), PYTHONDONTWRITEBYTECODE="1",
                     XDG_CONFIG_HOME=str(cell / "home/.config"), XDG_CACHE_HOME=str(cell / "home/.cache"),
                     XDG_STATE_HOME=str(cell / "home/.local/state"))
    driver = tmp_path / "zip_parent.py"
    driver.write_text(_ZIP_PARENT, encoding="utf-8")
    done = subprocess.run([sys.executable, "-B", str(driver), str(ROOT), str(cell)],
                          env=env, cwd=tmp_path, capture_output=True, text=True, timeout=240)
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    assert (home / ".update_exit_code").read_text().strip() == "0", output
    calls = cell / "parent-resume-calls.jsonl"
    assert not calls.exists(), calls.read_text() + output
    latest = json.loads((home / "logs/update_receipts/latest.json").read_text(encoding="utf-8-sig"))
    assert latest["outcome"] == "success"
    assert [row["step"] for row in latest["followups"]] == ["windows_resume"]


_RECEIPT_CHILD = textwrap.dedent('''
    import json, sys
    root_checkout, action = sys.argv[1:3]
    sys.path.insert(0, root_checkout)
    from hermes_cli import update_receipt as ur
    if action == "seed":
        ur.begin_update_receipt()
        ur.record_fact("plan", {"runtimes": [json.loads(sys.argv[3])]})
    else:
        # An older interpreter's frozen handoff payload: no ``carried_manual_serves`` field.
        old = ur.UpdateReceipt().data
        old.pop("carried_manual_serves", None)
        ur.begin_update_receipt(previous=old, correlation_id=old["update_id"])
    ur.finalize_pending_update_receipt(0)
''')


@pytest.mark.platforms("posix")
def test_old_schema_handoff_receipt_keeps_the_previous_runs_manual_serve_debt(tmp_path):
    from hermes_cli.process_identity import _process_create_time

    home = tmp_path / "home" / ".hermes"
    home.mkdir(parents=True)
    # The durable reminder store is obstructed, so the receipt is the only record of the debt.
    (home / "serve_restart_pending").write_text("not a directory\n", encoding="utf-8")
    env = _child_env(HOME=str(tmp_path / "home"), HERMES_HOME=str(home), PYTHONDONTWRITEBYTECODE="1")
    child = tmp_path / "receipt_child.py"
    child.write_text(_RECEIPT_CHILD, encoding="utf-8")
    serve = subprocess.Popen([sys.executable, "-I", "-c", "import time; print('ready', flush=True); time.sleep(120)"],
                             stdout=subprocess.PIPE, text=True)
    assert serve.stdout is not None
    try:
        assert serve.stdout.readline().strip() == "ready"
        row = {"kind": "serve", "profile": "fixture", "supervisor": "manual-serve", "restart_via": "respawn-argv",
               "pid": serve.pid, "detail": {"create_time": _process_create_time(serve.pid)}}

        def run(*args):
            done = subprocess.run([sys.executable, "-B", str(child), str(ROOT), *args],
                                  env=env, capture_output=True, text=True, timeout=120)
            assert done.returncode == 0, done.stdout + done.stderr

        def owed():
            receipt = json.loads((home / "logs/update_receipts/latest.json").read_text(encoding="utf-8-sig"))
            rows = (receipt.get("plan") or {}).get("runtimes", []) + receipt.get("pending_manual_serves", [])
            return [r["pid"] for r in rows if r.get("supervisor") == "manual-serve"]

        run("seed", json.dumps(row))
        assert owed() == [serve.pid]
        run("handoff")
        assert owed() == [serve.pid]
    finally:
        serve.terminate()
        serve.wait(timeout=10)
        serve.stdout.close()
