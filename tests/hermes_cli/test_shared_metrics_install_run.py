"""hermes.install.run: the installers' local receipt, counted by a later start only while collection is on."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import relay_runtime
from hermes_cli.observability import relay_shared_metrics
from hermes_cli.observability import shared_metrics as store_module
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability import shared_metrics_install_run as install_run
from hermes_cli.observability import shared_metrics_process as process_metrics

ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = ROOT / "scripts" / "install.sh"
INSTALL_PS1 = ROOT / "scripts" / "install.ps1"
SCHEMA = ROOT / "hermes_cli" / "observability" / "schemas" / "hermes.shared_metrics.v4.schema.json"


@pytest.fixture
def marks(tmp_path, monkeypatch):
    captured: list[tuple[str, dict]] = []
    policy = {"on": True}
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config_readonly",
        lambda: {"telemetry": {"shared_metrics": {"enabled": policy["on"]}}},
    )
    monkeypatch.setattr(relay_shared_metrics, "enabled", lambda: policy["on"])
    store = {"saves": True}

    def saved(rows):
        if store["saves"]:
            captured.extend(rows)
        return len(rows) if store["saves"] else 0

    monkeypatch.setattr(relay_shared_metrics, "record_process_marks_saved", saved, raising=False)
    yield SimpleNamespace(rows=captured, policy=policy, home=tmp_path / "home", store=store)


def _receipt(**overrides) -> dict:
    now = int(time.time())
    receipt = {"id": "ab" * 16, "installer": "install_sh", "outcome": "failed", "failed_stage": "repository",
               "failure_class": "git_clone_failed", "started_at": now - 200, "finished_at": now}
    return {**receipt, **overrides}


def _park(home: Path, receipt: dict, name: str | None = None) -> Path:
    directory = install_run.pending_installs_dir(home)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name or receipt.get('id', 'x')}.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_schema_and_installers_match_the_contract_exactly():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8-sig"))
    by_name = {d["properties"]["name"]["const"]: d for d in schema["$defs"].values() if "properties" in d}
    dims = by_name[contract.INSTALL_RUN_METRIC]["properties"]["dimensions"]
    assert {field: set(spec["enum"]) for field, spec in dims["properties"].items()} == {
        field: set(values) for field, values in contract._COUNTER_DIMENSION_VALUES[contract.INSTALL_RUN_METRIC].items()}
    assert set(dims["required"]) == set(dims["properties"])
    assert {"$ref": "#/$defs/install_run_counter"} in schema["properties"]["metrics"]["items"]["oneOf"]

    sh, ps1 = INSTALL_SH.read_text(encoding="utf-8-sig"), INSTALL_PS1.read_text(encoding="utf-8-sig")
    # Initialize-ResolvedPaths runs before the ladder's try, so its Fail can never write a receipt and
    # carries no class (a class there would be dead); every ladder call site must carry one.
    pre_ladder = re.search(r"\nfunction Initialize-ResolvedPaths \{\r?\n.*?\r?\n\}\r?\n", ps1, re.DOTALL).group(0)
    assert re.search(r'Fail "[^\n]*-InstallDir\."\r?\n', pre_ladder)
    ps1 = ps1.replace(pre_ladder, "\n")
    # Every fail()/Fail call site passes a class from the contract, and every class is used somewhere.
    sh_sites = re.findall(r'(?<![-\w])fail "(?:[^"\\$]|\\.|\$\([^)]*\)|\$)*" ?([a-z_]*)', sh)
    ps1_sites = re.findall(r'(?<![-\w])Fail "(?:[^"`$]|`.|\$\([^)]*\)|\$)*" ?([a-z_$(]*)', ps1)
    assert sh_sites and ps1_sites
    assert "" not in sh_sites, "an install.sh fail() call site passes no failure class"
    assert "" not in ps1_sites, "an install.ps1 Fail call site passes no failure class"
    used = {c for c in sh_sites + ps1_sites if not c.startswith("$")} | {"setup_failed", "gateway_failed"}
    assert used | {"interrupted", "none", "other"} == contract.INSTALL_RUN_FAILURE_CLASSES
    stage_names = re.search(r"stage_names\(\) \{\n\s+printf '%s\\n' ([a-z -]+)\n", sh).group(1).split()
    assert {s.replace("-", "_") for s in stage_names} == contract.INSTALL_RUN_STAGES
    ps1_stages = re.findall(r'@\{ name = "([a-z-]+)"', ps1)
    assert {s.replace("-", "_") for s in ps1_stages} == contract.INSTALL_RUN_STAGES


def test_receipt_is_reported_once_out_of_contract_ones_dropped(marks):
    good = _park(marks.home, _receipt())
    success = _park(marks.home, _receipt(id="cd" * 16, installer="install_ps1", outcome="success",
                                         failed_stage="none", failure_class="none"))
    bad = [
        _park(marks.home, _receipt(id="01" * 16, failure_class="git clone of https://host/x failed")),
        _park(marks.home, _receipt(id="02" * 16, failed_stage="/home/someone")),
        _park(marks.home, {**_receipt(id="03" * 16), "reason": "free text"}),
        _park(marks.home, _receipt(id="04" * 16, outcome="success")),  # success naming a failed stage
    ]
    marks.store["saves"] = False  # a busy store keeps every in-contract receipt for the next start
    install_run.report_pending_installs(marks.home)
    assert good.exists() and success.exists() and not any(p.exists() for p in bad)
    assert marks.rows == []

    marks.store["saves"] = True
    install_run.report_pending_installs(marks.home)
    _park(marks.home, _receipt())  # the same receipt left behind by a failed delete
    install_run.report_pending_installs(marks.home)
    assert sorted(marks.rows, key=lambda r: r[1]["outcome"]) == [
        (contract.INSTALL_RUN_MARK, {"duration_bucket": "2m_to_5m", "failed_stage": "repository",
                                     "failure_class": "git_clone_failed", "installer": "install_sh", "outcome": "failed"}),
        (contract.INSTALL_RUN_MARK, {"duration_bucket": "2m_to_5m", "failed_stage": "none",
                                     "failure_class": "none", "installer": "install_ps1", "outcome": "success"}),
    ]
    assert list(install_run.pending_installs_dir(marks.home).iterdir()) == []
    for mark, data in marks.rows:
        assert contract.counter_dimensions_are_valid(contract._DECISION_MARK_METRICS[mark], data)


def test_start_with_collection_off_purges_receipts_unreported(marks, monkeypatch):
    receipt = _park(marks.home, _receipt())
    monkeypatch.setattr(process_metrics, "_STATE", {})
    marks.policy["on"] = False
    process_metrics.begin_process("cli")
    assert not receipt.parent.exists()
    marks.policy["on"] = True
    install_run.report_pending_installs(marks.home)
    assert marks.rows == []


@pytest.mark.skipif(sys.platform == "win32", reason="runs the real install.sh")
def test_real_install_sh_ladder_leaves_only_closed_tokens(tmp_path):
    """The real ladder with git missing from PATH fails in prerequisites and parks one closed receipt."""
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ("bash", "uname", "date", "od", "tr", "mkdir", "mv", "curl", "id", "cat", "dirname"):
        found = shutil.which(name)
        if found:
            (tools / name).symlink_to(found)
    home = tmp_path / "hermes home"
    result = subprocess.run(
        [str(tools / "bash"), str(INSTALL_SH), "--non-interactive", "--hermes-home", str(home),
         "--dir", str(tmp_path / "checkout")],
        env={"HOME": str(tmp_path), "PATH": str(tools), "HERMES_REPO_URL": "https://example.invalid/x.git"},
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "git is required" in result.stderr
    receipts = list(install_run.pending_installs_dir(home).glob("*.json"))
    assert len(receipts) == 1
    text = receipts[0].read_text(encoding="utf-8-sig")
    assert "example.invalid" not in text and str(tmp_path) not in text and "required" not in text
    receipt = json.loads(text)
    assert install_run.install_run_fields(receipt) == {
        "duration_bucket": "lt_30s", "failed_stage": "prerequisites", "failure_class": "git_missing",
        "installer": "install_sh", "outcome": "failed",
    }


_REAL_RECORD_SAVED = relay_shared_metrics.record_process_marks_saved
_RESOURCE = {"architecture": "x86_64", "hermes_version": "0.0.0", "install_method": "git", "os_family": "linux"}


def test_receipt_is_recorded_only_on_a_day_the_sender_can_ever_send(marks, monkeypatch):
    """The opt-in usually happens inside the install (setup asks), so the first start is the opt-in day,
    whose package CONSENT_GATE_SQL never passes (period_start < opened_at). The receipt waits for the
    first start on a later day, and the package carrying it then passes the real gate."""
    from hermes_cli.observability.shared_metrics_sender import CONSENT_GATE_SQL, reconcile_send_consent
    from hermes_cli.sqlite_util import write_txn

    t0 = datetime(2026, 10, 6, tzinfo=timezone.utc)
    clock = {"now": t0 + timedelta(hours=9)}
    monkeypatch.setattr(store_module, "_utc_now", lambda: clock["now"])
    monkeypatch.setattr("hermes_cli.config.read_raw_config_readonly",
                        lambda: {"telemetry": {"shared_metrics": {"enabled": True, "send": True}}})
    store = store_module.SharedMetricsStore()

    def observe(send=True):
        with store._connection() as connection, write_txn(connection):
            reconcile_send_consent(connection, send, now=clock["now"])

    def record(rows):  # the real store, on the store's own clock
        for mark, data in rows:
            store.record_counter(contract._DECISION_MARK_METRICS[mark], data, _RESOURCE)
        return len(rows)

    monkeypatch.setattr(relay_shared_metrics, "record_process_marks_saved", record, raising=False)
    receipt = _park(marks.home, _receipt(finished_at=int(clock["now"].timestamp()) - 60,
                                         started_at=int(clock["now"].timestamp()) - 400))
    observe()  # "Yes, send" in the install's setup stage opens the window now
    clock["now"] += timedelta(hours=1)
    install_run.report_pending_installs(marks.home)  # first start, same UTC day
    assert receipt.exists() and store.counter_snapshot() == []

    clock["now"] = t0 + timedelta(days=1, hours=8)
    observe()
    install_run.report_pending_installs(marks.home)  # first start on a later day
    assert not receipt.exists()
    clock["now"] = t0 + timedelta(days=2, hours=1)
    observe()  # the heartbeat that confirms the whole recorded day
    store.create_and_export_package()
    with store._connection() as connection:
        packages = connection.execute(
            f"SELECT payload_json, {CONSENT_GATE_SQL} FROM package_outbox").fetchall()
    carrying = [bool(ok) for body, ok in packages
                if any(m["name"] == contract.INSTALL_RUN_METRIC for m in json.loads(body)["metrics"])]
    assert carrying == [True]


def test_receipt_waits_without_an_open_send_window_and_is_recorded_at_once_when_kept_local(marks, monkeypatch):
    config = {"enabled": True, "send": True}
    monkeypatch.setattr("hermes_cli.config.read_raw_config_readonly",
                        lambda: {"telemetry": {"shared_metrics": config}})
    receipt = _park(marks.home, _receipt())
    install_run.report_pending_installs(marks.home)  # send on, no window recorded yet
    assert receipt.exists() and marks.rows == []
    config["send"] = False  # collected on this machine only: nothing waits on a send window
    install_run.report_pending_installs(marks.home)
    assert not receipt.exists() and len(marks.rows) == 1


def test_unreadable_and_week_old_receipts_are_deleted_unreported(marks):
    directory = install_run.pending_installs_dir(marks.home)
    directory.mkdir(parents=True)
    bad = [directory / f"{'c' * 32}.json", directory / f"{'d' * 32}.json"]
    bad[0].write_text('{"id":"' + "c" * 32 + '","started_at":,"finished_at":}\n', encoding="utf-8")
    bad[1].write_text("[]", encoding="utf-8")
    old = int(time.time()) - install_run.MAX_RECEIPT_AGE_SECONDS - 3600
    stale = _park(marks.home, _receipt(id="e" * 32, started_at=old - 100, finished_at=old))
    install_run.report_pending_installs(marks.home)
    assert not any(p.exists() for p in [*bad, stale])
    assert list(directory.iterdir()) == [] and marks.rows == []


def test_relay_instrumentation_off_keeps_the_receipt_instead_of_latching_it_unrecorded(marks, monkeypatch):
    """With Relay off the real store call settles a row without recording it; the receipt must not be
    latched and deleted on that answer."""
    monkeypatch.setattr(relay_shared_metrics, "record_process_marks_saved", _REAL_RECORD_SAVED)
    monkeypatch.setattr(relay_runtime, "relay_instrumentation_enabled", lambda: False)
    receipt = _park(marks.home, _receipt())
    install_run.report_pending_installs(marks.home)
    assert receipt.exists()
    assert not install_run._recorded_latch(marks.home, receipt.stem).exists()


def test_every_opt_out_answer_purges_pending_receipts(marks, monkeypatch):
    """A "No" from setup, the pre-chat offer, the dashboard banner or `hermes config` goes through
    save_consent; none of them starts a process that purges, so the answer itself must."""
    from hermes_cli.observability import shared_metrics_consent as consent
    from hermes_cli.observability import shared_metrics_update as update_metrics

    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    monkeypatch.setattr("hermes_cli.setup._record_send_consent_change", lambda **_: None)
    install = _park(marks.home, _receipt())
    parked = update_metrics.pending_updates_dir(marks.home)
    parked.mkdir(parents=True)
    (parked / f"{'f' * 32}.json").write_text("{}", encoding="utf-8")

    consent.save_consent(True, False)
    assert install.exists() and parked.exists()
    consent.save_consent(False, False)
    assert not install.exists() and not parked.exists()


_MAIN_GUARD = 'if [ "${BASH_SOURCE[0]:-$0}" = "$0" ]; then\n'
_CHILD = """
import pathlib, sys, time
pathlib.Path(sys.argv[1]).write_text("ready")
try:
    time.sleep(30)
except KeyboardInterrupt:
    print("child handled Ctrl-C")
    sys.exit(int(sys.argv[2]))
"""


def _ctrl_c_ladder(tmp_path: Path, child_exit: int) -> tuple[int, str, list[dict]]:
    """The real install.sh main block and traps, with every stage body a no-op except python-deps,
    whose child catches Ctrl-C like pm.cli / `hermes setup` do. Ctrl-C reaches the whole process
    group, as from a terminal."""
    script = INSTALL_SH.read_text(encoding="utf-8-sig")
    assert script.count(_MAIN_GUARD) == 1
    stubs = "".join(f"stage_{name}() {{ echo 'stage {name} ok'; }}\n" for name in (
        "prerequisites", "repository", "venv", "config", "products", "setup", "gateway", "desktop", "complete"))
    stubs += ('stage_python_deps() { "$STUB_PY" -c "$STUB_CHILD" "$STUB_READY" "$STUB_EXIT" '
              '|| fail "pm install failed" deps_install_failed; }\nprint_banner() { :; }\n')
    harness = tmp_path / "install-harness.sh"
    harness.write_text(script.replace(_MAIN_GUARD, stubs + _MAIN_GUARD), encoding="utf-8")
    home, ready = tmp_path / "home", tmp_path / "ready"
    proc = subprocess.Popen(
        ["bash", str(harness), "--non-interactive", "--hermes-home", str(home), "--dir", str(tmp_path / "checkout")],
        env={"HOME": str(tmp_path), "PATH": os.environ.get("PATH", ""), "STUB_PY": sys.executable,
             "STUB_CHILD": _CHILD, "STUB_READY": str(ready), "STUB_EXIT": str(child_exit)},
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), "the stage child never started"
        time.sleep(0.2)  # the child is inside its sleep
        os.killpg(proc.pid, signal.SIGINT)  # windows-footgun: ok (POSIX-only test)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)  # windows-footgun: ok (POSIX-only test)
    receipts = [json.loads(p.read_text(encoding="utf-8-sig")) for p in install_run.pending_installs_dir(home).glob("*.json")]
    return proc.returncode, out.decode("utf-8", "replace"), receipts


@pytest.mark.skipif(sys.platform == "win32", reason="runs the real install.sh")
def test_a_ctrl_c_the_stage_handles_lets_the_install_finish(tmp_path):
    """The receipt's signal traps must never change the installer's control flow: before them, bash's
    cooperative exit let the ladder carry on when the child caught Ctrl-C and succeeded."""
    rc, out, receipts = _ctrl_c_ladder(tmp_path, child_exit=0)
    assert "child handled Ctrl-C" in out
    assert rc == 0, out
    assert "stage complete ok" in out
    assert [(r["outcome"], r["failed_stage"], r["failure_class"]) for r in receipts] == [("success", "none", "none")]


@pytest.mark.skipif(sys.platform == "win32", reason="runs the real install.sh")
def test_a_user_abort_the_child_turns_into_exit_1_is_recorded_as_interrupted(tmp_path):
    """pm.cli / source_completion / `hermes setup` catch KeyboardInterrupt and exit 1, which reaches the
    stage's `|| fail ... deps_install_failed`; the receipt must still say the user stopped it."""
    rc, out, receipts = _ctrl_c_ladder(tmp_path, child_exit=1)
    assert rc != 0 and "stage complete ok" not in out
    assert [(r["outcome"], r["failed_stage"], r["failure_class"]) for r in receipts] == [
        ("failed", "python_deps", "interrupted")]
