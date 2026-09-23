"""T0 UNIT REGRESSION: existing updater contracts, not an installation test.

Real receipt/marker persistence, real temporary Git generations, and a bounded
fixture child exercise the baseline without running an update or discovering a
host fleet. Fleet observations are injected at the existing collector seam.

The baseline can correlate a restart marker's expected_sha and owed profiles
with a checkout and observed fleet. It cannot bind a receipt to a future rollout
intent/target identifier: that acceptance gate REQUIRES T4. No invented target
fields, skipped future assertions, or implementation stubs stand in for it.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import build_info, update_cmd, update_cmd_fleet as fleet
from hermes_cli import update_handoff as handoff, update_receipt as receipt
from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan, record_plan_in_receipt


@pytest.fixture
def baseline_home(tmp_path, monkeypatch, isolated_update_runtime):
    """Reuse the updater's host-fleet guard; keep all persistence in this home."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(home / "gateway-locks"))
    # These tests exercise the legacy per-home marker reader.  Force the
    # current host-scoped writer down its documented compatibility fallback so
    # the assertion targets the marker format it names, not a shared host file.
    from hermes_cli import update_host_obligation
    monkeypatch.setattr(update_host_obligation, "write_host_obligation", lambda **_: False)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(receipt, "_current", None)
    yield home
    receipt._current = None


@pytest.fixture
def generations(baseline_home, tmp_path, monkeypatch):
    """Local-only Git fixture; never stages/commits in the candidate checkout."""
    root = tmp_path / "generation-checkout"
    root.mkdir()
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)

    def git(*args):
        return subprocess.run(
            ["git", "-c", "user.name=Baseline Fixture", "-c",
             "user.email=baseline@example.invalid", "-c", "commit.gpgsign=false",
             "-c", "core.hooksPath=" + os.devnull, *args],
            cwd=root, check=True, capture_output=True, text=True, timeout=20,
        ).stdout.strip()

    git("init", "--quiet")
    module = root / "rollout_generation_fixture.py"
    module.write_text("def old_symbol():\n    return 'old'\n", encoding="utf-8")
    git("add", module.name)
    git("commit", "--quiet", "-m", "fixture old generation")
    old = git("rev-parse", "HEAD")

    def advance():
        module.write_text(
            "def new_symbol():\n    return 'new generation'\n", encoding="utf-8"
        )
        git("add", module.name)
        git("commit", "--quiet", "-m", "fixture new generation")
        return git("rev-parse", "HEAD")

    # Preserve the real identity reader; only its checkout location changes.
    monkeypatch.setattr(build_info, "__file__", str(root / "hermes_cli" / "build_info.py"))
    monkeypatch.setattr(build_info, "_code_identity_cache", None)
    return SimpleNamespace(root=root, old=old, advance=advance)


def _gateway(profile="work"):
    return RuntimeRecord(kind="gateway", profile=profile, pid=123,
                         supervisor="manual", restart_via="manual")


def _row(sha, profile="work", state="current"):
    return {"profile": profile, "pid": 456, "code_sha": sha, "state": state}


@pytest.mark.parametrize("runtimes,armed", [([], False), (None, True)])
def test_known_empty_inventory_does_not_arm_unknown_restart_debt(
    baseline_home, runtimes, armed
):
    fleet._write_fleet_restart_pending_marker(expected_sha="a" * 40, runtimes=runtimes)
    marker = fleet._fleet_restart_pending_marker_path()
    assert marker.parent == baseline_home
    assert marker.exists() is armed
    if armed:
        # Empty observations cannot turn an uncaptured inventory into known-empty.
        assert fleet._marker_only_restart_obsolete() is False
        assert marker.is_file()


@pytest.mark.parametrize(
    "inventory,settled",
    [('{"version":1,"runtimes":[]}', True),
     (None, False), ('{broken', False),
     ('{"version":1}', False), ('{"version":2,"runtimes":[]}', False)],
    ids=["captured-empty", "missing", "unreadable-json", "missing-rows", "unknown-version"],
)
def test_empty_is_evidence_not_an_inventory_read_failure(baseline_home, inventory, settled):
    marker = fleet._fleet_restart_pending_marker_path()
    body = "expected_sha=" + "a" * 40 + "\n"
    if inventory is not None:
        body += "inventory=" + inventory + "\n"
    marker.write_text(body, encoding="utf-8")
    assert fleet._marker_only_restart_obsolete() is settled
    assert marker.exists() is not settled


def test_unreadable_marker_cannot_discharge_restart_debt(baseline_home):
    marker = fleet._fleet_restart_pending_marker_path()
    marker.write_bytes(b"inventory=\xff\xfe\n")
    assert fleet._marker_only_restart_obsolete() is False
    assert marker.read_bytes() == b"inventory=\xff\xfe\n"


@pytest.mark.parametrize(
    "observation,settled",
    [("matching", True), ("wrong-sha", False), ("wrong-profile", False),
     ("unknown", False), ("down", False), ("absent", False), ("moved-checkout", False)],
)
def test_captured_restart_debt_requires_its_target_and_profile(
    baseline_home, generations, monkeypatch, observation, settled
):
    target = generations.advance()
    plan = UpdatePlan(expected_sha=generations.old, runtimes=[_gateway()])
    fleet._write_fleet_restart_pending_marker(
        expected_sha=target, runtimes=plan.to_dict()["runtimes"]
    )
    marker = fleet._fleet_restart_pending_marker_path()
    before = marker.read_bytes()
    rows = [_row(target)]
    if observation == "wrong-sha":
        rows = [_row(generations.old)]
    elif observation == "wrong-profile":
        rows = [_row(target, profile="workbench")]
    elif observation in ("unknown", "down"):
        rows = [_row(target, state=observation)]
    elif observation == "absent":
        rows = []
    elif observation == "moved-checkout":
        # A marker belongs to its captured target, not whichever SHA is newest.
        marker.write_text(before.decode().replace(target, generations.old), encoding="utf-8")
        before = marker.read_bytes()
    monkeypatch.setattr(receipt, "collect_fleet_versions", lambda **kw: rows)

    assert fleet._marker_only_restart_obsolete() is settled
    if settled:
        assert not marker.exists()
    else:
        assert marker.read_bytes() == before


@pytest.mark.parametrize("covers", [True, False], ids=["settled", "wrong-profile"])
def test_receipt_recovery_preserves_failed_history_and_requires_coverage(
    baseline_home, generations, monkeypatch, covers
):
    receipt.begin_update_receipt()
    record_plan_in_receipt(UpdatePlan(expected_sha=generations.old, runtimes=[
        RuntimeRecord(kind="gateway", profile="work", pid=123, code_sha=generations.old)
    ]))
    target = generations.advance()
    receipt.record_gateway_restart(incomplete=True, phase_error="restart failed")
    archived = receipt.finalize_pending_update_receipt(1, "restart failed")
    assert archived is not None
    original = archived.read_bytes()
    latest_path = archived.parent / "latest.json"
    before = latest_path.read_bytes()
    rows = [_row(target, profile="work" if covers else "workbench")]
    monkeypatch.setattr(receipt, "collect_fleet_versions", lambda **kw: rows)

    # The persistence API delegates acceptance to its caller. Exercise the real
    # coverage predicate here, not a made-up rollout-intent validator. This is
    # helper composition coverage, not proof of every updater caller's policy.
    settled = receipt.settle_latest_receipt_fleet(
        rows,
        discharges=lambda candidate: fleet._live_fleet_covers_receipt(
            target, candidate, fleet._receipt_owed_gateways(candidate, [])
        ),
    )
    assert settled is covers
    assert archived.read_bytes() == original
    latest = receipt.read_latest_receipt()
    # Settlement repairs fleet accounting, never rewrites an unsuccessful run as success.
    assert latest["outcome"] == "failed"
    assert latest["exit_code"] == 1
    assert latest["post_update"]["sha"] == target
    assert latest["pre_update"]["sha"] == generations.old
    if covers:
        assert latest["fleet"] == rows
        assert latest["gateway_restart"]["incomplete"] is False
        assert latest["gateway_restart"]["phase_error"] == ""
        assert fleet._update_owes_fleet_restart(receipt=latest, pending_manual=[]) is False
    else:
        assert latest_path.read_bytes() == before


@pytest.mark.parametrize("phase", ["pip_install", "gateway_restart"])
def test_failed_phase_at_nonzero_boundary_is_never_a_successful_receipt(
    baseline_home, phase
):
    receipt.begin_update_receipt()
    receipt.record_step(phase, False, "fixture phase failure")
    if phase == "gateway_restart":
        receipt.record_gateway_restart(
            failed_units=["hermes-gateway-work.service"], incomplete=True,
            phase_error="fixture phase failure",
        )
    path = receipt.finalize_pending_update_receipt(1, "fixture phase failure")
    assert path is not None
    latest = receipt.read_latest_receipt()
    assert latest["outcome"] == "failed"
    assert latest["exit_code"] == 1
    assert latest["steps"][0]["ok"] is False
    assert fleet._receipt_looks_unfinished(latest) is True
    assert fleet._receipt_restart_phase_completed(latest) is None
    assert receipt.finalize_pending_update_receipt(0) is None
    assert receipt.read_latest_receipt() == latest


# Only this fixture program runs in the child. It imports real persistence helpers,
# never invokes the installer/update tail, and cannot enumerate or restart services.
_HANDOFF_CHILD = """
import os, sys
from pathlib import Path
from hermes_cli import build_info, update_handoff, update_receipt
payload = update_handoff.read_handoff(sys.argv[1])
root = Path(sys.argv[2])
sys.path.insert(0, str(root))
import rollout_generation_fixture as generation
build_info.__file__ = str(root / 'hermes_cli' / 'build_info.py')
build_info._code_identity_cache = None
assert update_handoff.is_post_swap_child()
assert generation.new_symbol() == 'new generation'
assert not hasattr(generation, 'old_symbol')
update_receipt.resume_update_receipt(payload['receipt'])
update_receipt.record_step('fixture_new_generation', True, generation.new_symbol())
code = int(sys.argv[3])
if code:
    update_receipt.record_step('pip_install', False, 'fixture dependency failure')
assert update_receipt.finalize_pending_update_receipt(code) is not None
Path(sys.argv[1]).unlink()
sys.exit(code)
"""


@pytest.mark.real_post_swap_handoff
@pytest.mark.parametrize("child_exit", [0, 1], ids=["success", "dependency-failure"])
def test_old_interpreter_hands_receipt_to_fresh_fixture_process(
    baseline_home, generations, monkeypatch, child_exit
):
    monkeypatch.syspath_prepend(str(generations.root))
    old_module = importlib.import_module("rollout_generation_fixture")
    try:
        assert old_module.old_symbol() == "old"
        receipt.begin_update_receipt()
        receipt.record_step("pre_update_backup", True, "fixture snapshot")
        plan = UpdatePlan(expected_sha=generations.old, runtimes=[])
        record_plan_in_receipt(plan)
        target = generations.advance()
        # This remains the pre-swap module: mixing new source into it is unsafe.
        assert not hasattr(old_module, "new_symbol")
        monkeypatch.setattr(handoff, "_running_from_windows_shim", lambda: False)
        command = handoff.post_swap_command(Path("payload.json"), ["--yes"])
        assert command[1:5] == ["-m", "hermes_cli.main", "update", "--yes"]

        def fixture_command(path, argv_tail):
            assert argv_tail == ["--yes"]
            return [sys.executable, "-c", _HANDOFF_CHILD, str(path),
                    str(generations.root), str(child_exit)]

        # Replace only the command destination: the production handoff still
        # serializes, spawns/waits, detaches the parent, and relays the real exit.
        monkeypatch.setattr(handoff, "post_swap_command", fixture_command)
        with open(os.devnull, "r", encoding="utf-8") as stdin:
            monkeypatch.setattr(sys, "stdin", stdin)
            with pytest.raises(SystemExit) as stopped:
                update_cmd._hand_off_post_swap(
                    SimpleNamespace(yes=True), swap="git", branch="main",
                    pre_pull_sha=generations.old,
                    opts=SimpleNamespace(pre_update_version="fixture-old",
                                         active_lazy_features=[], active_tool_dependencies={}),
                    gateway_mode=False, had_desktop_app_before_update=False,
                    _pre_update_plan=plan,
                )
        assert stopped.value.code == child_exit
        assert receipt._current is None
        assert receipt.finalize_pending_update_receipt(child_exit) is None
        latest = receipt.read_latest_receipt()
        assert latest["outcome"] == ("success" if child_exit == 0 else "failed")
        assert latest["exit_code"] == child_exit
        assert latest["pid"] == os.getpid()
        assert latest["post_swap_pid"] != latest["pid"]
        assert latest["pre_update"]["sha"] == generations.old
        assert latest["post_update"]["sha"] == target
        assert latest["plan"] == plan.to_dict()
        assert [step["name"] for step in latest["steps"]][:2] == [
            "pre_update_backup", "fixture_new_generation"
        ]
        directory = baseline_home / "logs" / "update_receipts"
        assert len(list(directory.glob("update_*.json"))) == 1
        assert not list(directory.glob("post_swap_*.json"))
        assert old_module.old_symbol() == "old"
    finally:
        sys.modules.pop("rollout_generation_fixture", None)
