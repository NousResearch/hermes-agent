"""pm.receipt: the universal machine-readable surface for venv ops.

Every pm sync writes one; same schema/dir as update receipts; the
updater embeds the sync sections via snapshot().
"""

from __future__ import annotations

import json

import pytest

import pm.receipt as receipt


@pytest.fixture(autouse=True)
def _isolated_receipt_context():
    import hermes_cli.update_receipt as update_receipt
    variables = (receipt._current, receipt._completed_by_update, update_receipt._current)
    tokens = [variable.set(None) for variable in variables]
    yield
    for variable, token in zip(variables, tokens):
        variable.reset(token)


@pytest.fixture
def homed(tmp_path, monkeypatch):
    """Receipt dir inside a temp hermes home."""
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    return tmp_path


@pytest.mark.parametrize("outcome,exit_code", [("failed", 1), ("refused", 2)])
def test_begin_record_finalize_roundtrip(homed, outcome, exit_code):
    assert receipt.latest() is None
    assert not (homed / "logs").exists()
    assert receipt.finalize("ok") is None
    receipt.record_warning("orphan")
    receipt.record_refusal("x", "y")
    assert receipt.latest() is None
    receipt.begin("sync")
    assert receipt.snapshot()["update_id"] is None
    receipt.record_step("uv-lock", True)
    receipt.record_venv_rebuild(True)
    receipt.record_feature_list(["web", "acp"])
    assert receipt.snapshot()["kind"] == "sync"
    path = receipt.finalize("ok")
    assert path is not None and path.is_file()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    assert receipt.latest() == data
    assert data["kind"] == "sync" and data["outcome"] == "ok"
    assert data["venv_rebuild"] == {"ok": True, "reason": ""}
    assert data["steps"][0]["name"] == "uv-lock"
    assert data["feature_list"] == ["web", "acp"]
    assert receipt.snapshot() is None
    assert receipt.finalize("ok") is None
    assert receipt.last_for_update("any-id") is None
    assert receipt.last_for_update(None) is None
    receipt.begin("sync")
    receipt.record_venv_rebuild(False, "uv sync exited 1")
    receipt.record_warning("first")
    receipt.record_warning("second")
    receipt.record_refusal("lazy-install", "extras outside frozen set")
    failed = receipt.finalize(outcome, exit_code)
    latest = receipt.latest()
    assert latest == json.loads(failed.read_text(encoding="utf-8-sig"))
    assert latest["outcome"] == outcome and latest["exit_code"] == exit_code
    assert latest["venv_rebuild"]["reason"] == "uv sync exited 1"
    assert latest["refusal"]["code"] == "lazy-install"
    assert latest["refusal"]["detail"] == "extras outside frozen set"
    assert [w["message"] for w in latest["warnings"]] == ["first", "second"]
    assert all(w["at"] for w in latest["warnings"])
    assert json.loads(path.read_text(encoding="utf-8-sig")) == data


def test_bare_python_can_report_a_failed_bootstrap(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    import subprocess
    import sys

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    repo = Path(__file__).resolve().parents[2]
    code = """
import json
from pm import receipt
from pm.package import InstallError
try:
    token = receipt.begin('sync')
    try:
        raise InstallError('venv', 'original dependency failure')
    except InstallError as exc:
        receipt.record_step('dependency-sync', False, str(exc))
        raise
    finally:
        receipt.finalize('failed', 1, token=token)
except InstallError as exc:
    assert 'original dependency failure' in str(exc)
row = receipt.latest()
assert row['outcome'] == 'failed' and row['exit_code'] == 1
assert 'original dependency failure' in row['steps'][0]['detail']
print(json.dumps(row))
"""
    child = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=repo, env=dict(os.environ),
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    row = json.loads(child.stdout)
    assert row["steps"][0]["ok"] is False


def test_concurrent_finalize_writes_unique_names(homed):
    """Overlapping syncs (threaded cadence/ensure) must each get their own
    receipt file — no stamp collision overwrites."""
    import threading

    paths: list = []
    errors: list = []

    def run(i):
        try:
            receipt.begin("sync")
            receipt.record_step(f"step-{i}", True)
            paths.append(receipt.finalize("ok"))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(p is not None and p.is_file() for p in paths)
    assert len({p.name for p in paths}) == 8


def test_rotation_only_removes_pm_receipts(homed):
    """The shared receipts dir also holds the updater's update_*.json —
    pm rotation must never delete those."""
    d = homed / "logs" / "update_receipts"
    d.mkdir(parents=True)
    updater = d / "update_20260101_000000_123.json"
    updater.write_text('{"kind": "update"}\n', encoding="utf-8")
    for i in range(25):
        (d / f"pm_20260101T0000{i:02d}Z-sync-1-0001.json").write_text(
            '{"kind": "sync"}\n', encoding="utf-8"
        )
    receipt._rotate(d)
    assert updater.is_file()
    kept = sorted(p.name for p in d.glob("pm_*.json"))
    assert len(kept) == 20  # oldest pm receipts rotated, updater receipt untouched


def test_receipt_state_is_thread_scoped(homed):
    """begin/record must not leak across threads: an overlapping sync in
    another thread neither sees nor clobbers this one's in-flight receipt."""
    import threading

    with receipt.worker_context("mine"):
        receipt.begin("sync")
    receipt.record_step("mine", True)
    seen: dict = {}

    def other():
        seen["snapshot_from_other"] = receipt.snapshot()
        with receipt.worker_context("theirs"):
            receipt.begin("update")
        receipt.record_step("theirs", False)
        receipt.finalize("failed")
        seen["after_other"] = receipt.snapshot()

    t = threading.Thread(target=other)
    t.start()
    t.join()
    # the other thread never saw our in-flight receipt
    assert seen["snapshot_from_other"] is None
    # its begin/finalize did not clobber ours
    snap = receipt.snapshot()
    assert snap is not None and snap["kind"] == "sync"
    assert [s["name"] for s in snap["steps"]] == ["mine"]
    path = receipt.finalize("ok")
    assert path is not None and path.is_file()
    assert receipt.last_for_update("theirs") is None
    assert receipt.last_for_update("mine")["steps"][0]["name"] == "mine"


def test_copied_context_does_not_corrupt_parent_receipt():
    """A copied context (asyncio.to_thread / task group pattern) inherits
    the SAME ContextVar dict — record_step must copy-on-write, never
    mutate the parent's in-flight receipt in place."""
    import contextvars

    receipt.begin("sync")
    receipt.record_step("parent", True)
    parent_before = receipt.snapshot()

    def child():
        receipt.record_step("child", False)
        return receipt.snapshot()

    ctx = contextvars.copy_context()
    child_seen = ctx.run(child)

    # the child saw the parent's receipt (ambient inheritance) and
    # appended its own step — in ITS copy only
    assert [s["name"] for s in child_seen["steps"]] == ["parent", "child"]
    # the parent's receipt is untouched by the child's record
    parent_after = receipt.snapshot()
    assert [s["name"] for s in parent_after["steps"]] == ["parent"]
    assert parent_after == parent_before
    receipt.finalize("ok")


def test_copied_context_finalize_does_not_finish_parent(homed):
    import contextvars

    receipt.begin("sync")
    contextvars.copy_context().run(receipt.finalize, "failed", 1)
    assert receipt.snapshot()["outcome"] is None


def test_recorded_values_are_not_mutable_through_the_input():
    receipt.begin("sync")
    checks = [{"plugin": "a", "result": {"compatible": True}}]
    receipt.record_plugin_checks(checks)
    checks[0]["result"]["compatible"] = False
    assert receipt.snapshot()["plugin_checks"][0]["result"]["compatible"] is True


def test_snapshot_returns_a_copy():
    """Mutating the snapshot must not touch the authoritative receipt."""
    receipt.begin("sync")
    receipt.record_step("a", True)
    snap = receipt.snapshot()
    snap["steps"].append({"name": "injected", "ok": True})
    snap["kind"] = "hijacked"
    live = receipt.snapshot()
    assert [s["name"] for s in live["steps"]] == ["a"]
    assert live["kind"] == "sync"


def test_nested_begin_with_token_restores_outer():
    """A nested begin (same context) with the token from finalize must
    restore the OUTER receipt instead of discarding it."""
    outer = receipt.begin("sync")
    receipt.record_step("outer-step", True)
    inner = receipt.begin("update")
    receipt.record_step("inner-step", True)
    receipt.finalize("ok", token=inner)
    # outer receipt survives the inner finalize
    live = receipt.snapshot()
    assert live is not None and live["kind"] == "sync"
    assert [s["name"] for s in live["steps"]] == ["outer-step"]
    path = receipt.finalize("ok", token=outer)
    assert path is not None and path.is_file()
    assert receipt.snapshot() is None


def test_sync_begin_after_update_finalized_has_no_update_id(homed, monkeypatch):
    """The correlation id comes from the OPEN update receipt: after its
    finalize pops it, a sync begun is standalone."""
    import hermes_cli.update_receipt as ur

    monkeypatch.setattr(ur, "_receipt_dir", lambda: homed / "logs" / "update_receipts")
    ur.begin_update_receipt()
    ur.finalize_update_receipt("success")
    assert ur.current_correlation_id() is None
    receipt.begin("sync")
    assert receipt.snapshot()["update_id"] is None


def test_last_for_update_returns_a_copy(homed):
    import hermes_cli.update_receipt as ur

    ur.begin_update_receipt()
    my_id = ur.current_correlation_id()
    receipt.begin("sync")
    receipt.finalize("ok")
    snap = receipt.last_for_update(my_id)
    snap["outcome"] = "tampered"
    assert receipt.last_for_update(my_id)["outcome"] == "ok"
