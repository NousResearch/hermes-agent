"""Regression suite: a FAILED ``hermes update`` keeps its receipt across state loss.

Regression for #112465 ("hermes update writes no receipt on v0.21.3 even with the
``hermes_cli.update_`` purge protection in place").

The bug: the module singleton ``_current`` was the *only* copy of an in-progress run.
When it was gone before finalization — module eviction/re-import, or the command
boundary running in a process that never began the receipt — ``finalize_update_receipt``
and ``finalize_pending_update_receipt`` both hit their ``receipt is None`` early return
and the failed run left NO artifact whatsoever: zero ``update_*.json`` and ``latest.json``
still pointing at an earlier update. The operator could not even see that an update had
been attempted and failed, and the write-failure path was swallowed into a DEBUG record
that production (INFO) discards.

Contract pinned here, at the receipt layer, with the real module functions against a
temp home:

1. RETENTION — a failed run whose singleton is lost still leaves a terminal
   ``update_<id>.json`` (outcome ``failed``/``refused``, exit code, stop reason, step
   evidence, ``update_id``, timestamps) and repoints ``latest.json`` at it.
2. RECOVERY IS THE REAL LOSS ROUTE — recovering works when the loss is an actual
   module eviction + re-import (the documented production route), not merely
   ``_current = None``.
3. PROCESS BOUNDARY — after a real fresh interpreter runs the command-boundary net,
   the failed run's record is still on disk with its metadata, and no reader can
   mistake it for a completed success.
4. MANY FAILURES — consecutive failed runs each keep their own receipt; retention
   still bounds the store.
5. CORRUPT DATA — a torn ``latest.json`` or an unparseable/malformed per-run record
   degrades to "skipped", never to a crash and never to a rewrite.
6. NO FALSE FAILURE — a successful update after a recovered failure stays ``success``,
   including under a later non-zero boundary exit.

The dashboard's HTTP gating (``running`` is not an outcome) is pinned in
``test_update_receipt_endpoint.py``; this suite adds only the end-to-end read of a
receipt that the *recovery path* produced.
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_cli.update_receipt as ur

_MODULE_NAME = "hermes_cli.update_receipt"
_REPO_ROOT = Path(ur.__file__).resolve().parents[1]


@pytest.fixture()
def receipt_home(tmp_path, monkeypatch):
    """Hermetic HERMES_HOME so receipts never touch the real profile."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(
        "hermes_cli.config.get_hermes_home", lambda: home, raising=False
    )
    ur._current = None
    yield home
    ur._current = None


def _receipt_dir(home):
    return home / "logs" / "update_receipts"


def _per_run_files(home):
    """The per-run records (``update_<id>.json``); ``latest.json`` is a pointer."""
    directory = _receipt_dir(home)
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.glob("*.json") if path.name.startswith("update_")
    )


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _lost_singleton_run(*steps):
    """A begun run that recorded ``steps``, then lost its module singleton.

    The loss is the reported failure mode: whatever held ``_current`` (module
    eviction, re-import, a process boundary) is gone, so finalization has only
    durable state left to work from.
    """
    ur.begin_update_receipt()
    for name, ok, detail in steps:
        ur.record_step(name, ok, detail)
    ur._current = None


_FAILED_STEPS = [
    ("git_pull", True, "updated checkout"),
    ("gateway_restart", False, "settlement check failed closed"),
]


class TestFailedReceiptSurvivesStateLoss:
    """Invariant 1 — the failed run outlives the singleton that recorded it."""

    def test_failed_update_receipt_is_retained_after_state_loss(self, receipt_home):
        _lost_singleton_run(*_FAILED_STEPS)

        path = ur.finalize_pending_update_receipt(1, "sys.exit(1)")

        assert path is not None and Path(path).is_file()
        payload = _read(path)
        assert payload["outcome"] == "failed"
        assert payload["exit_code"] == 1
        assert payload["stop_reason"] == "sys.exit(1)"
        assert payload["finished_at"] is not None
        # Identity and evidence survive the loss, so the record is diagnosable.
        assert payload["update_id"]
        assert payload["pid"] == os.getpid()
        assert payload["started_at"]
        assert [step["name"] for step in payload["steps"]] == [
            name for name, _, _ in _FAILED_STEPS
        ]
        assert payload["steps"][1]["ok"] is False
        assert payload["steps"][1]["detail"] == "settlement check failed closed"

        latest = ur.read_latest_receipt()
        assert latest is not None and latest["outcome"] == "failed"

    def test_recovery_leaves_no_dangling_running_record(self, receipt_home):
        """The recovered record is terminal: a reader polling ``running`` forever
        is the same operator-visible dead end as no record at all."""
        _lost_singleton_run(*_FAILED_STEPS)

        ur.finalize_pending_update_receipt(1, "sys.exit(1)")

        records = _per_run_files(receipt_home)
        assert len(records) == 1
        payload = _read(records[0])
        assert payload["outcome"] != "running"
        assert payload["finished_at"] is not None

    def test_recovery_is_idempotent(self, receipt_home):
        """A second boundary call after recovery must not rewrite or duplicate."""
        _lost_singleton_run(*_FAILED_STEPS)
        path = ur.finalize_pending_update_receipt(1, "sys.exit(1)")
        settled = Path(path).read_text(encoding="utf-8")

        assert ur.finalize_pending_update_receipt(1, "sys.exit(1)") is None

        assert len(_per_run_files(receipt_home)) == 1
        assert Path(path).read_text(encoding="utf-8") == settled

    def test_refused_run_keeps_its_own_vocabulary_through_recovery(self, receipt_home):
        _lost_singleton_run(("venv_preflight", False, "another hermes holds the venv"))

        payload = _read(ur.finalize_pending_update_receipt(2, "sys.exit(2)"))

        assert payload["outcome"] == "refused"
        assert payload["exit_code"] == 2


class TestModuleEvictionIsTheRealLossRoute:
    """Invariant 2 — the loss route is module eviction + re-import, not just a
    cleared variable, so recovery is pinned against a genuinely fresh module."""

    def test_reimported_module_recovers_the_evicted_run(self, receipt_home):
        _lost_singleton_run(*_FAILED_STEPS)
        seeded_before = [path.name for path in _per_run_files(receipt_home)]
        assert len(seeded_before) == 1

        original = sys.modules.get(_MODULE_NAME)
        try:
            # The documented eviction: the cached module object goes away and the
            # next import re-executes it with empty globals (_current is None by
            # construction — nobody re-begins the receipt).
            sys.modules.pop(_MODULE_NAME, None)
            fresh = importlib.import_module(_MODULE_NAME)

            assert fresh is not ur
            assert fresh._current is None

            path = fresh.finalize_pending_update_receipt(1, "sys.exit(1)")
        finally:
            if original is not None:
                sys.modules[_MODULE_NAME] = original
                original._current = None

        assert path is not None and Path(path).is_file()
        payload = _read(path)
        assert payload["outcome"] == "failed"
        assert payload["exit_code"] == 1
        assert [step["name"] for step in payload["steps"]] == [
            name for name, _, _ in _FAILED_STEPS
        ]
        # Recovered in place: the evicted module's record is the one rewritten,
        # so no dangling in-progress file is left beside it.
        assert [p.name for p in _per_run_files(receipt_home)] == seeded_before


_CHILD_BOUNDARY_SCRIPT = """
import json
import hermes_cli.update_receipt as ur

result = ur.finalize_pending_update_receipt(1, "sys.exit(1)")
print(json.dumps({
    "returned": None if result is None else str(result),
    "latest": ur.read_latest_receipt(),
}))
"""


class TestProcessBoundaryRetention:
    """Invariant 3 — a fresh interpreter (the restart/abort-recovery process)
    cannot erase the failed run: its durable record and metadata stay on disk,
    and it can never be read as a completed success.

    NOTE: a fresh process cannot yet *close* another pid's in-progress record
    (recovery is deliberately pid-scoped). Reconciling a dangling ``running``
    record whose owner is dead is a separate follow-up; what is pinned here is
    that the failed run is RETAINED rather than lost, which is the regression.
    """

    @pytest.fixture()
    def process_boundary_home(self, receipt_home, monkeypatch):
        # The child resolves the receipt dir from its own environment, so the
        # temp home has to be the real HERMES_HOME for the child too.
        monkeypatch.setenv("HERMES_HOME", str(receipt_home))
        return receipt_home

    def _run_boundary_in_fresh_process(self, home):
        env = {
            **os.environ,
            "HERMES_HOME": str(home),
            "PYTHONPATH": str(_REPO_ROOT),
        }
        return subprocess.run(
            [sys.executable, "-c", _CHILD_BOUNDARY_SCRIPT],
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_failed_run_is_still_on_disk_after_a_fresh_process(self, process_boundary_home):
        _lost_singleton_run(*_FAILED_STEPS)
        records = _per_run_files(process_boundary_home)
        assert len(records) == 1
        before = records[0].read_text(encoding="utf-8")

        child = self._run_boundary_in_fresh_process(process_boundary_home)

        assert child.returncode == 0, f"child failed: {child.stderr}"
        output = [line for line in child.stdout.splitlines() if line.strip()]
        assert output, f"child produced no output: {child.stderr}"
        report = json.loads(output[-1])
        # The fresh interpreter owns no receipt of its own, so it has nothing to
        # finalize — and must not invent one.
        assert report["returned"] is None

        # RETAINED: the failed run's own record is still there, byte-for-byte,
        # with its identity and step evidence intact.
        assert _per_run_files(process_boundary_home) == records
        assert records[0].read_text(encoding="utf-8") == before
        payload = _read(records[0])
        assert payload["pid"] == os.getpid()
        assert payload["update_id"]
        assert payload["started_at"]
        assert [step["name"] for step in payload["steps"]] == [
            name for name, _, _ in _FAILED_STEPS
        ]
        # ... and no reader is handed a false success: whatever the pointer
        # resolves to (nothing, or the last finalized run), it is never this run
        # claiming it completed.
        assert (report["latest"] or {}).get("outcome") != "success"
        assert payload["outcome"] != "success"


class TestMultipleFailedReceipts:
    """Invariant 4 — consecutive failures each keep their own receipt, and the
    retention cap still bounds the store after the per-run path change."""

    _RUNS = [
        ("git_pull", "remote unreachable"),
        ("pip_install", "dependency resolution failed"),
        ("gateway_restart", "settlement check failed closed"),
    ]

    def test_each_failed_update_keeps_its_own_receipt(self, receipt_home):
        paths = []
        for name, detail in self._RUNS:
            _lost_singleton_run((name, False, detail))
            paths.append(ur.finalize_pending_update_receipt(1, "sys.exit(1)"))

        assert len(set(paths)) == len(self._RUNS)
        assert len(_per_run_files(receipt_home)) == len(self._RUNS)

        # Each record is still traceable to the run that produced it: recovery
        # picked the current run's record, not an older (terminal) neighbour.
        for (name, detail), path in zip(self._RUNS, paths):
            payload = _read(path)
            assert payload["outcome"] == "failed"
            assert [step["name"] for step in payload["steps"]] == [name]
            assert payload["steps"][0]["detail"] == detail

        latest = ur.read_latest_receipt()
        assert latest is not None and latest["outcome"] == "failed"
        assert latest["steps"][0]["name"] == self._RUNS[-1][0]

    def test_retention_cap_still_bounds_failed_receipts(self, receipt_home, monkeypatch):
        monkeypatch.setattr(ur, "_RECEIPT_KEEP", 3)
        attempts = 5
        base = 1_700_000_000.0  # fixed stamps: ordering must not depend on the clock
        for index in range(attempts):
            _lost_singleton_run(("git_pull", False, f"attempt {index}"))
            path = ur.finalize_pending_update_receipt(1, "sys.exit(1)")
            os.utime(path, (base + index, base + index))

        remaining = _per_run_files(receipt_home)
        assert len(remaining) == 3
        # Compare in mtime order, not in ``_per_run_files`` order: the filename is
        # second-resolution, so two runs landing in the same second get a
        # ``update_<id>-2.json`` neighbour that sorts BEFORE ``update_<id>.json``
        # lexicographically. The stamped mtimes are the actual retention key.
        by_mtime = sorted(remaining, key=lambda path: path.stat().st_mtime)
        assert [p for p in (_read(path)["steps"][0]["detail"] for path in by_mtime)] == [
            "attempt 2",
            "attempt 3",
            "attempt 4",
        ]


class TestCorruptedReceiptData:
    """Invariant 5 — corrupt data degrades to "skipped"; it never crashes the
    updater, never blocks recovery of a healthy record, and is never rewritten."""

    def test_torn_latest_pointer_never_breaks_recovery_and_is_repaired(
        self, receipt_home
    ):
        # A previous FINALIZED run is what puts a `latest.json` pointer on disk.
        ur.begin_update_receipt()
        ur.record_step("git_pull", True, "updated checkout")
        ur.finalize_update_receipt("success")
        pointer = _receipt_dir(receipt_home) / "latest.json"
        assert _read(pointer)["outcome"] == "success"

        pointer.write_text('{"schema": 1, "outcome": "suc', encoding="utf-8")

        # The reader is exception-swallowing: a torn pointer is "no receipt",
        # not a traceback.
        assert ur.read_latest_receipt() is None

        _lost_singleton_run(*_FAILED_STEPS)
        path = ur.finalize_pending_update_receipt(1, "sys.exit(1)")

        # Recovery works off the per-run records, so the torn pointer cannot
        # cost the failed run its receipt...
        assert path is not None
        assert _read(path)["outcome"] == "failed"
        # ... and the pointer it repoints is readable again.
        assert _read(pointer)["outcome"] == "failed"
        assert ur.read_latest_receipt()["outcome"] == "failed"

    def test_malformed_records_are_skipped_and_never_rewritten(self, receipt_home):
        _lost_singleton_run(*_FAILED_STEPS)
        healthy = _per_run_files(receipt_home)[0]
        directory = healthy.parent
        pid = os.getpid()
        future = 2_000_000_000.0  # newest first: recovery meets the decoys first

        decoys = {
            "torn": '{"schema": 1, "update_id": "x", "pid": %d, "outcome": "run' % pid,
            "null": "null",
            "not_an_object": json.dumps([{"pid": pid, "outcome": "running"}]),
            "no_outcome": json.dumps(
                {"schema": 1, "update_id": "x", "pid": pid, "started_at": "t"}
            ),
        }
        written = {}
        for index, (label, text) in enumerate(sorted(decoys.items())):
            path = directory / f"update_2099010{index}_000000_{pid}.json"
            path.write_text(text, encoding="utf-8")
            os.utime(path, (future + index, future + index))
            written[path] = text

        path = ur.finalize_pending_update_receipt(1, "sys.exit(1)")

        assert path == healthy
        assert _read(healthy)["outcome"] == "failed"
        for decoy, text in written.items():
            assert decoy.read_text(encoding="utf-8") == text, (
                f"corrupt record {decoy.name} was rewritten instead of skipped"
            )

    def test_recovery_of_a_corrupt_only_store_reports_nothing(self, receipt_home):
        """Every candidate corrupt → "no receipt was open" (None), no crash, and
        the corrupt files are left exactly as found for a human to inspect."""
        directory = _receipt_dir(receipt_home)
        directory.mkdir(parents=True, exist_ok=True)
        torn = directory / f"update_20260101_000000_{os.getpid()}.json"
        torn.write_text('{"schema": 1, "pid":', encoding="utf-8")

        assert ur.finalize_pending_update_receipt(1, "sys.exit(1)") is None

        assert torn.read_text(encoding="utf-8") == '{"schema": 1, "pid":'


class TestSuccessfulUpdateAfterRecovery:
    """Invariant 6 — the fix must not turn good runs bad: a success after a
    recovered failure is recorded (and kept) as a success."""

    def _recovered_failure(self, receipt_home):
        _lost_singleton_run(("git_pull", False, "remote unreachable"))
        return ur.finalize_pending_update_receipt(1, "sys.exit(1)")

    def test_success_after_a_recovered_failure_is_recorded_as_success(self, receipt_home):
        failed_path = self._recovered_failure(receipt_home)

        ur.begin_update_receipt()
        ur.record_step("git_pull", True, "updated checkout")
        success_path = ur.finalize_update_receipt("success")

        assert success_path != failed_path
        assert _read(success_path)["outcome"] == "success"
        # The earlier failure is neither rewritten nor deleted.
        assert _read(failed_path)["outcome"] == "failed"
        assert len(_per_run_files(receipt_home)) == 2
        assert ur.read_latest_receipt()["outcome"] == "success"

    def test_later_boundary_exit_does_not_flip_a_success_to_failed(self, receipt_home):
        failed_path = self._recovered_failure(receipt_home)
        ur.begin_update_receipt()
        ur.record_step("git_pull", True, "updated checkout")
        success_path = ur.finalize_update_receipt("success")

        # The boundary net runs on the way out of a successful update too; it
        # must not adopt the *previous* run's terminal record.
        assert ur.finalize_pending_update_receipt(1, "sys.exit(1)") is None

        assert _read(success_path)["outcome"] == "success"
        assert _read(failed_path)["outcome"] == "failed"
        assert ur.read_latest_receipt()["outcome"] == "success"

    def test_recovery_never_adopts_a_finalized_record(self, receipt_home):
        """The narrowness that keeps recovery safe: only an in-progress record
        for THIS pid is admissible, so a finalized success can't be resurrected
        as a failure."""
        ur.begin_update_receipt()
        ur.record_step("git_pull", True)
        success_path = ur.finalize_update_receipt("success")

        ur._current = None
        assert ur.finalize_pending_update_receipt(1, "sys.exit(1)") is None

        assert _read(success_path)["outcome"] == "success"
        assert len(_per_run_files(receipt_home)) == 1


class TestDashboardReadsTheRecoveredFailure:
    """The point of retention: the dashboard/Desktop can actually SEE the
    failure. Driven end-to-end from a receipt the recovery path produced."""

    @pytest.fixture()
    def client(self, receipt_home, monkeypatch):
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")
        from hermes_cli import web_server_gateway
        from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

        # Dashboard restarted: the in-memory action registries are empty, so the
        # durable receipt is the only surviving evidence of the run.
        actions_dir = receipt_home / "actions"
        actions_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(web_server_gateway, "_ACTION_LOG_DIR", actions_dir)
        monkeypatch.setattr(web_server_gateway, "_ACTION_PROCS", {})
        monkeypatch.setattr(web_server_gateway, "_ACTION_RESULTS", {})
        monkeypatch.setattr(web_server_gateway, "_ACTION_COMMANDS", {})
        monkeypatch.setattr(web_server_gateway, "_ACTION_IDS", {})

        test_client = TestClient(app)
        test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        return test_client

    @pytest.fixture()
    def recovered_failure(self, receipt_home):
        _lost_singleton_run(*_FAILED_STEPS)
        ur.finalize_pending_update_receipt(1, "sys.exit(1)")
        return receipt_home

    def test_recovered_failed_receipt_is_served_to_the_dashboard(
        self, client, recovered_failure
    ):
        resp = client.get("/api/hermes/update/receipt")

        assert resp.status_code == 200
        receipt = resp.json()["receipt"]
        assert receipt["outcome"] == "failed"
        assert receipt["exit_code"] == 1
        assert [step["name"] for step in receipt["steps"]] == [
            name for name, _, _ in _FAILED_STEPS
        ]

    def test_recovered_failure_is_not_reported_as_still_running(
        self, client, recovered_failure
    ):
        """The failed run is over: the client must not be left polling a run
        that can never finish (the operator-visible dead end in #112465)."""
        status = client.get("/api/actions/hermes-update/status").json()

        assert status["running"] is False
        assert status["receipt"]["outcome"] == "failed"

    @pytest.mark.xfail(
        reason=(
            "actions.py::_completed_exit_code only consults a receipt whose outcome is "
            "success/partial, so a durable `failed` receipt reports exit_code None — the "
            "run is visibly not running, but its failure has no exit code. Needs "
            "hermes_cli/web_routers/actions.py to map failed/refused receipts too."
        ),
        strict=False,
    )
    def test_recovered_failure_has_an_exit_code(self, client, recovered_failure):
        status = client.get("/api/actions/hermes-update/status").json()

        assert status["running"] is False
        assert status["exit_code"] == 1
