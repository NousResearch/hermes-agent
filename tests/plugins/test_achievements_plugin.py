"""Tests for the bundled hermes-achievements dashboard plugin.

These target the two behaviors that matter for official integration:

* The 200-session scan cap is removed — the plugin now walks the entire
  session history by default. Lifetime badges (tens of thousands of
  tool calls) were unreachable before this fix on long-running installs.
* First-ever scans run in a background thread so the dashboard request
  path never blocks, even on 8000+ session databases where a cold scan
  takes minutes.

The upstream repo ships its own unittest suite under
``plugins/hermes-achievements/tests/`` covering the achievement engine
internals (tier math, secret-state handling, catalog invariants). These
tests live at the hermes-agent level and focus on the integration
contract: the plugin scans ALL of your sessions, not the first 200.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

PLUGIN_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "hermes-achievements"
    / "dashboard"
    / "plugin_api.py"
)


@pytest.fixture
def plugin_api(tmp_path, monkeypatch):
    """Load plugin_api with isolated ~/.hermes so state/snapshot files don't collide.

    We load the module fresh per test because the plugin keeps module-level
    caches (``_SNAPSHOT_CACHE``, ``_SCAN_STATUS``, background thread handle).
    Reloading gives each test a clean world.
    """
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    spec = importlib.util.spec_from_file_location(
        f"plugin_api_test_{id(tmp_path)}", PLUGIN_MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Stash monkeypatch so ``_install_fake_session_db`` can swap the storage
    # reader door with auto-restoration. Without this, a raw assignment would
    # leak the fake into later tests in the same process.
    module._test_monkeypatch = monkeypatch
    yield module


class _FakeReader:
    """Stand-in for the storage interface reader that records scan calls."""

    def __init__(self, session_count: int):
        self.session_count = session_count
        self.last_limit: Optional[int] = None
        self.last_include_children: Optional[bool] = None
        self.list_calls = 0
        self.messages_calls = 0
        self.sessions = self.messages = self

    def ready(self) -> None:
        pass

    def list_sessions(self, query, window) -> list[Any]:
        self.last_limit = window.limit
        self.last_include_children = query.include_children
        self.list_calls += 1
        effective = min(self.session_count, window.limit)
        now = int(time.time())
        return [
            SimpleNamespace(
                session=SimpleNamespace(
                    session_id=f"sess-{i}", title=f"Session {i}", started_at=now - (self.session_count - i) * 60,
                    source="cli", model="test-model"),
                preview=f"preview {i}",
                last_active=now - (self.session_count - i) * 60 + 30,
            )
            for i in range(effective)
        ]

    def read_transcript(self, session_id: str, request) -> list[Any]:
        self.messages_calls += 1
        return [
            SimpleNamespace(as_wire_row=lambda row=row: row)
            for row in (
                {"role": "user", "content": f"ask {session_id}"},
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"name": "terminal"}}],
                },
                {"role": "tool", "tool_name": "terminal", "content": "ok"},
            )
        ]


def _install_fake_session_db(plugin_api, fake_db):
    """Inject a fake reader so ``scan_sessions`` opens it through its local import of the door.

    Uses the monkeypatch stashed on ``plugin_api`` by the fixture, so the
    swap is auto-restored at test teardown and cannot leak into unrelated
    tests in the same process.
    """
    @contextmanager
    def fake_reader(_target):
        yield fake_db

    plugin_api._test_monkeypatch.setattr("persistence_assembly.open_persistence_reader", fake_reader)


def test_scan_sessions_default_scans_all_history_not_first_200(plugin_api):
    """Bug regression: ``scan_sessions()`` used to cap at limit=200.

    A user with 8000+ sessions would only see ~2% of their history in
    achievement totals, making lifetime badges unreachable. The default
    now asks ``list_sessions`` for a window of ``sys.maxsize`` rows (unlimited).
    """
    fake_db = _FakeReader(session_count=500)  # > old 200 cap
    _install_fake_session_db(plugin_api, fake_db)

    result = plugin_api.scan_sessions()

    assert fake_db.last_limit == sys.maxsize, (
        "scan_sessions() must ask list_sessions for an unlimited window "
        f"by default, got {fake_db.last_limit}"
    )
    assert fake_db.last_include_children is True, (
        "scan_sessions() must include subagent/compression child sessions so "
        "tool calls made in delegated agents still count toward achievements"
    )
    assert len(result["sessions"]) == 500
    assert result["scan_meta"]["sessions_total"] == 500


def test_evaluate_all_first_run_returns_pending_and_starts_background_scan(plugin_api):
    """First-ever evaluate_all with no cache returns a pending placeholder
    immediately and kicks off a background scan thread. Cold scans on
    large DBs take minutes — blocking the dashboard request path is not
    acceptable.
    """
    fake_db = _FakeReader(session_count=50)
    _install_fake_session_db(plugin_api, fake_db)

    # Wrap _run_scan_and_update_cache so we can release it on demand,
    # simulating a slow cold scan without actually waiting.
    scan_started = threading.Event()
    allow_scan_finish = threading.Event()
    original_run = plugin_api._run_scan_and_update_cache

    def gated_run(*args, **kwargs):
        scan_started.set()
        allow_scan_finish.wait(timeout=5)
        original_run(*args, **kwargs)

    plugin_api._run_scan_and_update_cache = gated_run

    t0 = time.time()
    result = plugin_api.evaluate_all()
    elapsed = time.time() - t0

    # Immediate return — should not block waiting for the scan.
    assert elapsed < 1.0, f"evaluate_all blocked for {elapsed:.2f}s on first run"
    assert result["scan_meta"]["mode"] == "pending"
    assert result["unlocked_count"] == 0
    # Catalog still rendered so UI has something to draw.
    assert result["total_count"] >= 60

    # Background scan is running.
    assert scan_started.wait(timeout=2), "background scan did not start"

    # Let the scan complete, then a second call returns real data.
    allow_scan_finish.set()
    # Wait for thread to finish.
    thread = plugin_api._BACKGROUND_SCAN_THREAD
    assert thread is not None
    thread.join(timeout=5)
    assert not thread.is_alive()

    second = plugin_api.evaluate_all()
    assert second["scan_meta"]["mode"] != "pending"
    assert second["scan_meta"].get("sessions_total") == 50


def test_start_background_scan_is_idempotent_while_running(plugin_api):
    """Multiple concurrent dashboard requests must not spawn duplicate scans."""
    fake_db = _FakeReader(session_count=5)
    _install_fake_session_db(plugin_api, fake_db)

    release = threading.Event()
    original_run = plugin_api._run_scan_and_update_cache

    def gated_run(*args, **kwargs):
        release.wait(timeout=5)
        original_run(*args, **kwargs)

    plugin_api._run_scan_and_update_cache = gated_run

    plugin_api._start_background_scan()
    first_thread = plugin_api._BACKGROUND_SCAN_THREAD
    assert first_thread is not None and first_thread.is_alive()

    plugin_api._start_background_scan()
    plugin_api._start_background_scan()

    assert plugin_api._BACKGROUND_SCAN_THREAD is first_thread

    release.set()
    first_thread.join(timeout=5)


def test_background_scan_publishes_partial_snapshots(plugin_api):
    """The background scanner publishes intermediate snapshots to the cache
    every ~N sessions. Each dashboard refresh during a long cold scan sees
    more badges unlocked instead of staring at zeros for minutes and then
    having everything pop at the end.
    """
    fake_db = _FakeReader(session_count=750)
    _install_fake_session_db(plugin_api, fake_db)

    # Record every partial snapshot the scanner publishes.
    partial_snapshots: list[dict[str, Any]] = []
    original_compute_from_scan = plugin_api._compute_from_scan

    def recording_compute(scan, *, is_partial=False):
        result = original_compute_from_scan(scan, is_partial=is_partial)
        if is_partial:
            partial_snapshots.append(result)
        return result

    plugin_api._compute_from_scan = recording_compute

    # scan 750 sessions with progress_every=250 → expect 2 intermediate
    # publications (at 250 and 500; the final 750 call goes through the
    # finished, non-partial path).
    plugin_api._run_scan_and_update_cache(publish_partial_snapshots=True)

    assert len(partial_snapshots) >= 2, (
        f"expected at least 2 partial publications on a 750-session scan with "
        f"progress_every=250, got {len(partial_snapshots)}"
    )
    # Partial snapshots should report growing session counts.
    counts = [p["scan_meta"].get("sessions_scanned_so_far") for p in partial_snapshots]
    assert counts == sorted(counts), f"partial session counts not monotonic: {counts}"
    assert counts[0] < 750 and counts[-1] < 750, (
        f"partial counts should be less than the final total; got {counts}"
    )
    # Every partial reports the expected end-state total so the UI can
    # show an accurate progress bar.
    for p in partial_snapshots:
        assert p["scan_meta"].get("sessions_expected_total") == 750

    # Final snapshot in cache is the real (non-partial) one.
    final = plugin_api._SNAPSHOT_CACHE
    assert final is not None
    assert final["scan_meta"].get("mode") != "in_progress"
    assert final["scan_meta"].get("sessions_total") == 750


def test_partial_snapshots_do_not_persist_unlock_timestamps(plugin_api):
    """Intermediate snapshots must not write to state.json — an unlock
    that appears at 30% scan progress could disappear when a later session
    rebalances the aggregate. Only the final snapshot records ``unlocked_at``.
    """
    fake_db = _FakeReader(session_count=10)
    _install_fake_session_db(plugin_api, fake_db)

    # Seed empty state, then invoke partial compute directly.
    plugin_api.save_state({"unlocks": {}})
    partial_scan = {
        "sessions": [{"session_id": "x", "tool_call_count": 99999, "tool_names": set()}],
        "aggregate": {"max_tool_calls_in_session": 99999, "total_tool_calls": 99999},
        "scan_meta": {"mode": "in_progress"},
    }
    result = plugin_api._compute_from_scan(partial_scan, is_partial=True)

    # Some achievements should evaluate as unlocked in this aggregate...
    assert any(a["unlocked"] for a in result["achievements"])

    # ...but state.json on disk stays empty (no timestamps were recorded).
    persisted = plugin_api.load_state()
    assert persisted.get("unlocks", {}) == {}, (
        "partial scans must not record unlock timestamps — a later session "
        "could change whether the badge deserves to be unlocked yet"
    )


def test_scan_sessions_never_opens_a_writable_session_db(plugin_api, tmp_path, monkeypatch):
    """The scan is a pure read that runs inside the dashboard process (per background scan,
    per /rescan). A writable store open there was one more writer connection on the
    dashboard's own state.db each time — the same-process handle leak behind the
    ``N live SessionDB handles`` precursor (#100896). Real store, real open, no fakes."""
    import persistence_assembly
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_messages, seed_session

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with open_persistence(profile_scope(home=tmp_path)) as seed:
        seed_session(seed, "s1", "cli")
        seed_messages(seed, "s1", {"role": "user", "content": "hello"})

    writable_opens = []
    real_assemble = persistence_assembly._assemble

    def spy(provider, stores, *, read_only=False, **options):
        if not read_only:
            writable_opens.append(dict(stores))
        return real_assemble(provider, stores, read_only=read_only, **options)

    monkeypatch.setattr(persistence_assembly, "_assemble", spy)

    result = plugin_api.scan_sessions()

    assert result.get("error") is None
    assert [s["session_id"] for s in result["sessions"]] == ["s1"]
    assert writable_opens == [], "the achievements scan must attach read-only"


def test_scan_sessions_names_the_store_it_could_not_open(plugin_api):
    """A store that cannot open reports main's ``Could not open SessionDB: <reason>``."""
    def broken_reader(_target):
        raise RuntimeError("disk gone")

    plugin_api._test_monkeypatch.setattr("persistence_assembly.open_persistence_reader", broken_reader)

    result = plugin_api.scan_sessions()

    assert result["error"] == "Could not open SessionDB: disk gone"
    assert result["sessions"] == []
