"""Concurrent discovery cannot shrink durable adoption membership or invert locks."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest

from hermes_state import SessionDB
from tests.hermes_state.test_adoption_transfer_records import _open, _retire_fails_for, _seed_compression_lineage


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


def test_concurrent_absence_observation_cannot_shrink_saved_membership(home, monkeypatch):
    donor, target = _open(home, default=True), _open(home)
    discovered, release = threading.Event(), threading.Event()
    try:
        _seed_compression_lineage(donor, "concurrent-root", "concurrent-tip")
        export = donor.export_session_lineage
        paused = False

        def delayed_export(session_id, **kwargs):
            nonlocal paused
            if threading.current_thread().name.startswith("stale-adopter") and not paused:
                paused = True
                discovered.set()
                assert release.wait(15), "second request did not release the first"
            return export(session_id, **kwargs)

        monkeypatch.setattr(donor, "export_session_lineage", delayed_export)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="stale-adopter") as pool:
            future = pool.submit(target.adopt_session_lineage_from, donor, "concurrent-root")
            try:
                assert discovered.wait(15), "first request did not reach optimistic discovery"
                with monkeypatch.context() as patch:
                    patch.setattr(SessionDB, "_retire_donor_segment", _retire_fails_for("concurrent-tip"))
                    partial = target.adopt_session_lineage_from(donor, "concurrent-tip")
                assert partial["adopted"] and not partial["donor_retired"]
            finally:
                release.set()
            result = future.result(timeout=20)
        members = target._conn.execute("SELECT member_id FROM adoption_transfers ORDER BY member_order").fetchall()
        assert [row[0] for row in members] == ["concurrent-root", "concurrent-tip"]
        if result.get("donor_retired"):
            assert donor.get_session("concurrent-tip")["end_reason"] == "adopted_by_profile"
        completed = target.adopt_session_lineage_from(donor, "concurrent-root")
        assert completed["donor_retired"]
        assert completed["member_ids"] == ["concurrent-root", "concurrent-tip"]
    finally:
        release.set()
        donor.close()
        target.close()


def test_expanding_discovery_releases_prior_locks_before_reacquiring(home, monkeypatch):
    import hermes_state_portability as portability
    donor, target = _open(home, default=True), _open(home)
    original = portability.session_lineage_guard
    active, expanded_under_lock = [], []
    grew = False
    try:
        donor.create_session("growing-root", "tui")
        donor.append_message("growing-root", "user", "first segment")

        @contextmanager
        def observe_guard(path, ids, **kwargs):
            nonlocal grew
            current = frozenset(ids)
            if not grew:
                grew = True
                donor._conn.execute(
                    "INSERT INTO sessions(id,source,started_at,parent_session_id,message_count) VALUES (?,'tui',?,?,0)",
                    ("growing-tip", time.time(), "growing-root"),
                )
                donor._conn.commit()
                donor.append_message("growing-tip", "user", "new segment")
                donor.end_session("growing-root", "compression")
            if active and not current.issubset(frozenset().union(*active)):
                expanded_under_lock.append(current)
            with original(path, current, **kwargs) as snapshot:
                active.append(current)
                try:
                    yield snapshot
                finally:
                    active.pop()

        monkeypatch.setattr(portability, "session_lineage_guard", observe_guard)
        result = target.adopt_session_lineage_from(donor, "growing-root")
        assert not expanded_under_lock, "expanded lock set was acquired while old locks remained held"
        assert result["donor_retired"]
        assert result["member_ids"] == ["growing-root", "growing-tip"]
    finally:
        donor.close()
        target.close()
