"""Durable work probes use existing leases, respect profiles and fail awake."""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import pytest

from gateway.scale_to_zero import gateway_is_idle
from gateway.scale_to_zero_work import has_durable_work, profile_has_durable_work
from hermes_state import SessionDB
from hermes_state_pidns import holder_namespace_token
from tests.gateway.test_scale_to_zero_external_work import _idle_runner


def test_turn_lease_stays_busy_without_recent_progress_and_releases(tmp_path):
    with closing(SessionDB(tmp_path / "state.db")) as db:
        holder = f"pid={os.getpid()}{holder_namespace_token()}:turn=inert"
        assert db.try_acquire_session_turn_lease(
            "conversation", holder, ttl_seconds=3600
        )
        assert profile_has_durable_work(tmp_path, now=time.time() + 1800)
        db.release_session_turn_lease("conversation", holder)
        assert not profile_has_durable_work(tmp_path, now=time.time())


def test_expired_turn_lease_does_not_pin_machine(tmp_path):
    with closing(SessionDB(tmp_path / "state.db")) as db:
        assert db.try_acquire_session_turn_lease("conversation", "unknown-holder")
        assert not profile_has_durable_work(tmp_path, now=time.time() + 600)


def test_probe_never_creates_a_missing_store_and_accepts_older_schema(tmp_path):
    assert not profile_has_durable_work(tmp_path, now=time.time())
    assert not (tmp_path / "state.db").exists()
    with closing(sqlite3.connect(tmp_path / "state.db")) as conn:
        conn.execute("CREATE TABLE old_schema (value TEXT)")
    assert not profile_has_durable_work(tmp_path, now=time.time())


def test_unreadable_store_keeps_gateway_awake(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "state.db").write_text("corrupt", encoding="utf-8")
    assert not gateway_is_idle(_idle_runner())


def test_unreadable_store_identity_keeps_gateway_awake(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    stat = Path.stat

    def denied_store(path, *args, **kwargs):
        if path == tmp_path / "state.db":
            raise PermissionError("synthetic access denial")
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied_store)
    assert not gateway_is_idle(_idle_runner())


def test_locked_store_keeps_gateway_awake_then_recovers(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with closing(sqlite3.connect(tmp_path / "state.db")) as conn:
        conn.execute("CREATE TABLE old_schema (value TEXT)")
        conn.commit()
        conn.execute("BEGIN EXCLUSIVE")
        assert not gateway_is_idle(_idle_runner())
        conn.rollback()
        assert gateway_is_idle(_idle_runner())


def test_profile_work_ignores_request_override_and_includes_parked_profile(
    tmp_path, monkeypatch
):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    root = tmp_path / "install"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text("", encoding="utf-8")
    (profile / "gateway.parked").touch()
    foreign = tmp_path / "another-install"
    foreign.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    with closing(SessionDB(profile / "state.db")) as db:
        assert db.try_acquire_session_turn_lease("conversation", "unknown-holder")
        token = set_hermes_home_override(str(foreign))
        try:
            assert has_durable_work()
            db.release_session_turn_lease("conversation", "unknown-holder")
            with closing(SessionDB(foreign / "state.db")) as other:
                assert other.try_acquire_session_turn_lease("foreign", "unknown-holder")
                assert not has_durable_work()
        finally:
            reset_hermes_home_override(token)


@pytest.mark.parametrize(
    "started,current,expected",
    [(100, 100, True), (100, 1000, False), (None, 200, True), (100, None, True)],
)
def test_delegation_owner_identity_is_conservative(
    monkeypatch, started, current, expected
):
    from gateway.scale_to_zero_work import _delegation_owner_may_be_live

    monkeypatch.setattr("gateway.status._pid_exists", lambda _pid: True)
    monkeypatch.setattr("gateway.status.get_process_start_time", lambda _pid: current)
    assert _delegation_owner_may_be_live(123, started) is expected
