"""Idle Desktop ownership and host execution capacity are different resources."""

from types import SimpleNamespace

import pytest

from hermes_cli.active_sessions import active_session_registry_snapshot
from tui_gateway import server


@pytest.fixture
def budget_home(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("max_concurrent_sessions: 2\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(server, "_load_cfg", lambda: {"max_concurrent_sessions": 2})
    monkeypatch.setattr(server, "_canonical_ledger_refusal", lambda session: None)
    return tmp_path


def test_two_idle_desktop_writers_do_not_refuse_a_cloud_job(budget_home):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner

    sessions = [{"session_key": key, "source": "desktop"} for key in ("idle-one", "idle-two")]
    cloud = None
    try:
        for index, session in enumerate(sessions):
            assert server._ensure_active_session_slot(f"viewer-{index}", session) is None
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(max_concurrent_sessions=2)
        runner._is_session_running = lambda key: False
        runner._resolve_profile_home_for_source = lambda source: budget_home
        source = SimpleNamespace(platform=Platform.TELEGRAM, chat_id="synthetic", user_id="synthetic")
        cloud, refusal = runner._claim_active_session_slot("cloud-job", source)
        assert refusal is None, "idle Desktop ownership must not consume cloud execution slots"
        assert cloud is not None
        assert len(active_session_registry_snapshot()) == 3
    finally:
        if cloud is not None:
            cloud.release()
        for session in sessions:
            server._release_active_session_slot(session)


def test_execution_capacity_is_shared_across_profiles_and_surfaces(budget_home):
    from hermes_cli.execution_budget import try_acquire_execution_permit

    profile = budget_home / "profiles" / "secondary"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("max_concurrent_sessions: 99\n", encoding="utf-8")
    permits = []
    try:
        for home, surface in ((budget_home, "desktop"), (profile, "gateway:telegram")):
            permit, refusal = try_acquire_execution_permit(profile_home=home, execution_id=surface, surface=surface)
            assert refusal is None and permit is not None
            permits.append(permit)
        permit, refusal = try_acquire_execution_permit(profile_home=profile, execution_id="third", surface="cli")
        assert permit is None and refusal.reason == "MAX_CONCURRENT_SESSIONS"
        permits.pop().release()
        permit, refusal = try_acquire_execution_permit(profile_home=profile, execution_id="third", surface="cli")
        assert refusal is None and permit is not None
        permits.append(permit)
    finally:
        for permit in permits:
            permit.release()


def test_unknown_execution_registry_refuses_work(budget_home):
    from hermes_cli.execution_budget import try_acquire_execution_permit

    runtime = budget_home / "runtime"
    runtime.mkdir()
    (runtime / "execution_permits.json").write_text("{", encoding="utf-8")
    permit, refusal = try_acquire_execution_permit(profile_home=budget_home, execution_id="new", surface="desktop")
    assert permit is None and refusal.reason == "SESSION_COORDINATION_UNAVAILABLE"


def test_status_reports_execution_instead_of_idle_writer_capacity(budget_home, capsys):
    from hermes_cli.status import _render_execution_slots
    from hermes_cli.execution_budget import try_acquire_execution_permit

    permit, refusal = try_acquire_execution_permit(execution_id="working", surface="gateway")
    assert refusal is None
    try:
        _render_execution_slots()
        assert "1/2 in use" in capsys.readouterr().out
    finally:
        permit.release()


def test_unknown_own_process_birth_refuses_new_execution(budget_home, monkeypatch):
    from hermes_cli import execution_budget

    monkeypatch.setattr(execution_budget.registry, "_own_start_time", lambda: None)
    permit, refusal = execution_budget.try_acquire_execution_permit(execution_id="unknown", surface="cli")
    try:
        assert permit is None
        assert refusal.reason == "SESSION_COORDINATION_UNAVAILABLE"
    finally:
        if permit is not None:
            permit.release()


def test_borrowed_worker_keeps_slot_after_parent_releases_and_stale_release_cannot_free_successor(budget_home):
    from hermes_cli.execution_budget import borrow_execution_permit, try_acquire_execution_permit

    parent, _ = try_acquire_execution_permit(profile_home=budget_home, execution_id="parent", surface="desktop")
    child, refusal = borrow_execution_permit(parent.vouch())
    assert refusal is None
    other, _ = try_acquire_execution_permit(profile_home=budget_home, execution_id="other", surface="managed")
    successor = None
    try:
        parent.release()
        permit, refusal = try_acquire_execution_permit(profile_home=budget_home, execution_id="third", surface="cli")
        assert permit is None and refusal.reason == "MAX_CONCURRENT_SESSIONS"
        child.release()
        successor, refusal = try_acquire_execution_permit(profile_home=budget_home, execution_id="third", surface="cli")
        assert refusal is None
        parent.release()
        child.release()
        stale, refusal = borrow_execution_permit(parent.vouch())
        assert stale is None and refusal.reason == "SESSION_COORDINATION_UNAVAILABLE"
        permit, refusal = try_acquire_execution_permit(profile_home=budget_home, execution_id="fourth", surface="desktop")
        assert permit is None and refusal.reason == "MAX_CONCURRENT_SESSIONS"
    finally:
        parent.release()
        child.release()
        other.release()
        if successor is not None:
            successor.release()
