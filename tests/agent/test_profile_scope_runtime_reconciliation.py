"""Immutable profile scopes compose with request defaults and no-I/O persistence."""
from __future__ import annotations

import os
import threading

import pytest


def test_scoped_send_defaults_keep_owner_precedence_and_parent_environment(tmp_path, monkeypatch):
    from agent import secret_scope as ss
    from hermes_cli import env_loader
    from hermes_cli.send_cmd import _load_hermes_env
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    launch, target = tmp_path / "launch", tmp_path / "target"
    launch.mkdir()
    target.mkdir()
    (target / ".env").write_text("SERVICE_TOKEN=old-dotenv\n", encoding="utf-8")
    (target / "config.yaml").write_text(
        "SERVICE_TOKEN: yaml-must-not-win\nSEND_DEFAULT: target-only\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("SEND_DEFAULT", raising=False)
    env_loader._record_external_secret_snapshot(target, data={"SERVICE_TOKEN": "fresh-vault"}, status="ready")
    prior_mode = ss.is_multiplex_active()
    ss.set_multiplex_active(True)
    home_token = set_hermes_home_override(target)
    scope = ss.build_profile_secret_scope(target)
    token = ss.set_secret_scope(scope, profile_home=str(target))
    before = dict(os.environ)
    try:
        _load_hermes_env()
        published = ss.current_secret_scope()
        assert published is not None
        assert published["SERVICE_TOKEN"] == "fresh-vault"
        assert published["SEND_DEFAULT"] == "target-only"
        assert published.profile_home == scope.profile_home
        assert published.external_generation == scope.external_generation
        assert published.source_status == scope.source_status
        assert published.generation != scope.generation
        assert "SEND_DEFAULT" not in scope
        assert dict(os.environ) == before
        with pytest.raises(RuntimeError, match="home"):
            ss.add_secret_scope_defaults({"FOREIGN_DEFAULT": "refused"}, profile_home=launch)
        assert "FOREIGN_DEFAULT" not in ss.current_secret_scope()
    finally:
        ss.reset_secret_scope(token)
        reset_hermes_home_override(home_token)
        ss.set_multiplex_active(prior_mode)


@pytest.mark.parametrize("target_kind", ["launch", "served"])
def test_persistence_scope_neither_hydrates_nor_waits_for_hydrator(tmp_path, monkeypatch, target_kind):
    from agent import secret_scope as ss
    from hermes_cli import env_loader
    from tui_gateway import server
    from tui_gateway import launch_profile_policy as lpp

    launch, served = tmp_path / "launch", tmp_path / "served"
    launch.mkdir()
    served.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    monkeypatch.setattr(lpp, "_snapshot", None)
    def forbidden(_home):
        raise AssertionError("persistence attempted external hydration")
    monkeypatch.setattr(env_loader, "hydrate_profile_secret_sources", forbidden)
    locked, release = threading.Event(), threading.Event()
    def hydrator():
        with env_loader._SECRET_SOURCE_CACHE_LOCK:
            locked.set()
            release.wait(10)
    worker = threading.Thread(target=hydrator)
    worker.start()
    assert locked.wait(5)
    observed = []
    class Agent:
        _session_messages = [{"role": "user", "content": "preserve me"}]
        def _persist_session(self, messages):
            from hermes_constants import get_hermes_home
            observed.append((get_hermes_home(), messages))
    home = launch if target_kind == "launch" else served
    session = {"agent": Agent(), "profile_home": None if target_kind == "launch" else str(served)}
    previous_sessions = server._sessions
    monkeypatch.setattr(server, "_sessions", {"sid": session})
    monkeypatch.setattr(server, "_sessions_lock", threading.RLock())
    try:
        assert server._flush_sessions_before_exit(budget_s=2.0) == 1
        assert observed == [(home, Agent._session_messages)]
        with pytest.raises(RuntimeError, match="snapshot is failed"):
            ss.build_profile_secret_scope(home, hydrate_external=False, fail_closed_external=True)
    finally:
        release.set()
        worker.join(5)
        server._sessions = previous_sessions
    assert not worker.is_alive()
