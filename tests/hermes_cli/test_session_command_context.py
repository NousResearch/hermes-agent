"""Persistence and capability contracts of host-owned plugin session commands."""

import json
from types import SimpleNamespace

import pytest

from hermes_cli.session_commands import (
    SessionCommandHandler, SessionCommandRequest, make_session_context,
    prepare_session_command, run_session_command,
)
from hermes_state import SessionDB

KEY = "example_override"


@pytest.fixture
def policy(monkeypatch):
    entry = {"config_path": "feature.enabled", "default": False,
             "apply": lambda runtime, value: runtime.set_context_option("example", value)}
    monkeypatch.setattr("agent.session_policies.policy_entries", lambda: {KEY: entry})


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    value = SessionDB(db_path=tmp_path / "state.db")
    yield value
    value.close()


def context(db, *, agent=None, config=None, ensure=None, sid="session"):
    return make_session_context(agent=agent, session_db=db, session_id=sid,
                                ensure_session=ensure,
                                profile_home=str(db.db_path.parent),
                                config_loader=config or (lambda: {"feature": {"enabled": False}}))


def invoke(ctx, *, write, fn):
    handler = SessionCommandHandler("example", lambda raw: SessionCommandRequest(write, raw), fn)
    return run_session_command(handler, prepare_session_command(handler, "value"), ctx)


def test_commit_precedes_engine_and_preserves_unrelated_json(policy, db):
    db.create_session("session", source="cli", model_config={"unrelated": {"value": 7}})
    calls = []

    class Engine:
        def set_session_option(self, name, enabled):
            calls.append(enabled)
            assert SessionDB.session_policy_override(db.get_session("session"), KEY) is enabled
            return True

    agent = SimpleNamespace(session_id="session", context_compressor=Engine(),
                            _session_init_model_config={"keep": 5})
    ctx = context(db, agent=agent, ensure=lambda: None)
    assert invoke(ctx, write=True, fn=lambda req, session: session.set_override(KEY, True)) is True
    assert calls == [True]
    assert agent._session_init_model_config == {"keep": 5, KEY: True}
    assert json.loads(db.get_session("session")["model_config"])["unrelated"] == {"value": 7}
    assert ctx.writes == {KEY: True}
    invoke(ctx, write=True, fn=lambda req, session: session.set_override(KEY, None))
    assert KEY not in json.loads(db.get_session("session")["model_config"])


def test_status_does_not_ensure_or_create(policy, db):
    ctx = context(db, ensure=lambda: pytest.fail("status ensured a row"))
    assert invoke(ctx, write=False, fn=lambda req, session: session.get_override(KEY)) is None
    assert db.get_session("session") is None
    with pytest.raises(ValueError, match="read-only"):
        invoke(ctx, write=False, fn=lambda req, session: session.set_override(KEY, True))
    assert db.get_session("session") is None


def test_config_failure_precedes_row_ensure(policy, db):
    def unavailable():
        raise OSError("unavailable config")

    ctx = context(db, config=unavailable, ensure=lambda: pytest.fail("created before config read"))
    with pytest.raises(OSError, match="unavailable config"):
        invoke(ctx, write=True, fn=lambda req, session: session.set_override(KEY, True))
    assert not ctx.writes
    assert db.get_session("session") is None


def test_db_failure_leaves_live_metadata_and_engine_unchanged(policy, db, monkeypatch):
    db.create_session("session", source="cli")
    agent = SimpleNamespace(session_id="session", _session_init_model_config={"keep": True})
    ctx = context(db, agent=agent, ensure=lambda: None)
    monkeypatch.setattr(db, "set_session_policy_override", lambda *args: (_ for _ in ()).throw(OSError("write failed")))
    with pytest.raises(OSError, match="write failed"):
        invoke(ctx, write=True, fn=lambda req, session: session.set_override(KEY, True))
    assert agent._session_init_model_config == {"keep": True}
    assert not ctx.writes


@pytest.mark.parametrize("value", ["true", 1, 0, {}, []])
def test_override_requires_literal_boolean_before_ensure(policy, db, value):
    ctx = context(db, ensure=lambda: pytest.fail("invalid value ensured row"))
    with pytest.raises(ValueError, match="literal bool"):
        invoke(ctx, write=True, fn=lambda req, session: session.set_override(KEY, value))


@pytest.mark.parametrize("value", ["true", 1, 0, {}, [], None])
def test_malformed_legacy_value_inherits(value):
    assert SessionDB.session_policy_override({"model_config": json.dumps({KEY: value})}, KEY) is None


def test_foreign_agent_and_profile_are_refused(db, tmp_path):
    with pytest.raises(ValueError, match="identity mismatch"):
        context(db, agent=SimpleNamespace(session_id="foreign"))
    with pytest.raises(ValueError, match="profile mismatch"):
        make_session_context(agent=None, session_db=db, session_id="session", profile_home=tmp_path / "foreign")


def test_missing_exact_row_and_corrupt_model_config_are_refused(policy, db):
    ctx = context(db, ensure=lambda: None)
    with pytest.raises(ValueError, match="row not found"):
        invoke(ctx, write=True, fn=lambda req, session: session.set_override(KEY, True))
    db.create_session("session", source="cli")
    db._execute_write(lambda conn: conn.execute("UPDATE sessions SET model_config='invalid' WHERE id='session'"))
    with pytest.raises(ValueError, match="malformed JSON"):
        invoke(ctx, write=True, fn=lambda req, session: session.set_override(KEY, True))
    assert not ctx.writes


def test_engine_failure_retains_durable_commit(policy, db):
    db.create_session("session", source="cli")

    class Engine:
        def set_session_option(self, name, enabled):
            raise RuntimeError("unsupported operation")

    agent = SimpleNamespace(session_id="session", context_compressor=Engine())
    ctx = context(db, agent=agent, ensure=lambda: None)
    assert invoke(ctx, write=True, fn=lambda req, session: session.set_override(KEY, True)) is False
    assert SessionDB.session_policy_override(db.get_session("session"), KEY) is True
    assert ctx.writes == {KEY: True}


def test_unregistered_policy_cannot_be_written(policy, db):
    ctx = context(db, ensure=lambda: pytest.fail("foreign key ensured row"))
    with pytest.raises(ValueError, match="unregistered"):
        invoke(ctx, write=True, fn=lambda req, session: session.set_override("unknown", True))


def test_active_turn_lease_refuses_write_and_releases_command_lease(policy, db):
    db.create_session("session", source="cli")
    assert db.try_acquire_session_turn_lease("session", "live-turn") is True
    ctx = context(db, ensure=lambda: pytest.fail("busy command ensured row"))
    try:
        with pytest.raises(ValueError, match="session busy"):
            invoke(ctx, write=True, fn=lambda req, session: session.set_override(KEY, True))
        assert SessionDB.session_policy_override(db.get_session("session"), KEY) is None
        assert not ctx.writes
    finally:
        db.release_session_turn_lease("session", "live-turn")
    ctx = context(db, ensure=lambda: None)
    invoke(ctx, write=True, fn=lambda req, session: session.set_override(KEY, True))
    assert db.try_acquire_session_turn_lease("session", "next-turn") is True
    db.release_session_turn_lease("session", "next-turn")


def test_retained_capability_cannot_be_used_after_command_returns(policy, db):
    ctx = context(db, ensure=lambda: None)
    invoke(ctx, write=False, fn=lambda req, session: session.get_override(KEY))
    with pytest.raises(ValueError, match="no longer active"):
        ctx.get_override(KEY)
    with pytest.raises(ValueError, match="no longer active"):
        ctx.set_override(KEY, True)


def test_ensure_rotation_cannot_retarget_setting(policy, db):
    db.create_session("session", source="cli")
    agent = SimpleNamespace(session_id="session")
    ctx = context(db, agent=agent, ensure=lambda: setattr(agent, "session_id", "replacement"))
    with pytest.raises(ValueError, match="identity changed"):
        invoke(ctx, write=True, fn=lambda req, session: session.set_override(KEY, True))
    assert SessionDB.session_policy_override(db.get_session("session"), KEY) is None
    assert db.get_session("replacement") is None
    assert not ctx.writes


def test_command_config_missing_home_and_invalid_yaml_do_not_write(tmp_path, monkeypatch):
    from hermes_cli.session_commands import load_session_command_config

    home = tmp_path / 'missing-profile'
    monkeypatch.setenv('HERMES_HOME', str(home))
    assert load_session_command_config() == {}
    assert not home.exists()
    home.mkdir()
    path = home / 'config.yaml'
    path.write_text('compression: [unclosed')
    with pytest.raises(Exception):
        load_session_command_config()
    assert sorted(p.name for p in home.iterdir()) == ['config.yaml']
