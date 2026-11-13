"""The shared owner's ``config.set`` refuses a key the administrator's managed scope pins, like
``hermes config set``; before, it wrote the pinned key and answered success while the managed value
kept winning on every read. A managed scope that pins something else must not block other keys.
"""
from types import SimpleNamespace

import pytest
import hermes_agent.hermes_yaml as yaml
from hermes_agent.hermes_state_runtime import RuntimeStoreError

from hermes_agent.gateway.session_tui_config import _set_display, _write_profile_config


@pytest.fixture
def home(tmp_path, monkeypatch):
    from hermes_agent.hermes_cli import managed_scope
    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    (managed / "config.yaml").write_text("display:\n  skin: corp\n")
    (home / "config.yaml").write_text("display:\n  tui_theme: dark\n")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    yield home
    managed_scope.invalidate_managed_cache()


def _owner_write(home, key, value, calls=None):
    connection, calls = SimpleNamespace(authority=SimpleNamespace(profile_id=str(home))), [] if calls is None else calls

    def edit(cfg):
        calls.append(key)
        return _set_display(cfg, key, value)
    return _write_profile_config(connection, edit)


@pytest.mark.parametrize("raw", [None, "display:\n  skin: mono\n"], ids=["unset", "stale-raw-equal"])
def test_owner_config_set_refuses_a_managed_key_and_writes_nothing(home, raw):
    if raw:  # the raw file already holds the requested value: no change there, the managed one still wins
        (home / "config.yaml").write_text(raw)
    before = (home / "config.yaml").read_text()
    with pytest.raises(RuntimeStoreError) as refused:
        _owner_write(home, "skin", "mono")
    assert (refused.value.reason, refused.value.field) == ("permission_denied", "display.skin")
    assert (home / "config.yaml").read_text() == before


def test_owner_config_set_still_writes_a_key_the_scope_does_not_pin(home):
    assert _owner_write(home, "theme", "light") == {"value": "light"}
    assert yaml.safe_load((home / "config.yaml").read_text())["display"]["tui_theme"] == "light"


@pytest.mark.parametrize("managed", [True, False], ids=["managed", "unmanaged"])
@pytest.mark.parametrize(("key", "value", "expected"), [
    ("theme", "light", {"value": "light"}),
    ("focus", "status", {"value": "off", "tool_progress": "all"}),  # read-only: sets no leaf, even a pinned one
])
def test_owner_config_set_runs_its_edit_once(home, monkeypatch, managed, key, value, expected):
    from hermes_agent.hermes_cli import managed_scope
    if managed:
        (home.parent / "managed" / "config.yaml").write_text("display:\n  focus_view: true\n")
    else:
        monkeypatch.delenv("HERMES_MANAGED_DIR")
    managed_scope.invalidate_managed_cache()
    calls = []
    assert _owner_write(home, key, value, calls) == expected
    assert calls == [key]


@pytest.mark.parametrize(("scope", "pins_effort"), [("global", True), ("global", False), ("session", True)])
def test_owner_global_reasoning_level_is_refused_before_anything_changes(home, monkeypatch, scope, pins_effort):
    """``config.set reasoning <level> scope=global`` saves agent.reasoning_effort: a pinned one is refused
    before the session's level is committed or config.yaml is written. A session level writes no config key."""
    import asyncio
    from hermes_agent.gateway import session_setting_commands, session_tui_config
    from hermes_agent.hermes_cli import managed_scope
    if pins_effort:
        (home.parent / "managed" / "config.yaml").write_text("agent:\n  reasoning_effort: low\n")
        managed_scope.invalidate_managed_cache()
    calls = []

    async def commit(authority, actor, ref, value, persist_global):
        calls.append(("commit", persist_global))
    monkeypatch.setattr(session_tui_config, "_authorize_write", lambda *a: None)
    monkeypatch.setattr(session_setting_commands, "commit_reasoning_command", commit)
    runner = SimpleNamespace(_apply_reasoning_selection=lambda route, platform, value, persist_global: calls.append(
        ("apply", persist_global)))
    authority = SimpleNamespace(profile_id=str(home), authorize=lambda *a: None, runner=runner,
                                sessions={"s": SimpleNamespace(route="s")})
    connection = SimpleNamespace(authority=authority, actor=SimpleNamespace(subject="owner"))
    params = {"session_id": "s", "key": "reasoning", "value": "high", "scope": scope}

    def run():
        return asyncio.run(session_tui_config._set_command_setting(connection, SimpleNamespace(session_id="s"), params))
    if scope == "global" and pins_effort:
        with pytest.raises(RuntimeStoreError) as refused:
            run()
        assert (refused.value.reason, refused.value.field) == ("permission_denied", "agent.reasoning_effort")
        assert calls == []
    else:
        assert run() == {"key": "reasoning", "value": "high", "scope": scope}
        assert calls == [("commit", scope == "global"), ("apply", scope == "global")]


@pytest.mark.parametrize("pins_mode", [True, False], ids=["managed", "unmanaged"])
def test_owner_approval_mode_refusal_names_a_pinned_key(home, monkeypatch, pins_mode):
    """A refused ``approvals.mode`` write reports ``permission_denied`` naming the key when the managed
    scope pins it; any other refusal keeps ``storage_unavailable``. Nothing is written either way."""
    from hermes_agent.gateway import session_tui_config
    from hermes_agent.hermes_cli import approval_mode, managed_scope
    if pins_mode:
        (home.parent / "managed" / "config.yaml").write_text("approvals:\n  mode: manual\n")
        managed_scope.invalidate_managed_cache()
    else:
        monkeypatch.setattr(approval_mode, "run_approval_mode_command",
                            lambda mode: SimpleNamespace(ok=False, mode="manual"))
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(home)))
    with pytest.raises(RuntimeStoreError) as refused:
        session_tui_config._set_approval_mode(connection, "off")
    expected = ("permission_denied", "approvals.mode") if pins_mode else ("storage_unavailable", None)
    assert (refused.value.reason, refused.value.field) == expected
    assert yaml.safe_load((home / "config.yaml").read_text()) == {"display": {"tui_theme": "dark"}}
