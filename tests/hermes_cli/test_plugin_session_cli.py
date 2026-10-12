"""Frozen native plugin exercised through the classic CLI session dispatcher."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import plugins
from hermes_state import SessionDB

POLICY_KEY = "fixture_mode_override"
FROZEN_PLUGIN = '''
from hermes_cli.session_commands import SessionCommandRequest

KEY = "fixture_mode_override"
USAGE = "Usage: /session-mode on|off|inherit|status"

def prepare(raw):
    words = raw.lower().strip().split()
    if len(words) != 1 or words[0] not in ("on", "off", "inherit", "status"):
        raise ValueError(USAGE)
    return SessionCommandRequest(words[0] != "status", words[0])

def execute(request, session):
    config = session.get_config()
    if request.write:
        session.set_override(KEY, {"on": True, "off": False, "inherit": None}[request.data])
    override = session.get_override(KEY)
    enabled = config["fixture_session"]["enabled"] if override is None else override
    return f"state={enabled}; override={override}; id={session.session_id}; live={session.live}; home={session.profile_home}"

def register(ctx):
    ctx.register_session_policy(
        KEY, config_path="fixture_session.enabled",
        apply=lambda runtime, enabled: runtime.set_context_option("fixture_enabled", enabled))
    ctx.register_session_command(
        "session-mode", execute, prepare=prepare, description="Control this session",
        args_hint="[on|off|inherit|status]", subcommands=("on", "off", "inherit", "status"),
        advertise_in_gateway_menu=False)
    ctx.register_command("legacy-echo", lambda raw: "legacy:" + raw, description="Echo text")
'''


@pytest.fixture
def plugin_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    directory = home / "plugins" / "frozen-session"
    directory.mkdir(parents=True)
    (directory / "plugin.yaml").write_text(
        "name: frozen-session\nversion: 0.1.0\n", encoding="utf-8")
    (directory / "__init__.py").write_text(FROZEN_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [frozen-session]\nfixture_session:\n  enabled: false\n",
        encoding="utf-8")
    bundled = tmp_path / "empty-bundled"
    bundled.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    plugins._reset_plugin_managers_for_tests()
    plugins.discover_plugins(force=True)
    loaded = plugins.get_plugin_manager()._plugins["frozen-session"]
    assert loaded.error is None
    assert loaded.module is not None
    yield home
    plugins._reset_plugin_managers_for_tests()


@pytest.fixture
def db(plugin_home):
    value = SessionDB(db_path=plugin_home / "state.db")
    yield value
    value.close()


class _Agent:
    def __init__(self, db, session_id):
        self.session_id = session_id
        self.session_start = datetime.now()
        self._session_db = db
        self._session_init_model_config = {"keep": "live"}
        self._session_db_created = False
        self._last_flushed_db_idx = 0
        self.ensure_calls = 0
        self.engine_calls = []
        self.reset_calls = 0
        self.context_compressor = SimpleNamespace(set_session_option=self._set_option)
        self._cached_system_prompt = "stable system prompt"

    def _set_option(self, name, enabled):
        row = self._session_db.get_session(self.session_id)
        override = SessionDB.session_policy_override(row, POLICY_KEY)
        self.engine_calls.append((name, enabled, override))
        return True

    def _ensure_db_session(self):
        self.ensure_calls += 1
        if self._session_db.get_session(self.session_id) is None:
            self._session_db.create_session(
                self.session_id, source="cli", model_config={"keep": "durable"})

    def reset_session_state(self):
        self.reset_calls += 1

    def _flush_messages_to_session_db(self, *args, **kwargs):
        pass


def _cli(db, monkeypatch, *, session_id="cli-current"):
    import cli as facade

    value = object.__new__(facade.HermesCLI)
    value.session_id = session_id
    value.session_start = datetime.now()
    value.agent = _Agent(db, session_id)
    value._session_db = db
    value.config = {"quick_commands": {}}
    value._slash_metrics_surface = ""
    value._pending_resume_sessions = None
    value.conversation_history = []
    value._pending_title = None
    value._resumed = False
    value.reasoning_config = {}
    value.max_turns = 7
    value.model = "fixture/model"
    value.provider = "fixture"
    value.requested_provider = "fixture"
    value.base_url = "http://fixture"
    value.api_key = ""
    value.api_mode = "chat_completions"
    value.service_tier = None
    value._pending_one_turn_model_restore = None
    value._notify_session_boundary = lambda *args: None
    value._launch_session_boundary_memory_flush = lambda *args, **kwargs: None
    value._write_terminal_breadcrumb = lambda: None
    value._restore_session_cwd = lambda *args: None
    value._restore_session_yolo = lambda *args: None
    value._restore_session_model = lambda *args: None
    value._display_resumed_history = lambda: None
    output = []
    monkeypatch.setattr(facade, "_cprint", lambda *args, **kwargs: output.extend(map(str, args)))
    monkeypatch.setattr(facade, "_ensure_skill_commands", dict)
    monkeypatch.setattr(facade, "get_skill_bundles", dict)
    monkeypatch.setattr(facade, "_sync_process_session_id", lambda *args: None)
    return value, output


def _override(db, session_id):
    return SessionDB.session_policy_override(db.get_session(session_id), POLICY_KEY)


def test_real_plugin_cli_dispatch_persists_before_live_without_prompt_changes(db, monkeypatch):
    cli, output = _cli(db, monkeypatch)
    history = cli.conversation_history
    prompt = cli.agent._cached_system_prompt

    assert cli.process_command("/session-mode ON") is True
    assert _override(db, cli.session_id) is True
    assert cli.agent.engine_calls == [("fixture_enabled", True, True)]
    assert "state=True; override=True" in output[-1]
    assert cli.process_command("/session-mode off") is True
    assert _override(db, cli.session_id) is False
    assert cli.agent.engine_calls[-1] == ("fixture_enabled", False, False)
    assert cli.process_command("/session-mode inherit") is True
    assert _override(db, cli.session_id) is None
    assert POLICY_KEY not in json.loads(db.get_session(cli.session_id)["model_config"])
    assert cli.agent._session_init_model_config == {"keep": "live"}
    assert json.loads(db.get_session(cli.session_id)["model_config"])["keep"] == "durable"
    assert cli.conversation_history is history
    assert cli.agent._cached_system_prompt == prompt


def test_cli_status_without_row_is_read_only(db, monkeypatch):
    cli, output = _cli(db, monkeypatch)
    assert cli.process_command("/session-mode status") is True
    assert "state=False; override=None" in output[-1]
    assert db.get_session(cli.session_id) is None
    assert cli.agent.ensure_calls == 0
    assert cli.agent.engine_calls == []


@pytest.mark.parametrize("args", ["", "maybe", "on off", "status extra"])
def test_invalid_cli_request_parses_before_foreground_or_db_access(plugin_home, monkeypatch, args):
    import cli as facade

    class _UnexpectedForeground:
        @property
        def agent(self):
            pytest.fail("invalid requests must not inspect the foreground agent")

    output = []
    monkeypatch.setattr(facade, "_cprint", output.append)
    facade.HermesCLI._run_plugin_slash_command(
        _UnexpectedForeground(), "/session-mode", args)
    assert output == ["Usage: /session-mode on|off|inherit|status"]
    assert not (plugin_home / "state.db").exists()


def test_cli_foreground_identity_and_profile_are_exact(db, monkeypatch, tmp_path):
    cli, output = _cli(db, monkeypatch)
    cli.session_id = "foreign"
    cli.process_command("/session-mode on")
    assert "identity mismatch" in output[-1]
    assert db.get_session("cli-current") is None
    assert cli.agent.ensure_calls == 0

    foreign_home = tmp_path / "foreign-home"
    foreign_home.mkdir()
    foreign = SessionDB(db_path=foreign_home / "state.db")
    try:
        cli.session_id = cli.agent.session_id
        cli.agent._session_db = foreign
        cli.process_command("/session-mode on")
        assert "profile mismatch" in output[-1]
        assert foreign.get_session(cli.session_id) is None
        assert cli.agent.ensure_calls == 0
    finally:
        foreign.close()


def test_cli_config_and_db_failures_never_apply_live_state(db, monkeypatch):
    cli, output = _cli(db, monkeypatch)

    def unavailable():
        raise RuntimeError("configuration unavailable")

    with monkeypatch.context() as patcher:
        patcher.setattr("hermes_cli.session_commands.load_session_command_config", unavailable)
        cli.process_command("/session-mode on")
    assert "configuration unavailable" in output[-1]
    assert cli.agent.ensure_calls == 0
    assert db.get_session(cli.session_id) is None

    def failed_write(*args):
        raise RuntimeError("durable write refused")

    monkeypatch.setattr(db, "set_session_policy_override", failed_write)
    cli.process_command("/session-mode on")
    assert "durable write refused" in output[-1]
    assert _override(db, cli.session_id) is None
    assert cli.agent.engine_calls == []


def test_cli_existing_raw_plugin_handler_keeps_its_argument_contract(db, monkeypatch):
    cli, output = _cli(db, monkeypatch)
    assert cli.process_command("/legacy-echo Some RAW args") is True
    assert output[-1] == "legacy:Some RAW args"
    assert db.get_session(cli.session_id) is None


@pytest.mark.parametrize("override", [True, False])
def test_new_preserves_empty_policy_row_and_resets_live_override(db, monkeypatch, override):
    import cli as facade

    cli, _ = _cli(db, monkeypatch)
    db.create_session(cli.session_id, source="cli", model_config={POLICY_KEY: override})
    from agent.session_policies import hydrate_session_policies
    hydrate_session_policies(cli.agent, db, cli.session_id)
    old_session_id = cli.session_id
    monkeypatch.setattr(
        facade, "CLI_CONFIG", {"agent": {"service_tier": ""}})
    monkeypatch.setattr(
        "hermes_cli.cli_session_mixin._reset_model_to_config_default", lambda *args: None)
    monkeypatch.setattr(
        "hermes_cli.cli_model_switch_mixin._resolve_cli_reasoning", lambda *args: None)

    cli.new_session(silent=True)

    assert cli.session_id != old_session_id
    assert _override(db, old_session_id) is override
    assert db.get_session(cli.session_id) is not None
    assert _override(db, cli.session_id) is None
    assert cli.agent._session_policy_overrides[POLICY_KEY] is None
    assert cli.agent.engine_calls[-1] == ("fixture_enabled", False, None)
    assert POLICY_KEY not in cli.agent._session_init_model_config


def test_resume_hydrates_exact_row_and_branch_excludes_parent_policy(db, monkeypatch):
    cli, _ = _cli(db, monkeypatch)
    cli.process_command("/session-mode on")
    db.create_session("target", source="cli", model_config={POLICY_KEY: False})
    monkeypatch.setattr(
        "hermes_cli.main._resolve_session_by_name_or_id", lambda target: target)

    assert cli.process_command("/resume target") is True
    assert cli.agent.session_id == "target"
    assert cli.agent._session_policy_overrides[POLICY_KEY] is False
    assert cli.agent.engine_calls[-1] == ("fixture_enabled", False, False)
    cli.conversation_history = [{"role": "user", "content": "branch this"}]
    assert cli.process_command("/branch copied") is True
    assert cli.session_id != "target"
    assert _override(db, "target") is False
    assert _override(db, cli.session_id) is None
    assert cli.agent._session_policy_overrides[POLICY_KEY] is None
    assert cli.agent.engine_calls[-1] == ("fixture_enabled", False, None)


def test_plugin_subcommands_and_hidden_menus_preserve_typed_help(plugin_home):
    from hermes_cli.commands import (
        SUBCOMMANDS, gateway_help_lines, is_gateway_known_command,
        resolve_command, should_bypass_active_session,
    )
    from hermes_cli.commands_platforms import (
        slack_native_slashes, slack_subcommand_map,
        telegram_bot_commands, telegram_menu_commands,
    )

    assert resolve_command("session-mode") is None
    assert SUBCOMMANDS["/session-mode"] == ["on", "off", "inherit", "status"]
    assert is_gateway_known_command("session-mode")
    assert is_gateway_known_command("session_mode")
    assert should_bypass_active_session("session_mode")
    assert any(line.startswith(chr(96) + "/session-mode ") for line in gateway_help_lines())
    assert not any("session-mode" in line for line in gateway_help_lines(["help"]))
    assert slack_subcommand_map()["session-mode"] == "/session-mode"
    assert "session_mode" not in {name for name, _ in telegram_bot_commands()}
    assert "session_mode" not in {name for name, _ in telegram_menu_commands()[0]}
    assert "session-mode" not in {row[0] for row in slack_native_slashes()}
