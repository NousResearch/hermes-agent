"""``/skills approval`` and ``/memory approval``: set and show the write-approval scope.

``on`` gates every write, ``background`` exempts the user's own turns, ``off`` disables
the gate. Each setter (classic CLI, messaging gateway, TUI/desktop) persists exactly the keys the
shared parser hands it: the boolean ``write_approval`` and, for on/background, the string
``write_approval_scope``.
"""

import threading
import types

import pytest

import hermes_yaml as yaml


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _approval(subsystem, *words, setter=True):
    from hermes_cli.write_approval_commands import handle_pending_subcommand

    captured = []
    out = handle_pending_subcommand(
        subsystem, ["approval", *words],
        set_mode_fn=(lambda settings: captured.append(dict(settings))) if setter else None)
    return out, captured


@pytest.mark.parametrize("word", ["background", "background_review", "background_only", "BACKGROUND"])
def test_background_sets_the_gate_on_with_background_scope(home, word):
    out, captured = _approval("skills", word)
    assert captured == [{"write_approval_scope": "background", "write_approval": True}]
    # Scope first: a per-key setter failing half way leaves the old gate with the new scope, never
    # a gate narrower than both the old and the requested state.
    assert list(captured[0]) == ["write_approval_scope", "write_approval"]
    assert "set to 'on' (background writes only; your own turns apply directly)" in out


@pytest.mark.parametrize("word", ["on", "true", "enable"])
def test_on_resets_the_scope_to_all(home, word):
    out, captured = _approval("memory", word)
    assert captured == [{"write_approval_scope": "all", "write_approval": True}]
    assert list(captured[0]) == ["write_approval_scope", "write_approval"]
    assert "set to 'on' (all writes)" in out


def test_off_turns_the_gate_off_and_leaves_the_scope(home):
    out, captured = _approval("skills", "off")
    assert captured == [{"write_approval": False}]
    assert "set to 'off'" in out


def test_invalid_value_is_refused_without_writing(home):
    out, captured = _approval("skills", "sometimes")
    assert captured == []
    assert "Invalid value 'sometimes'" in out and "background" in out


def test_without_a_setter_it_prints_every_config_command(home):
    out, _ = _approval("skills", "background", setter=False)
    assert out.splitlines()[1:] == ["  hermes config set skills.write_approval_scope background",
                                    "  hermes config set skills.write_approval true"]


def test_setter_failure_is_reported(home):
    from hermes_cli.write_approval_commands import handle_pending_subcommand

    def _boom(_settings):
        raise OSError("read-only config")

    out = handle_pending_subcommand("skills", ["approval", "background"], set_mode_fn=_boom)
    assert "Failed to set skills.write_approval" in out and "read-only config" in out


@pytest.mark.parametrize(("config", "line"), [
    ({}, "skills.write_approval = off"),
    ({"write_approval": True}, "skills.write_approval = on (all writes)"),
    ({"write_approval": True, "write_approval_scope": "background"},
     "skills.write_approval = on (background writes only; your own turns apply directly)"),
    ({"write_approval": True, "write_approval_scope": "bogus"}, "skills.write_approval = on (all writes)"),
    ({"write_approval": False, "write_approval_scope": "background"}, "skills.write_approval = off"),
])
def test_state_line_names_the_scope(home, config, line):
    import hermes_cli.config as cfg

    c = cfg.load_config()
    c["skills"] = {**c.get("skills", {}), **config}
    cfg.save_config(c)
    out, _ = _approval("skills")
    assert out.splitlines()[0] == line


def _saved(home):
    return yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize(("word", "expected"), [
    ("background", {"write_approval": True, "write_approval_scope": "background"}),
    ("on", {"write_approval": True, "write_approval_scope": "all"}),
])
def test_classic_cli_setter_persists_both_keys(home, capsys, word, expected):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    handler = CLICommandsMixin.__new__(CLICommandsMixin)
    handler._handle_skills_command(f"/skills approval {word}")
    assert "set to 'on'" in capsys.readouterr().out
    saved = _saved(home)["skills"]
    assert {k: saved.get(k) for k in expected} == expected


def test_classic_cli_setter_reports_a_failed_save(home, capsys, monkeypatch):
    """``save_config_value`` returns False instead of raising; the handler must not claim success."""
    import cli
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    monkeypatch.setattr(cli, "save_config_value", lambda key, value: False)
    handler = CLICommandsMixin.__new__(CLICommandsMixin)
    handler._handle_memory_command("/memory approval background")
    out = capsys.readouterr().out
    assert "Failed to set memory.write_approval" in out and "set to 'on'" not in out


@pytest.mark.parametrize(("failing_key", "attempted"), [
    ("skills.write_approval_scope", ["skills.write_approval_scope"]),
    ("skills.write_approval", ["skills.write_approval_scope", "skills.write_approval"]),
])
def test_settings_saver_saves_in_order_and_stops_at_the_first_failure(failing_key, attempted):
    from hermes_cli.write_approval_commands import settings_saver

    calls = []

    def save(key, value):
        calls.append((key, value))
        return key != failing_key

    with pytest.raises(RuntimeError, match=failing_key):
        settings_saver(save, "skills")({"write_approval_scope": "background", "write_approval": True})
    values = {"skills.write_approval_scope": "background", "skills.write_approval": True}
    assert calls == [(key, values[key]) for key in attempted]


@pytest.mark.parametrize(("failing_key", "scope_after"), [
    ("skills.write_approval_scope", "all"),         # first key fails: nothing changes
    ("skills.write_approval", "background"),        # second key fails: new scope, gate still off
])
def test_classic_cli_partial_save_never_turns_the_gate_on(home, capsys, monkeypatch, failing_key, scope_after):
    """Start with the gate off and scope ``all``; one config save fails. The gate must stay off
    (the scope is written first and the saver stops at the failure), and the user is told."""
    import cli
    import hermes_cli.config as cfg
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    c = cfg.load_config()
    c["skills"] = {**c.get("skills", {}), "write_approval": False, "write_approval_scope": "all"}
    cfg.save_config(c)
    real_save, attempted = cli.save_config_value, []

    def save(key, value):
        attempted.append(key)
        return False if key == failing_key else real_save(key, value)

    monkeypatch.setattr(cli, "save_config_value", save)
    CLICommandsMixin.__new__(CLICommandsMixin)._handle_skills_command("/skills approval background")
    out = capsys.readouterr().out
    assert "Failed to set skills.write_approval" in out and "set to 'on'" not in out
    assert attempted == ["skills.write_approval_scope", "skills.write_approval"][:1 + (scope_after == "background")]
    # Effective state through the gate's own readers (save_config omits values equal to defaults).
    from tools import write_approval as wa
    assert wa.write_approval_enabled(wa.SKILLS) is False
    assert wa.write_approval_scope(wa.SKILLS) == scope_after
    assert (_saved(home) or {}).get("skills", {}).get("write_approval", False) is False


def _tui_session(home):
    return {"agent": types.SimpleNamespace(), "session_key": "session-key", "history": [],
            "history_lock": threading.Lock(), "history_version": 0, "running": False,
            "attached_images": [], "image_counter": 0, "cols": 80, "slash_worker": None,
            "show_reasoning": False, "tool_progress_mode": "all", "profile_home": str(home)}


def test_tui_dispatch_persists_both_keys(home):
    from tui_gateway import server

    server._sessions["sid-scope"] = _tui_session(home)
    try:
        res = server._methods["command.dispatch"](
            "d1", {"name": "skills", "arg": "approval background", "session_id": "sid-scope"})
        assert "set to 'on'" in res["result"]["output"], res
    finally:
        server._sessions.pop("sid-scope", None)
    saved = _saved(home)["skills"]
    assert saved.get("write_approval") is True and saved.get("write_approval_scope") == "background"
