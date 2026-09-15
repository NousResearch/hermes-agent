from __future__ import annotations

from queue import Queue

from hermes_cli.commands import resolve_command
from hermes_cli.cli_commands_mixin import CLICommandsMixin


def test_trigger_is_registered_cli_only():
    command = resolve_command("trigger")
    assert command is not None
    assert command.cli_only
    assert command.args_hint == "[trigger [arguments]]"


def test_trigger_preserves_explicit_arguments(capsys):
    mixin = CLICommandsMixin()
    pending = Queue()
    setattr(mixin, "_pending_input", pending)
    mixin._handle_trigger_command("/trigger todo tracking: migrate catalog")
    assert pending.get_nowait() == "todo tracking: migrate catalog"
    assert "Trigger queued" in capsys.readouterr().out


def test_trigger_rejects_unknown_name(capsys):
    mixin = CLICommandsMixin()
    mixin._handle_trigger_command("/trigger unknown: thing")
    assert "Unknown trigger" in capsys.readouterr().out


def test_default_triggers_are_stable_baseline():
    """With no config, the handler exposes exactly the built-in set."""
    mixin = CLICommandsMixin()
    triggers = mixin._configured_triggers()
    assert {p for p, _ in triggers} == {
        "todo tracking", "shepherd the flock", "Gate approved", "gate notebook",
    }
    # Built-ins all carry a non-empty description for the chooser rows.
    assert all(desc for _, desc in triggers)


def test_configured_triggers_extend_builtins():
    mixin = CLICommandsMixin()
    setattr(mixin, "config", {
        "triggers": [
            {"phrase": "forensics", "description": "Forensic investigation of a failure"},
            {"phrase": "deploy calypso", "description": "Promote the calypso service"},
        ],
    })
    phrases = dict(mixin._configured_triggers())
    # User triggers are added, built-ins are never dropped.
    assert phrases["forensics"] == "Forensic investigation of a failure"
    assert phrases["deploy calypso"] == "Promote the calypso service"
    assert phrases["todo tracking"]  # still present


def test_configured_trigger_overrides_builtin_description():
    mixin = CLICommandsMixin()
    setattr(mixin, "config", {
        "triggers": [{"phrase": "todo tracking", "description": "custom todo desc"}],
    })
    phrases = dict(mixin._configured_triggers())
    assert phrases["todo tracking"] == "custom todo desc"
    # Dedup by phrase: exactly one row for the redefined phrase.
    assert [p for p, _ in mixin._configured_triggers()].count("todo tracking") == 1


def test_config_malformed_entries_are_skipped():
    mixin = CLICommandsMixin()
    setattr(mixin, "config", {
        "triggers": [
            "not-a-dict",
            {"phrase": 123, "description": "numeric phrase"},
            {"phrase": "   ", "description": "blank phrase"},
            {"phrase": "valid", "description": "ok"},
        ],
    })
    triggers = mixin._configured_triggers()
    assert dict(triggers)["valid"] == "ok"
    # Malformed entries don't pollute the list.
    assert len(triggers) == len(mixin._default_triggers()) + 1


def test_configured_trigger_routes_explicit_arg(capsys):
    mixin = CLICommandsMixin()
    pending = Queue()
    setattr(mixin, "_pending_input", pending)
    setattr(mixin, "config", {
        "triggers": [{"phrase": "forensics", "description": "Forensic investigation"}],
    })
    mixin._handle_trigger_command("/trigger forensics: investigate pane 3")
    assert pending.get_nowait() == "forensics: investigate pane 3"
    assert "Trigger queued" in capsys.readouterr().out