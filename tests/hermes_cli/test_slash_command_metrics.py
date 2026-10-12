"""Classic-CLI slash commands reach shared metrics once each: canonical name, surface ``cli``."""

from cli import HermesCLI


def test_cli_counts_each_typed_slash_command_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_events.record_slash_command", lambda **kw: calls.append(kw))
    ran = []
    monkeypatch.setattr(HermesCLI, "_cmd_new", lambda self, cmd: ran.append(cmd))
    cli = object.__new__(HermesCLI)
    cli.config = {"quick_commands": {"rs": {"type": "alias", "target": "/reset"}}}

    cli.process_command("/reset")
    cli.process_command("/rs")  # a user alias re-dispatches /reset internally

    assert ran == ["/reset", "/reset"]
    assert calls == [{"command": "new", "surface": "cli"}, {"command": "rs", "surface": "cli"}]
