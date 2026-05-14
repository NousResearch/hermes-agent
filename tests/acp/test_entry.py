"""Tests for acp_adapter.entry startup wiring."""

import acp

from acp_adapter import entry


def test_main_enables_unstable_protocol(monkeypatch):
    calls = {}

    async def fake_run_agent(agent, **kwargs):
        calls["kwargs"] = kwargs

    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(acp, "run_agent", fake_run_agent)

    entry.main([])

    assert calls["kwargs"]["use_unstable_protocol"] is True


def test_main_version_prints_without_starting_server(monkeypatch, capsys):
    monkeypatch.setattr(entry, "_setup_logging", lambda: (_ for _ in ()).throw(AssertionError("started server")))

    entry.main(["--version"])

    output = capsys.readouterr().out.strip()
    assert output
    assert "Starting hermes-agent ACP adapter" not in output


def test_main_check_prints_ok_without_starting_server(monkeypatch, capsys):
    monkeypatch.setattr(entry, "_setup_logging", lambda: (_ for _ in ()).throw(AssertionError("started server")))

    entry.main(["--check"])

    assert capsys.readouterr().out.strip() == "Hermes ACP check OK"


def test_main_setup_runs_model_configuration(monkeypatch):
    calls = {}

    def fake_hermes_main():
        import sys

        calls["argv"] = sys.argv[:]

    monkeypatch.setattr("hermes_cli.main.main", fake_hermes_main)

    entry.main(["--setup"])

    assert calls["argv"][1:] == ["model"]
