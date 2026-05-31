"""Tests for acp_adapter.entry startup wiring."""

import sys

import acp
import pytest

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
        calls["argv"] = sys.argv[:]

    monkeypatch.setattr("hermes_cli.main.main", fake_hermes_main)
    # Pretend stdin is not a TTY so the follow-up browser prompt is skipped.
    # That keeps this test focused on the model-setup wiring; the
    # browser-prompt path has its own test below.
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    entry.main(["--setup"])

    assert calls["argv"][1:] == ["model"]


def test_main_setup_offers_browser_install_when_tty(monkeypatch):
    """When stdin is a TTY and the user answers yes, model setup is followed
    by a browser-tools bootstrap call."""
    monkeypatch.setattr("hermes_cli.main.main", lambda: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "y")

    bootstrap_calls = []
    monkeypatch.setattr(
        entry,
        "_run_setup_browser",
        lambda assume_yes=False: bootstrap_calls.append(assume_yes) or 0,
    )

    entry.main(["--setup"])

    assert bootstrap_calls == [False]


def test_main_setup_skips_browser_prompt_on_no(monkeypatch):
    monkeypatch.setattr("hermes_cli.main.main", lambda: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "")

    called = []
    monkeypatch.setattr(
        entry,
        "_run_setup_browser",
        lambda assume_yes=False: called.append(assume_yes) or 0,
    )

    entry.main(["--setup"])

    assert called == []


def test_main_setup_browser_calls_ensure_dependency(monkeypatch):
    """`hermes-acp --setup-browser` routes through dep_ensure.ensure_dependency."""
    calls = []

    def fake_ensure(dep, interactive=True):
        calls.append((dep, interactive))
        return True

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    entry.main(["--setup-browser"])

    assert ("node", True) in calls
    assert ("browser", True) in calls


def test_main_setup_browser_forwards_yes_flag(monkeypatch):
    """--yes suppresses interactive prompts in ensure_dependency."""
    calls = []

    def fake_ensure(dep, interactive=True):
        calls.append((dep, interactive))
        return True

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    entry.main(["--setup-browser", "--yes"])

    assert ("node", False) in calls
    assert ("browser", False) in calls


def test_main_setup_browser_stops_on_node_failure(monkeypatch):
    """If node install fails, browser install is not attempted."""
    calls = []

    def fake_ensure(dep, interactive=True):
        calls.append(dep)
        return dep != "node"  # node fails

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1
    assert "node" in calls
    assert "browser" not in calls


def test_main_setup_browser_propagates_browser_failure(monkeypatch):
    """If browser install fails, exit code is 1."""
    def fake_ensure(dep, interactive=True):
        return dep != "browser"  # browser fails

    monkeypatch.setattr("hermes_cli.dep_ensure.ensure_dependency", fake_ensure)

    with pytest.raises(SystemExit) as excinfo:
        entry.main(["--setup-browser"])
    assert excinfo.value.code == 1
