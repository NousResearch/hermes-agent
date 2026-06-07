"""Tests for acp_adapter.entry startup wiring."""

import sys
from unittest import mock

import pytest

from acp_adapter import entry


def test_main_enables_unstable_protocol():
    calls = {}

    async def fake_run_agent(agent, **kwargs):
        calls["kwargs"] = kwargs

    # Create a mock acp module with all required attributes
    mock_acp = mock.MagicMock()
    mock_acp.run_agent = fake_run_agent
    # Mock acp.schema and all the classes that server.py imports from it
    mock_schema = mock.MagicMock()
    mock_acp.schema = mock_schema
    
    with mock.patch.dict(sys.modules, {"acp": mock_acp, "acp.schema": mock_schema}):
        with mock.patch.object(entry, "_setup_logging", return_value=None):
            with mock.patch.object(entry, "_load_env", return_value=None):
                entry.main([])

    assert calls["kwargs"]["use_unstable_protocol"] is True


def test_main_version_prints_without_starting_server(capsys):
    with mock.patch.object(entry, "_setup_logging", side_effect=AssertionError("started server")):
        entry.main(["--version"])

    output = capsys.readouterr().out.strip()
    assert output
    assert "Starting hermes-agent ACP adapter" not in output


def test_main_check_prints_ok_without_starting_server(capsys):
    # Mock acp module, acp.schema, and HermesACPAgent
    mock_acp = mock.MagicMock()
    mock_schema = mock.MagicMock()
    mock_acp.schema = mock_schema
    
    with mock.patch.dict(sys.modules, {"acp": mock_acp, "acp.schema": mock_schema}):
        entry.main(["--check"])

    assert capsys.readouterr().out.strip() == "Hermes ACP check OK"


def test_main_setup_runs_model_configuration():
    calls = {}

    def fake_hermes_main():
        calls["argv"] = sys.argv[:]

    with mock.patch("hermes_cli.main.main", fake_hermes_main):
        with mock.patch("sys.stdin.isatty", return_value=False):
            entry.main(["--setup"])

    assert calls["argv"][1:] == ["model"]


def test_main_setup_offers_browser_install_when_tty():
    """When stdin is a TTY and the user answers yes, model setup is followed
    by a browser-tools bootstrap call."""
    bootstrap_calls = []
    
    def fake_setup_browser(assume_yes=False):
        bootstrap_calls.append(assume_yes)
        return 0

    with mock.patch("hermes_cli.main.main", return_value=None):
        with mock.patch("sys.stdin.isatty", return_value=True):
            with mock.patch("builtins.input", return_value="y"):
                with mock.patch.object(entry, "_run_setup_browser", fake_setup_browser):
                    entry.main(["--setup"])

    assert bootstrap_calls == [False]


def test_main_setup_skips_browser_prompt_on_no():
    called = []
    
    def fake_setup_browser(assume_yes=False):
        called.append(assume_yes)
        return 0

    with mock.patch("hermes_cli.main.main", return_value=None):
        with mock.patch("sys.stdin.isatty", return_value=True):
            with mock.patch("builtins.input", return_value=""):
                with mock.patch.object(entry, "_run_setup_browser", fake_setup_browser):
                    entry.main(["--setup"])

    assert called == []


def test_main_setup_browser_calls_ensure_dependency():
    """`hermes-acp --setup-browser` routes through dep_ensure.ensure_dependency."""
    calls = []

    def fake_ensure(dep, interactive=True):
        calls.append((dep, interactive))
        return True

    with mock.patch("hermes_cli.dep_ensure.ensure_dependency", fake_ensure):
        entry.main(["--setup-browser"])

    assert ("node", True) in calls
    assert ("browser", True) in calls


def test_main_setup_browser_forwards_yes_flag():
    """--yes suppresses interactive prompts in ensure_dependency."""
    calls = []

    def fake_ensure(dep, interactive=True):
        calls.append((dep, interactive))
        return True

    with mock.patch("hermes_cli.dep_ensure.ensure_dependency", fake_ensure):
        entry.main(["--setup-browser", "--yes"])

    assert ("node", False) in calls
    assert ("browser", False) in calls


def test_main_setup_browser_stops_on_node_failure():
    """If node install fails, browser install is not attempted."""
    calls = []

    def fake_ensure(dep, interactive=True):
        calls.append(dep)
        return dep != "node"  # node fails

    with mock.patch("hermes_cli.dep_ensure.ensure_dependency", fake_ensure):
        with pytest.raises(SystemExit) as excinfo:
            entry.main(["--setup-browser"])
        assert excinfo.value.code == 1
        assert "node" in calls
        assert "browser" not in calls


def test_main_setup_browser_propagates_browser_failure():
    """If browser install fails, exit code is 1."""
    def fake_ensure(dep, interactive=True):
        return dep != "browser"  # browser fails

    with mock.patch("hermes_cli.dep_ensure.ensure_dependency", fake_ensure):
        with pytest.raises(SystemExit) as excinfo:
            entry.main(["--setup-browser"])
        assert excinfo.value.code == 1
