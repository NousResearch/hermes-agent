"""`hermes config set/unset` must refuse agent-owned sessions (#128524).

The terminal tool bridges the gateway session's HERMES_SESSION_* context into the child
environment, so an agent that was refused a file-tool edit of config.yaml (and told to use
``hermes config``) can relay through this CLI and pin ``delegation.model`` on a paid route.
The CLI write handlers must fail closed on those markers; a human shell has none of them.
"""

import os
from unittest.mock import patch

import pytest

from hermes_cli.config import _cmd_config_set, _cmd_config_unset

AGENT_MARKERS = ["HERMES_SESSION_KEY", "HERMES_SESSION_ID", "HERMES_SESSION_PLATFORM"]


class _Args:
    def __init__(self, *, key, value=None, force=False):
        self.key = key
        self.value = value
        self.force = force


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    """Point HERMES_HOME at a temp dir so tests never touch real config, and scrub any
    session markers the outer environment might carry so the negative control is clean."""
    (tmp_path / ".env").touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}, clear=False):
        for name in AGENT_MARKERS:
            os.environ.pop(name, None)
        yield tmp_path


def _config_written(tmp_path):
    path = tmp_path / "config.yaml"
    return path.exists() and path.read_text() != ""


class TestAgentSessionConfigWriteGuard:

    @pytest.mark.parametrize("marker", AGENT_MARKERS)
    def test_set_refused_under_each_agent_marker(self, marker, _isolated_hermes_home):
        os.environ[marker] = "agent-owned"
        with pytest.raises(SystemExit) as exc:
            _cmd_config_set(_Args(key="delegation.model", value="openai/gpt-6-luna-pro"))
        assert exc.value.code == 1
        assert not _config_written(_isolated_hermes_home)

    @pytest.mark.parametrize("marker", AGENT_MARKERS)
    def test_unset_refused_under_each_agent_marker(self, marker, _isolated_hermes_home):
        os.environ[marker] = "agent-owned"
        with pytest.raises(SystemExit) as exc:
            _cmd_config_unset(_Args(key="model.default"))
        assert exc.value.code == 1

    def test_set_refusal_names_the_marker_and_the_human_path(self, _isolated_hermes_home, capsys):
        os.environ["HERMES_SESSION_ID"] = "sess-123"
        with pytest.raises(SystemExit):
            _cmd_config_set(_Args(key="approvals.mode", value="auto"))
        err = capsys.readouterr().err
        assert "HERMES_SESSION_ID" in err
        assert "agent session" in err
        assert "#128524" in err

    def test_human_shell_still_writes(self, _isolated_hermes_home):
        for name in AGENT_MARKERS:
            os.environ.pop(name, None)
        # Route through a key that lands in config.yaml, not the .env credential path.
        _cmd_config_set(_Args(key="display.language", value="en"))
        assert _config_written(_isolated_hermes_home)

    def test_empty_marker_value_still_refuses_fail_closed(self, _isolated_hermes_home):
        # The session bridge writes "" for a None value — an engaged context must still refuse.
        os.environ["HERMES_SESSION_KEY"] = ""
        with pytest.raises(SystemExit):
            _cmd_config_set(_Args(key="model.provider", value="openrouter"))

    def test_usage_error_takes_precedence(self, _isolated_hermes_home):
        os.environ["HERMES_SESSION_ID"] = "sess-123"
        with pytest.raises(SystemExit) as exc:
            _cmd_config_set(_Args(key="", value="x"))
        assert exc.value.code == 1
