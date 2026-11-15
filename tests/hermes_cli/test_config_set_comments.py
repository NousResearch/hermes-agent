"""``hermes config set`` on a commented config.yaml, end to end through ``set_config_value``.

Every set rewrites the whole document through the round-trip writer, so a comment in one place can
break a key the user never touched. These cover the two shapes that did, each through the real
command rather than the writer alone.
"""

import os
from unittest.mock import patch

import pytest

import hermes_yaml as yaml
from hermes_cli.config import set_config_value


@pytest.fixture
def home(tmp_path):
    (tmp_path / ".env").touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


def test_emptying_a_commented_list_keeps_the_file_valid(home):
    """#148524: the gateway restarted on a config.yaml that ``config set ... '[]'`` had left as
    ``# some comment`` / ``[]`` at column 0 / the next two keys joined on one line."""
    config_path = home / "config.yaml"
    config_path.write_text(
        "kanban:\n"
        "  dispatch_profiles:     # some comment\n"
        "    - mordecai\n"
        "  dispatch_in_gateway: true\n"
        "  dispatch_interval_seconds: 60\n",
        encoding="utf-8",
    )

    set_config_value("kanban.dispatch_profiles", "[]")

    assert config_path.read_text(encoding="utf-8") == (
        "kanban:\n"
        "  dispatch_profiles: []  # some comment\n"
        "  dispatch_in_gateway: true\n"
        "  dispatch_interval_seconds: 60\n"
    )


def test_setting_one_key_leaves_a_commented_list_of_mappings_intact(home):
    """Setting an unrelated key re-emits the whole file. A commented list whose first entry is
    ``{}`` used to come back with its next two entries fused into one string, silently."""
    config_path = home / "config.yaml"
    config_path.write_text(
        "mcp_servers:\n"
        "  local:\n"
        "    env:   # per-call overrides\n"
        "      - {}\n"
        "      - DEBUG\n"
        "      - TRACE\n"
        "agent:\n"
        "  max_turns: 60\n",
        encoding="utf-8",
    )

    set_config_value("agent.max_turns", "90")

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["mcp_servers"]["local"]["env"] == [{}, "DEBUG", "TRACE"]
    assert saved["agent"]["max_turns"] == 90


def test_emptying_a_commented_mapping_keeps_the_file_valid(home):
    """The mapping twin of #148524: ``config set --force display '{}'`` left ``display:  # note``
    followed by ``{}`` at column 0. Mapping case reported in #148569 and #148572."""
    config_path = home / "config.yaml"
    config_path.write_text(
        "display:   # display stuff\n"
        "  skin: default\n"
        "agent:\n"
        "  max_turns: 60\n",
        encoding="utf-8",
    )

    set_config_value("display", "{}", force=True)

    text = config_path.read_text(encoding="utf-8")
    assert yaml.safe_load(text) == {"display": {}, "agent": {"max_turns": 60}}
    assert "# display stuff" in text


def test_unsetting_the_last_child_of_a_commented_key_keeps_the_file_valid(home):
    """``config unset`` of a commented mapping's only child emptied the parent in place, which
    dumped the same ``{}``-at-column-0 shape as #148524."""
    from utils import atomic_roundtrip_yaml_update

    config_path = home / "config.yaml"
    config_path.write_text(
        "display:   # display stuff\n"
        "  skin: default\n"
        "agent:\n"
        "  max_turns: 60\n",
        encoding="utf-8",
    )

    atomic_roundtrip_yaml_update(config_path, "display.skin", None)

    text = config_path.read_text(encoding="utf-8")
    assert yaml.safe_load(text) == {"display": {}, "agent": {"max_turns": 60}}
    assert "# display stuff" in text
