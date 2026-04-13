"""Tests for PR #4788 feat/queue-followup.

Covers: Alt+Enter queues followup messages.
Attributes on HermesCLI: _followup_queue, _cancelled_followups, _followup_recall_count.
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_cli(env_overrides=None, config_overrides=None, **kwargs):
    """Create a HermesCLI instance with minimal mocking (mirrors test_cli_init.py)."""
    import importlib

    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    if config_overrides:
        for key, value in config_overrides.items():
            if key in _clean_config and isinstance(_clean_config[key], dict) and isinstance(value, dict):
                _clean_config[key] = {**_clean_config[key], **value}
            else:
                _clean_config[key] = value

    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    if env_overrides:
        clean_env.update(env_overrides)

    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), \
         patch.dict("os.environ", clean_env, clear=False):
        import cli as _cli_mod
        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), \
             patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}):
            return _cli_mod.HermesCLI(**kwargs)


class TestFollowupQueueInit:
    """_followup_queue must be initialized as an empty list."""

    def test_followup_queue_attribute_exists(self):
        cli = _make_cli()
        assert hasattr(cli, "_followup_queue"), (
            "_followup_queue must be initialized in HermesCLI.__init__"
        )

    def test_followup_queue_is_empty_list(self):
        cli = _make_cli()
        assert cli._followup_queue == []

    def test_followup_queue_is_list_type(self):
        cli = _make_cli()
        assert isinstance(cli._followup_queue, list), (
            "_followup_queue must be a list (not a queue or deque)"
        )


class TestCancelledFollowupsInit:
    """_cancelled_followups must be initialized as an empty set."""

    def test_cancelled_followups_attribute_exists(self):
        cli = _make_cli()
        assert hasattr(cli, "_cancelled_followups"), (
            "_cancelled_followups must be initialized in HermesCLI.__init__"
        )

    def test_cancelled_followups_is_empty_set(self):
        cli = _make_cli()
        assert cli._cancelled_followups == set()

    def test_cancelled_followups_is_set_type(self):
        cli = _make_cli()
        assert isinstance(cli._cancelled_followups, set), (
            "_cancelled_followups must be a set (not a list)"
        )


class TestFollowupRecallCountInit:
    """_followup_recall_count must be initialized to 0."""

    def test_followup_recall_count_attribute_exists(self):
        cli = _make_cli()
        assert hasattr(cli, "_followup_recall_count"), (
            "_followup_recall_count must be initialized in HermesCLI.__init__"
        )

    def test_followup_recall_count_is_zero(self):
        cli = _make_cli()
        assert cli._followup_recall_count == 0

    def test_followup_recall_count_is_int(self):
        cli = _make_cli()
        assert isinstance(cli._followup_recall_count, int)


class TestFollowupQueueIndependence:
    """Each HermesCLI instance has its own queue/set (no shared mutable defaults)."""

    def test_two_instances_have_independent_queues(self):
        cli_a = _make_cli()
        cli_b = _make_cli()
        cli_a._followup_queue.append("item")
        assert cli_b._followup_queue == [], (
            "_followup_queue must not be shared between instances"
        )

    def test_two_instances_have_independent_cancelled_sets(self):
        cli_a = _make_cli()
        cli_b = _make_cli()
        cli_a._cancelled_followups.add("uid-abc")
        assert cli_b._cancelled_followups == set(), (
            "_cancelled_followups must not be shared between instances"
        )
