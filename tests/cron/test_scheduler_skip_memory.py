"""Tests for cron scheduler skip_memory configuration.

Regression for issue #9763: skip_memory was hardcoded to True in
cron/scheduler.py, making external memory providers (e.g. mem0) unusable
in cron jobs. The resolution is now centralized in
``_resolve_cron_skip_memory`` and wired into the cron AIAgent construction.
"""

from cron.scheduler import _resolve_cron_skip_memory


class TestResolveCronSkipMemory:
    def test_defaults_to_true_when_no_cron_section(self):
        assert _resolve_cron_skip_memory({}) is True

    def test_defaults_to_true_when_skip_memory_missing(self):
        assert _resolve_cron_skip_memory({"cron": {}}) is True

    def test_false_when_configured(self):
        assert _resolve_cron_skip_memory({"cron": {"skip_memory": False}}) is False

    def test_true_when_explicitly_set(self):
        assert _resolve_cron_skip_memory({"cron": {"skip_memory": True}}) is True

    def test_other_cron_settings_do_not_affect_default(self):
        assert _resolve_cron_skip_memory({"cron": {"timeout": 300}}) is True

    def test_tolerates_non_dict_cron_section(self):
        assert _resolve_cron_skip_memory({"cron": None}) is True
        assert _resolve_cron_skip_memory(None) is True
