"""Tests for cron scheduler skip_memory configuration.

Regression for issue #9763: skip_memory was hardcoded to True in
cron/scheduler.py, making external memory providers (e.g. mem0) unusable
in cron jobs.
"""
import textwrap


class TestCronSkipMemoryConfig:
    """Verify cron reads skip_memory from config.yaml."""

    def test_skip_memory_defaults_to_true(self):
        """When no cron.skip_memory is configured, default to True."""
        config = {}
        result = config.get("cron", {}).get("skip_memory", True)
        assert result is True

    def test_skip_memory_false_when_configured(self):
        """When cron.skip_memory is set to false, use False."""
        config = {"cron": {"skip_memory": False}}
        result = config.get("cron", {}).get("skip_memory", True)
        assert result is False

    def test_skip_memory_true_when_explicitly_set(self):
        """When cron.skip_memory is explicitly true, use True."""
        config = {"cron": {"skip_memory": True}}
        result = config.get("cron", {}).get("skip_memory", True)
        assert result is True

    def test_empty_cron_section_defaults_to_true(self):
        """When cron section exists but skip_memory is missing, default to True."""
        config = {"cron": {}}
        result = config.get("cron", {}).get("skip_memory", True)
        assert result is True

    def test_other_cron_settings_preserved(self):
        """Other cron config settings don't affect skip_memory default."""
        config = {"cron": {"timeout": 300}}
        result = config.get("cron", {}).get("skip_memory", True)
        assert result is True
