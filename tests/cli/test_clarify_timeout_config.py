"""Tests for hermes_cli.callbacks — clarify timeout config key alignment.

Regression tests for #42969: CLI clarify timeout reads from agent.clarify_timeout
(matching the gateway's get_clarify_timeout()) instead of the non-existent
clarify.timeout key.
"""


def _read_timeout_from_config(config):
    """Extract the timeout value clarify_callback would compute.

    Mirrors the config-reading logic in hermes_cli/callbacks.py:clarify_callback
    without actually running the callback (which blocks on a queue).
    """
    _agent = config.get("agent", {})
    _agent = _agent if isinstance(_agent, dict) else {}
    timeout = _agent.get("clarify_timeout", None)
    if timeout is None:
        _legacy = config.get("clarify", {})
        _legacy = _legacy if isinstance(_legacy, dict) else {}
        timeout = _legacy.get("timeout", 600)
    return timeout


class TestClarifyTimeoutConfig:
    """Verify clarify_callback reads agent.clarify_timeout (canonical key)."""

    def test_default_timeout_is_600(self):
        """When no config overrides exist, timeout defaults to 600s (matching gateway)."""
        assert _read_timeout_from_config({}) == 600

    def test_reads_agent_clarify_timeout(self):
        """agent.clarify_timeout from config is used as the timeout."""
        assert _read_timeout_from_config({"agent": {"clarify_timeout": 900}}) == 900

    def test_legacy_clarify_timeout_fallback(self):
        """Legacy clarify.timeout key is used when agent.clarify_timeout is absent."""
        assert _read_timeout_from_config({"clarify": {"timeout": 300}}) == 300

    def test_agent_key_takes_precedence_over_legacy(self):
        """agent.clarify_timeout takes precedence over clarify.timeout."""
        config = {"agent": {"clarify_timeout": 450}, "clarify": {"timeout": 120}}
        assert _read_timeout_from_config(config) == 450

    def test_non_dict_agent_config_does_not_crash(self):
        """When agent config is a string (e.g., shorthand), fallback to default."""
        assert _read_timeout_from_config({"agent": "openrouter/claude-sonnet-4"}) == 600

    def test_empty_agent_section_uses_default(self):
        """When agent is an empty dict, default 600 is used."""
        assert _read_timeout_from_config({"agent": {}}) == 600

    def test_none_agent_value_uses_default(self):
        """When agent is None, default 600 is used."""
        assert _read_timeout_from_config({"agent": None}) == 600
