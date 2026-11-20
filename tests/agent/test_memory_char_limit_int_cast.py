"""The live agent must build its ``MemoryStore`` with integer char limits.

``memory.user_char_limit`` / ``memory.memory_char_limit`` reach the store as whatever
the config file held: an int in plain YAML, but a string when the value is quoted or
came from a ``${VAR}`` expansion (``hermes_cli.config._expand_env_vars`` rewrites string
values in place). The no-agent path (``tools.memory_tool.load_on_disk_store``) already
coerced both limits on the way in; the live-agent path in ``agent_init`` did not, so a
string limit reached ``MemoryStore`` and every comparison against a char count raised
``TypeError`` instead of enforcing the cap (#53834).

These tests drive the real ``_init_memory`` construction, not a mock.
"""

from types import SimpleNamespace

import pytest

from agent.agent_init import _init_memory


class _StartupAgent(SimpleNamespace):
    """The attribute surface ``_init_memory`` reads off the agent."""

    enabled_toolsets: list = []
    disabled_toolsets: list = []
    tools: list = []
    valid_tool_names: set = set()
    log_prefix = ""
    quiet_mode = True
    status_callback = None
    tool_progress_callback = None
    _emit_startup_warning = None


@pytest.fixture()
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))


def _live_store(memory_section):
    """The store ``_init_memory`` builds for a live agent under ``memory_section``."""
    agent = _StartupAgent()
    _init_memory(agent, {"memory": memory_section}, False, "cli")
    return agent._memory_store


class TestLiveAgentCharLimits:
    def test_string_limits_are_coerced(self, _isolated_home):
        store = _live_store({"memory_char_limit": "20000", "user_char_limit": "30000"})
        assert store.memory_char_limit == 20000
        assert store.user_char_limit == 30000

    def test_int_limits_pass_through(self, _isolated_home):
        store = _live_store({"memory_char_limit": 20000, "user_char_limit": 30000})
        assert store.memory_char_limit == 20000
        assert store.user_char_limit == 30000

    def test_string_limit_enforces_the_cap(self, _isolated_home):
        """The limit is compared against a char count on every write; a string limit
        raised ``TypeError`` there instead of refusing the over-budget entry."""
        store = _live_store({"memory_char_limit": "20", "user_char_limit": "20"})
        assert store._char_limit("memory") == 20
        assert store._char_count("memory") <= store._char_limit("memory")
