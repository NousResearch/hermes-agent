"""A memory plugin may stop the host loading its own two memory files.

``memory.minir.block_builtin_memory`` (on by default) resolves through the same flag
function ``agent_init`` uses, so while the plugin owns the memory surface no ``MemoryStore``
is built and MEMORY.md / USER.md are neither read nor rendered into the prompt. A missing
plugin block, a different selected provider, or an explicit ``false`` keeps the built-in
files exactly as before.
"""

import hermes_yaml as yaml
import pytest

from agent import agent_init
from model_tools import get_tool_definitions


@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch):
    """The env override wins by design; a developer shell must not decide these tests."""
    monkeypatch.delenv("MINIR_BLOCK_BUILTIN_MEMORY", raising=False)


@pytest.fixture(autouse=True)
def _clear_caches():
    """check_fn results and tool definitions are cached; a config write must be visible."""
    from model_tools import _clear_tool_defs_cache
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    yield
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    home.mkdir(parents=True)
    return home


def _write_memory_config(home, **memory_section):
    (home / "config.yaml").write_text(
        yaml.safe_dump({"memory": memory_section}), encoding="utf-8"
    )


def _flags(**memory_section):
    from tools.memory_tool import get_builtin_memory_store_flags

    return get_builtin_memory_store_flags({"memory": memory_section})


def _memory_tool_names():
    return {
        tool["function"]["name"]
        for tool in get_tool_definitions(enabled_toolsets=["memory"], quiet_mode=True)
    }


class TestPluginSwitchResolution:
    def test_block_defaults_on_inside_the_plugin_block(self):
        assert _flags(provider="minir", minir={}) == (False, False)

    def test_block_applies_without_an_explicit_provider(self):
        """A context-engine install has no ``memory.provider``; the block is the opt-in."""
        assert _flags(minir={}) == (False, False)

    def test_explicit_false_keeps_the_builtin_files(self):
        assert _flags(provider="minir", minir={"block_builtin_memory": "false"}) == (True, True)
        assert _flags(provider="minir", minir={"block_builtin_memory": False}) == (True, True)

    def test_another_selected_provider_wins_over_a_stale_block(self):
        assert _flags(provider="mem0", minir={"block_builtin_memory": "true"}) == (True, True)

    def test_no_plugin_block_changes_nothing(self):
        assert _flags(provider="minir") == (True, True)
        assert _flags() == (True, True)

    def test_env_override_wins_in_both_directions(self, monkeypatch):
        monkeypatch.setenv("MINIR_BLOCK_BUILTIN_MEMORY", "false")
        assert _flags(provider="minir", minir={}) == (True, True)
        monkeypatch.setenv("MINIR_BLOCK_BUILTIN_MEMORY", "true")
        assert _flags(provider="minir", minir={"block_builtin_memory": "false"}) == (False, False)


class TestRealConfigChain:
    def test_memory_tool_disappears_while_the_plugin_owns_memory(self, hermes_home):
        _write_memory_config(hermes_home, provider="minir", minir={})
        assert "memory" not in _memory_tool_names()

    def test_memory_tool_stays_when_the_operator_opts_out(self, hermes_home):
        _write_memory_config(
            hermes_home, provider="minir", minir={"block_builtin_memory": False}
        )
        assert "memory" in _memory_tool_names()


class TestAgentInitDoesNotLoadTheFiles:
    @pytest.fixture(autouse=True)
    def _ordinary_worker(self, monkeypatch):
        monkeypatch.setattr("agent.safe_worker_policy.safe_worker_enabled", lambda: False)

    def _agent(self):
        class _Agent:
            enabled_toolsets = ["memory"]
            disabled_toolsets = []
            _memory_manager = None
            tools = None

            def _emit_startup_warning(self, *args, **kwargs):
                pass

        return _Agent()

    def _seed_files(self, home):
        memories = home / "memories"
        memories.mkdir(parents=True, exist_ok=True)
        (memories / "MEMORY.md").write_text("an agent note\n", encoding="utf-8")
        (memories / "USER.md").write_text("a user fact\n", encoding="utf-8")

    def test_no_store_is_built_for_the_plugin_block(self, hermes_home):
        self._seed_files(hermes_home)
        agent = self._agent()
        agent_init._init_memory(agent, {"memory": {"minir": {}}}, True, "cli")

        assert agent._memory_store is None
        assert (agent._memory_enabled, agent._user_profile_enabled) == (False, False)

    def test_files_still_load_when_the_operator_opts_out(self, hermes_home):
        self._seed_files(hermes_home)
        agent = self._agent()
        agent_init._init_memory(
            agent, {"memory": {"minir": {"block_builtin_memory": False}}}, True, "cli"
        )

        assert agent._memory_store is not None
        assert "a user fact" in (agent._memory_store.format_for_system_prompt("user") or "")
        assert "an agent note" in (
            agent._memory_store.format_for_system_prompt("memory") or ""
        )
