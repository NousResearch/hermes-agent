"""Regression tests for issue #80646: agent_context derivation.

The memory-provider contract (agent/memory_provider.py) documents ``agent_context``
as "primary" | "subagent" | "cron" | "flush" with skip-writes semantics for
non-primary contexts, but ``_memory_provider_init_kwargs`` hardcoded "primary" for
every session — leaving every provider's context-skip logic (supermemory's
``_write_enabled``, honcho's cron/flush skip, external providers' ``skip_contexts``)
dead. Cron runs since ef04d846e9 (skip_memory=False) wrote their turns into stores
operators had configured to skip them.

These tests pin the contract through the REAL production function so a future
re-hardcode fails loudly.
"""

from types import SimpleNamespace

from agent.agent_init import _memory_provider_init_kwargs


def _fake_agent():
    """Minimum attribute surface _memory_provider_init_kwargs reads."""
    return SimpleNamespace(
        session_id="sess-80646",
        _session_db=None,
        _emit_warning=None,
        _emit_status=None,
        **{
            f"_{name}": None
            for name in (
                "user_id",
                "user_id_alt",
                "user_name",
                "chat_id",
                "chat_name",
                "chat_type",
                "thread_id",
                "gateway_session_key",
            )
        },
    )


class TestAgentContextDerivation:
    def test_cron_platform_yields_cron_context(self):
        """Scheduler passes platform='cron' (cron/scheduler.py) — providers must
        receive agent_context='cron' so skip-configured stores stop capturing
        cron turns (the reported production symptom)."""
        kwargs = _memory_provider_init_kwargs(_fake_agent(), "cron")
        assert kwargs["agent_context"] == "cron"
        assert kwargs["platform"] == "cron"

    def test_subagent_platform_yields_subagent_context(self):
        """delegate_task children pass platform='subagent' (tools/delegate_tool.py)
        — providers must receive agent_context='subagent' (supermemory branches on it)."""
        kwargs = _memory_provider_init_kwargs(_fake_agent(), "subagent")
        assert kwargs["agent_context"] == "subagent"
        assert kwargs["platform"] == "subagent"

    def test_primary_platforms_stay_primary(self):
        """Interactive surfaces (cli + gateway platforms + batch) are primary
        contexts: memory writes must keep flowing exactly as before."""
        for platform in ("cli", "telegram", "discord", "slack", "batch"):
            kwargs = _memory_provider_init_kwargs(_fake_agent(), platform)
            assert kwargs["agent_context"] == "primary", platform
            assert kwargs["platform"] == platform

    def test_none_and_empty_platform_fall_back_to_primary(self):
        """platform=None / '' must never crash nor misclassify: they normalize to
        cli (as kwargs['platform'] already does) and stay a primary context."""
        for platform in (None, ""):
            kwargs = _memory_provider_init_kwargs(_fake_agent(), platform)
            assert kwargs["agent_context"] == "primary", repr(platform)
            assert kwargs["platform"] == "cli", repr(platform)

    def test_derivation_reaches_initialize_all(self):
        """End-to-end through MemoryManager.initialize_all: a recording provider
        initialized with the cron-scoping kwargs actually observes
        agent_context='cron' — proving the plumbing providers rely on."""
        from agent.memory_manager import MemoryManager
        from tests.agent.test_memory_user_id import RecordingProvider

        mgr = MemoryManager()
        provider = RecordingProvider()
        mgr.add_provider(provider)
        mgr.initialize_all(**_memory_provider_init_kwargs(_fake_agent(), "cron"))
        assert provider._init_kwargs.get("agent_context") == "cron"


class TestSupermemorySkipContract:
    """The bundled provider whose skip logic was dead: initialize() must disable
    writes for cron contexts once agent_context is live. Uses an empty hermes_home
    (config defaults, no API key) so no network or env dependency."""

    def test_cron_context_disables_supermemory_writes(self, tmp_path):
        from plugins.memory.supermemory import SupermemoryMemoryProvider

        provider = SupermemoryMemoryProvider()
        provider.initialize(
            session_id="sess-cron",
            hermes_home=str(tmp_path),
            agent_context="cron",
        )
        assert provider._write_enabled is False

    def test_primary_context_keeps_supermemory_writes(self, tmp_path):
        from plugins.memory.supermemory import SupermemoryMemoryProvider

        provider = SupermemoryMemoryProvider()
        provider.initialize(
            session_id="sess-primary",
            hermes_home=str(tmp_path),
            agent_context="primary",
        )
        assert provider._write_enabled is True
