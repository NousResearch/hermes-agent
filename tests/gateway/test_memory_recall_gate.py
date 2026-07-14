"""End-to-end gateway-to-provider coverage for the memory-recall gate (#40170).

Drives the real production chain: gateway identity decision
(``_memory_recall_requester_is_operator``) → ``_apply_memory_recall_gate``
(the wire-up both agent-construction sites call) → the ``_skip_memory_injection``
flag → the actual prefetch site in ``build_turn_context`` → the memory
provider's ``prefetch_all``.

Because it runs the real gateway decision and the real prefetch boundary, it
fails if the wire-up is dropped, if the operator/customer boundary regresses,
or if the prefetch site moves — the gaps teknium's review called out.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


# --------------------------------------------------------------------------
# Gateway side: identity decision + the wire-up both construction sites call.
# --------------------------------------------------------------------------

def _runner(config: GatewayConfig):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = config
    return runner


def _source(platform: Platform, *, user_id: str, chat_type: str = "dm") -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id="chat-1",
        user_name="tester",
        chat_type=chat_type,
    )


def _config_with_admins(admins):
    return GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True, extra={"allow_admin_from": list(admins)}
            )
        }
    )


def test_operator_admin_is_recognized():
    runner = _runner(_config_with_admins(["op-1"]))
    assert runner._memory_recall_requester_is_operator(
        _source(Platform.TELEGRAM, user_id="op-1")
    ) is True
    assert runner._memory_recall_requester_is_operator(
        _source(Platform.TELEGRAM, user_id="customer-9")
    ) is False


def test_no_admin_list_means_no_operator():
    """With gating disabled (no allow_admin_from), we cannot confirm an
    operator, so nobody is treated as one — the safe default."""
    runner = _runner(GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)}))
    assert runner._memory_recall_requester_is_operator(
        _source(Platform.TELEGRAM, user_id="anyone")
    ) is False


def test_apply_gate_operator_dm_allows_recall():
    runner = _runner(_config_with_admins(["op-1"]))
    agent = types.SimpleNamespace(_skip_memory_injection=None)
    runner._apply_memory_recall_gate(agent, _source(Platform.TELEGRAM, user_id="op-1"))
    assert agent._skip_memory_injection is False


def test_apply_gate_customer_dm_suppresses_recall():
    runner = _runner(_config_with_admins(["op-1"]))
    agent = types.SimpleNamespace(_skip_memory_injection=None)
    runner._apply_memory_recall_gate(agent, _source(Platform.TELEGRAM, user_id="customer-9"))
    assert agent._skip_memory_injection is True


def test_apply_gate_group_suppresses_even_for_operator():
    runner = _runner(_config_with_admins(["op-1"]))
    agent = types.SimpleNamespace(_skip_memory_injection=None)
    runner._apply_memory_recall_gate(
        agent, _source(Platform.TELEGRAM, user_id="op-1", chat_type="group")
    )
    assert agent._skip_memory_injection is True


def test_apply_gate_local_cli_allows_recall():
    runner = _runner(_config_with_admins(["op-1"]))
    agent = types.SimpleNamespace(_skip_memory_injection=None)
    runner._apply_memory_recall_gate(agent, _source(Platform.LOCAL, user_id="op-1"))
    assert agent._skip_memory_injection is False


# --------------------------------------------------------------------------
# Full chain: gateway decision → flag → build_turn_context → prefetch_all.
# --------------------------------------------------------------------------

class _FakeGuardrails:
    def reset_for_turn(self):
        pass


class _FakeAgent:
    """Minimal stand-in the gate + build_turn_context both operate on."""

    def __init__(self):
        self.session_id = "sess-1"
        self.model = "test/model"
        self.provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-x"
        self.api_mode = "chat_completions"
        self.platform = "telegram"
        self.quiet_mode = True
        self.max_iterations = 90
        self.tools = []
        self.valid_tool_names = set()
        self.enabled_toolsets = None
        self.disabled_toolsets = None
        self._skip_mcp_refresh = False
        self.compression_enabled = False
        self.context_compressor = types.SimpleNamespace(protect_first_n=2, protect_last_n=2)
        self._cached_system_prompt = "SYSTEM"
        self._memory_store = None
        self._memory_manager = None
        self._memory_nudge_interval = 0
        self._turns_since_memory = 0
        self._user_turn_count = 0
        self._todo_store = types.SimpleNamespace(has_items=lambda: True, _hydrate=lambda *a, **k: None)
        self._tool_guardrails = _FakeGuardrails()
        self._compression_warning = None
        self._interrupt_requested = False
        self._memory_write_origin = "assistant_tool"
        self._stream_context_scrubber = None
        self._stream_think_scrubber = None
        self._invalid_tool_retries = -1
        self._vision_supported = None
        self._skip_memory_injection = False

    def _ensure_db_session(self):
        pass

    def _restore_primary_runtime(self):
        pass

    def _cleanup_dead_connections(self):
        return False

    def _emit_status(self, _msg):
        pass

    def _replay_compression_warning(self):
        pass

    def _hydrate_todo_store(self, *_a, **_k):
        pass

    def _safe_print(self, *_a, **_k):
        pass

    def _persist_session(self, *_a, **_k):
        pass


def _run_turn(agent):
    from agent.turn_context import build_turn_context

    return build_turn_context(
        agent=agent,
        user_message="hello",
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s,
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


def _memory_manager():
    mm = MagicMock()
    mm.prefetch_all = MagicMock(return_value="OPERATOR MEMORY")
    mm.on_turn_start = MagicMock()
    return mm


def test_customer_dm_end_to_end_suppresses_prefetch():
    """Gateway customer DM → gate suppresses → build_turn_context never
    prefetches operator memory."""
    runner = _runner(_config_with_admins(["op-1"]))
    agent = _FakeAgent()
    agent._memory_manager = _memory_manager()

    runner._apply_memory_recall_gate(agent, _source(Platform.TELEGRAM, user_id="customer-9"))
    ctx = _run_turn(agent)

    assert agent._skip_memory_injection is True
    agent._memory_manager.prefetch_all.assert_not_called()
    assert ctx.ext_prefetch_cache == ""


def test_operator_dm_end_to_end_injects_recall():
    """Gateway operator DM → gate allows → build_turn_context prefetches and
    caches the recall into the turn."""
    runner = _runner(_config_with_admins(["op-1"]))
    agent = _FakeAgent()
    agent._memory_manager = _memory_manager()

    runner._apply_memory_recall_gate(agent, _source(Platform.TELEGRAM, user_id="op-1"))
    ctx = _run_turn(agent)

    assert agent._skip_memory_injection is False
    agent._memory_manager.prefetch_all.assert_called_once()
    assert ctx.ext_prefetch_cache == "OPERATOR MEMORY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
