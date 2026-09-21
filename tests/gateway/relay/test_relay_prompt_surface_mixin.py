"""Tests for the extracted RelayPromptMixin (gateway/relay/adapter.py shard, #79990).

The Phase-3 interactive prompt cluster (mint / send / consume / resolve plus the
in-channel lifecycle acks) was lifted byte-verbatim out of ``gateway/relay/adapter.py``
into ``gateway/relay/prompt_surface.py``. These tests pin the two halves of that
contract: ``RelayAdapter`` resolves every moved method through the MRO, and the origin
module still re-exports the cluster's module-level names with the SAME object identity
(so ``gateway.relay.adapter.<name>`` keeps working, unchanged).
"""

from __future__ import annotations

import gateway.relay.adapter as adapter
import gateway.relay.prompt_surface as prompt_surface

PROMPT_METHODS = [
    "_mint_prompt",
    "_minted_here",
    "_pop_prompt",
    "_note_prompt_resolved",
    "_send_prompt",
    "_mint_and_send_prompt",
    "_send_exec_approval_prompt",
    "send_slash_confirm",
    "send_clarify",
    "_consume_prompt_response",
    "_resolve_exec_approval",
    "_resolve_slash_confirm",
    "_resolve_clarify",
    "_send_lifecycle_ack",
    "_notify_prompt_expired",
    "_prompt_reply_metadata",
]

REEXPORTS = (
    "_PROMPT_RESOLVERS",
    "_EXEC_APPROVAL_LABELS",
    "_SLASH_CONFIRM_LABELS",
)


def test_adapter_resolves_prompt_methods_from_the_mixin():
    for name in PROMPT_METHODS:
        assert getattr(adapter.RelayAdapter, name) is getattr(prompt_surface.RelayPromptMixin, name), name


def test_origin_module_reexports_cluster_names_by_identity():
    for name in REEXPORTS:
        assert getattr(adapter, name) is getattr(prompt_surface, name), name
    assert adapter._RESOLVED_PROMPT_MEMORY == prompt_surface._RESOLVED_PROMPT_MEMORY


def test_prompt_resolvers_point_at_the_live_methods():
    expected = {
        "exec_approval": "_resolve_exec_approval",
        "slash_confirm": "_resolve_slash_confirm",
        "clarify": "_resolve_clarify",
    }
    assert set(adapter._PROMPT_RESOLVERS) == set(expected)
    for kind, method in expected.items():
        assert adapter._PROMPT_RESOLVERS[kind] is getattr(adapter.RelayAdapter, method), kind


def test_class_level_prompt_constants_survive_the_move():
    assert adapter.RelayAdapter._EA_HEADER == prompt_surface.RelayPromptMixin._EA_HEADER
    assert adapter.RelayAdapter._EA_CMD_BUDGET == 1500
    assert adapter.RelayAdapter._PROMPT_UNAVAILABLE is prompt_surface.RelayPromptMixin._PROMPT_UNAVAILABLE


def test_moved_code_logs_to_the_origin_modules_logger():
    """Log-record parity: the extracted module must not rename the logger."""
    assert prompt_surface.logger is adapter.logger
    assert prompt_surface.logger.name == "gateway.relay.adapter"
