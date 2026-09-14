from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.evaluator_policy import SubprocessEvaluatorPolicy
from gateway.pre_delivery import PreDeliveryGate
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


EVALUATOR_ADAPTER = Path(__file__).resolve().parents[1] / "fixtures" / "evaluator_policy.py"


def _policy():
    return {
        "required_patterns": ["evidence"],
        "forbidden_patterns": [],
    }


def test_subprocess_policy_connects_to_evaluator_pass():
    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )
    gate = PreDeliveryGate(mode="strict", policy=policy)

    decision = gate.evaluate_sync(
        final_text="Answer includes evidence.",
        metadata={"turn_id": "bridge-pass-001", "platform": "test"},
    )

    assert decision.allowed is True
    assert decision.status == "passed"
    assert decision.final_text == "Answer includes evidence."


def test_subprocess_policy_connects_to_evaluator_block():
    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )
    gate = PreDeliveryGate(mode="strict", policy=policy)

    decision = gate.evaluate_sync(
        final_text="Answer without the marker.",
        metadata={"turn_id": "bridge-block-001", "platform": "test"},
    )

    assert decision.allowed is False
    assert decision.status == "blocked"
    assert decision.final_text is None


def test_subprocess_policy_does_not_use_shell_commands():
    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )
    assert policy._command[0] == sys.executable


def test_subprocess_policy_rejects_mismatched_request_id(monkeypatch):
    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )

    class Completed:
        stdout = '{"schema_version": 1, "request_id": "other-turn", "status": "passed", "allowed": true, "final_text": "ok"}'

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Completed())
    with pytest.raises(RuntimeError, match="mismatched request_id"):
        policy(final_text="Answer includes evidence.", metadata={"turn_id": "turn-1"})


def test_subprocess_policy_rejects_unknown_schema(monkeypatch):
    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )

    class Completed:
        stdout = '{"schema_version": 999, "request_id": "turn-1", "status": "passed", "allowed": true, "final_text": "ok"}'

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Completed())
    with pytest.raises(RuntimeError, match="unsupported schema_version"):
        policy(final_text="Answer includes evidence.", metadata={"turn_id": "turn-1"})


def test_subprocess_policy_requires_turn_identity():
    policy = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )
    with pytest.raises(RuntimeError, match="requires a non-empty turn_id"):
        policy(final_text="Answer includes evidence.", metadata={})


def test_gateway_runner_exposes_supported_gate_injection_seam():
    from gateway.run import GatewayRunner

    parameter = inspect.signature(GatewayRunner.__init__).parameters["pre_delivery_gate"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.asyncio
async def test_local_e2e_gate_evaluator_and_real_consumer():
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    Adapter = type("E2EAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 4096})
    Adapter.__abstractmethods__ = frozenset()
    adapter = Adapter.__new__(Adapter)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    adapter.draft_calls = []

    async def send_draft(*, chat_id, draft_id, content, metadata=None):
        adapter.draft_calls.append(content)
        return SendResult(success=True, message_id=None)

    adapter.supports_draft_streaming = lambda chat_type=None, metadata=None: True
    adapter.send_draft = send_draft
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg-e2e"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))

    evaluator = SubprocessEvaluatorPolicy(
        [sys.executable, str(EVALUATOR_ADAPTER)],
        policy=_policy(),
        agent_configuration_id="hermes-test-v1",
        evaluator_configuration_id="evaluator-test-v1",
    )
    gate = PreDeliveryGate(mode="strict", policy=evaluator)
    consumer = GatewayStreamConsumer(
        adapter,
        "chat-e2e",
        StreamConsumerConfig(transport="auto", chat_type="dm", cursor=""),
    )
    consumer.quarantine_content_delivery()
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Answer includes ")
    consumer.on_delta("evidence.")
    await asyncio.sleep(0.08)
    decision = await gate.evaluate(
        final_text="Answer includes evidence.",
        metadata={"turn_id": "local-e2e-001", "platform": "test"},
    )
    assert decision.allowed is True
    consumer.finish(decision.final_text)
    await task

    assert adapter.draft_calls == []
    adapter.send.assert_awaited_once()
    assert adapter.send.call_args.kwargs["content"] == "Answer includes evidence."
