from __future__ import annotations

import builtins
import json
from types import SimpleNamespace

import pytest

from agent.local_muncho.backends import (
    UnavailableCanonicalBrain,
    build_canonical_brain_backend,
    describe_configured_backends,
)
from agent.local_muncho.brain import CanonicalBrainUnavailable
from agent.local_muncho.evidence import validate_final_output
from agent.local_muncho.runtime import (
    LocalMunchoRuntime,
    reset_current_canonical_brain,
    reset_current_runtime_context,
    set_current_canonical_brain,
    set_current_runtime_context,
    validate_final_response_for_agent,
)
from agent.local_muncho.testing import InMemoryCanonicalBrain
from agent.local_muncho.types import (
    LeaseState,
    RuntimeContext,
    ToolEvidence,
    VisibleSendIntent,
    WorkerContract,
    utc_ts,
)
from gateway.local_muncho_guard import guard_stream_consumer_config, guard_visible_send
from gateway.stream_consumer import StreamConsumerConfig


def _cfg(**overrides):
    cfg = {
        "enabled": True,
        "lane": "internal-support",
        "runtime_id": "local-muncho",
        "runtime_kind": "local-primary",
        "lease_ttl_seconds": 90,
        "fail_open": False,
    }
    cfg.update(overrides)
    return cfg


def _ctx(lane: str | None = "internal-support") -> RuntimeContext:
    return RuntimeContext(
        lane=lane,
        platform="discord",
        chat_id="chat",
        thread_id="thread",
    )


def _lease(**overrides) -> LeaseState:
    data = {
        "lease_owner": "local-muncho",
        "active_runtime": "local-primary",
        "expires_at": utc_ts() + 90,
    }
    data.update(overrides)
    return LeaseState(**data)


def _valid_frame() -> str:
    return "\n".join(
        [
            "VERDICT: PASS",
            "TL;DR: ok",
            "CATEGORY: runtime_guard",
            "EVIDENCE_CHECKED: unit-test",
            "EVIDENCE_GAP: none",
            "STATUS: done",
            "NEXT_ACTION: none",
            "APPROVAL_NEEDED: no",
            "RISK: low",
        ]
    )


def test_disabled_runtime_is_noop_for_final_validation() -> None:
    class Agent:
        _local_muncho_config = {"enabled": False}
        _local_muncho_context = _ctx()
        _local_muncho_brain = None

    text = "ordinary Hermes response without Muncho frame"
    assert validate_final_response_for_agent(Agent(), text, messages=[]) == text


def test_backend_boundary_is_no_network_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUNCHO_POSTGRES_DSN", "postgres://placeholder")
    monkeypatch.delenv("MUNCHO_REDIS_URL", raising=False)

    descriptors = describe_configured_backends(_cfg())
    assert descriptors[0].kind == "postgres"
    assert descriptors[0].configured is True
    assert descriptors[1].kind == "redis"
    assert descriptors[1].configured is False

    backend = build_canonical_brain_backend(_cfg())
    assert isinstance(backend, UnavailableCanonicalBrain)
    with pytest.raises(CanonicalBrainUnavailable):
        backend.read_active_lease()


def test_enabled_runtime_without_brain_fails_closed_for_mutating_tool() -> None:
    runtime = LocalMunchoRuntime(_cfg(), _ctx(), brain=None)
    decision = runtime.guard_tool_action("send_message", {"action": "send"})
    assert not decision.allowed
    assert "canonical brain unavailable" in decision.reason


def test_tool_guard_internal_error_blocks_when_runtime_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.tool_executor import _local_muncho_tool_block

    class Agent:
        _local_muncho_config = _cfg()
        _local_muncho_context = _ctx()
        _local_muncho_brain = InMemoryCanonicalBrain(lease=_lease())

    def raise_guard_error(self, tool_name, args):
        raise RuntimeError("guard exploded")

    monkeypatch.setattr(
        LocalMunchoRuntime,
        "guard_tool_action",
        raise_guard_error,
    )

    result, reason = _local_muncho_tool_block(
        Agent(),
        "send_message",
        {"target": "discord:123", "message": "hello"},
    )
    assert result is not None
    parsed = json.loads(result)
    assert parsed["status"] == "blocked"
    assert parsed["blocked_by"] == "local_muncho_runtime"
    assert parsed["code"] == "local_muncho_guard_error"
    assert "tool_action guard internal error" in parsed["error"]
    assert "guard exploded" in reason


def test_tool_guard_internal_error_noops_when_runtime_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.tool_executor import _local_muncho_tool_block

    class Agent:
        _local_muncho_config = {"enabled": False}
        _local_muncho_context = _ctx()
        _local_muncho_brain = None

    def raise_guard_error(self, tool_name, args):
        raise RuntimeError("guard exploded")

    monkeypatch.setattr(
        LocalMunchoRuntime,
        "guard_tool_action",
        raise_guard_error,
    )

    result, reason = _local_muncho_tool_block(
        Agent(),
        "send_message",
        {"target": "discord:123", "message": "hello"},
    )
    assert result is None
    assert reason is None


def test_tool_guard_import_failure_blocks_when_runtime_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.tool_executor import _local_muncho_tool_block

    class Agent:
        _local_muncho_config = _cfg()
        _local_muncho_context = _ctx()
        _local_muncho_brain = None

    real_import = builtins.__import__

    def fail_runtime_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "agent.local_muncho.runtime":
            raise ImportError("runtime helper unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_runtime_import)

    result, reason = _local_muncho_tool_block(
        Agent(),
        "send_message",
        {"target": "discord:123", "message": "hello"},
    )
    assert result is not None
    parsed = json.loads(result)
    assert parsed["status"] == "blocked"
    assert parsed["blocked_by"] == "local_muncho_runtime"
    assert parsed["code"] == "local_muncho_guard_error"
    assert "tool_action guard internal error" in parsed["error"]
    assert "runtime helper unavailable" in reason


def test_tool_guard_import_failure_noops_when_runtime_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.tool_executor import _local_muncho_tool_block

    class Agent:
        _local_muncho_config = {"enabled": False}
        _local_muncho_context = _ctx()
        _local_muncho_brain = None

    real_import = builtins.__import__

    def fail_runtime_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "agent.local_muncho.runtime":
            raise ImportError("runtime helper unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_runtime_import)

    result, reason = _local_muncho_tool_block(
        Agent(),
        "send_message",
        {"target": "discord:123", "message": "hello"},
    )
    assert result is None
    assert reason is None


def test_sequential_tool_block_preserves_local_muncho_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.tool_executor as tool_executor

    class Hints:
        def check_tool_call(self, function_name, function_args):
            return ""

    class Agent:
        _interrupt_requested = False
        _local_muncho_config = _cfg()
        _local_muncho_context = _ctx()
        _local_muncho_brain = None
        _subdirectory_hints = Hints()
        quiet_mode = True
        verbose_logging = False
        tool_progress_mode = "off"
        tool_progress_callback = None
        tool_start_callback = None
        tool_complete_callback = None
        tool_delay = 0
        session_id = "session"

        def _tool_result_content_for_active_model(self, function_name, result):
            return result

        def _apply_pending_steer_to_tool_results(self, messages, count):
            return None

        def _touch_activity(self, message):
            return None

    monkeypatch.setattr(tool_executor, "_emit_terminal_post_tool_call", lambda *a, **kw: None)
    monkeypatch.setattr(
        tool_executor,
        "maybe_persist_tool_result",
        lambda content, **kwargs: content,
    )
    monkeypatch.setattr(tool_executor, "enforce_turn_budget", lambda *a, **kw: None)

    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="send_message",
            arguments=json.dumps({"target": "discord:123", "message": "hello"}),
        ),
    )
    messages: list[dict] = []

    tool_executor.execute_tool_calls_sequential(
        Agent(),
        SimpleNamespace(tool_calls=[tool_call]),
        messages,
        "task-1",
    )

    parsed = json.loads(messages[-1]["content"])
    assert parsed["status"] == "blocked"
    assert parsed["blocked_by"] == "local_muncho_runtime"
    assert parsed["code"] == "local_muncho_block"
    assert "canonical brain unavailable" in parsed["error"]


def test_active_local_lease_allows_tool_and_force_cloud_blocks() -> None:
    runtime = LocalMunchoRuntime(
        _cfg(),
        _ctx(),
        brain=InMemoryCanonicalBrain(lease=_lease()),
    )
    assert runtime.guard_tool_action("send_message", {"action": "send"}).allowed

    blocked = LocalMunchoRuntime(
        _cfg(),
        _ctx(),
        brain=InMemoryCanonicalBrain(lease=_lease(flags=("force-cloud",))),
    )
    decision = blocked.guard_tool_action("send_message", {"action": "send"})
    assert not decision.allowed
    assert "force-cloud" in decision.reason


def test_final_output_requires_structured_evidence_frame() -> None:
    result = validate_final_output(
        "Completed the task successfully.",
        context=_ctx(),
        evidence=(),
    )
    assert not result.allowed
    assert result.replacement_text is not None
    assert "VERDICT: BLOCKED" in result.replacement_text
    assert "missing required frame fields" in result.reason


def test_completion_claim_requires_durable_evidence() -> None:
    result = validate_final_output(
        _valid_frame() + "\ncompleted",
        context=_ctx(),
        evidence=(),
    )
    assert not result.allowed
    assert "no durable evidence" in result.reason

    ok = validate_final_output(
        _valid_frame() + "\ncompleted",
        context=_ctx(),
        evidence=(
            ToolEvidence(
                tool_name="send_message",
                result={"success": True},
                success=True,
            ),
        ),
    )
    assert ok.allowed


def test_final_response_rechecks_lease_before_return() -> None:
    brain = InMemoryCanonicalBrain(
        lease_sequence=(_lease(), _lease(flags=("pause-all",))),
    )
    runtime = LocalMunchoRuntime(_cfg(), _ctx(), brain=brain)
    result = runtime.validate_final_output(_valid_frame(), evidence=())
    assert not result.allowed
    assert result.replacement_text is not None
    assert "pause-all" in result.replacement_text


def test_final_output_guard_internal_error_replaces_when_runtime_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Agent:
        _local_muncho_config = _cfg()
        _local_muncho_context = _ctx()
        _local_muncho_brain = InMemoryCanonicalBrain(lease=_lease())

    def raise_validation_error(self, text, *, evidence):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(
        LocalMunchoRuntime,
        "validate_final_output",
        raise_validation_error,
    )

    result = validate_final_response_for_agent(Agent(), "original response", messages=[])
    assert result.startswith("VERDICT: BLOCKED")
    assert "final_output guard internal error" in result
    assert "validator exploded" in result
    assert result != "original response"


def test_final_output_guard_internal_error_noops_when_runtime_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Agent:
        _local_muncho_config = {"enabled": False}
        _local_muncho_context = _ctx()
        _local_muncho_brain = None

    def raise_validation_error(self, text, *, evidence):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(
        LocalMunchoRuntime,
        "validate_final_output",
        raise_validation_error,
    )

    assert (
        validate_final_response_for_agent(Agent(), "original response", messages=[])
        == "original response"
    )


@pytest.mark.asyncio
async def test_visible_send_guard_replaces_unverified_text_before_send() -> None:
    runtime = LocalMunchoRuntime(
        _cfg(),
        _ctx(),
        brain=InMemoryCanonicalBrain(lease=_lease()),
    )
    decision = await guard_visible_send(
        VisibleSendIntent(
            kind="send_message",
            platform="discord",
            chat_id="chat",
            thread_id="thread",
            text="done",
        ),
        runtime=runtime,
    )
    assert decision.allowed
    assert decision.replacement_text is not None
    assert "VERDICT: BLOCKED" in decision.replacement_text


def test_stream_config_guard_is_disabled_by_default_noop() -> None:
    runtime = LocalMunchoRuntime({"enabled": False}, _ctx(), brain=None)
    config = StreamConsumerConfig(buffer_only=False, transport="auto")
    assert guard_stream_consumer_config(config, runtime=runtime) is config


def test_stream_config_guard_can_force_buffer_only_copy() -> None:
    runtime = LocalMunchoRuntime(
        _cfg(streaming={"buffer_only": True, "drafts_enabled": False}),
        _ctx(),
        brain=InMemoryCanonicalBrain(lease=_lease()),
    )
    config = StreamConsumerConfig(buffer_only=False, transport="auto")
    guarded = guard_stream_consumer_config(config, runtime=runtime)
    assert guarded is not config
    assert guarded.buffer_only is True
    assert guarded.transport == "edit"
    assert config.buffer_only is False
    assert config.transport == "auto"


def test_send_message_tool_guard_noops_when_runtime_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gateway.config as gateway_config
    import tools.send_message_tool as send_message_tool
    from gateway.config import Platform, PlatformConfig

    class FakeGatewayConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="token")}

        def get_home_channel(self, platform):
            return None

    sent: dict[str, str] = {}

    async def fake_send_to_platform(platform, pconfig, chat_id, message, **kwargs):
        sent["message"] = message
        return {
            "success": True,
            "platform": platform.value,
            "chat_id": chat_id,
            "message_id": "m1",
        }

    monkeypatch.setattr(
        gateway_config,
        "load_gateway_config",
        lambda: FakeGatewayConfig(),
    )
    monkeypatch.setattr(send_message_tool, "_send_to_platform", fake_send_to_platform)
    monkeypatch.setattr(
        "agent.local_muncho.runtime._load_runtime_config",
        lambda: {"enabled": False},
    )

    parsed = json.loads(
        send_message_tool._handle_send({"target": "discord:123", "message": "hello"})
    )
    assert parsed["success"] is True
    assert sent["message"] == "hello"


def test_send_message_tool_guard_error_noops_when_runtime_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gateway.config as gateway_config
    import gateway.local_muncho_guard as local_muncho_guard
    import tools.send_message_tool as send_message_tool
    from gateway.config import Platform, PlatformConfig

    class FakeGatewayConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="token")}

        def get_home_channel(self, platform):
            return None

    sent: dict[str, str] = {}

    async def fake_send_to_platform(platform, pconfig, chat_id, message, **kwargs):
        sent["message"] = message
        return {
            "success": True,
            "platform": platform.value,
            "chat_id": chat_id,
            "message_id": "m1",
        }

    async def raise_visible_guard(*args, **kwargs):
        raise RuntimeError("visible guard exploded")

    monkeypatch.setattr(
        gateway_config,
        "load_gateway_config",
        lambda: FakeGatewayConfig(),
    )
    monkeypatch.setattr(send_message_tool, "_send_to_platform", fake_send_to_platform)
    monkeypatch.setattr(local_muncho_guard, "guard_visible_send", raise_visible_guard)
    monkeypatch.setattr(
        "agent.local_muncho.runtime._load_runtime_config",
        lambda: {"enabled": False},
    )

    parsed = json.loads(
        send_message_tool._handle_send({"target": "discord:123", "message": "hello"})
    )
    assert parsed["success"] is True
    assert sent["message"] == "hello"


def test_send_message_tool_replaces_visible_text_before_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gateway.config as gateway_config
    import tools.send_message_tool as send_message_tool
    from gateway.config import Platform, PlatformConfig

    class FakeGatewayConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="token")}

        def get_home_channel(self, platform):
            return None

    sent: dict[str, str] = {}

    async def fake_send_to_platform(platform, pconfig, chat_id, message, **kwargs):
        sent["message"] = message
        return {
            "success": True,
            "platform": platform.value,
            "chat_id": chat_id,
            "message_id": "m1",
        }

    monkeypatch.setattr(
        gateway_config,
        "load_gateway_config",
        lambda: FakeGatewayConfig(),
    )
    monkeypatch.setattr(send_message_tool, "_send_to_platform", fake_send_to_platform)
    monkeypatch.setattr("agent.local_muncho.runtime._load_runtime_config", lambda: _cfg())
    ctx_token = set_current_runtime_context(_ctx())
    brain_token = set_current_canonical_brain(InMemoryCanonicalBrain(lease=_lease()))
    try:
        parsed = json.loads(
            send_message_tool._handle_send({"target": "discord:123", "message": "done"})
        )
    finally:
        reset_current_canonical_brain(brain_token)
        reset_current_runtime_context(ctx_token)

    assert parsed["success"] is True
    assert sent["message"].startswith("VERDICT: BLOCKED")


def test_send_message_tool_guard_error_blocks_when_runtime_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gateway.config as gateway_config
    import gateway.local_muncho_guard as local_muncho_guard
    import tools.send_message_tool as send_message_tool
    from gateway.config import Platform, PlatformConfig

    class FakeGatewayConfig:
        platforms = {Platform.DISCORD: PlatformConfig(enabled=True, token="token")}

        def get_home_channel(self, platform):
            return None

    sent: dict[str, str] = {}

    async def fake_send_to_platform(platform, pconfig, chat_id, message, **kwargs):
        sent["message"] = message
        return {
            "success": True,
            "platform": platform.value,
            "chat_id": chat_id,
            "message_id": "m1",
        }

    async def raise_visible_guard(*args, **kwargs):
        raise RuntimeError("visible guard exploded")

    monkeypatch.setattr(
        gateway_config,
        "load_gateway_config",
        lambda: FakeGatewayConfig(),
    )
    monkeypatch.setattr(send_message_tool, "_send_to_platform", fake_send_to_platform)
    monkeypatch.setattr(local_muncho_guard, "guard_visible_send", raise_visible_guard)
    monkeypatch.setattr("agent.local_muncho.runtime._load_runtime_config", lambda: _cfg())
    ctx_token = set_current_runtime_context(_ctx())
    brain_token = set_current_canonical_brain(InMemoryCanonicalBrain(lease=_lease()))
    try:
        parsed = json.loads(
            send_message_tool._handle_send({"target": "discord:123", "message": "done"})
        )
    finally:
        reset_current_canonical_brain(brain_token)
        reset_current_runtime_context(ctx_token)

    assert parsed["status"] == "blocked"
    assert parsed["blocked_by"] == "local_muncho_runtime"
    assert parsed["code"] == "local_muncho_guard_error"
    assert "visible_send guard internal error" in parsed["error"]
    assert sent == {}


def test_worker_spawn_requires_active_lease() -> None:
    runtime = LocalMunchoRuntime(
        _cfg(),
        _ctx(),
        brain=InMemoryCanonicalBrain(lease=None),
    )
    decision = runtime.guard_worker_spawn(
        WorkerContract(goals=("x",), task_count=1),
        source="delegate_task",
    )
    assert not decision.allowed
    assert "active lease missing" in decision.reason


def test_tool_block_result_is_json() -> None:
    runtime = LocalMunchoRuntime(_cfg(), _ctx(), brain=InMemoryCanonicalBrain(lease=None))
    decision = runtime.guard_tool_action("send_message", {"action": "send"})
    from agent.local_muncho.runtime import tool_block_result

    parsed = json.loads(tool_block_result(decision))
    assert parsed["blocked_by"] == "local_muncho_runtime"
    assert parsed["status"] == "blocked"
