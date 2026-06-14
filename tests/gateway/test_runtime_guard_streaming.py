from gateway.runtime_guard import (
    GuardContext,
    GuardDecision,
    RuntimeGuardConfig,
    RuntimeGuardScope,
    RuntimeGuardStreamingConfig,
    get_runtime_guard_manager,
    register_runtime_guard_provider,
)


def test_disabled_config_does_not_disable_streaming():
    manager = get_runtime_guard_manager(RuntimeGuardConfig())
    ctx = GuardContext(surface="assistant_stream", platform="discord", chat_id="chat-1")

    assert manager.should_disable_streaming(ctx) is False
    assert manager.requires_first_visible_stream_guard(ctx) is False


def test_dry_run_config_does_not_disable_streaming():
    manager = get_runtime_guard_manager(
        RuntimeGuardConfig(
            enabled=True,
            dry_run=True,
            scope=RuntimeGuardScope(platforms=("discord",), chat_ids=("chat-1",)),
            streaming=RuntimeGuardStreamingConfig(policy="disable"),
        )
    )
    ctx = GuardContext(surface="assistant_stream", platform="discord", chat_id="chat-1")

    assert manager.should_disable_streaming(ctx) is False


def test_enabled_scoped_disable_policy_disables_streaming():
    manager = get_runtime_guard_manager(
        RuntimeGuardConfig(
            enabled=True,
            dry_run=False,
            scope=RuntimeGuardScope(platforms=("discord",)),
            streaming=RuntimeGuardStreamingConfig(policy="disable"),
        )
    )
    ctx = GuardContext(surface="assistant_stream", platform="discord", chat_id="chat-1")

    assert manager.should_disable_streaming(ctx) is True
    assert manager.requires_first_visible_stream_guard(ctx) is False


def test_guard_first_visible_policy_keeps_streaming_but_requires_guard():
    seen = []

    class RecordingGuard:
        def check(self, context):
            seen.append(context)
            return GuardDecision(allowed=True, reason="ok", status="allowed")

    register_runtime_guard_provider("record_stream_guard_test", RecordingGuard())
    manager = get_runtime_guard_manager(
        RuntimeGuardConfig(
            enabled=True,
            provider="record_stream_guard_test",
            dry_run=False,
            scope=RuntimeGuardScope(platforms=("discord",)),
            streaming=RuntimeGuardStreamingConfig(policy="guard_first_visible"),
        )
    )
    ctx = GuardContext(surface="assistant_final", platform="discord", chat_id="chat-1")

    assert manager.should_disable_streaming(ctx) is False
    assert manager.requires_first_visible_stream_guard(ctx) is True

    decision = manager.check_first_visible_stream(ctx)

    assert decision.allowed is True
    assert seen[0].surface == "assistant_stream"
