from gateway.runtime_guard import (
    GuardContext,
    RuntimeGuardConfig,
    RuntimeGuardScope,
    get_runtime_guard_manager,
)


def test_surface_policy_blocks_scoped_send_message_tool():
    manager = get_runtime_guard_manager(
        RuntimeGuardConfig(
            enabled=True,
            dry_run=False,
            scope=RuntimeGuardScope(platforms=("discord",)),
        )
    )
    ctx = GuardContext(surface="send_message_tool", platform="discord", chat_id="chat-1")
    final_ctx = GuardContext(surface="assistant_final", platform="discord", chat_id="chat-1")

    assert manager.surface_action(final_ctx) == "guard"
    assert manager.should_guard_surface(final_ctx) is True
    assert manager.surface_action(ctx) == "block"
    assert manager.should_block_surface(ctx) is True

    decision = manager.check_surface_policy(ctx)

    assert decision.allowed is False
    assert decision.status == "surface_blocked"
    assert decision.reason == "surface_policy_block"


def test_command_ack_surface_allowed():
    manager = get_runtime_guard_manager(
        RuntimeGuardConfig(
            enabled=True,
            dry_run=False,
            scope=RuntimeGuardScope(platforms=("discord",)),
        )
    )
    ctx = GuardContext(surface="command_ack", platform="discord", chat_id="chat-1")

    assert manager.surface_action(ctx) == "allow"
    assert manager.should_block_surface(ctx) is False
    assert manager.check_surface_policy(ctx).allowed is True
