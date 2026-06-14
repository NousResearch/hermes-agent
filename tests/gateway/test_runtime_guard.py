from gateway.runtime_guard import (
    GuardContext,
    GuardDecision,
    RuntimeGuardConfig,
    RuntimeGuardScope,
    get_runtime_guard_manager,
    register_runtime_guard_provider,
)


def test_runtime_guard_config_defaults_disabled():
    cfg = RuntimeGuardConfig()

    assert cfg.enabled is False
    assert cfg.provider == "noop"
    assert cfg.dry_run is True
    assert cfg.fail_closed is True
    assert cfg.streaming.policy == "disable"
    assert cfg.delivery_surfaces.policy_for("assistant_final") == "guard"
    assert cfg.delivery_surfaces.policy_for("assistant_stream") == "disable"
    assert cfg.delivery_surfaces.policy_for("send_message_tool") == "block"
    assert cfg.delivery_surfaces.policy_for("command_ack") == "allow"

    manager = get_runtime_guard_manager(cfg)
    ctx = GuardContext(surface="assistant_final", platform="discord", chat_id="chat-1")

    assert manager.is_scoped(ctx) is False
    decision = manager.check(ctx)
    assert decision.allowed is True
    assert decision.status == "disabled"
    assert decision.provider == "noop"


def test_noop_guard_allows_disabled_context():
    cfg = RuntimeGuardConfig(
        enabled=False,
        scope=RuntimeGuardScope(platforms=("discord",), chat_ids=("guarded-chat",)),
    )
    manager = get_runtime_guard_manager(cfg)
    ctx = GuardContext(
        surface="assistant_final",
        platform="discord",
        chat_id="guarded-chat",
        session_key="discord:guarded-chat",
    )

    assert manager.check(ctx).allowed is True
    assert manager.check_surface_policy(ctx).allowed is True


def test_scope_matches_discord_thread_and_session():
    cfg = RuntimeGuardConfig(
        enabled=True,
        scope=RuntimeGuardScope(
            platforms=("discord",),
            guild_ids=("guild-1",),
            chat_ids=("channel-1",),
            parent_chat_ids=("channel-1",),
            thread_ids=("thread-1",),
            session_keys=("discord:guild-1:channel-1:thread-1:user-1",),
        ),
    )
    manager = get_runtime_guard_manager(cfg)
    matching = GuardContext(
        surface="assistant_final",
        platform="discord",
        guild_id="guild-1",
        chat_id="channel-1",
        parent_chat_id="channel-1",
        thread_id="thread-1",
        session_key="discord:guild-1:channel-1:thread-1:user-1",
    )
    wrong_thread = GuardContext(
        surface="assistant_final",
        platform="discord",
        guild_id="guild-1",
        chat_id="channel-1",
        parent_chat_id="channel-1",
        thread_id="thread-2",
        session_key="discord:guild-1:channel-1:thread-1:user-1",
    )

    assert manager.is_scoped(matching) is True
    assert manager.is_scoped(wrong_thread) is False


def test_enabled_scoped_provider_exception_fails_closed():
    class ExplodingGuard:
        def check(self, context):
            raise RuntimeError("provider unavailable")

    register_runtime_guard_provider("explode_for_test", ExplodingGuard())
    manager = get_runtime_guard_manager(
        RuntimeGuardConfig(
            enabled=True,
            provider="explode_for_test",
            dry_run=False,
            fail_closed=True,
            scope=RuntimeGuardScope(platforms=("discord",)),
        )
    )
    ctx = GuardContext(surface="assistant_final", platform="discord", chat_id="chat-1")

    decision = manager.check(ctx)

    assert decision.allowed is False
    assert decision.status == "provider_error"
    assert decision.fail_closed is True
    assert "provider unavailable" in decision.reason


def test_mapping_decision_string_false_does_not_allow():
    class StringFalseGuard:
        def check(self, context):
            return {"allowed": "false", "reason": "string_false", "status": "denied"}

    register_runtime_guard_provider("string_false_for_test", StringFalseGuard())
    manager = get_runtime_guard_manager(
        RuntimeGuardConfig(
            enabled=True,
            provider="string_false_for_test",
            dry_run=False,
            scope=RuntimeGuardScope(platforms=("discord",)),
        )
    )

    decision = manager.check(GuardContext(surface="assistant_final", platform="discord"))

    assert decision.allowed is False
    assert decision.status == "denied"
    assert decision.reason == "string_false"


def test_invalid_provider_return_fails_closed_without_raising():
    class InvalidGuard:
        def check(self, context):
            return object()

    register_runtime_guard_provider("invalid_return_for_test", InvalidGuard())
    manager = get_runtime_guard_manager(
        RuntimeGuardConfig(
            enabled=True,
            provider="invalid_return_for_test",
            dry_run=False,
            fail_closed=True,
            scope=RuntimeGuardScope(platforms=("discord",)),
        )
    )

    decision = manager.check(GuardContext(surface="assistant_final", platform="discord"))

    assert decision.allowed is False
    assert decision.status == "provider_error"
    assert decision.fail_closed is True
    assert "unsupported decision" in decision.reason


def test_invalid_mapping_allowed_value_fails_closed_without_dry_run_allow():
    class InvalidMappingGuard:
        def check(self, context):
            return {"allowed": ["not", "a", "bool"]}

    register_runtime_guard_provider("invalid_mapping_for_test", InvalidMappingGuard())
    manager = get_runtime_guard_manager(
        RuntimeGuardConfig(
            enabled=True,
            provider="invalid_mapping_for_test",
            dry_run=True,
            fail_closed=True,
            scope=RuntimeGuardScope(platforms=("discord",)),
        )
    )

    decision = manager.check(GuardContext(surface="assistant_final", platform="discord"))

    assert decision.allowed is False
    assert decision.status == "provider_error"
    assert decision.fail_closed is True
    assert "provider_invalid_allowed" in decision.reason


def test_dry_run_denial_allows_but_records_block():
    class DenyingGuard:
        def check(self, context):
            return GuardDecision(
                allowed=False,
                reason="lease_conflict",
                status="denied",
                audit={"safe": "kept", "lease_token": "raw-secret"},
            )

    register_runtime_guard_provider("deny_for_dry_run_test", DenyingGuard())
    manager = get_runtime_guard_manager(
        RuntimeGuardConfig(
            enabled=True,
            provider="deny_for_dry_run_test",
            dry_run=True,
            fail_closed=True,
            scope=RuntimeGuardScope(platforms=("discord",)),
        )
    )
    ctx = GuardContext(surface="assistant_final", platform="discord", chat_id="chat-1")

    decision = manager.check(ctx)

    assert decision.allowed is True
    assert decision.dry_run is True
    assert decision.status == "dry_run_allowed"
    assert "lease_conflict" in decision.reason
    assert decision.audit["would_block"] is True
    assert decision.audit["original_status"] == "denied"
    assert decision.audit["original_reason"] == "lease_conflict"
    assert decision.audit["original_audit"] == {"safe": "kept"}
    assert "lease_token" not in decision.audit
