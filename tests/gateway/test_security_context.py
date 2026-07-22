import asyncio
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import gateway.security_context as security_module
from gateway.config import (
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
    SecurityContextTrustGrant,
)
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.security_context import (
    SecurityContext,
    SecurityContextError,
    apply_security_context_to_agent,
    bind_context_to_adapter,
    bind_security_context,
    current_security_context,
    enforce_agent_tool_dispatch,
    issue_adapter_security_capability,
    reset_security_context,
    require_action_for_context,
)
from gateway.session import SessionSource, build_session_key

_AUTH_TIME = datetime.now(timezone.utc)


def _context(
    *,
    principal="user-a",
    tenant="tenant-a",
    platform="telegram",
    tools=("web_search",),
    denied=(),
    revision="r1",
    authority="tier1_staff",
    bundle="tier1_staff",
    domains=("daily",),
    toolsets=("web",),
    actions=("teams.reply",),
    expose_private_context=False,
    expose_memory=False,
    expose_identity=True,
    authenticated_at=None,
    evidence_source="test",
):
    return SecurityContext(
        principal_id=principal,
        tenant_id=tenant,
        platform=platform,
        authority=authority,
        domains=frozenset(domains),
        capability_bundle=bundle,
        allowed_toolsets=frozenset(toolsets),
        allowed_tools=frozenset(tools),
        denied_tools=frozenset(denied),
        allowed_actions=frozenset(actions),
        expose_private_context=expose_private_context,
        expose_memory=expose_memory,
        expose_identity=expose_identity,
        policy_revision=revision,
        authenticated_at=authenticated_at or datetime.now(timezone.utc),
        evidence_source=evidence_source,
    )


class _Adapter:
    platform = Platform.TELEGRAM

    def __init__(self):
        self.revalidated = []

    async def revalidate_security_context(self, context, tool_name):
        self.revalidated.append((context.capability_hash, tool_name))


class _OtherAdapter(_Adapter):
    pass


class _LiveTrustedAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="test-token"), Platform.TELEGRAM
        )
        self.effects = []

    async def connect(self, *, is_reconnect: bool = False):
        return None

    async def disconnect(self):
        return None

    async def send(self, chat_id, text, **kwargs):
        self.effects.append(("send", chat_id, text))
        return SendResult(success=True, message_id="sent")

    async def get_chat_info(self, chat_id):
        return {}

    async def revalidate_security_context(self, context, tool_name):
        self.effects.append(("revalidate", tool_name))


def _grant(
    adapter_class: type = _Adapter,
    *,
    platform="telegram",
    plugin_name="secure-plugin",
):
    return SecurityContextTrustGrant(
        platform=platform,
        adapter_module=adapter_class.__module__,
        adapter_class=adapter_class.__qualname__,
        plugin_name=plugin_name,
    )


def _entry(adapter, *, plugin_name="secure-plugin", trusted=False):
    return PlatformEntry(
        name="telegram",
        label="Test",
        adapter_factory=lambda cfg: adapter,
        check_fn=lambda: True,
        source="plugin",
        plugin_name=plugin_name,
        trusted_security_context=trusted,
    )


def test_capability_hash_ignores_freshness_metadata():
    base = _context(authenticated_at=_AUTH_TIME)
    refreshed = _context(
        authenticated_at=_AUTH_TIME + timedelta(minutes=10),
        evidence_source="graph-live-refresh",
    )
    assert refreshed.capability_hash == base.capability_hash
    source = dict(
        platform=Platform.TELEGRAM, chat_id="room", chat_type="group", user_id="user-a"
    )
    assert build_session_key(SessionSource(**source, security_context=refreshed)) == (
        build_session_key(SessionSource(**source, security_context=base))
    )


@pytest.mark.asyncio
async def test_context_freshness_is_enforced_at_admission_and_dispatch():
    adapter = _Adapter()
    capability = issue_adapter_security_capability(
        adapter, asyncio.get_running_loop(), adapter.revalidate_security_context
    )
    stale = _context(
        authenticated_at=datetime.now(timezone.utc) - timedelta(minutes=6)
    )

    with pytest.raises(SecurityContextError, match="security_context_expired"):
        bind_context_to_adapter(stale, capability)

    future = _context(
        authenticated_at=datetime.now(timezone.utc) + timedelta(seconds=31)
    )
    with pytest.raises(SecurityContextError, match="security_context_from_future"):
        bind_context_to_adapter(future, capability)

    bound = bind_context_to_adapter(_context(), capability)
    stale_after_admission = replace(
        bound,
        authenticated_at=datetime.now(timezone.utc) - timedelta(minutes=6),
    )
    with pytest.raises(SecurityContextError, match="security_context_expired"):
        await capability.revalidate_async(stale_after_admission, "web_search")
    with pytest.raises(SecurityContextError, match="security_context_expired"):
        capability.revalidate_sync(stale_after_admission, "web_search")


@pytest.mark.asyncio
async def test_context_expiring_during_revalidation_denies_async_and_sync_dispatch(
    monkeypatch,
):
    adapter = _Adapter()
    capability = issue_adapter_security_capability(
        adapter, asyncio.get_running_loop(), adapter.revalidate_security_context
    )
    bound = bind_context_to_adapter(_context(), capability)

    async_freshness = Mock(
        side_effect=[None, SecurityContextError("security_context_expired")]
    )
    monkeypatch.setattr(
        security_module, "require_fresh_security_context", async_freshness
    )
    with pytest.raises(SecurityContextError, match="security_context_expired"):
        await capability.revalidate_async(bound, "teams.reply")

    sync_freshness = Mock(
        side_effect=[None, SecurityContextError("security_context_expired")]
    )
    monkeypatch.setattr(
        security_module, "require_fresh_security_context", sync_freshness
    )
    with pytest.raises(SecurityContextError, match="security_context_expired"):
        await asyncio.to_thread(capability.revalidate_sync, bound, "web_search")


def test_context_freshness_exact_age_and_clock_skew_boundaries():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    security_module.require_fresh_security_context(
        _context(authenticated_at=now - timedelta(seconds=300)), now=now
    )
    security_module.require_fresh_security_context(
        _context(authenticated_at=now + timedelta(seconds=30)), now=now
    )
    with pytest.raises(SecurityContextError, match="security_context_expired"):
        security_module.require_fresh_security_context(
            _context(authenticated_at=now - timedelta(seconds=300, microseconds=1)),
            now=now,
        )
    with pytest.raises(SecurityContextError, match="security_context_from_future"):
        security_module.require_fresh_security_context(
            _context(authenticated_at=now + timedelta(seconds=30, microseconds=1)),
            now=now,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"principal": "user-b"},
        {"tenant": "tenant-b"},
        {"platform": "discord"},
        {"authority": "tier2"},
        {"domains": ("daily", "restricted")},
        {"bundle": "tier2"},
        {"toolsets": ("web", "files")},
        {"tools": ("web_extract",)},
        {"denied": ("terminal",)},
        {"actions": ("teams.reply", "teams.delete")},
        {"expose_private_context": True},
        {"expose_memory": True},
        {"expose_identity": False},
        {"revision": "r2"},
    ],
)
def test_capability_hash_changes_for_every_effective_policy_field(change):
    base = _context()
    assert len(base.capability_hash) == 64
    assert _context(**change).capability_hash != base.capability_hash


def test_capability_hash_rejects_stale_supplied_digest():
    base = _context()
    with pytest.raises(SecurityContextError, match="capability_hash_mismatch"):
        replace(base, capability_hash="0" * 64, policy_revision="r2")


@pytest.mark.parametrize(
    "field", ["expose_private_context", "expose_memory", "expose_identity"]
)
def test_exposure_capabilities_require_booleans(field):
    with pytest.raises(SecurityContextError, match=f"invalid_{field}"):
        _context(**{field: "yes"})


def test_agent_context_options_are_explicit_and_legacy_context_is_unchanged():
    from gateway.security_context import agent_context_options

    assert agent_context_options(None) == {
        "skip_context_files": False,
        "load_soul_identity": False,
        "skip_memory": False,
    }
    assert agent_context_options(
        _context(
            authority="arbitrary-label",
            bundle="arbitrary-bundle",
            expose_private_context=True,
            expose_memory=True,
            expose_identity=False,
        )
    ) == {
        "skip_context_files": False,
        "load_soul_identity": False,
        "skip_memory": False,
    }
    assert agent_context_options(
        _context(
            authority="another-label",
            bundle="another-bundle",
            expose_private_context=False,
            expose_memory=False,
            expose_identity=True,
        )
    ) == {
        "skip_context_files": True,
        "load_soul_identity": True,
        "skip_memory": True,
    }


def test_session_key_isolated_with_collision_free_security_discriminator():
    base = dict(platform=Platform.TELEGRAM, chat_id="room", chat_type="group")
    contexts = (
        _context(principal="user-a"),
        _context(principal="user-b"),
        _context(principal="user-a", tools=("web_extract",)),
        _context(principal="a:tenant:b", tenant="c"),
        _context(principal="a", tenant="b:tenant:c"),
    )
    keys = {
        build_session_key(
            SessionSource(**base, user_id=context.principal_id, security_context=context)
        )
        for context in contexts
    }
    assert len(keys) == len(contexts)
    assert all(":security:" in key and len(key.rsplit(":", 1)[-1]) == 64 for key in keys)


def test_security_context_is_wire_invisible():
    context = _context()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="dm",
        user_id="user-a",
        security_context=context,
    )
    encoded = source.to_dict()
    assert "security_context" not in encoded
    assert SessionSource.from_dict(encoded).security_context is None


def test_apply_context_intersects_tool_schemas_and_denies_exact_other_name():
    agent = SimpleNamespace(
        tools=[
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "terminal"}},
        ],
        valid_tool_names={"web_search", "terminal"},
    )
    apply_security_context_to_agent(agent, _context())
    assert agent.valid_tool_names == {"web_search"}
    with pytest.raises(SecurityContextError, match="tool_denied"):
        enforce_agent_tool_dispatch(agent, "terminal")


@pytest.mark.asyncio
async def test_allowed_dispatch_revalidates_on_gateway_loop_from_executor():
    adapter = _Adapter()
    capability = issue_adapter_security_capability(
        adapter, asyncio.get_running_loop(), adapter.revalidate_security_context
    )
    context = bind_context_to_adapter(_context(), capability)
    agent = SimpleNamespace(
        tools=[{"type": "function", "function": {"name": "web_search"}}],
        valid_tool_names={"web_search"},
    )
    apply_security_context_to_agent(agent, context)
    await asyncio.to_thread(enforce_agent_tool_dispatch, agent, "web_search")
    assert adapter.revalidated == [(context.capability_hash, "web_search")]


@pytest.mark.asyncio
async def test_live_revalidation_times_out_and_cancels_hung_callback():
    cancelled = asyncio.Event()

    class HungAdapter(_Adapter):
        async def revalidate_security_context(self, context, tool_name):
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

    adapter = HungAdapter()
    capability = issue_adapter_security_capability(
        adapter,
        asyncio.get_running_loop(),
        adapter.revalidate_security_context,
        revalidation_timeout_seconds=0.02,
    )
    context = bind_context_to_adapter(_context(), capability)
    agent = SimpleNamespace(
        tools=[{"type": "function", "function": {"name": "web_search"}}],
        valid_tool_names={"web_search"},
    )
    apply_security_context_to_agent(agent, context)
    with pytest.raises(SecurityContextError, match="security_revalidation_timeout"):
        await asyncio.to_thread(enforce_agent_tool_dispatch, agent, "web_search")
    await asyncio.wait_for(cancelled.wait(), timeout=1)


def test_capability_change_or_provenance_change_cannot_mutate_cached_agent():
    agent = SimpleNamespace(
        tools=[{"type": "function", "function": {"name": "web_search"}}],
        valid_tool_names={"web_search"},
    )
    apply_security_context_to_agent(agent, _context())
    with pytest.raises(SecurityContextError, match="cached_agent_capability_mismatch"):
        apply_security_context_to_agent(agent, _context(tools=("web_extract",), revision="r2"))


def test_contextvar_binding_resets_without_leak():
    context = _context()
    token = bind_security_context(context)
    try:
        assert current_security_context(required=True) is context
    finally:
        reset_security_context(token)
    assert current_security_context() is None


def test_invalid_overlapping_and_denied_contexts_fail_closed():
    with pytest.raises(SecurityContextError):
        _context(principal="bad principal")
    with pytest.raises(SecurityContextError, match="tool_allow_deny_overlap"):
        _context(tools=("terminal",), denied=("terminal",))
    denied = _context(authority="denied", bundle="denied", tools=())
    assert denied.denied
    assert not denied.permits_tool("web_search")


def test_platform_trust_flag_defaults_false():
    entry = PlatformEntry(
        name="example",
        label="Example",
        adapter_factory=lambda cfg: None,
        check_fn=lambda: True,
    )
    assert entry.trusted_security_context is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "grant,registered_class,entry_plugin,source_platform",
    [
        (None, _Adapter, "secure-plugin", Platform.TELEGRAM),
        (replace(_grant(), adapter_module="wrong.module"), _Adapter, "secure-plugin", Platform.TELEGRAM),
        (replace(_grant(), adapter_class="WrongAdapter"), _Adapter, "secure-plugin", Platform.TELEGRAM),
        (_grant(plugin_name="other-plugin"), _Adapter, "secure-plugin", Platform.TELEGRAM),
        (_grant(platform="discord"), _Adapter, "secure-plugin", Platform.TELEGRAM),
        (_grant(), _OtherAdapter, "secure-plugin", Platform.TELEGRAM),
        (_grant(), _Adapter, "other-plugin", Platform.TELEGRAM),
        (_grant(), _Adapter, "secure-plugin", Platform.DISCORD),
    ],
)
async def test_gateway_trust_requires_exact_host_registry_class_plugin_and_platform(
    grant, registered_class, entry_plugin, source_platform
):
    from gateway.run import GatewayRunner

    prior = platform_registry.get("telegram")
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        security_context_trust_grants=(() if grant is None else (grant,))
    )
    runner._startup_restore_in_progress = True
    runner._queue_startup_restore_event = lambda event: pytest.fail("untrusted event queued")
    try:
        registered = registered_class()
        platform_registry.register(
            _entry(registered, plugin_name=entry_plugin, trusted=True)
        )
        adapter = platform_registry.create_adapter("telegram", object())
        assert adapter is registered
        handler = runner._make_adapter_message_handler(adapter)
        source = SessionSource(
            platform=source_platform, chat_id="dm", user_id="user-a"
        )
        assert await handler(
            MessageEvent(text="hello", source=source, security_context=_context())
        ) is None
    finally:
        platform_registry.unregister("telegram")
        if prior is not None:
            platform_registry.register(prior)


@pytest.mark.asyncio
async def test_gateway_binds_exact_adapter_and_registry_reregistration_cannot_spoof():
    from gateway.run import GatewayRunner

    prior = platform_registry.get("telegram")
    queued = []
    runner = object.__new__(GatewayRunner)
    runner._startup_restore_in_progress = True
    runner._queue_startup_restore_event = queued.append
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="dm", user_id="user-a")
    context = _context()
    adapter = _Adapter()
    runner.config = GatewayConfig(security_context_trust_grants=(_grant(),))
    try:
        # Direct/unprovenanced invocation is denied even if a plugin self-asserts trust.
        platform_registry.register(_entry(adapter, trusted=True))
        assert await runner._handle_message(
            MessageEvent(text="hello", source=source, security_context=context)
        ) is None
        assert queued == []

        assert platform_registry.create_adapter("telegram", object()) is adapter
        handler = runner._make_adapter_message_handler(adapter)
        await handler(MessageEvent(text="hello", source=source, security_context=context))
        assert len(queued) == 1
        assert queued[0].source.security_context._adapter_capability.belongs_to(adapter)
        queued.clear()

        # Re-registering/spoofing the same platform after the adapter was created
        # invalidates both newly-created and already-issued handler provenance.
        platform_registry.register(_entry(_OtherAdapter(), plugin_name="attacker"))
        spoofed_handler = runner._make_adapter_message_handler(adapter)
        assert await spoofed_handler(
            MessageEvent(text="hello", source=source, security_context=context)
        ) is None
        assert queued == []
        assert await handler(
            MessageEvent(text="hello", source=source, security_context=context)
        ) is None
        assert queued == []

        # Even an exact-identity replacement entry cannot claim an already
        # created instance (registry creation provenance is first-writer-wins).
        platform_registry.register(_entry(adapter))
        assert platform_registry.create_adapter("telegram", object()) is adapter
        replacement_handler = runner._make_adapter_message_handler(adapter)
        assert await replacement_handler(
            MessageEvent(text="hello", source=source, security_context=context)
        ) is None
        assert queued == []
        assert await handler(
            MessageEvent(text="hello", source=source, security_context=context)
        ) is None
        assert queued == []
    finally:
        platform_registry.unregister("telegram")
        if prior is not None:
            platform_registry.register(prior)


@pytest.mark.asyncio
async def test_denied_and_principal_mismatch_stop_before_session_or_command_paths():
    from gateway.run import GatewayRunner

    prior = platform_registry.get("telegram")
    runner = object.__new__(GatewayRunner)
    runner._startup_restore_in_progress = True
    runner._queue_startup_restore_event = lambda event: pytest.fail("must stop before queue")
    adapter = _Adapter()
    runner.config = GatewayConfig(security_context_trust_grants=(_grant(),))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="dm", user_id="user-a")
    try:
        platform_registry.register(_entry(adapter))
        platform_registry.create_adapter("telegram", object())
        handler = runner._make_adapter_message_handler(adapter)
        assert await handler(MessageEvent(
            text="/reset", source=source,
            security_context=_context(authority="denied", bundle="denied", tools=()),
        )) is None
        assert await handler(MessageEvent(
            text="hello", source=source,
            security_context=_context(principal="someone-else"),
        )) is None
    finally:
        platform_registry.unregister("telegram")
        if prior is not None:
            platform_registry.register(prior)


@pytest.mark.asyncio
async def test_direct_synthetic_entry_cannot_bypass_trusted_adapter_context(monkeypatch):
    from gateway.run import GatewayRunner

    prior = platform_registry.get("telegram")
    runner = object.__new__(GatewayRunner)
    adapter = _Adapter()
    runner.config = GatewayConfig(security_context_trust_grants=(_grant(),))
    runner.adapters = {Platform.TELEGRAM: adapter}
    entered = []

    async def impl(_event):
        entered.append(True)
        return "unsafe"

    monkeypatch.setattr(runner, "_handle_message_impl", impl)
    try:
        platform_registry.register(_entry(adapter))
        assert platform_registry.create_adapter("telegram", object()) is adapter
        event = MessageEvent(
            text="internal continuation",
            source=SessionSource(
                platform=Platform.TELEGRAM,
                user_id="user-1",
                chat_id="chat-1",
            ),
            internal=True,
        )
        assert await runner._handle_message(event) is None
        assert entered == []
    finally:
        platform_registry.unregister("telegram")
        if prior is not None:
            platform_registry.register(prior)


@pytest.mark.asyncio
async def test_trusted_adapter_handoff_denies_before_destination_mutation():
    """A context-free handoff must fail before creating or rebinding its lane."""
    from gateway.run import GatewayRunner

    prior = platform_registry.get("telegram")
    adapter = _LiveTrustedAdapter()
    adapter.create_handoff_thread = AsyncMock(return_value="new-thread")
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                token="test-token",
                home_channel=HomeChannel(
                    platform=Platform.TELEGRAM,
                    chat_id="home-chat",
                    name="Home",
                ),
            )
        },
        security_context_trust_grants=(
            _grant(adapter_class=_LiveTrustedAdapter),
        ),
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(),
        switch_session=AsyncMock(),
    )
    runner._evict_cached_agent = Mock()
    runner._release_running_agent_state = Mock()

    try:
        platform_registry.register(
            PlatformEntry(
                name="telegram",
                label="Trusted Base Test",
                adapter_factory=lambda cfg: adapter,
                check_fn=lambda: True,
                source="plugin",
                plugin_name="secure-plugin",
                trusted_security_context=True,
            )
        )
        assert platform_registry.create_adapter("telegram", object()) is adapter
        runner._install_adapter_security_preflight(adapter)

        handoff = {
            "id": "cli-session",
            "title": "CLI work",
            "handoff_platform": "telegram",
        }
        with pytest.raises(
            SecurityContextError, match="handoff_security_context_missing"
        ):
            await runner._process_handoff(handoff)

        runner.config = replace(
            runner.config, security_context_trust_grants=()
        )
        with pytest.raises(
            SecurityContextError, match="handoff_security_context_missing"
        ):
            await runner._process_handoff(handoff)

        platform_registry.register(
            PlatformEntry(
                name="telegram",
                label="Replacement",
                adapter_factory=lambda cfg: _OtherAdapter(),
                check_fn=lambda: True,
                source="plugin",
                plugin_name="replacement",
                trusted_security_context=False,
            )
        )
        with pytest.raises(
            SecurityContextError, match="handoff_security_context_missing"
        ):
            await runner._process_handoff(handoff)

        adapter.create_handoff_thread.assert_not_awaited()
        runner.async_session_store.get_or_create_session.assert_not_awaited()
        runner.async_session_store.switch_session.assert_not_awaited()
        runner._evict_cached_agent.assert_not_called()
        runner._release_running_agent_state.assert_not_called()
    finally:
        platform_registry.unregister("telegram")
        if prior is not None:
            platform_registry.register(prior)


@pytest.mark.asyncio
async def test_action_requires_exact_grant_and_live_revalidation_before_effect():
    adapter = _Adapter()
    capability = issue_adapter_security_capability(
        adapter, asyncio.get_running_loop(), adapter.revalidate_security_context
    )
    context = bind_context_to_adapter(
        _context(actions=("command.reset",)), capability
    )
    with pytest.raises(SecurityContextError, match="action_denied"):
        await require_action_for_context(context, "command.restart")
    assert adapter.revalidated == []

    await require_action_for_context(context, "command.reset")
    assert adapter.revalidated == [(context.capability_hash, "action:command.reset")]


@pytest.mark.asyncio
async def test_dispatch_revalidation_uses_live_registry_and_live_runner_grants():
    from gateway.run import GatewayRunner

    prior = platform_registry.get("telegram")
    runner = object.__new__(GatewayRunner)
    adapter = _Adapter()
    runner.config = GatewayConfig(security_context_trust_grants=(_grant(),))
    try:
        entry = _entry(adapter)
        platform_registry.register(entry)
        assert platform_registry.create_adapter("telegram", object()) is adapter
        capability = runner._issue_live_adapter_security_capability(adapter)
        context = bind_context_to_adapter(
            _context(actions=("command.reset",)), capability
        )

        await require_action_for_context(context, "command.reset")
        runner.config = GatewayConfig(security_context_trust_grants=())
        with pytest.raises(SecurityContextError):
            await require_action_for_context(context, "command.reset")

        runner.config = GatewayConfig(security_context_trust_grants=(_grant(),))
        platform_registry.register(_entry(_OtherAdapter(), plugin_name="attacker"))
        with pytest.raises(SecurityContextError):
            await require_action_for_context(context, "command.reset")
    finally:
        platform_registry.unregister("telegram")
        if prior is not None:
            platform_registry.register(prior)


def test_registry_provenance_reads_are_race_safe_during_replacement():
    import threading

    prior = platform_registry.get("telegram")
    adapter = _Adapter()
    entry = _entry(adapter)
    failures = []
    try:
        platform_registry.register(entry)
        assert platform_registry.create_adapter("telegram", object()) is adapter

        def replace_entries():
            for _ in range(200):
                platform_registry.register(_entry(_OtherAdapter(), plugin_name="attacker"))
                platform_registry.register(entry)

        thread = threading.Thread(target=replace_entries)
        thread.start()
        while thread.is_alive():
            try:
                matched = platform_registry.adapter_matches_current_entry(
                    adapter, "telegram"
                )
                assert matched is entry or matched is None
            except BaseException as exc:  # pragma: no cover - regression evidence
                failures.append(exc)
                break
        thread.join()
        assert failures == []
    finally:
        platform_registry.unregister("telegram")
        if prior is not None:
            platform_registry.register(prior)


@pytest.mark.parametrize("text", ["stop", "new", "reset"])
def test_security_preflight_preserves_ordinary_plaintext_conversation(text):
    from gateway.run import GatewayRunner

    event = MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id="user-1",
            chat_id="chat-1",
            chat_type="dm",
        ),
    )
    assert GatewayRunner._preflight_command_name(event) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/stop", "/new"])
async def test_revoked_live_adapter_preflight_denies_active_session_command_before_effects(
    command,
):
    """The real Base adapter must not begin its command handoff after revocation."""
    from gateway.run import GatewayRunner

    prior = platform_registry.get("telegram")
    adapter = _LiveTrustedAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        security_context_trust_grants=(
            _grant(adapter_class=_LiveTrustedAdapter),
        )
    )
    context = _context(
        actions=("command.stop", "command.reset"),
        tools=("web_search",),
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="dm",
        chat_type="dm",
        user_id=context.principal_id,
    )
    event = MessageEvent(
        text=command,
        message_type=MessageType.TEXT,
        source=source,
        security_context=context,
    )
    session_key = build_session_key(source)
    worker = asyncio.create_task(asyncio.Event().wait())
    guard = asyncio.Event()
    pending = MessageEvent(text="pending", source=source)
    debounce = object()
    handler_calls = []
    hook_calls = []

    async def handler(inbound):
        handler_calls.append(inbound.text)
        return "must not send"

    try:
        platform_registry.register(
            PlatformEntry(
                name="telegram",
                label="Trusted Base Test",
                adapter_factory=lambda cfg: adapter,
                check_fn=lambda: True,
                source="plugin",
                plugin_name="secure-plugin",
                trusted_security_context=True,
            )
        )
        assert platform_registry.create_adapter("telegram", object()) is adapter
        adapter.set_message_handler(handler)
        runner._install_adapter_security_preflight(adapter)
        adapter._topic_recovery_fn = lambda _event: hook_calls.append("topic")
        adapter._active_sessions[session_key] = guard
        adapter._session_tasks[session_key] = worker
        adapter._pending_messages[session_key] = pending
        adapter._text_debounce[session_key] = debounce

        # Revoke after the exact live adapter and callback were installed.
        runner.config = GatewayConfig(security_context_trust_grants=())
        adapter.effects.clear()

        await adapter.handle_message(event)
        await asyncio.sleep(0)

        assert adapter._active_sessions == {session_key: guard}
        assert adapter._session_tasks == {session_key: worker}
        assert adapter._pending_messages == {session_key: pending}
        assert adapter._text_debounce == {session_key: debounce}
        assert not worker.cancelled() and not worker.done()
        assert event.text == command
        assert handler_calls == []
        assert hook_calls == []
        assert adapter.effects == []
        assert adapter._background_tasks == set()
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        platform_registry.unregister("telegram")
        if prior is not None:
            platform_registry.register(prior)


@pytest.mark.asyncio
async def test_multiplex_profile_configuration_preserves_trusted_preflight(
    monkeypatch,
):
    """The upstream multiplex helper must not bypass adapter provenance binding."""
    from gateway.run import GatewayRunner

    prior = platform_registry.get("telegram")
    adapter = _LiveTrustedAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        security_context_trust_grants=(
            _grant(adapter_class=_LiveTrustedAdapter),
        )
    )
    runner.session_store = object()
    runner._handle_active_session_busy_message = object()
    runner._recover_telegram_topic_thread_id = object()
    runner._busy_text_mode = "queue"
    runner._make_adapter_auth_check = lambda platform, profile_name=None: object()
    seen = []

    async def handle(inbound):
        seen.append((inbound.source.profile, inbound.security_context))
        return "ok"

    runner._handle_message = handle
    monkeypatch.setattr(
        "gateway.run._profile_runtime_scope", lambda _home: nullcontext()
    )
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir", lambda _name: object()
    )
    context = _context(tools=("web_search",))
    event = MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="dm",
            chat_type="dm",
            user_id=context.principal_id,
        ),
        security_context=context,
    )
    try:
        platform_registry.register(
            PlatformEntry(
                name="telegram",
                label="Trusted Multiplex Test",
                adapter_factory=lambda cfg: adapter,
                check_fn=lambda: True,
                source="plugin",
                plugin_name="secure-plugin",
                trusted_security_context=True,
            )
        )
        assert platform_registry.create_adapter("telegram", object()) is adapter
        runner._configure_profile_adapter(adapter, "coder", Platform.TELEGRAM)
        assert callable(adapter._security_preflight)
        stamped = await adapter._security_preflight(event)
        assert stamped is not None
        assert await adapter._message_handler(stamped) == "ok"
        assert seen == [("coder", stamped.security_context)]
    finally:
        platform_registry.unregister("telegram")
        if prior is not None:
            platform_registry.register(prior)
