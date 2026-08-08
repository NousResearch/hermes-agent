"""Public contracts for external-plugin interactive gateway actions."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _plugin_context(name: str, manager: PluginManager) -> PluginContext:
    return PluginContext(
        PluginManifest(name=name, key=name, source="user"),
        manager,
    )


def test_plugin_interactive_action_registration_is_namespaced_and_conflict_safe():
    from gateway.interactive_actions import (
        InteractiveActionRegistrationError,
    )

    manager = PluginManager()
    context = _plugin_context("proposal-plugin", manager)

    def handler(_action_context):
        return None

    assert (
        context.register_interactive_action("apply", handler) == "proposal-plugin/apply"
    )

    with pytest.raises(InteractiveActionRegistrationError, match="already registered"):
        context.register_interactive_action("proposal-plugin/apply", handler)

    with pytest.raises(InteractiveActionRegistrationError, match="namespace"):
        context.register_interactive_action("another-plugin/apply", handler)


def test_nested_plugin_registry_key_can_own_an_action_namespace():
    manager = PluginManager()
    context = PluginContext(
        PluginManifest(
            name="proposal-service",
            key="business/proposal-service",
            source="user",
        ),
        manager,
    )

    assert (
        context.register_interactive_action("apply", lambda _ctx: None)
        == "business/proposal-service/apply"
    )

    from gateway.interactive_actions import InteractiveCardAction

    action = InteractiveCardAction(
        label="Apply",
        action="business/proposal-service/apply",
        external_action_id="proposal-v1",
        payload={"proposal_id": "prop_1"},
    )
    assert action.action == "business/proposal-service/apply"


def test_plugin_manager_exposes_action_registry_as_a_snapshot():
    manager = PluginManager()
    context = _plugin_context("proposal-plugin", manager)
    context.register_interactive_action("apply", lambda _ctx: None)

    snapshot = manager.get_interactive_actions()
    snapshot.clear()

    assert "proposal-plugin/apply" in manager.get_interactive_actions()


@pytest.mark.parametrize(
    ("action_name", "handler", "policy"),
    [
        ("", lambda _ctx: None, "initiator_only"),
        ("bad action", lambda _ctx: None, "initiator_only"),
        ("apply", None, "initiator_only"),
        ("apply", lambda _ctx: None, "owner_only"),
    ],
)
def test_plugin_interactive_action_registration_rejects_invalid_contracts(
    action_name,
    handler,
    policy,
):
    from gateway.interactive_actions import InteractiveActionRegistrationError

    context = _plugin_context("proposal-plugin", PluginManager())

    with pytest.raises(InteractiveActionRegistrationError):
        context.register_interactive_action(
            action_name,
            handler,
            authorization_policy=policy,
        )


def test_failed_plugin_registration_does_not_leave_live_action_handler(monkeypatch):
    manager = PluginManager()
    manifest = PluginManifest(
        name="proposal-plugin",
        key="proposal-plugin",
        source="user",
        path="/not-used",
    )

    def register(context):
        context.register_interactive_action("apply", lambda _ctx: None)
        raise RuntimeError("registration failed after action")

    monkeypatch.setattr(
        manager,
        "_load_directory_module",
        lambda _manifest: SimpleNamespace(register=register),
    )

    manager._load_plugin(manifest)

    assert manager._interactive_actions == {}


def _proposal_envelope():
    from gateway.interactive_actions import (
        InteractiveCardAction,
        InteractiveCardEnvelope,
        InteractiveCardFact,
        InteractiveCardSection,
    )

    return InteractiveCardEnvelope(
        version=1,
        title="Apply pricing proposal?",
        summary="This changes the customer contract after confirmation.",
        facts=(InteractiveCardFact("Customer", "Acme"),),
        sections=(
            InteractiveCardSection(
                title="Commercial terms",
                body="Annual price: CNY 120,000",
            ),
        ),
        fallback_text="Apply pricing proposal for Acme? Reply in the admin console.",
        expires_in_seconds=900,
        actions=(
            InteractiveCardAction(
                label="Apply proposal",
                action="proposal-plugin/apply",
                external_action_id="proposal-acme-v7",
                payload={"proposal_id": "prop_7", "revision": 7},
                style="primary",
            ),
        ),
    )


def test_post_tool_call_hook_gets_bound_send_capability_without_destination_ids():
    from gateway.interactive_actions import bind_interactive_card_sender

    manager = PluginManager()
    context = _plugin_context("proposal-plugin", manager)
    context.register_interactive_action("apply", lambda _ctx: None)
    envelope = _proposal_envelope()
    sent = []
    transcript = [{"role": "user", "content": "validate proposal"}]
    system_prompt = "byte-stable-system-prompt"
    toolsets = ("proposal", "terminal")

    def sender(*, plugin_id, envelope):
        sent.append((plugin_id, envelope))
        return "delivery-1"

    def observer(*, result, **_kwargs):
        assert result == '{"proposal_id":"prop_7"}'
        assert context.send_interactive_card(envelope) == "delivery-1"

    context.register_hook("post_tool_call", observer)
    with bind_interactive_card_sender(sender):
        manager.invoke_hook(
            "post_tool_call",
            tool_name="proposal_validate",
            args={},
            result='{"proposal_id":"prop_7"}',
        )

    assert sent == [("proposal-plugin", envelope)]
    assert transcript == [{"role": "user", "content": "validate proposal"}]
    assert system_prompt == "byte-stable-system-prompt"
    assert toolsets == ("proposal", "terminal")


def test_post_tool_call_worker_propagates_bound_gateway_sender(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    from gateway.interactive_actions import bind_interactive_card_sender
    from model_tools import _emit_post_tool_call_hook
    from tools.thread_context import propagate_context_to_thread
    import hermes_cli.lifecycle as lifecycle

    manager = PluginManager()
    context = _plugin_context("proposal-plugin", manager)
    context.register_interactive_action("apply", lambda _ctx: None)
    envelope = _proposal_envelope()
    sent = []

    def sender(*, plugin_id, envelope):
        sent.append((plugin_id, envelope))
        return "delivery-1"

    def observer(**_kwargs):
        assert context.send_interactive_card(envelope) == "delivery-1"

    context.register_hook("post_tool_call", observer)
    monkeypatch.setattr(lifecycle, "has_hook", manager.has_hook)
    monkeypatch.setattr(lifecycle, "invoke_hook", manager.invoke_hook)

    with bind_interactive_card_sender(sender):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                propagate_context_to_thread(_emit_post_tool_call_hook),
                function_name="proposal_validate",
                function_args={},
                result='{"status":"ready"}',
            )
            future.result(timeout=2)

    assert sent == [("proposal-plugin", envelope)]


def test_send_interactive_card_outside_gateway_turn_fails_closed():
    from gateway.interactive_actions import InteractiveCardUnavailableError

    context = _plugin_context("proposal-plugin", PluginManager())

    with pytest.raises(InteractiveCardUnavailableError, match="gateway turn"):
        context.send_interactive_card(_proposal_envelope())


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": 2},
        {"version": True},
        {"title": "x" * 121},
        {"fallback_text": ""},
        {"expires_in_seconds": 0},
        {"expires_in_seconds": True},
        {"actions": ()},
    ],
)
def test_interactive_card_envelope_is_versioned_and_bounded(overrides):
    from dataclasses import replace
    from gateway.interactive_actions import InteractiveCardValidationError

    with pytest.raises(InteractiveCardValidationError):
        replace(_proposal_envelope(), **overrides)


def test_interactive_action_payload_rejects_secret_like_fields():
    from gateway.interactive_actions import (
        InteractiveCardAction,
        InteractiveCardValidationError,
    )

    with pytest.raises(InteractiveCardValidationError, match="secret"):
        InteractiveCardAction(
            label="Apply",
            action="proposal-plugin/apply",
            external_action_id="proposal-acme-v7",
            payload={"proposal_id": "prop_7", "api_key": "must-not-store"},
        )


def test_interactive_action_payload_rejects_non_text_object_keys():
    from gateway.interactive_actions import (
        InteractiveCardAction,
        InteractiveCardValidationError,
    )

    with pytest.raises(InteractiveCardValidationError, match="keys must be text"):
        InteractiveCardAction(
            label="Apply",
            action="proposal-plugin/apply",
            external_action_id="proposal-acme-v7",
            payload={1: "ambiguous with string key"},
        )


def _registered_manager(
    tmp_path,
    handler,
    *,
    policy="initiator_only",
    clock=lambda: 1_000.0,
    instance_id="ia_instance_1",
):
    from gateway.interactive_actions import (
        InteractiveActionManager,
        SQLiteInteractiveActionStorage,
    )

    plugins = PluginManager()
    context = _plugin_context("proposal-plugin", plugins)
    context.register_interactive_action(
        "apply",
        handler,
        authorization_policy=policy,
    )
    manager = InteractiveActionManager(
        storage=SQLiteInteractiveActionStorage(tmp_path / "state.db"),
        registrations=plugins._interactive_actions,
        clock=clock,
        id_source=lambda: instance_id,
    )
    return plugins, manager


def _origin(**overrides):
    from dataclasses import replace
    from gateway.interactive_actions import InteractiveCardOrigin

    value = InteractiveCardOrigin(
        platform="feishu",
        profile_id="work",
        chat_id="oc_chat",
        thread_id="om_root",
        initiator_id="ou_initiator",
        initiator_name="Alice",
        message_id="om_trigger",
    )
    return replace(value, **overrides)


def _callback(action_instance_id="ia_instance_1", **overrides):
    from dataclasses import replace
    from gateway.interactive_actions import InteractiveActionCallback

    value = InteractiveActionCallback(
        action_instance_id=action_instance_id,
        platform="feishu",
        profile_id="work",
        operator_id="ou_initiator",
        operator_name="Alice",
        chat_id="oc_chat",
        thread_id="om_root",
        card_id="om_card",
    )
    return replace(value, **overrides)


async def _issue_and_dispatch(manager, *, callback=None, authorize=lambda: True):
    prepared = manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    manager.activate_card(prepared, card_id="om_card")
    result = await manager.dispatch(
        callback or _callback(),
        gateway_authorize=authorize,
    )
    return prepared, result


@pytest.mark.asyncio
async def test_authorized_initiator_gets_exact_immutable_stored_context(tmp_path):
    from gateway.interactive_actions import InteractiveActionResult

    contexts = []

    async def handler(action_context):
        contexts.append(action_context)
        return InteractiveActionResult.succeeded()

    _plugins, manager = _registered_manager(tmp_path, handler)
    prepared, result = await _issue_and_dispatch(manager)

    assert prepared.action_instance_ids == ("ia_instance_1",)
    assert result.status == "succeeded"
    assert len(contexts) == 1
    action_context = contexts[0]
    assert action_context.platform == "feishu"
    assert action_context.profile_id == "work"
    assert action_context.operator_id == "ou_initiator"
    assert action_context.operator_name == "Alice"
    assert action_context.chat_id == "oc_chat"
    assert action_context.thread_id == "om_root"
    assert action_context.message_id == "om_trigger"
    assert action_context.card_id == "om_card"
    assert action_context.action_instance_id == "ia_instance_1"
    assert action_context.external_action_id == "proposal-acme-v7"
    assert dict(action_context.payload) == {"proposal_id": "prop_7", "revision": 7}
    with pytest.raises(TypeError):
        action_context.payload["revision"] = 8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "callback",
    [
        _callback(operator_id=""),
        _callback(operator_id="ou_attacker"),
        _callback(chat_id="oc_other"),
        _callback(profile_id="personal"),
        _callback(card_id="om_tampered"),
        _callback(action_instance_id="ia_unknown"),
    ],
)
async def test_invalid_or_unauthorized_click_never_executes(tmp_path, callback):
    calls = []
    _plugins, manager = _registered_manager(tmp_path, lambda ctx: calls.append(ctx))

    _prepared, result = await _issue_and_dispatch(manager, callback=callback)

    assert result.status in {"denied", "unknown"}
    assert calls == []


@pytest.mark.asyncio
async def test_gateway_authorization_runs_before_policy_and_handler(tmp_path):
    events = []

    def handler(_ctx):
        events.append("handler")

    _plugins, manager = _registered_manager(tmp_path, handler)
    prepared = manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    manager.activate_card(prepared, card_id="om_card")

    result = await manager.dispatch(
        _callback(),
        gateway_authorize=lambda: events.append("gateway-auth") or False,
    )

    assert result.status == "denied"
    assert events == ["gateway-auth"]


@pytest.mark.asyncio
async def test_authorized_user_policy_allows_non_initiator_after_gateway_auth(tmp_path):
    calls = []
    _plugins, manager = _registered_manager(
        tmp_path,
        lambda ctx: calls.append(ctx),
        policy="authorized_user",
    )

    _prepared, result = await _issue_and_dispatch(
        manager,
        callback=_callback(operator_id="ou_admin", operator_name="Bob"),
    )

    assert result.status == "succeeded"
    assert [ctx.operator_id for ctx in calls] == ["ou_admin"]


@pytest.mark.asyncio
async def test_action_instance_keeps_issuance_policy_across_registration_change(
    tmp_path,
):
    from dataclasses import replace

    calls = []
    plugins, manager = _registered_manager(
        tmp_path,
        lambda ctx: calls.append(ctx),
        policy="initiator_only",
    )
    prepared = manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    manager.activate_card(prepared, card_id="om_card")
    registration = plugins._interactive_actions["proposal-plugin/apply"]
    plugins._interactive_actions["proposal-plugin/apply"] = replace(
        registration,
        authorization_policy="authorized_user",
    )

    result = await manager.dispatch(
        _callback(operator_id="ou_admin", operator_name="Bob"),
        gateway_authorize=lambda: True,
    )

    assert result.status == "denied"
    assert calls == []


@pytest.mark.asyncio
async def test_current_policy_tightening_also_applies_to_issued_action(tmp_path):
    from dataclasses import replace

    calls = []
    plugins, manager = _registered_manager(
        tmp_path,
        lambda ctx: calls.append(ctx),
        policy="authorized_user",
    )
    prepared = manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    manager.activate_card(prepared, card_id="om_card")
    registration = plugins._interactive_actions["proposal-plugin/apply"]
    plugins._interactive_actions["proposal-plugin/apply"] = replace(
        registration,
        authorization_policy="initiator_only",
    )

    result = await manager.dispatch(
        _callback(operator_id="ou_admin", operator_name="Bob"),
        gateway_authorize=lambda: True,
    )

    assert result.status == "denied"
    assert calls == []


@pytest.mark.asyncio
async def test_expired_action_does_not_execute(tmp_path):
    now = [1_000.0]
    calls = []
    _plugins, manager = _registered_manager(
        tmp_path,
        lambda ctx: calls.append(ctx),
        clock=lambda: now[0],
    )
    prepared = manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    manager.activate_card(prepared, card_id="om_card")
    now[0] += 901

    result = await manager.dispatch(_callback(), gateway_authorize=lambda: True)

    assert result.status == "expired"
    assert calls == []


def test_ledger_coexists_with_existing_sessiondb_schema(tmp_path):
    import sqlite3

    from gateway.interactive_actions import SQLiteInteractiveActionStorage
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    session_db = SessionDB(db_path)
    session_db.close()

    storage = SQLiteInteractiveActionStorage(db_path)
    assert storage.get("missing") is None

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"sessions", "messages", "interactive_actions"}.issubset(tables)

    reopened = SessionDB(db_path)
    reopened.close()


def test_ledger_migrates_pre_policy_schema_without_losing_rows(tmp_path):
    import sqlite3

    from gateway.interactive_actions import SQLiteInteractiveActionStorage

    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE interactive_actions (
                action_instance_id TEXT PRIMARY KEY,
                plugin_action TEXT NOT NULL,
                external_action_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                thread_id TEXT,
                initiator_id TEXT NOT NULL,
                initiator_name TEXT NOT NULL,
                message_id TEXT NOT NULL,
                card_id TEXT,
                payload_json TEXT NOT NULL,
                state TEXT NOT NULL,
                outcome TEXT,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        conn.execute(
            """INSERT INTO interactive_actions VALUES (
                'ia_old', 'proposal-plugin/apply', 'proposal-v1',
                'work', 'feishu', 'oc_chat', NULL, 'ou_initiator', 'Alice',
                'om_trigger', 'om_card', '{}', 'active', NULL, 2000, 1000, 1000
            )"""
        )

    record = SQLiteInteractiveActionStorage(db_path).get("ia_old")

    assert record is not None
    assert record.authorization_policy == "initiator_only"
    assert record.card_id == "om_card"


@pytest.mark.asyncio
async def test_ledger_migrates_existing_schema_and_replays_safe_terminal_result(
    tmp_path,
):
    import sqlite3

    from gateway.interactive_actions import (
        InteractiveActionManager,
        InteractiveActionResult,
        SQLiteInteractiveActionStorage,
    )

    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE interactive_actions (
                action_instance_id TEXT PRIMARY KEY,
                plugin_action TEXT NOT NULL,
                external_action_id TEXT NOT NULL,
                authorization_policy TEXT NOT NULL DEFAULT 'initiator_only',
                profile_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                thread_id TEXT,
                initiator_id TEXT NOT NULL,
                initiator_name TEXT NOT NULL,
                message_id TEXT NOT NULL,
                card_id TEXT,
                payload_json TEXT NOT NULL,
                state TEXT NOT NULL,
                outcome TEXT,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        conn.execute(
            """INSERT INTO interactive_actions VALUES (
                'ia_instance_1', 'proposal-plugin/apply', 'proposal-v1',
                'initiator_only', 'work', 'feishu', 'oc_chat', 'om_root',
                'ou_initiator', 'Alice', 'om_trigger', 'om_card', '{}',
                'finished', 'conflict', 2000, 1000, 1000
            )"""
        )

    plugins = PluginManager()
    context = _plugin_context("proposal-plugin", plugins)
    context.register_interactive_action("apply", lambda _ctx: None)
    manager = InteractiveActionManager(
        storage=SQLiteInteractiveActionStorage(db_path),
        registrations=plugins._interactive_actions,
        clock=lambda: 1_001.0,
    )

    replay = await manager.dispatch(_callback(), gateway_authorize=lambda: True)

    assert replay == InteractiveActionResult.conflict()
    with sqlite3.connect(db_path) as conn:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(interactive_actions)")
        }
    assert "user_message" in columns


def test_concurrent_ledger_schema_initialization_is_idempotent(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import sqlite3

    from gateway.interactive_actions import SQLiteInteractiveActionStorage

    db_path = tmp_path / "state.db"

    def initialize(_index):
        return SQLiteInteractiveActionStorage(db_path).get("missing")

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(initialize, range(16))) == [None] * 16

    with sqlite3.connect(db_path) as conn:
        columns = [
            row[1] for row in conn.execute("PRAGMA table_info(interactive_actions)")
        ]
    assert columns.count("authorization_policy") == 1
    assert columns.count("user_message") == 1


def test_default_ledger_resolves_profile_home_on_each_operation(tmp_path):
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    from gateway.interactive_actions import (
        InteractiveActionManager,
        SQLiteInteractiveActionStorage,
    )

    plugins = PluginManager()
    context = _plugin_context("proposal-plugin", plugins)
    context.register_interactive_action("apply", lambda _ctx: None)
    storage = SQLiteInteractiveActionStorage()
    ids = iter(("ia_work", "ia_personal"))
    manager = InteractiveActionManager(
        storage=storage,
        registrations=plugins._interactive_actions,
        clock=lambda: 1_000.0,
        id_source=ids.__next__,
    )
    work_home = tmp_path / "work"
    personal_home = tmp_path / "personal"

    token = set_hermes_home_override(work_home)
    try:
        manager.prepare_card(
            plugin_id="proposal-plugin",
            envelope=_proposal_envelope(),
            origin=_origin(profile_id="work"),
        )
        assert storage.get("ia_work") is not None
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(personal_home)
    try:
        assert storage.get("ia_work") is None
        manager.prepare_card(
            plugin_id="proposal-plugin",
            envelope=_proposal_envelope(),
            origin=_origin(profile_id="personal"),
        )
        assert storage.get("ia_personal") is not None
    finally:
        reset_hermes_home_override(token)

    assert (work_home / "state.db").exists()
    assert (personal_home / "state.db").exists()


def test_stale_crash_rows_do_not_exhaust_bounded_ledger_forever(tmp_path):
    from gateway.interactive_actions import InteractiveActionManager

    now = [1_000.0]
    plugins, manager = _registered_manager(
        tmp_path,
        lambda _ctx: None,
        clock=lambda: now[0],
        instance_id="ia_first",
    )
    manager.storage._MAX_ROWS = 1
    manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )

    now[0] += (
        _proposal_envelope().expires_in_seconds + manager.storage._RETENTION_SECONDS + 1
    )
    replacement = InteractiveActionManager(
        storage=manager.storage,
        registrations=plugins._interactive_actions,
        clock=lambda: now[0],
        id_source=lambda: "ia_second",
    )
    replacement.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )

    assert manager.storage.get("ia_first") is None
    assert manager.storage.get("ia_second") is not None


def test_profile_ledger_hard_capacity_fails_closed_without_unbounded_growth(tmp_path):
    from gateway.interactive_actions import InteractiveActionCapacityError

    _plugins, manager = _registered_manager(tmp_path, lambda _ctx: None)
    manager.storage._MAX_ROWS = 1
    ids = iter(("ia_first", "ia_second"))
    manager._id_source = ids.__next__
    manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )

    with pytest.raises(InteractiveActionCapacityError, match="capacity"):
        manager.prepare_card(
            plugin_id="proposal-plugin",
            envelope=_proposal_envelope(),
            origin=_origin(),
        )

    assert manager.storage.get("ia_first") is not None
    assert manager.storage.get("ia_second") is None


@pytest.mark.asyncio
async def test_atomic_double_click_executes_handler_once(tmp_path):
    from gateway.interactive_actions import InteractiveActionResult

    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def handler(ctx):
        calls.append(ctx)
        entered.set()
        await release.wait()
        return InteractiveActionResult.succeeded()

    _plugins, manager = _registered_manager(tmp_path, handler)
    prepared = manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    manager.activate_card(prepared, card_id="om_card")

    first = asyncio.create_task(
        manager.dispatch(_callback(), gateway_authorize=lambda: True)
    )
    await entered.wait()
    duplicate = await manager.dispatch(_callback(), gateway_authorize=lambda: True)
    release.set()
    completed = await first

    assert completed.status == "succeeded"
    assert duplicate.status == "processing"
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["processing", "finished"])
@pytest.mark.parametrize(
    "copied_callback",
    [
        _callback(card_id="om_copied_card"),
        _callback(chat_id="oc_other"),
        _callback(profile_id="personal"),
        _callback(platform="slack"),
        _callback(thread_id="om_other_thread"),
    ],
    ids=("wrong-card", "wrong-chat", "wrong-profile", "wrong-platform", "wrong-thread"),
)
async def test_processing_and_finished_replay_require_original_card_binding(
    tmp_path,
    state,
    copied_callback,
):
    from gateway.interactive_actions import (
        ClaimedInteractiveAction,
        InteractiveActionResult,
    )

    _plugins, manager = _registered_manager(
        tmp_path,
        lambda _ctx: InteractiveActionResult.succeeded(
            "Proposal applied from the original card."
        ),
    )
    prepared = manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    manager.activate_card(prepared, card_id="om_card")

    if state == "processing":
        claim = manager.claim_action(_callback(), gateway_authorize=lambda: True)
        assert isinstance(claim, ClaimedInteractiveAction)
        expected_original = InteractiveActionResult.processing()
    else:
        expected_original = await manager.dispatch(
            _callback(),
            gateway_authorize=lambda: True,
        )

    copied_replay = await manager.dispatch(
        copied_callback,
        gateway_authorize=lambda: True,
    )
    original_replay = await manager.dispatch(
        _callback(),
        gateway_authorize=lambda: True,
    )

    assert copied_replay == InteractiveActionResult.denied()
    assert original_replay == expected_original


@pytest.mark.asyncio
async def test_durable_replay_after_manager_restart_returns_exact_sanitized_result(
    tmp_path,
):
    from gateway.interactive_actions import (
        InteractiveActionManager,
        InteractiveActionResult,
        SQLiteInteractiveActionStorage,
    )

    calls = []

    def handler(ctx):
        calls.append(ctx)
        return InteractiveActionResult.succeeded(
            "  Proposal applied.\nReceipt 42.  "
        )

    plugins, first_manager = _registered_manager(
        tmp_path,
        handler,
    )
    _prepared, completed = await _issue_and_dispatch(first_manager)
    assert completed == InteractiveActionResult.succeeded(
        "Proposal applied. Receipt 42."
    )
    assert (
        first_manager.storage.get("ia_instance_1").user_message
        == "Proposal applied. Receipt 42."
    )

    reconstructed = InteractiveActionManager(
        storage=SQLiteInteractiveActionStorage(tmp_path / "state.db"),
        registrations=plugins._interactive_actions,
        clock=lambda: 1_001.0,
        id_source=lambda: "unused",
    )
    replay = await reconstructed.dispatch(
        _callback(),
        gateway_authorize=lambda: True,
    )

    assert replay == completed
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retryable_failure_reactivates_same_bound_action_for_success(tmp_path):
    from gateway.interactive_actions import InteractiveActionResult

    calls = []

    def handler(ctx):
        calls.append(ctx)
        if len(calls) == 1:
            return InteractiveActionResult.retryable_failure(
                "Proposal service is temporarily unavailable."
            )
        return InteractiveActionResult.succeeded("Proposal applied on retry.")

    _plugins, manager = _registered_manager(tmp_path, handler)
    prepared = manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    manager.activate_card(prepared, card_id="om_card")

    transient = await manager.dispatch(_callback(), gateway_authorize=lambda: True)
    after_transient = manager.storage.get("ia_instance_1")
    retried = await manager.dispatch(_callback(), gateway_authorize=lambda: True)

    assert transient == InteractiveActionResult.retryable_failure(
        "Proposal service is temporarily unavailable."
    )
    assert after_transient.state == "active"
    assert after_transient.outcome == "retryable_failure"
    assert retried == InteractiveActionResult.succeeded(
        "Proposal applied on retry."
    )
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_cancellation_during_handler_persists_terminal_unknown_outcome(tmp_path):
    from gateway.interactive_actions import InteractiveActionResult

    entered = asyncio.Event()

    async def handler(_ctx):
        entered.set()
        await asyncio.Event().wait()

    _plugins, manager = _registered_manager(tmp_path, handler)
    prepared = manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    manager.activate_card(prepared, card_id="om_card")
    task = asyncio.create_task(
        manager.dispatch(_callback(), gateway_authorize=lambda: True)
    )
    await entered.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    replay = await manager.dispatch(_callback(), gateway_authorize=lambda: True)
    assert replay == InteractiveActionResult.unknown_outcome()
    assert manager.storage.get("ia_instance_1").state == "finished"


@pytest.mark.asyncio
async def test_storage_startup_reconciles_prior_processing_to_unknown_outcome(
    tmp_path,
):
    from gateway.interactive_actions import (
        ClaimedInteractiveAction,
        InteractiveActionManager,
        InteractiveActionResult,
        SQLiteInteractiveActionStorage,
    )

    calls = []
    plugins, first_manager = _registered_manager(
        tmp_path,
        lambda ctx: calls.append(ctx),
    )
    prepared = first_manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    first_manager.activate_card(prepared, card_id="om_card")
    claim = first_manager.claim_action(
        _callback(),
        gateway_authorize=lambda: True,
    )
    assert isinstance(claim, ClaimedInteractiveAction)
    assert first_manager.storage.get("ia_instance_1").state == "processing"

    restarted = InteractiveActionManager(
        storage=SQLiteInteractiveActionStorage(tmp_path / "state.db"),
        registrations=plugins._interactive_actions,
        clock=lambda: 1_001.0,
    )
    replay = await restarted.dispatch(_callback(), gateway_authorize=lambda: True)

    assert replay == InteractiveActionResult.unknown_outcome()
    assert restarted.storage.get("ia_instance_1").state == "finished"
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler_factory", "expected_status", "forbidden_text"),
    [
        (
            lambda result: lambda _ctx: result.downstream_replay(),
            "downstream_replay",
            "proposal_id",
        ),
        (
            lambda result: lambda _ctx: result.conflict(),
            "conflict",
            "proposal_id",
        ),
        (
            lambda result: lambda _ctx: result.retryable_failure(),
            "retryable_failure",
            "proposal_id",
        ),
        (
            lambda _result: (
                lambda _ctx: (_ for _ in ()).throw(
                    RuntimeError("secret traceback proposal_id=prop_7")
                )
            ),
            "retryable_failure",
            "prop_7",
        ),
    ],
)
async def test_handler_outcomes_are_truthful_and_sanitized(
    tmp_path,
    handler_factory,
    expected_status,
    forbidden_text,
):
    from gateway.interactive_actions import InteractiveActionResult

    handler = handler_factory(InteractiveActionResult)
    _plugins, manager = _registered_manager(tmp_path, handler)

    _prepared, result = await _issue_and_dispatch(manager)

    assert result.status == expected_status
    assert forbidden_text not in result.user_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_name", "expected_status", "safe_message"),
    [
        ("InteractiveActionConflictError", "conflict", "Proposal revision changed."),
        (
            "InteractiveActionRetryableError",
            "retryable_failure",
            "The proposal service is temporarily unavailable.",
        ),
    ],
)
async def test_typed_handler_errors_expose_only_explicit_bounded_public_text(
    tmp_path,
    error_name,
    expected_status,
    safe_message,
):
    import gateway.interactive_actions as actions

    error_type = getattr(actions, error_name)

    def handler(_ctx):
        raise error_type(f"  {safe_message}\n")

    _plugins, manager = _registered_manager(tmp_path, handler)
    _prepared, result = await _issue_and_dispatch(manager)

    assert result.status == expected_status
    assert result.user_message == safe_message


@pytest.mark.asyncio
async def test_unsupported_adapter_sends_exact_fallback_without_ledger_state(tmp_path):
    sent = []

    async def send(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(success=True, message_id="fallback-message")

    adapter = SimpleNamespace(
        supports_interactive_cards=False,
        send=send,
    )
    _plugins, manager = _registered_manager(
        tmp_path,
        lambda _ctx: None,
        instance_id="must-not-be-created",
    )

    delivery = await manager.deliver_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
        adapter=adapter,
        metadata={"thread_id": "om_root"},
    )

    assert delivery.mode == "fallback"
    assert delivery.action_instance_ids == ()
    assert sent == [
        {
            "chat_id": "oc_chat",
            "content": _proposal_envelope().fallback_text,
            "reply_to": "om_trigger",
            "metadata": {"thread_id": "om_root"},
        }
    ]
    assert manager.storage.get("must-not-be-created") is None


@pytest.mark.asyncio
async def test_successful_fallback_does_not_require_adapter_message_id(tmp_path):
    async def send(**_kwargs):
        return SimpleNamespace(success=True, message_id=None)

    adapter = SimpleNamespace(
        supports_interactive_cards=False,
        send=send,
    )
    _plugins, manager = _registered_manager(
        tmp_path,
        lambda _ctx: None,
        instance_id="must-not-be-created",
    )

    delivery = await manager.deliver_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
        adapter=adapter,
    )

    assert delivery.mode == "fallback"
    assert delivery.message_id == ""
    assert manager.storage.get("must-not-be-created") is None


@pytest.mark.asyncio
async def test_fallback_delivery_does_not_require_native_identity_or_origin_message(
    tmp_path,
):
    sent = []

    async def send(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(success=True, message_id=None)

    adapter = SimpleNamespace(supports_interactive_cards=False, send=send)
    _plugins, manager = _registered_manager(
        tmp_path,
        lambda _ctx: None,
        instance_id="must-not-be-created",
    )

    delivery = await manager.deliver_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(initiator_id="", initiator_name="", message_id=""),
        adapter=adapter,
    )

    assert delivery.mode == "fallback"
    assert sent[0]["content"] == _proposal_envelope().fallback_text
    assert sent[0]["reply_to"] is None
    assert manager.storage.get("must-not-be-created") is None


@pytest.mark.asyncio
async def test_native_delivery_requires_stable_initiator_and_origin_message(tmp_path):
    from gateway.interactive_actions import InteractiveCardDeliveryError

    native_send = AsyncMock()
    adapter = SimpleNamespace(
        supports_interactive_cards=True,
        send_interactive_card=native_send,
    )
    _plugins, manager = _registered_manager(tmp_path, lambda _ctx: None)

    with pytest.raises(InteractiveCardDeliveryError, match="prepared"):
        await manager.deliver_card(
            plugin_id="proposal-plugin",
            envelope=_proposal_envelope(),
            origin=_origin(initiator_id="", message_id=""),
            adapter=adapter,
        )

    native_send.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_delivery_activates_opaque_action_only_after_send_success(
    tmp_path,
):
    send_native = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="om_card")
    )
    adapter = SimpleNamespace(
        supports_interactive_cards=True,
        send_interactive_card=send_native,
    )
    _plugins, manager = _registered_manager(tmp_path, lambda _ctx: None)

    delivery = await manager.deliver_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
        adapter=adapter,
        metadata={"thread_id": "om_root"},
    )

    assert delivery.mode == "native"
    assert delivery.message_id == "om_card"
    assert delivery.action_instance_ids == ("ia_instance_1",)
    send_native.assert_awaited_once_with(
        chat_id="oc_chat",
        envelope=_proposal_envelope(),
        action_instance_ids=("ia_instance_1",),
        reply_to="om_trigger",
        metadata={"thread_id": "om_root"},
    )
    assert manager.storage.get("ia_instance_1").state == "active"


@pytest.mark.asyncio
async def test_native_delivery_failure_is_explicit_and_never_clickable(tmp_path):
    from gateway.interactive_actions import InteractiveCardDeliveryError

    adapter = SimpleNamespace(
        supports_interactive_cards=True,
        send_interactive_card=AsyncMock(
            return_value=SimpleNamespace(
                success=False,
                message_id=None,
                error="raw provider secret details",
            )
        ),
    )
    _plugins, manager = _registered_manager(tmp_path, lambda _ctx: None)

    with pytest.raises(
        InteractiveCardDeliveryError, match="could not be delivered"
    ) as exc_info:
        await manager.deliver_card(
            plugin_id="proposal-plugin",
            envelope=_proposal_envelope(),
            origin=_origin(),
            adapter=adapter,
        )

    assert "raw provider" not in str(exc_info.value)
    assert manager.storage.get("ia_instance_1").state == "delivery_failed"


@pytest.mark.asyncio
async def test_delivery_and_failure_persistence_errors_remain_generic(
    tmp_path,
    monkeypatch,
):
    from gateway.interactive_actions import InteractiveCardDeliveryError

    adapter = SimpleNamespace(
        supports_interactive_cards=True,
        send_interactive_card=AsyncMock(
            side_effect=RuntimeError("provider api_key=hidden")
        ),
    )
    _plugins, manager = _registered_manager(tmp_path, lambda _ctx: None)
    monkeypatch.setattr(
        manager,
        "fail_card_delivery",
        lambda _prepared: (_ for _ in ()).throw(
            RuntimeError("state.db=/secret/profile")
        ),
    )

    with pytest.raises(
        InteractiveCardDeliveryError,
        match="could not be delivered",
    ) as exc_info:
        await manager.deliver_card(
            plugin_id="proposal-plugin",
            envelope=_proposal_envelope(),
            origin=_origin(),
            adapter=adapter,
        )

    assert "api_key" not in str(exc_info.value)
    assert "/secret/profile" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_native_post_send_binding_failure_disables_card_and_hides_error(
    tmp_path,
    monkeypatch,
):
    from gateway.interactive_actions import InteractiveCardDeliveryError

    adapter = SimpleNamespace(
        supports_interactive_cards=True,
        send_interactive_card=AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="om_card")
        ),
        update_interactive_card=AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="om_card")
        ),
    )
    _plugins, manager = _registered_manager(tmp_path, lambda _ctx: None)
    monkeypatch.setattr(
        manager,
        "activate_card",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("state.db path and provider secret")
        ),
    )

    with pytest.raises(
        InteractiveCardDeliveryError,
        match="could not be activated",
    ) as exc_info:
        await manager.deliver_card(
            plugin_id="proposal-plugin",
            envelope=_proposal_envelope(),
            origin=_origin(),
            adapter=adapter,
        )

    assert "provider secret" not in str(exc_info.value)
    assert manager.storage.get("ia_instance_1").state == "delivery_failed"
    disabled = adapter.update_interactive_card.await_args.kwargs["result"]
    assert disabled.status == "retryable_failure"
    assert "state.db" not in disabled.user_message


def test_ledger_sqlite_contention_fails_before_platform_callback_timeout(tmp_path):
    import sqlite3
    import time
    from gateway.interactive_actions import SQLiteInteractiveActionStorage

    db_path = tmp_path / "state.db"
    storage = SQLiteInteractiveActionStorage(db_path)
    assert storage.get("initialize-schema") is None
    locker = sqlite3.connect(db_path, timeout=0)
    locker.execute("BEGIN IMMEDIATE")
    started = time.monotonic()
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            storage.claim(_callback(), now=1_000.0)
    finally:
        elapsed = time.monotonic() - started
        locker.rollback()
        locker.close()

    assert elapsed < 3.0


@pytest.mark.asyncio
async def test_gateway_callback_returns_processing_only_after_claim_and_edits_final(
    tmp_path,
):
    from gateway.config import Platform
    from gateway.interactive_actions import InteractiveActionResult
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    handled = asyncio.Event()

    async def handler(_ctx):
        handled.set()
        return InteractiveActionResult.succeeded()

    _plugins, manager = _registered_manager(tmp_path, handler)
    prepared = manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    manager.activate_card(prepared, card_id="om_card")

    runner = object.__new__(GatewayRunner)
    runner._interactive_action_manager = manager
    runner._interactive_action_tasks = set()
    runner._is_user_authorized = lambda _source: True
    runner.config = SimpleNamespace(multiplex_profiles=False)
    updates = []

    async def update_interactive_card(*, chat_id, card_id, result):
        updates.append((chat_id, card_id, result))
        return SimpleNamespace(success=True)

    adapter = SimpleNamespace(update_interactive_card=update_interactive_card)
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        chat_type="group",
        user_id="ou_initiator",
        user_name="Alice",
        thread_id="om_root",
        message_id="om_card",
        profile="work",
    )

    initial = await runner._begin_interactive_action_from_adapter(
        callback=_callback(),
        source=source,
        adapter=adapter,
    )

    assert initial.status == "processing"
    assert manager.storage.get("ia_instance_1").state == "processing"
    await asyncio.wait_for(handled.wait(), timeout=1)
    await asyncio.gather(*runner._interactive_action_tasks)
    assert len(updates) == 1
    final = updates[0][2]
    assert final.status == "succeeded"
    assert manager.storage.get("ia_instance_1").state == "finished"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "let_task_start",
    [False, True],
    ids=("before-task-start", "during-initial-sleep"),
)
async def test_gateway_cancellation_during_initial_sleep_marks_unknown_outcome(
    tmp_path,
    let_task_start,
):
    from gateway.config import Platform
    from gateway.interactive_actions import InteractiveActionResult
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    calls = []
    _plugins, manager = _registered_manager(
        tmp_path,
        lambda ctx: calls.append(ctx),
    )
    prepared = manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    manager.activate_card(prepared, card_id="om_card")
    runner = object.__new__(GatewayRunner)
    runner._interactive_action_manager = manager
    runner._interactive_action_tasks = set()
    runner._is_user_authorized = lambda _source: True
    runner.config = SimpleNamespace(multiplex_profiles=False)
    adapter = SimpleNamespace(update_interactive_card=AsyncMock())
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        chat_type="group",
        user_id="ou_initiator",
        user_name="Alice",
        thread_id="om_root",
        message_id="om_card",
        profile="work",
    )

    initial = await runner._begin_interactive_action_from_adapter(
        callback=_callback(),
        source=source,
        adapter=adapter,
    )
    task = next(iter(runner._interactive_action_tasks))
    if let_task_start:
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert initial.status == "processing"
    replay = await manager.dispatch(_callback(), gateway_authorize=lambda: True)
    assert replay == InteractiveActionResult.unknown_outcome()
    assert calls == []
    adapter.update_interactive_card.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_cancellation_during_final_card_update_preserves_finish(
    tmp_path,
):
    from gateway.config import Platform
    from gateway.interactive_actions import InteractiveActionResult
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    update_started = asyncio.Event()

    async def update_interactive_card(**_kwargs):
        update_started.set()
        await asyncio.Event().wait()

    _plugins, manager = _registered_manager(
        tmp_path,
        lambda _ctx: InteractiveActionResult.succeeded(
            "Proposal applied before card refresh."
        ),
    )
    prepared = manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    manager.activate_card(prepared, card_id="om_card")
    runner = object.__new__(GatewayRunner)
    runner._interactive_action_manager = manager
    runner._interactive_action_tasks = set()
    runner._interactive_action_task_claims = {}
    runner._is_user_authorized = lambda _source: True
    runner.config = SimpleNamespace(multiplex_profiles=False)
    adapter = SimpleNamespace(update_interactive_card=update_interactive_card)
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        chat_type="group",
        user_id="ou_initiator",
        user_name="Alice",
        thread_id="om_root",
        message_id="om_card",
        profile="work",
    )

    await runner._begin_interactive_action_from_adapter(
        callback=_callback(),
        source=source,
        adapter=adapter,
    )
    await update_started.wait()
    task = next(iter(runner._interactive_action_tasks))
    assert manager.storage.get("ia_instance_1").outcome == "succeeded"

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    replay = await manager.dispatch(_callback(), gateway_authorize=lambda: True)

    assert replay == InteractiveActionResult.succeeded(
        "Proposal applied before card refresh."
    )
    assert manager.storage.get("ia_instance_1").outcome == "succeeded"


@pytest.mark.asyncio
async def test_gateway_handler_failure_edits_generic_retryable_state_without_leak(
    tmp_path,
):
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    def handler(_ctx):
        raise RuntimeError("secret proposal_id=prop_7 api_key=hidden")

    _plugins, manager = _registered_manager(tmp_path, handler)
    prepared = manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    manager.activate_card(prepared, card_id="om_card")
    runner = object.__new__(GatewayRunner)
    runner._interactive_action_manager = manager
    runner._interactive_action_tasks = set()
    runner._is_user_authorized = lambda _source: True
    runner.config = SimpleNamespace(multiplex_profiles=False)
    adapter = SimpleNamespace(
        update_interactive_card=AsyncMock(return_value=SimpleNamespace(success=True))
    )
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        chat_type="group",
        user_id="ou_initiator",
        user_name="Alice",
        thread_id="om_root",
        message_id="om_card",
        profile="work",
    )

    initial = await runner._begin_interactive_action_from_adapter(
        callback=_callback(),
        source=source,
        adapter=adapter,
    )
    await asyncio.gather(*runner._interactive_action_tasks)

    assert initial.status == "processing"
    final = adapter.update_interactive_card.await_args.kwargs["result"]
    assert final.status == "retryable_failure"
    assert "prop_7" not in final.user_message
    assert "api_key" not in final.user_message
    assert manager.storage.get("ia_instance_1").outcome == "retryable_failure"


@pytest.mark.asyncio
async def test_gateway_retries_transient_final_card_edit_without_rerunning_handler(
    tmp_path,
):
    from gateway.config import Platform
    from gateway.interactive_actions import InteractiveActionResult
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    calls = []

    def handler(ctx):
        calls.append(ctx)
        return InteractiveActionResult.succeeded()

    _plugins, manager = _registered_manager(tmp_path, handler)
    prepared = manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    manager.activate_card(prepared, card_id="om_card")
    runner = object.__new__(GatewayRunner)
    runner._interactive_action_manager = manager
    runner._interactive_action_tasks = set()
    runner._is_user_authorized = lambda _source: True
    runner.config = SimpleNamespace(multiplex_profiles=False)
    update = AsyncMock(
        side_effect=(
            SimpleNamespace(success=False),
            RuntimeError("provider secret must not escape"),
            SimpleNamespace(success=True),
        )
    )
    adapter = SimpleNamespace(update_interactive_card=update)
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        chat_type="group",
        user_id="ou_initiator",
        user_name="Alice",
        thread_id="om_root",
        message_id="om_card",
        profile="work",
    )

    initial = await runner._begin_interactive_action_from_adapter(
        callback=_callback(),
        source=source,
        adapter=adapter,
    )
    await asyncio.gather(*runner._interactive_action_tasks)

    assert initial.status == "processing"
    assert len(calls) == 1
    assert update.await_count == 3
    assert manager.storage.get("ia_instance_1").outcome == "succeeded"


@pytest.mark.asyncio
async def test_gateway_rejects_callback_source_identity_mismatch_before_auth_or_claim():
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    class RejectUnexpectedClaim:
        def claim_action(self, *_args, **_kwargs):
            raise AssertionError("mismatched callback reached the claim path")

    auth_calls = []
    runner = object.__new__(GatewayRunner)
    runner._interactive_action_manager = RejectUnexpectedClaim()
    runner._is_user_authorized = lambda _source: auth_calls.append(True) or True
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        user_id="ou_authenticated",
        message_id="om_card",
        profile="work",
    )

    result = await runner._begin_interactive_action_from_adapter(
        callback=_callback(operator_id="ou_attacker"),
        source=source,
        adapter=SimpleNamespace(),
    )

    assert result.status == "denied"
    assert auth_calls == []


@pytest.mark.asyncio
async def test_gateway_denied_callback_never_claims_or_edits(tmp_path):
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    calls = []
    _plugins, manager = _registered_manager(tmp_path, lambda ctx: calls.append(ctx))
    prepared = manager.prepare_card(
        plugin_id="proposal-plugin",
        envelope=_proposal_envelope(),
        origin=_origin(),
    )
    manager.activate_card(prepared, card_id="om_card")
    runner = object.__new__(GatewayRunner)
    runner._interactive_action_manager = manager
    runner._interactive_action_tasks = set()
    runner._is_user_authorized = lambda _source: False
    runner.config = SimpleNamespace(multiplex_profiles=False)
    adapter = SimpleNamespace(update_interactive_card=AsyncMock())
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        user_id="ou_initiator",
        profile="work",
    )

    result = await runner._begin_interactive_action_from_adapter(
        callback=_callback(),
        source=source,
        adapter=adapter,
    )

    assert result.status == "denied"
    assert manager.storage.get("ia_instance_1").state == "active"
    assert calls == []
    adapter.update_interactive_card.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_derives_callback_profile_from_secondary_adapter_owner(
    tmp_path,
    monkeypatch,
):
    from contextlib import nullcontext
    from gateway.config import Platform
    from gateway.interactive_actions import InteractiveActionResult
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    captured = []

    class RecordingManager:
        def claim_action(self, callback, *, gateway_authorize):
            assert gateway_authorize() is True
            captured.append(callback)
            return InteractiveActionResult.denied()

    runner = object.__new__(GatewayRunner)
    runner._interactive_action_manager = RecordingManager()
    runner._interactive_action_tasks = set()
    runner._is_user_authorized = lambda _source: True
    runner.config = SimpleNamespace(multiplex_profiles=True, profile_routes=())
    runner._resolve_profile_home_for_source = lambda _source: tmp_path
    adapter = SimpleNamespace(update_interactive_card=AsyncMock())
    runner._profile_adapters = {"work": {Platform.FEISHU: adapter}}
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_chat",
        chat_type="group",
        user_id="ou_initiator",
        user_name="Alice",
        message_id="om_card",
    )
    monkeypatch.setattr(
        "gateway.run._profile_runtime_scope",
        lambda _home: nullcontext(),
    )

    result = await runner._begin_interactive_action_from_adapter(
        callback=_callback(profile_id="default", thread_id=None),
        source=source,
        adapter=adapter,
    )

    assert result.status == "denied"
    assert source.profile == "work"
    assert captured[0].profile_id == "work"
