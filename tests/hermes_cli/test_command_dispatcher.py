from __future__ import annotations

from dataclasses import dataclass

import pytest

from hermes_cli.command_dispatcher import (
    CommandBinding,
    CommandDispatcher,
    CommandInvocation,
    CommandPolicyState,
    CommandResult,
    InMemoryCommandReceiptStore,
)


@dataclass(frozen=True)
class FakeCommand:
    command_id: str
    name: str = "new"
    aliases: tuple[str, ...] = ("reset",)
    execution_owner: str = "server"
    handler_id: str = "session.new"


@dataclass(frozen=True)
class FakeCatalog:
    revision: str
    commands: tuple[FakeCommand, ...]

    def resolve_command_id(self, command_id: str):
        return next(
            (
                command
                for command in self.commands
                if command.command_id.casefold() == command_id.casefold()
            ),
            None,
        )

    def to_dict(self):
        return {
            "schema_version": 2,
            "revision": self.revision,
            "commands": [command.__dict__ for command in self.commands],
        }


def invocation(**overrides) -> CommandInvocation:
    values = {
        "request_id": "request-1",
        "catalog_revision": "rev-1",
        "command_id": "session.new",
        "entered_name": "reset",
        "raw_input": "/reset now",
        "raw_arguments": "now",
        "parsed_arguments": {"name": "now"},
        "surface": "desktop",
        "platform": "desktop",
        "actor": {"id": "user-1"},
        "session_id": "session-1",
        "attachments": ({"id": "attachment-1"},),
        "capabilities": {"live_session": True},
    }
    values.update(overrides)
    return CommandInvocation(**values)


def dispatcher(
    *,
    catalog: FakeCatalog | None = None,
    policy: CommandPolicyState | None = None,
    handler=None,
    receipt_store=None,
) -> CommandDispatcher:
    catalog = catalog or FakeCatalog("rev-1", (FakeCommand("session.new"),))
    policy = policy or CommandPolicyState()
    handler = handler or (
        lambda inv, _command: CommandResult(
            status="ok",
            command_id=inv.command_id,
            content=({"type": "text", "text": "created"},),
        )
    )
    return CommandDispatcher(
        catalog_provider=lambda _inv: catalog,
        policy_resolver=lambda _inv, _command: policy,
        bindings=(
            CommandBinding(
                command_id="session.new",
                execution_owner="server",
                handler_id="session.new",
                handler=handler,
            ),
        ),
        receipt_store=receipt_store,
    )


def test_invocation_is_detached_and_effect_fingerprint_ignores_request_id():
    parsed = {"nested": ["one"]}
    first = invocation(parsed_arguments=parsed)
    parsed["nested"].append("two")

    assert first.to_dict()["parsed_arguments"] == {"nested": ["one"]}
    with pytest.raises(AttributeError):
        first.parsed_arguments["nested"].append("blocked")
    assert first.effect_fingerprint() == invocation(
        request_id="request-2", parsed_arguments={"nested": ["one"]}
    ).effect_fingerprint()


def test_stale_catalog_refuses_before_policy_or_handler():
    called = []
    runtime = CommandDispatcher(
        catalog_provider=lambda _inv: FakeCatalog(
            "rev-2", (FakeCommand("session.new"),)
        ),
        policy_resolver=lambda *_args: called.append("policy"),
        bindings=(),
    )

    result = runtime.dispatch(invocation())

    assert result.status == "catalog_stale"
    assert result.current_catalog_revision == "rev-2"
    assert result.catalog_invalidation is True
    assert called == []


def test_unknown_command_is_typed_and_never_falls_through():
    result = dispatcher().dispatch(invocation(command_id="missing.command"))

    assert result.status == "unknown_command"
    assert result.command_id == "missing.command"


@pytest.mark.parametrize(
    ("policy", "status", "reason_or_code"),
    [
        (
            CommandPolicyState(available=False, unavailable_reason="unsupported"),
            "unavailable",
            "unsupported",
        ),
        (
            CommandPolicyState(authorized=False, authorization_reason="denied"),
            "error",
            "unauthorized",
        ),
        (
            CommandPolicyState(
                confirmation_required=True,
                confirmation={"kind": "confirm"},
            ),
            "confirmation_required",
            None,
        ),
        (
            CommandPolicyState(mutation_allowed=False, mutation_reason="read-only"),
            "error",
            "mutation_refused",
        ),
        (
            CommandPolicyState(busy_allowed=False, busy_reason="session_busy"),
            "unavailable",
            "session_busy",
        ),
        (
            CommandPolicyState(live_session_satisfied=False),
            "unavailable",
            "live_session_required",
        ),
        (
            CommandPolicyState(idempotency_allowed=False),
            "error",
            "idempotency_refused",
        ),
    ],
)
def test_policy_refusals_settle_before_execution(policy, status, reason_or_code):
    called = []
    result = dispatcher(policy=policy, handler=lambda *_args: called.append(True)).dispatch(
        invocation()
    )

    assert result.status == status
    if status == "unavailable":
        assert result.unavailable_reason == reason_or_code
    elif reason_or_code is not None:
        assert result.error["code"] == reason_or_code
    assert called == []


def test_retry_policy_applies_only_to_retry_attempts():
    policy = CommandPolicyState(retry_allowed=False, retry_reason="not safe")

    assert dispatcher(policy=policy).dispatch(invocation()).status == "ok"
    retried = dispatcher(policy=policy).dispatch(invocation(retry_of="request-0"))
    assert retried.status == "error"
    assert retried.error["code"] == "retry_refused"


def test_binding_owner_and_handler_are_attested_to_catalog():
    catalog = FakeCatalog(
        "rev-1",
        (FakeCommand("session.new", execution_owner="client"),),
    )
    result = dispatcher(catalog=catalog).dispatch(invocation())

    assert result.status == "error"
    assert result.error["code"] == "execution_owner_mismatch"


def test_handler_must_return_typed_result_for_same_command():
    invalid = dispatcher(handler=lambda *_args: {"status": "ok"}).dispatch(invocation())
    wrong = dispatcher(
        handler=lambda *_args: CommandResult(status="ok", command_id="session.stop")
    ).dispatch(invocation())

    assert invalid.error["code"] == "invalid_handler_result"
    assert wrong.error["code"] == "result_identity_mismatch"


def test_handler_exception_becomes_typed_error_without_escaping():
    def broken(*_args):
        raise RuntimeError("boom")

    result = dispatcher(handler=broken).dispatch(invocation())

    assert result.status == "error"
    assert result.error == {"code": "handler_error", "message": "boom"}


def test_idempotency_replays_same_effect_and_rejects_key_rebinding():
    calls = []
    store = InMemoryCommandReceiptStore()

    def handler(inv, _command):
        calls.append(inv.raw_arguments)
        return CommandResult(status="ok", command_id=inv.command_id)

    runtime = dispatcher(handler=handler, receipt_store=store)
    first = runtime.dispatch(invocation(idempotency_key="idem-1"))
    replay = runtime.dispatch(
        invocation(request_id="request-2", idempotency_key="idem-1")
    )
    conflict = runtime.dispatch(
        invocation(raw_arguments="different", idempotency_key="idem-1")
    )

    assert first is replay
    assert calls == ["now"]
    assert conflict.error["code"] == "idempotency_conflict"


def test_receipt_settlement_failure_is_indeterminate_after_the_effect():
    class BrokenStore:
        def begin(self, _key, _fingerprint):
            return "new", None

        def settle(self, _key, _fingerprint, _result):
            raise RuntimeError("receipt sink unavailable")

    result = dispatcher(receipt_store=BrokenStore()).dispatch(
        invocation(idempotency_key="idem-1")
    )

    assert result.status == "indeterminate"
    assert result.error["code"] == "receipt_settlement_failed"


def test_idempotent_attempt_fails_closed_without_receipt_store():
    result = dispatcher().dispatch(invocation(idempotency_key="idem-1"))

    assert result.error["code"] == "idempotency_store_unavailable"


def test_duplicate_bindings_fail_catalog_construction():
    binding = CommandBinding(
        command_id="session.new",
        execution_owner="server",
        handler_id="session.new",
        handler=lambda *_args: CommandResult(status="ok", command_id="session.new"),
    )

    with pytest.raises(ValueError, match="duplicate command binding"):
        CommandDispatcher(
            catalog_provider=lambda _inv: FakeCatalog("rev-1", ()),
            policy_resolver=lambda *_args: CommandPolicyState(),
            bindings=(binding, binding),
        )
