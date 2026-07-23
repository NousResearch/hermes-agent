from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from communication_core.adapters import (
    FakeCommunicationAdapter,
    DatingCommunicationAdapter,
    NormalizedConversation,
    NormalizedIdentity,
    NormalizedMessage,
)
from communication_core.errors import AccountUnavailableError, ApprovalInvalidError, CapabilityUnsupportedError, RouteDeniedError, ScopeViolationError
from communication_core.repository import CommunicationRepository
from communication_core.service import CommunicationService


def _service(tmp_path):
    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    service = CommunicationService(repository, register_builtin_adapters=False)
    adapter = FakeCommunicationAdapter(
        contacts=[NormalizedIdentity("contact", "Contact", observed_at="2026-01-01T00:00:00Z")],
        conversations=[
            NormalizedConversation(
                "thread", "contact", observed_at="2026-01-01T00:00:01Z"
            )
        ],
        messages=[
            NormalizedMessage(
                "message",
                "thread",
                "contact",
                "incoming",
                "hello",
                "2026-01-01T00:00:02Z",
                observed_at="2026-01-01T00:00:03Z",
            )
        ],
    )
    service.register_adapter(adapter)
    return repository, service, adapter


def test_full_and_incremental_sync_are_idempotent_and_account_scoped(tmp_path):
    repository, service, _ = _service(tmp_path)
    first = repository.add_account(
        provider="fake", account_namespace="one", label="one", owner_profile="test"
    )
    second = repository.add_account(
        provider="fake", account_namespace="two", label="two", owner_profile="test"
    )

    assert service.sync(first["id"], mode="full")["stats"]["messages_inserted"] == 1
    assert service.sync(first["id"], mode="incremental")["stats"]["messages_inserted"] == 0
    assert service.sync(second["id"], mode="full")["stats"]["messages_inserted"] == 1

    first_message = repository.get_conversation_by_external(first["id"], "thread")
    second_message = repository.get_conversation_by_external(second["id"], "thread")
    assert first_message["id"] != second_message["id"]
    assert repository.get_cursor(first["id"], "messages") == "2026-01-01T00:00:03Z"
    assert repository.get_cursor(second["id"], "messages") == "2026-01-01T00:00:03Z"
    with repository.read_connection() as connection:
        cursor_rows = connection.execute(
            """SELECT COUNT(*), MAX(sync_version) FROM sync_cursors
               WHERE connected_account_id = ? AND resource = 'messages'
               AND endpoint_id IS NULL""",
            (first["id"],),
        ).fetchone()
    assert tuple(cursor_rows) == (1, 2)


def test_route_is_directed_default_deny_and_disable_has_no_fallback(tmp_path):
    repository, service, _ = _service(tmp_path)
    source_account = repository.add_account(
        provider="fake", account_namespace="source", label="source", owner_profile="test"
    )
    target_account = repository.add_account(
        provider="fake", account_namespace="target", label="target", owner_profile="test"
    )
    person = repository.create_person("Person")
    _, source = repository.upsert_identity(
        connected_account_id=source_account["id"],
        external_id="source-contact",
        display_name="Person",
        person_id=person["id"],
    )
    _, target = repository.upsert_identity(
        connected_account_id=target_account["id"],
        external_id="target-contact",
        display_name="Person",
        person_id=person["id"],
    )

    denied = service.route_dry_run(
        person_id=person["id"],
        source_endpoint_id=source["id"],
        target_endpoint_id=target["id"],
    )
    assert denied["allowed"] is False
    with pytest.raises(RouteDeniedError):
        service.apply_route(
            person_id=person["id"],
            source_endpoint_id=source["id"],
            target_endpoint_id=target["id"],
        )

    repository.allow_account_link(
        source_account["id"],
        target_account["id"],
        allowed=True,
        actor="test",
        reason="test",
    )
    route = service.apply_route(
        person_id=person["id"],
        source_endpoint_id=source["id"],
        target_endpoint_id=target["id"],
    )
    assert route["target_endpoint_id"] == target["id"]
    assert not repository.account_link_allowed(target_account["id"], source_account["id"])

    repository.disable_account(target_account["id"])
    assert repository.get_endpoint(target["id"])["status"] == "disabled"
    assert repository.get_route(person["id"], source["id"]) is None


def test_exact_approval_and_fake_test_sink_only(tmp_path):
    repository, service, adapter = _service(tmp_path)
    source_account = repository.add_account(
        provider="fake", account_namespace="source", label="source", owner_profile="test"
    )
    target_account = repository.add_account(
        provider="fake", account_namespace="target", label="target", owner_profile="test",
        write_policy="approval_required",
    )
    person = repository.create_person("Person")
    _, source = repository.upsert_identity(
        connected_account_id=source_account["id"], external_id="s", display_name="Person",
        person_id=person["id"],
    )
    _, target = repository.upsert_identity(
        connected_account_id=target_account["id"], external_id="t", display_name="Person",
        person_id=person["id"],
    )
    repository.allow_account_link(
        source_account["id"], target_account["id"], allowed=True, actor="test", reason="test"
    )
    service.apply_route(
        person_id=person["id"], source_endpoint_id=source["id"], target_endpoint_id=target["id"]
    )
    draft = service.create_draft(
        person_id=person["id"], source_endpoint_id=source["id"], payload="safe test"
    )
    approval = service.approve_draft(draft["id"], ttl_minutes=5)
    outbox = service.enqueue(approval["id"], idempotency_key="proof")
    sent = service.execute_test_sink(outbox["id"])
    assert sent["status"] == "sent"
    assert adapter.deliveries["proof"]["payload"] == "safe test"

    with pytest.raises(Exception):
        service.execute_test_sink(outbox["id"])


def test_payload_mutation_invalidates_exact_approval(tmp_path):
    repository, service, _ = _service(tmp_path)
    source_account = repository.add_account(
        provider="fake", account_namespace="source", label="source", owner_profile="test"
    )
    target_account = repository.add_account(
        provider="fake", account_namespace="target", label="target", owner_profile="test"
    )
    person = repository.create_person("Person")
    _, source = repository.upsert_identity(
        connected_account_id=source_account["id"], external_id="s", display_name="Person",
        person_id=person["id"],
    )
    _, target = repository.upsert_identity(
        connected_account_id=target_account["id"], external_id="t", display_name="Person",
        person_id=person["id"],
    )
    repository.allow_account_link(
        source_account["id"], target_account["id"], allowed=True, actor="test", reason="test"
    )
    service.apply_route(
        person_id=person["id"], source_endpoint_id=source["id"], target_endpoint_id=target["id"]
    )
    draft = service.create_draft(
        person_id=person["id"], source_endpoint_id=source["id"], payload="before"
    )
    approval = service.approve_draft(draft["id"], ttl_minutes=5)
    with repository.transaction() as connection:
        connection.execute(
            "UPDATE drafts SET payload = ?, payload_hash = ? WHERE id = ?",
            ("after", hashlib.sha256(b"after").hexdigest(), draft["id"]),
        )

    with pytest.raises(ApprovalInvalidError):
        service.enqueue(approval["id"], idempotency_key="must-not-send")

    with repository.read_connection() as connection:
        status = connection.execute(
            "SELECT status FROM approvals WHERE id = ?", (approval["id"],)
        ).fetchone()[0]
        outbox_count = connection.execute("SELECT COUNT(*) FROM outbox_items").fetchone()[0]
    assert status == "invalidated"
    assert outbox_count == 0


def test_production_provider_cannot_execute_outbox(tmp_path):
    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    service = CommunicationService(repository)
    source_account = repository.add_account(
        provider="facebook", account_namespace="source", label="source", owner_profile="test"
    )
    target_account = repository.add_account(
        provider="facebook", account_namespace="target", label="target", owner_profile="test",
        write_policy="approval_required",
    )
    person = repository.create_person("Person")
    _, source = repository.upsert_identity(
        connected_account_id=source_account["id"], external_id="s", display_name="Person", person_id=person["id"]
    )
    _, target = repository.upsert_identity(
        connected_account_id=target_account["id"], external_id="t", display_name="Person", person_id=person["id"]
    )
    repository.allow_account_link(
        source_account["id"], target_account["id"], allowed=True, actor="test", reason="test"
    )
    service.apply_route(
        person_id=person["id"], source_endpoint_id=source["id"], target_endpoint_id=target["id"]
    )
    draft = service.create_draft(
        person_id=person["id"], source_endpoint_id=source["id"], payload="must stay queued"
    )
    approval = service.approve_draft(draft["id"], ttl_minutes=5)
    outbox = service.enqueue(approval["id"], idempotency_key="production-denied")

    with pytest.raises(CapabilityUnsupportedError, match="fake test sink"):
        service.execute_test_sink(outbox["id"])
    assert repository.get_outbox(outbox["id"])["status"] == "pending"


def test_channel_state_machine_requires_person_or_inbound_evidence(tmp_path):
    repository, _, _ = _service(tmp_path)
    account = repository.add_account(
        provider="fake", account_namespace="one", label="one", owner_profile="test"
    )
    other_account = repository.add_account(
        provider="fake", account_namespace="two", label="two", owner_profile="test"
    )
    person = repository.create_person("Person")
    identity_a, endpoint_a = repository.upsert_identity(
        connected_account_id=account["id"], external_id="a", display_name="Person", person_id=person["id"]
    )
    identity_b, endpoint_b = repository.upsert_identity(
        connected_account_id=other_account["id"], external_id="b", display_name="Person", person_id=person["id"]
    )
    conversation = repository.upsert_conversation(
        connected_account_id=other_account["id"], endpoint_id=endpoint_b["id"], external_id="thread",
        kind="direct", title=None, provenance={}, observed_at="2026-01-01T00:00:00Z",
    )
    message, _ = repository.upsert_message(
        connected_account_id=other_account["id"], endpoint_id=endpoint_b["id"], conversation_id=conversation["id"],
        external_id="incoming", direction="incoming", body="back here", sent_at="2026-01-01T00:00:00Z",
        sender_identity_id=identity_b["id"], provenance={}, observed_at="2026-01-01T00:00:00Z",
    )
    first_resume = repository.resume_from_inbound(
        person_id=person["id"], endpoint_id=endpoint_b["id"], message_id=message["id"]
    )
    assert first_resume["state"] == "active"
    transition = repository.record_transition(
        person_id=person["id"], from_endpoint_id=endpoint_b["id"], to_endpoint_id=endpoint_a["id"],
        initiator="person", evidence_type="person_request", evidence_ref="request-message",
    )
    assert transition["initiator"] == "person"
    resumed = repository.resume_from_inbound(
        person_id=person["id"], endpoint_id=endpoint_b["id"], message_id=message["id"]
    )
    assert resumed["initiator"] == "inbound_resume"
    with pytest.raises(ScopeViolationError, match="delivery failure"):
        repository.record_transition(
            person_id=person["id"], from_endpoint_id=endpoint_b["id"], to_endpoint_id=endpoint_a["id"],
            initiator="user", evidence_type="delivery_failure", evidence_ref="failed",
        )


def test_read_adapter_contracts_and_named_dating_pilot(tmp_path):
    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    telegram = repository.add_account(
        provider="telegram", account_namespace="tg", label="tg", owner_profile="test"
    )
    vk = repository.add_account(
        provider="vk", account_namespace="vk", label="vk", owner_profile="test"
    )
    service = CommunicationService(repository)
    telegram_health = service.account_health(telegram["id"])
    vk_health = service.account_health(vk["id"])
    assert telegram_health["provider"] == "telegram"
    assert vk_health["provider"] == "vk"
    assert telegram_health["health_status"] == "failed"
    assert vk_health["health_status"] == "failed"
    assert telegram_health["capabilities"] == []
    assert vk_health["capabilities"] == []
    with pytest.raises(ValueError, match="explicitly user-confirmed"):
        DatingCommunicationAdapter("dating", pilot_confirmed=False)
    pilot = DatingCommunicationAdapter("example-pilot", pilot_confirmed=True)
    assert "messages.send" not in pilot.capabilities.names()


def test_parallel_sync_is_locked_per_account_without_cross_account_locking(tmp_path):
    class BlockingAdapter(FakeCommunicationAdapter):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def sync_contacts(self, account, *, cursor=None):
            self.started.set()
            assert self.release.wait(timeout=5)
            return super().sync_contacts(account, cursor=cursor)

    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    first = repository.add_account(
        provider="fake", account_namespace="first", label="first", owner_profile="test"
    )
    second = repository.add_account(
        provider="fake", account_namespace="second", label="second", owner_profile="test"
    )
    service = CommunicationService(repository, register_builtin_adapters=False)
    adapter = BlockingAdapter()
    service.register_adapter(adapter)

    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(service.sync, first["id"], mode="full")
        assert adapter.started.wait(timeout=5)
        with pytest.raises(AccountUnavailableError, match="sync already running"):
            service.sync(first["id"], mode="incremental")
        with repository.account_sync_lock(second["id"]):
            pass
        adapter.release.set()
        assert running.result(timeout=5)["status"] == "succeeded"
