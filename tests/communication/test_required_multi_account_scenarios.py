from __future__ import annotations

import pytest

from communication_core.adapters import FakeCommunicationAdapter
from communication_core.errors import ScopeViolationError
from communication_core.repository import CommunicationRepository
from communication_core.service import CommunicationService


def _endpoint(repository, account, person, external_id):
    return repository.upsert_identity(
        connected_account_id=account["id"],
        external_id=external_id,
        display_name=person["display_name"],
        person_id=person["id"],
    )[1]


def test_required_five_account_four_person_route_matrix(tmp_path):
    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    accounts = {
        "F1": repository.add_account(provider="facebook", account_namespace="F1", label="F1", owner_profile="test"),
        "F2": repository.add_account(provider="facebook", account_namespace="F2", label="F2", owner_profile="test"),
        "T1": repository.add_account(provider="telegram", account_namespace="T1", label="T1", owner_profile="test"),
        "T2": repository.add_account(provider="telegram", account_namespace="T2", label="T2", owner_profile="test"),
        "V1": repository.add_account(provider="vk", account_namespace="V1", label="V1", owner_profile="test"),
    }
    service = CommunicationService(repository)
    specification = {
        "A": ("F1", "T1"),
        "B": ("F1", "T2"),
        "C": ("F1", "V1"),
        "D": ("F2", "T2"),
    }
    routes = {}
    for name, (source_name, target_name) in specification.items():
        person = repository.create_person(f"Person-{name}")
        source = _endpoint(repository, accounts[source_name], person, f"{name}-source")
        target = _endpoint(repository, accounts[target_name], person, f"{name}-target")
        repository.allow_account_link(
            accounts[source_name]["id"], accounts[target_name]["id"],
            allowed=True, actor="test", reason="required scenario",
        )
        route = service.apply_route(
            person_id=person["id"], source_endpoint_id=source["id"], target_endpoint_id=target["id"]
        )
        routes[name] = (person, source, target, route)

    assert routes["A"][3]["target_endpoint_id"] == routes["A"][2]["id"]
    assert routes["B"][3]["target_endpoint_id"] == routes["B"][2]["id"]
    assert routes["A"][2]["connected_account_id"] != routes["B"][2]["connected_account_id"]
    assert not repository.account_link_allowed(accounts["T1"]["id"], accounts["F1"]["id"])

    repository.disable_account(accounts["V1"]["id"])
    assert repository.get_route(routes["C"][0]["id"], routes["C"][1]["id"]) is None
    for name in ("A", "B", "D"):
        assert repository.get_route(routes[name][0]["id"], routes[name][1]["id"]) is not None


def test_partial_failure_and_retry_stay_in_damaged_account_scope(tmp_path):
    class ConditionalAdapter(FakeCommunicationAdapter):
        def __init__(self):
            super().__init__()
            self.failed_accounts = set()

        def sync_contacts(self, account, *, cursor=None):
            if account["id"] in self.failed_accounts:
                raise TimeoutError("synthetic timeout with private detail")
            return super().sync_contacts(account, cursor=cursor)

    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    damaged = repository.add_account(
        provider="fake", account_namespace="damaged", label="damaged", owner_profile="test"
    )
    healthy = repository.add_account(
        provider="fake", account_namespace="healthy", label="healthy", owner_profile="test"
    )
    adapter = ConditionalAdapter()
    adapter.failed_accounts.add(damaged["id"])
    service = CommunicationService(repository, register_builtin_adapters=False)
    service.register_adapter(adapter)

    damaged_run = service.sync(damaged["id"], mode="incremental")
    healthy_run = service.sync(healthy["id"], mode="incremental")
    assert damaged_run["status"] == "partial"
    assert healthy_run["status"] == "succeeded"
    assert repository.sync_status(healthy["id"])["issues"] == []
    issue = repository.sync_status(damaged["id"])["issues"][0]
    assert issue["connected_account_id"] == damaged["id"]
    assert "private detail" not in issue["detail_redacted"]

    adapter.failed_accounts.clear()
    retry = service.sync(damaged["id"], mode="retry")
    assert retry["status"] == "succeeded"
    assert retry["run_id"] != healthy_run["run_id"]


def test_cross_contact_message_identity_is_rejected(tmp_path):
    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    account = repository.add_account(
        provider="fake", account_namespace="one", label="one", owner_profile="test"
    )
    first = repository.create_person("First")
    second = repository.create_person("Second")
    first_identity, first_endpoint = repository.upsert_identity(
        connected_account_id=account["id"], external_id="first", display_name="First", person_id=first["id"]
    )
    second_identity, _ = repository.upsert_identity(
        connected_account_id=account["id"], external_id="second", display_name="Second", person_id=second["id"]
    )
    conversation = repository.upsert_conversation(
        connected_account_id=account["id"], endpoint_id=first_endpoint["id"], external_id="thread",
        kind="direct", title=None, provenance={}, observed_at="2026-01-01T00:00:00Z",
    )
    with pytest.raises(ScopeViolationError, match="sender identity"):
        repository.upsert_message(
            connected_account_id=account["id"], endpoint_id=first_endpoint["id"], conversation_id=conversation["id"],
            external_id="wrong", direction="incoming", body="wrong contact", sent_at="2026-01-01T00:00:00Z",
            sender_identity_id=second_identity["id"], provenance={}, observed_at="2026-01-01T00:00:00Z",
        )
