from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import pytest

from gateway.mac_ops_edge_client import (
    MacOpsEdgeClient,
    MacOpsEdgeClientConfig,
    MacOpsEdgeClientError,
    load_mac_ops_edge_client_config,
)
from gateway.mac_ops_edge_protocol import (
    PROTOCOL_VERSION,
    MacOpsEdgeProtocolError,
    MacOpsEdgeReceipt,
    MacOpsEdgeRequest,
    MacOpsEdgeState,
)
from gateway.mac_ops_edge_service import (
    DEFAULT_PROJECT_ID,
    MacOpsEdgeConfig,
    MacOpsEdgeRuntime,
    MacOpsEdgePeer,
    MacOpsEdgeServer,
    MacOpsGitLabTransportError,
    MacOpsJournal,
)


NOW = 1_800_000_000_000
IDENTITY = "a" * 64


def _request(
    *,
    request_id: str | None = None,
    sequence: int = 0,
    idempotency_key: str = "case:bitrix:read:1",
    contract: str = "Objective\nRead.\nAllowed scope\nA.\nForbidden actions\nWrites.\nSecrets handling\nNone.\nVerification\nReadback.\nExpected report\nFacts.",
    task_class: str = "readonly.browser",
) -> MacOpsEdgeRequest:
    return MacOpsEdgeRequest.from_mapping(
        {
            "protocol": PROTOCOL_VERSION,
            "request_id": request_id or str(uuid.uuid4()),
            "sequence": sequence,
            "deadline_unix_ms": NOW + 20_000,
            "operation": "readonly.submit",
            "idempotency_key": idempotency_key,
            "payload": {
                "title": "Read selected Bitrix evidence",
                "task_class": task_class,
                "contract": contract,
                "contract_sha256": hashlib.sha256(contract.encode()).hexdigest(),
            },
        },
        now_ms=NOW,
    )


def _read_request(issue_iid: int = 17) -> MacOpsEdgeRequest:
    return MacOpsEdgeRequest.from_mapping(
        {
            "protocol": PROTOCOL_VERSION,
            "request_id": str(uuid.uuid4()),
            "sequence": 1,
            "deadline_unix_ms": NOW + 20_000,
            "operation": "task.read",
            "idempotency_key": f"case:bitrix:observe:{issue_iid}",
            "payload": {"issue_iid": issue_iid},
        },
        now_ms=NOW,
    )


def _ping_request(nonce: str = "f" * 64) -> MacOpsEdgeRequest:
    return MacOpsEdgeRequest.from_mapping(
        {
            "protocol": PROTOCOL_VERSION,
            "request_id": str(uuid.uuid4()),
            "sequence": 0,
            "deadline_unix_ms": NOW + 20_000,
            "operation": "ping",
            "idempotency_key": f"prerequisite:ping:{nonce[:24]}",
            "payload": {"nonce": nonce},
        },
        now_ms=NOW,
    )


def _config(tmp_path: Path, *, socket_path: Path | None = None) -> MacOpsEdgeConfig:
    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700, exist_ok=True)
    state = tmp_path / "state"
    state.mkdir(mode=0o700, exist_ok=True)
    return MacOpsEdgeConfig(
        socket_path=socket_path or runtime / "edge.sock",
        gateway_uid=os.getuid(),
        socket_gid=os.getgid(),
        service_identity_sha256=IDENTITY,
        max_connections=4,
        gitlab_env_file=tmp_path / "unused.env",
        gitlab_project_id=DEFAULT_PROJECT_ID,
        gitlab_timeout_seconds=5,
        journal_path=state / "journal.sqlite3",
        journal_busy_timeout_ms=1000,
    )


def _issue(iid: int = 17, *, description: str = "") -> dict[str, Any]:
    return {
        "iid": iid,
        "state": "opened",
        "title": "queued",
        "web_url": "https://gitlab.example/issues/17",
        "confidential": True,
        "updated_at": "2026-07-14T10:00:00Z",
        "description": description,
    }


class FakeApi:
    def __init__(self) -> None:
        self.create_calls = 0
        self.create_error = False
        self.created = _issue()
        self.search: list[Mapping[str, Any]] = []
        self.notes: list[Mapping[str, Any]] = []

    def create_issue(self, payload: Mapping[str, str]) -> Mapping[str, Any]:
        self.create_calls += 1
        if self.create_error:
            raise MacOpsGitLabTransportError(
                "gitlab_transport_unavailable", ambiguous=True
            )
        self.created = {**self.created, "description": payload["description"]}
        return self.created

    def search_issues(self, _search: str) -> list[Mapping[str, Any]]:
        return list(self.search)

    def read_issue(self, issue_iid: int) -> Mapping[str, Any]:
        return {**self.created, "iid": issue_iid}

    def read_notes(self, _issue_iid: int) -> list[Mapping[str, Any]]:
        return list(self.notes)


def test_ping_is_protocol_bound_and_performs_no_external_io(tmp_path: Path) -> None:
    api = FakeApi()
    runtime = _runtime(tmp_path, api)
    request = _ping_request()

    response = runtime.execute(request)

    assert response["state"] == "completed"
    assert response["result"] == {"nonce": "f" * 64, "external_io": False}
    assert api.create_calls == 0
    MacOpsEdgeReceipt.from_mapping(response["receipt"], request=request)


def _runtime(tmp_path: Path, api: FakeApi) -> MacOpsEdgeRuntime:
    config = _config(tmp_path)
    return MacOpsEdgeRuntime(
        config=config,
        api=api,
        journal=MacOpsJournal(
            config.journal_path, busy_timeout_ms=config.journal_busy_timeout_ms
        ),
        now_ms=lambda: NOW + 1,
    )


def test_protocol_binds_semantic_intent_across_fresh_transport_envelopes() -> None:
    first = _request(sequence=1)
    second = _request(sequence=2)

    assert first.request_sha256 != second.request_sha256
    assert first.intent_sha256 == second.intent_sha256


@pytest.mark.parametrize(
    "change,code",
    [
        ({"task_class": "prod.deploy"}, "invalid_readonly_task_class"),
        ({"contract_sha256": "0" * 64}, "contract_sha256_mismatch"),
        ({"extra": True}, "invalid_submit_payload"),
    ],
)
def test_protocol_rejects_mutation_hash_drift_and_unknown_fields(
    change: dict[str, Any], code: str
) -> None:
    request = _request().to_mapping()
    request["payload"] = {**request["payload"], **change}
    with pytest.raises(MacOpsEdgeProtocolError, match=code):
        MacOpsEdgeRequest.from_mapping(request, now_ms=NOW)


def test_receipt_is_exact_and_bound_to_current_request() -> None:
    request = _request()
    receipt = MacOpsEdgeReceipt.build(
        request=request,
        state=MacOpsEdgeState.QUEUED,
        issue_iid=17,
        external_updated_at="2026-07-14T10:00:00Z",
        service_identity_sha256=IDENTITY,
        recorded_at_unix_ms=NOW,
    )

    assert MacOpsEdgeReceipt.from_mapping(
        receipt.to_mapping(), request=request
    ).value["intent_sha256"] == request.intent_sha256
    changed = _request(idempotency_key="case:other")
    with pytest.raises(MacOpsEdgeProtocolError, match="receipt_binding_mismatch"):
        MacOpsEdgeReceipt.from_mapping(receipt.to_mapping(), request=changed)


def test_submit_replay_does_not_create_a_second_issue(tmp_path: Path) -> None:
    api = FakeApi()
    runtime = _runtime(tmp_path, api)
    first = _request(sequence=1)
    retry = _request(sequence=2)

    initial = runtime.execute(first)
    replay = runtime.execute(retry)

    assert initial["state"] == "queued"
    assert replay["state"] == "queued"
    assert replay["replayed"] is True
    assert api.create_calls == 1
    MacOpsEdgeReceipt.from_mapping(replay["receipt"], request=retry)


def test_ambiguous_submit_reconciles_external_issue_without_resend(
    tmp_path: Path,
) -> None:
    api = FakeApi()
    api.create_error = True
    runtime = _runtime(tmp_path, api)
    request = _request()
    api.search = [
        _issue(
            description=(
                f"MAC_OPS_BRIDGE_TASK {request.idempotency_key} "
                f"{request.intent_sha256}"
            )
        )
    ]

    response = runtime.execute(request)

    assert response["state"] == "queued"
    assert response["result"]["issue_iid"] == 17
    assert api.create_calls == 1


def test_unreconciled_pending_key_never_blindly_resends(tmp_path: Path) -> None:
    api = FakeApi()
    api.create_error = True
    runtime = _runtime(tmp_path, api)

    first = runtime.execute(_request(sequence=1))
    second = runtime.execute(_request(sequence=2))

    assert first["state"] == "dispatch_uncertain"
    assert second["state"] == "dispatch_uncertain"
    assert second["replayed"] is True
    assert api.create_calls == 1


def test_read_returns_all_bounded_non_system_evidence_for_model_interpretation(
    tmp_path: Path,
) -> None:
    api = FakeApi()
    api.notes = [
        {"id": 1, "system": True, "body": "system", "created_at": "x"},
        {"id": 2, "system": False, "body": "arbitrary evidence A", "created_at": "y"},
        {"id": 3, "system": False, "body": "different evidence B", "created_at": "z"},
    ]
    runtime = _runtime(tmp_path, api)

    response = runtime.execute(_read_request())

    assert response["state"] == "observed"
    assert [item["body"] for item in response["result"]["notes"]] == [
        "arbitrary evidence A",
        "different evidence B",
    ]


def test_client_config_is_static_exact_and_disabled_shape_is_minimal() -> None:
    assert load_mac_ops_edge_client_config({}) is None
    assert load_mac_ops_edge_client_config({"mac_ops_edge": {"enabled": False}}) is None
    with pytest.raises(ValueError, match="disabled_mac_ops_edge_config_not_exact"):
        load_mac_ops_edge_client_config(
            {"mac_ops_edge": {"enabled": False, "socket_path": "/tmp/x"}}
        )


class _Pid:
    def main_pid(self, _unit: str) -> int:
        return os.getpid()


def test_real_unix_transport_authenticates_peer_and_receipt(tmp_path: Path) -> None:
    api = FakeApi()
    config = _config(tmp_path)
    runtime = MacOpsEdgeRuntime(
        config=config,
        api=api,
        journal=MacOpsJournal(
            config.journal_path, busy_timeout_ms=config.journal_busy_timeout_ms
        ),
    )
    peer = MacOpsEdgePeer(pid=os.getpid(), uid=os.getuid(), gid=os.getgid())
    server = MacOpsEdgeServer(
        config=config, runtime=runtime, peer_getter=lambda _sock: peer
    )
    server.bind()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    deadline = time.time() + 2
    while not config.socket_path.exists() and time.time() < deadline:
        time.sleep(0.01)
    client = MacOpsEdgeClient(
        MacOpsEdgeClientConfig(
            socket_path=config.socket_path,
            service_unit="muncho-mac-ops-edge.service",
            service_uid=os.getuid(),
            socket_gid=os.getgid(),
            service_identity_sha256=IDENTITY,
            connect_timeout_seconds=1,
            request_timeout_seconds=2,
        ),
        main_pid_provider=_Pid(),
        peer_getter=lambda _sock: peer,
    )
    try:
        response = client.submit_readonly(
            title="Read selected Bitrix evidence",
            task_class="readonly.browser",
            contract=_request().payload.contract,
            idempotency_key="case:transport:1",
        )
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert response["state"] == "queued"
    assert response["receipt"]["service_identity_sha256"] == IDENTITY


def test_client_rejects_wrong_service_identity_before_accepting_receipt(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = MacOpsEdgeClient(
        MacOpsEdgeClientConfig(
            socket_path=config.socket_path,
            service_unit="muncho-mac-ops-edge.service",
            service_uid=os.getuid(),
            socket_gid=os.getgid(),
            service_identity_sha256="b" * 64,
        ),
        main_pid_provider=_Pid(),
    )
    with pytest.raises(MacOpsEdgeClientError, match="mac_ops_edge_unavailable"):
        client.submit_readonly(
            title="Read",
            task_class="readonly.browser",
            contract=_request().payload.contract,
            idempotency_key="case:missing:1",
        )
