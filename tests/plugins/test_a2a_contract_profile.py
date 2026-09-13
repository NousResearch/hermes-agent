from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import secret_scope
from gateway.config import PlatformConfig
from plugins.platforms.a2a import adapter as a2a_adapter
from plugins.platforms.a2a import contract, protocol, tools
from plugins.platforms.a2a.adapter import A2AAdapter


@pytest.fixture(autouse=True)
def _contract_path(monkeypatch):
    local_checkout = Path.home() / ".hermes" / "a2a-contracts"
    if local_checkout.is_dir():
        monkeypatch.setenv("A2A_CONTRACTS_PATH", str(local_checkout))
    else:
        monkeypatch.delenv("A2A_CONTRACTS_PATH", raising=False)


def _whatsapp_input() -> dict:
    return {
        "recipient": {"type": "group", "alias": "team-example"},
        "content": [{"type": "text", "text": "Hello"}],
        "idempotency_key": "test-001",
    }


def test_peer_auth_can_reference_a_service_environment_secret(monkeypatch):
    monkeypatch.setenv("YEOMAN_A2A_TOKEN", "test-secret")
    assert tools._auth_header({"type": "bearer", "token_env": "YEOMAN_A2A_TOKEN"}) == {
        "Authorization": "Bearer test-secret"
    }
    with pytest.raises(ValueError, match="environment variable"):
        tools._auth_header({"type": "bearer", "token_env": "../../secret"})


def test_peer_auth_uses_active_profile_secret_in_multiplex_mode(monkeypatch):
    monkeypatch.setenv("YEOMAN_A2A_TOKEN", "default-profile-secret")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"YEOMAN_A2A_TOKEN": "routed-profile-secret"})
    try:
        assert tools._auth_header({"type": "bearer", "token_env": "YEOMAN_A2A_TOKEN"}) == {
            "Authorization": "Bearer routed-profile-secret"
        }
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_contract_loader_rejects_previous_release(tmp_path, monkeypatch):
    """The Hermes consumer must reject a v1.0.0 checkout after the pin bump."""
    old_checkout = tmp_path / "a2a-contracts"
    (old_checkout / "schemas" / "common").mkdir(parents=True)
    (old_checkout / "schemas" / "common" / "invocation.schema.json").write_text("{}\n", encoding="utf-8")
    (old_checkout / "pyproject.toml").write_text(
        "[project]\nname = 'hermes-yeoman-a2a-contracts'\nversion = '1.0.0'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("A2A_CONTRACTS_PATH", str(old_checkout))

    with pytest.raises(contract.ContractViolation, match="expected 1.0.1"):
        contract.contract_root()


def test_structured_message_has_one_authoritative_data_part():
    invocation = {"skill": "whatsapp.send", "input": _whatsapp_input()}
    message = protocol.structured_message(protocol.ROLE_USER, invocation, context_id="ctx-test")

    assert [part["mediaType"] for part in message["parts"] if "data" in part] == ["application/json"]
    assert contract.extract_invocation(message) == invocation


def test_malformed_data_part_cannot_downgrade_to_conversation():
    message = protocol.message_with_parts_v1(
        protocol.ROLE_USER,
        [{"data": {"skill": "conversation", "input": {"text": "hello"}}, "mediaType": "text/plain"}],
    )
    with pytest.raises(contract.ContractViolation, match="mediaType application/json"):
        contract.extract_invocation(message)


def test_contract_rejects_unknown_fields_and_raw_jids():
    with pytest.raises(contract.ContractViolation):
        contract.validate_skill_request("whatsapp.send", {**_whatsapp_input(), "target": "third-party"})
    with pytest.raises(contract.ContractViolation):
        contract.validate_skill_request("whatsapp.send", {
            **_whatsapp_input(),
            "recipient": {"type": "contact", "alias": "15551234567@s.whatsapp.net"},
        })


def test_contract_result_is_validated_by_skill():
    result = contract.result_for_reply("research.deep", "A report")
    assert result["output"] == {"report": "A report", "sources": []}
    with pytest.raises(contract.ContractViolation):
        contract.validate_result({
            "skill": "research.deep",
            "status": "completed",
            "output": {"report": "missing sources"},
        })


def test_adapter_rejects_multiple_authoritative_data_parts():
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    invocation = {"skill": "conversation", "input": {"text": "hello"}}
    params = {"message": protocol.message_with_parts_v1(
        protocol.ROLE_USER,
        [protocol.json_data_part(invocation), protocol.json_data_part(invocation)],
        context_id="ctx-multi",
    )}

    task, pending = adapter._prepare_task(params, "yeoman")
    assert pending is None
    assert task["status"]["state"] == protocol.STATE_REJECTED
    assert task["artifacts"][0]["parts"][-1]["data"]["error"]["code"] == "INVALID_CONTRACT"


def test_adapter_rejects_text_only_message_on_advertised_profile_endpoint():
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    task, pending = adapter._prepare_task(
        {"message": protocol.text_message(protocol.ROLE_USER, "send_whatsapp team-example hello")},
        "yeoman",
    )

    assert pending is None
    assert task["status"]["state"] == protocol.STATE_REJECTED
    assert task["artifacts"][0]["parts"][0]["data"]["error"]["code"] == "INVALID_CONTRACT"


@pytest.mark.parametrize("parts", [{"not": "a list"}, [{"data": {"skill": 123}, "mediaType": "application/json"}]])
def test_adapter_returns_rejection_for_malformed_part_container_or_skill(parts):
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    task, pending = adapter._prepare_task({"message": {"contextId": "ctx-malformed", "parts": parts}}, "yeoman")
    assert pending is None
    assert task["status"]["state"] == protocol.STATE_REJECTED
    result = task["artifacts"][0]["parts"][0]["data"]
    assert result["status"] == "rejected"
    assert result["error"]["code"] == "INVALID_CONTRACT"


def test_adapter_rejects_whatsapp_on_hermes():
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    invocation = {"skill": "whatsapp.send", "input": _whatsapp_input()}

    task, pending = adapter._prepare_task(
        {"message": protocol.structured_message(protocol.ROLE_USER, invocation, context_id="ctx-wa")},
        "yeoman",
    )
    assert pending is None
    assert task["status"]["state"] == protocol.STATE_REJECTED
    data = task["artifacts"][0]["parts"][-1]["data"]
    assert data["error"]["code"] == "SKILL_NOT_AVAILABLE"


def test_profile_error_status_matches_rejected_task_and_keeps_valid_unknown_skill():
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    invocation = {"skill": "custom.lookup", "input": {}}
    task, pending = adapter._prepare_task(
        {"message": protocol.structured_message(protocol.ROLE_USER, invocation, context_id="ctx-unknown")}, "yeoman"
    )
    assert pending is None
    result = task["artifacts"][0]["parts"][0]["data"]
    assert task["status"]["state"] == protocol.STATE_REJECTED
    assert result["status"] == "rejected"
    assert result["skill"] == "custom.lookup"


def test_profile_error_status_matches_failed_task():
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    task, pending = adapter._prepare_task(
        {"message": protocol.structured_message(protocol.ROLE_USER, {"skill": "conversation", "input": {"text": "hi"}})}, "yeoman"
    )
    assert pending is None
    result = task["artifacts"][0]["parts"][0]["data"]
    assert task["status"]["state"] == protocol.STATE_FAILED
    assert result["status"] == "failed"


def test_profile_client_sends_only_the_structured_datapart(monkeypatch):
    card = {
        "version": "1.0.1",
        "capabilities": {"extensions": [{"uri": contract.PROFILE_URI, "required": False}]},
        "skills": [{"id": "whatsapp.send", "inputModes": ["application/json"], "outputModes": ["application/json"]}],
        "defaultInputModes": ["text/plain", "application/json"],
        "defaultOutputModes": ["text/plain", "application/json"],
        "supportedInterfaces": [{"url": "http://peer/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
    }
    response_data = {
        "skill": "whatsapp.send",
        "status": "completed",
        "output": {
            "delivery_id": "delivery-1",
            "status": "accepted",
            "recipient": {"type": "group", "alias": "team-example"},
        },
    }
    captured = {}
    monkeypatch.setattr(tools, "_fetch_card", lambda *args: card)
    monkeypatch.setattr(tools.security, "audit", lambda *args: None)
    monkeypatch.setattr(tools.protocol, "persist_message", lambda *args: None)

    def fake_post(_url, body, _headers, _timeout, follow_redirects=True):
        assert follow_redirects is False
        captured["body"] = body
        response_data["correlation"] = {"task_id": "task-1", "context_id": body["params"]["message"]["contextId"]}
        return {
            "jsonrpc": "2.0", "id": body["id"],
            "result": {"task": {"id": "task-1", "contextId": body["params"]["message"]["contextId"], "status": {"state": protocol.STATE_COMPLETED},
                                  "artifacts": [{"artifactId": "artifact-1", "parts": [{"data": response_data, "mediaType": "application/json"}]}]}}
        }

    monkeypatch.setattr(tools, "_http_post_json", fake_post)
    outcome = tools._send_profile_invocation("yeoman", {"url": "http://peer", "auth": {}},
                                             "whatsapp.send", _whatsapp_input())
    parts = captured["body"]["params"]["message"]["parts"]
    assert len(parts) == 1
    assert parts[0]["mediaType"] == "application/json"
    assert parts[0]["data"] == {"skill": "whatsapp.send", "input": _whatsapp_input()}
    assert outcome["result"] == response_data


def test_profile_client_refuses_unadvertised_skill(monkeypatch):
    monkeypatch.setattr(tools, "_fetch_card", lambda *args: {
        "version": "1.0.1",
        "capabilities": {"extensions": [{"uri": contract.PROFILE_URI, "required": False}]},
        "skills": [{"id": "conversation"}],
    })
    with pytest.raises(ValueError, match="does not advertise contract skill"):
        tools._send_profile_invocation("yeoman", {"url": "http://peer", "auth": {}},
                                       "whatsapp.send", _whatsapp_input())


def test_card_advertises_profile_and_structured_modes(monkeypatch):
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"advertised_toolsets": ["web_search"]}))
    from tools.registry import registry
    monkeypatch.setattr(registry, "get_definitions", lambda _names, quiet=True: {"web_search": object()})
    card = adapter._build_card("http://localhost:9900/")
    assert card["defaultInputModes"] == ["text/plain", "application/json"]
    assert card["defaultOutputModes"] == ["text/plain", "application/json"]
    assert card["version"] == contract.CONTRACT_VERSION
    assert card["supportedInterfaces"][0]["protocolVersion"] == protocol.PROTOCOL_VERSION
    assert card["extensions"][0] == {"uri": contract.PROFILE_URI, "required": False}
    assert card["capabilities"]["extensions"][0] == {"uri": contract.PROFILE_URI, "required": False}
    assert {skill["id"] for skill in card["skills"]} >= {"toolset.web_search", "conversation", "search.web", "research.deep"}
    assert not {"whatsapp.send", "media.voice.generate"} & {skill["id"] for skill in card["skills"]}


def test_card_advertises_web_profile_skills_only_for_live_web_search(monkeypatch):
    from tools.registry import registry
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    monkeypatch.setattr(adapter, "_advertised_skills", lambda _agent: [
        {"id": "toolset.web", "name": "web", "description": "", "tags": ["web", "web_search"]},
    ])
    monkeypatch.setattr(registry, "get_definitions", lambda _names, quiet=True: {})
    assert {skill["id"] for skill in adapter._build_card()["skills"]} == {"toolset.web", "conversation"}
    monkeypatch.setattr(registry, "get_definitions", lambda _names, quiet=True: {"web_search": object()})
    assert {"search.web", "research.deep"} <= {skill["id"] for skill in adapter._build_card()["skills"]}


def test_card_web_profile_skills_respect_advertised_toolset_policy(monkeypatch):
    from tools.registry import registry
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"advertised_toolsets": ["terminal"]}))
    monkeypatch.setattr(registry, "get_definitions", lambda _names, quiet=True: {"web_search": object()})
    monkeypatch.setattr(registry, "get_tool_names_for_toolset", lambda _name: [])
    assert not {"search.web", "research.deep"} & {skill["id"] for skill in adapter._build_card()["skills"]}


def test_profile_rpc_endpoint_stays_on_configured_origin():
    assert tools._profile_rpc_url("https://peer.example:8443/base", "/a2a") == "https://peer.example:8443/a2a"
    with pytest.raises(ValueError, match="configured peer origin"):
        tools._profile_rpc_url("https://peer.example", "https://other.example/a2a")
    with pytest.raises(ValueError, match="safe HTTP"):
        tools._profile_rpc_url("https://peer.example", "javascript:alert(1)")


def test_profile_client_rejects_missing_profile_extension(monkeypatch):
    monkeypatch.setattr(tools, "_fetch_card", lambda *args: {
        "version": "1.0.1",
        "capabilities": {"extensions": []},
        "skills": [{"id": "conversation"}],
    })
    with pytest.raises(ValueError, match="does not advertise the Hermes/Yeoman A2A profile"):
        tools._send_profile_invocation("yeoman", {"url": "http://peer", "auth": {}}, "conversation", {"text": "hi"})


def test_profile_retry_reuses_persisted_outbound_context(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    card = {
        "version": "1.0.1",
        "extensions": [{"uri": contract.PROFILE_URI, "required": False}],
        "skills": [{"id": "whatsapp.send", "inputModes": ["application/json"], "outputModes": ["application/json"]}],
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "supportedInterfaces": [{"url": "http://peer/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
    }
    first = {}
    monkeypatch.setattr(tools, "_fetch_card", lambda *args: card)
    monkeypatch.setattr(tools.security, "audit", lambda *args: None)

    def fake_post(_url, body, _headers, _timeout, _follow_redirects):
        context_id = body["params"]["message"]["contextId"]
        first.setdefault("context_id", context_id)
        result = {
            "skill": "whatsapp.send",
            "status": "completed",
            "output": {"delivery_id": "delivery-1", "status": "accepted",
                       "recipient": {"type": "group", "alias": "team-example"}},
            "correlation": {"task_id": "task-original", "context_id": first["context_id"]},
        }
        return {"jsonrpc": "2.0", "id": body["id"], "result": {"task": {
            "id": "task-original", "contextId": first["context_id"],
            "status": {"state": protocol.STATE_COMPLETED},
            "artifacts": [{"artifactId": "artifact-1", "parts": [contract.json_data_part(result)]}],
        }}}

    monkeypatch.setattr(tools, "_http_post_json", fake_post)
    peer = {"url": "http://peer", "auth": {}}
    first_result = tools._send_profile_invocation("yeoman", peer, "whatsapp.send", _whatsapp_input())
    retry_result = tools._send_profile_invocation("same-peer-alias", peer, "whatsapp.send", _whatsapp_input())

    assert retry_result == first_result
    assert retry_result["context_id"] == first["context_id"]


def test_profile_client_rejects_same_idempotency_key_with_different_input(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    card = {
        "version": "1.0.1",
        "extensions": [{"uri": contract.PROFILE_URI, "required": False}],
        "skills": [{"id": "research.deep", "inputModes": ["application/json"], "outputModes": ["application/json"]}],
        "supportedInterfaces": [{"url": "http://peer/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
    }
    monkeypatch.setattr(tools, "_fetch_card", lambda *args: card)
    monkeypatch.setattr(tools, "_http_post_json", lambda *_args: (_ for _ in ()).throw(RuntimeError("offline")))
    with pytest.raises(RuntimeError, match="offline"):
        tools._send_profile_invocation(
            "yeoman", {"url": "http://peer", "auth": {}}, "research.deep",
            {"question": "first", "idempotency_key": "same-key"},
        )
    with pytest.raises(ValueError, match="Idempotency key"):
        tools._send_profile_invocation(
            "yeoman", {"url": "http://peer", "auth": {}}, "research.deep",
            {"question": "different", "idempotency_key": "same-key"},
        )


def test_profile_client_omits_missing_interface_tenant(monkeypatch):
    card = {
        "version": "1.0.1",
        "extensions": [{"uri": contract.PROFILE_URI, "required": False}],
        "skills": [{"id": "research.deep", "inputModes": ["application/json"], "outputModes": ["application/json"]}],
        "supportedInterfaces": [{"url": "http://peer/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}],
    }
    captured = {}
    monkeypatch.setattr(tools, "_fetch_card", lambda *args: card)
    monkeypatch.setattr(tools.protocol, "persist_message", lambda *args: None)

    def fake_post(_url, body, _headers, _timeout, _follow_redirects):
        captured["params"] = body["params"]
        return {"jsonrpc": "2.0", "id": body["id"], "result": {"task": {
            "id": "task-1", "contextId": body["params"]["message"]["contextId"],
            "status": {"state": protocol.STATE_WORKING},
        }}}

    monkeypatch.setattr(tools, "_http_post_json", fake_post)
    tools._send_profile_invocation(
        "yeoman", {"url": "http://peer", "auth": {}}, "research.deep",
        {"question": "question", "idempotency_key": "tenant-test"},
    )
    assert "tenant" not in captured["params"]


def test_profile_tool_sanitizes_remote_error_payload(monkeypatch):
    secret = "sk-abcdefghijklmnopqrstuv"
    signed_url = "https://private.example/report?token=do-not-leak"
    monkeypatch.setattr(tools, "_resolve_peer", lambda _agent: {"url": "http://peer", "auth": {}})
    monkeypatch.setattr(
        tools, "_send_profile_invocation",
        lambda *_args: (_ for _ in ()).throw(ValueError(f"remote error {secret} at {signed_url}")),
    )

    output = tools.a2a_skill_call({"agent": "yeoman", "skill": "conversation", "input": {"text": "hi"}})

    assert output == "Error: profile call failed."
    assert secret not in output
    assert signed_url not in output


def test_research_invocation_returns_working_task_without_waiting(monkeypatch):
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    adapter._web_search_is_available = lambda _agent=None: True  # type: ignore[method-assign]
    adapter._loop = object()
    adapter._message_handler = object()
    adapter._background_finalize = lambda _pending: None  # type: ignore[method-assign]

    def fake_handle(_event):
        return None

    def fake_submit(_coro, _loop):
        return None

    adapter.handle_message = fake_handle  # type: ignore[method-assign]
    monkeypatch.setattr("plugins.platforms.a2a.adapter.asyncio.run_coroutine_threadsafe", fake_submit)
    invocation = {
        "skill": "research.deep",
        "input": {"question": "What changed?", "idempotency_key": "research-001", "max_duration_seconds": 1800},
    }

    task, pending = adapter._prepare_task(
        {"message": protocol.structured_message(protocol.ROLE_USER, invocation, context_id="ctx-research")},
        "yeoman",
    )
    assert pending is None
    assert task["status"]["state"] == protocol.STATE_WORKING
    assert adapter.tasks.get(task["id"])["orphan_timeout"] == 1800


def test_routed_research_returns_working_before_profile_forward(monkeypatch):
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={
        "agents": {"research": {"profile": "research", "tenant": "research"}}
    }))
    adapter._web_search_is_available = lambda _agent=None: True  # type: ignore[method-assign]
    agent = adapter._agents["research"]
    forwarded = []
    background = []
    adapter._forward_to_profile = lambda *_args, **_kwargs: forwarded.append(True) or ("report", protocol.STATE_COMPLETED)  # type: ignore[method-assign]
    monkeypatch.setattr("plugins.platforms.a2a.adapter._daemon_thread", lambda target, _name: background.append(target))
    invocation = {
        "skill": "research.deep",
        "input": {"question": "What changed?", "idempotency_key": "routed-research"},
    }

    task, pending = adapter._prepare_task(
        {"tenant": "research", "message": protocol.structured_message(
            protocol.ROLE_USER, invocation, context_id="ctx-routed-research"
        )},
        "yeoman",
        agent=agent,
    )

    assert pending is None
    assert task["status"]["state"] == protocol.STATE_WORKING
    assert forwarded == []
    assert len(background) == 1
    background[0]()
    completed = adapter.tasks.get(task["id"])
    assert completed["state"] == protocol.STATE_COMPLETED
    correlation = completed["result_data"]["correlation"]
    assert correlation == {"task_id": task["id"], "context_id": "ctx-routed-research"}


def test_routed_research_uses_requested_duration_for_profile_process(monkeypatch):
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={
        "agents": {"research": {"profile": "research", "tenant": "research"}}
    }))
    adapter._web_search_is_available = lambda _agent=None: True  # type: ignore[method-assign]
    agent = adapter._agents["research"]
    background = []
    subprocess_timeouts = []
    monkeypatch.setattr(a2a_adapter, "_daemon_thread", lambda target, _name: background.append(target))
    monkeypatch.setattr(a2a_adapter, "_state_db", lambda *_args, **_kwargs: None)

    def fake_run(*_args, **kwargs):
        subprocess_timeouts.append(kwargs["timeout"])
        return SimpleNamespace(returncode=0, stdout="report", stderr="")

    monkeypatch.setattr(a2a_adapter.subprocess, "run", fake_run)
    invocation = {
        "skill": "research.deep",
        "input": {
            "question": "What changed?",
            "idempotency_key": "routed-research-timeout",
            "max_duration_seconds": 1800,
        },
    }

    task, pending = adapter._prepare_task(
        {"tenant": "research", "message": protocol.structured_message(
            protocol.ROLE_USER, invocation, context_id="ctx-routed-research-timeout"
        )},
        "yeoman",
        agent=agent,
    )
    assert pending is None
    assert task["status"]["state"] == protocol.STATE_WORKING

    background[0]()

    assert subprocess_timeouts == [1800]


def test_inbound_search_uses_structured_skill_prompt_and_result_artifact(monkeypatch):
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    adapter._web_search_is_available = lambda _agent=None: True  # type: ignore[method-assign]
    adapter._loop = object()
    adapter._message_handler = object()
    captured = {}

    def fake_handle(event):
        captured["event"] = event
        return None

    adapter.handle_message = fake_handle  # type: ignore[method-assign]
    monkeypatch.setattr("plugins.platforms.a2a.adapter.asyncio.run_coroutine_threadsafe", lambda _coro, _loop: None)
    invocation = {"skill": "search.web", "input": {
        "query": "A2A", "max_results": 3, "recency_days": 7, "language": "de",
    }}

    task, pending = adapter._prepare_task(
        {"message": protocol.structured_message(protocol.ROLE_USER, invocation, context_id="ctx-search")},
        "yeoman",
    )
    assert task is None
    assert pending["skill"] == "search.web"
    assert pending["invocation"] == invocation
    assert pending["future"] is not None
    assert "within 7 days" in captured["event"].text
    assert "language: de" in captured["event"].text
    adapter._finalize_task(pending, protocol.STATE_COMPLETED, "Search answer")
    stored = adapter.tasks.get(pending["task_id"])
    assert stored["result_data"]["output"]["answer"] == "Search answer"


def test_empty_research_success_becomes_structured_profile_failure():
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    rec = adapter.tasks.create("task-empty", "ctx-empty", "yeoman")
    adapter.tasks.set_state(rec["task_id"], protocol.STATE_WORKING)
    pending = {"task_id": rec["task_id"], "context_id": rec["context_id"], "peer": "yeoman",
               "skill": "research.deep", "reference_task_ids": [], "started": 0.0}
    state, _ = adapter._finalize_task(pending, protocol.STATE_COMPLETED, "")
    task = protocol.TaskStore.to_task(adapter.tasks.get(pending["task_id"]))
    assert state == protocol.STATE_FAILED
    assert task["artifacts"][0]["parts"][0]["data"]["status"] == "failed"


def test_inbound_search_rejects_when_live_web_search_is_unavailable():
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    adapter._web_search_is_available = lambda _agent=None: False  # type: ignore[method-assign]
    task, pending = adapter._prepare_task(
        {"message": protocol.structured_message(protocol.ROLE_USER, {"skill": "search.web", "input": {"query": "A2A"}})},
        "yeoman",
    )
    assert pending is None
    assert task["status"]["state"] == protocol.STATE_REJECTED
    assert task["artifacts"][0]["parts"][0]["data"]["status"] == "rejected"


def test_profile_task_preserves_reference_correlation_and_stable_artifact():
    store = protocol.TaskStore()
    store.create("task-1", "ctx-1", "yeoman", reference_task_ids=["task-a", "task-b"])
    result = contract.result_for_reply("conversation", "Done", task_id="task-1", context_id="ctx-1",
                                       reference_task_ids=["task-a", "task-b"])
    store.complete("task-1", protocol.STATE_COMPLETED, "Done", result)
    first, second = protocol.TaskStore.to_task(store.get("task-1")), protocol.TaskStore.to_task(store.get("task-1"))
    assert "referenceTaskIds" not in first
    assert result["correlation"]["reference_task_ids"] == ["task-a", "task-b"]
    assert first["artifacts"] == second["artifacts"]
    assert first["artifacts"][0]["parts"] == [{"data": result, "mediaType": "application/json"}]


def test_profile_reference_task_ids_are_read_from_message_and_not_reemitted_on_task():
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    message = protocol.structured_message(protocol.ROLE_USER, {"skill": "conversation", "input": {"text": "hi"}}, "ctx-ref")
    message["referenceTaskIds"] = ["task-a", "task-b"]
    task, pending = adapter._prepare_task({"message": message}, "yeoman")
    assert pending is None
    assert "referenceTaskIds" not in task
    assert task["artifacts"][0]["parts"][0]["data"]["correlation"]["reference_task_ids"] == ["task-a", "task-b"]


@pytest.mark.parametrize("references", [["task-a", "task-a"], [""], ["x" * 201], [f"task-{i}" for i in range(21)]])
def test_profile_rejects_malformed_reference_task_ids(references):
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    message = protocol.structured_message(
        protocol.ROLE_USER, {"skill": "conversation", "input": {"text": "hi"}}, "ctx-bad-ref"
    )
    message["referenceTaskIds"] = references
    task, pending = adapter._prepare_task({"message": message}, "yeoman")
    assert pending is None
    assert task["status"]["state"] == protocol.STATE_REJECTED
    result = task["artifacts"][0]["parts"][0]["data"]
    assert result["status"] == "rejected"
    assert result["error"]["code"] == "INVALID_CONTRACT"


@pytest.mark.parametrize("context_id", [123, "x" * 201])
def test_profile_rejects_malformed_context_id(context_id):
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    message = protocol.structured_message(protocol.ROLE_USER, {"skill": "conversation", "input": {"text": "hi"}})
    message["contextId"] = context_id
    task, pending = adapter._prepare_task({"message": message}, "yeoman")
    assert pending is None
    assert task["status"]["state"] == protocol.STATE_REJECTED
    assert task["artifacts"][0]["parts"][0]["data"]["error"]["code"] == "INVALID_CONTRACT"


def test_research_idempotency_reuses_task_and_rejects_conflict(monkeypatch):
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    adapter._web_search_is_available = lambda _agent=None: True  # type: ignore[method-assign]
    adapter._loop = object()
    adapter._message_handler = object()
    adapter._background_finalize = lambda _pending: None  # type: ignore[method-assign]
    dispatched = []
    adapter.handle_message = lambda event: dispatched.append(event)  # type: ignore[method-assign]
    monkeypatch.setattr("plugins.platforms.a2a.adapter.asyncio.run_coroutine_threadsafe", lambda coro, _loop: coro)

    def request(question):
        invocation = {"skill": "research.deep", "input": {"question": question, "idempotency_key": "research-same"}}
        return {"message": protocol.structured_message(protocol.ROLE_USER, invocation, "ctx-idem")}

    first, _ = adapter._prepare_task(request("one"), "yeoman")
    duplicate, _ = adapter._prepare_task(request("one"), "yeoman")
    conflict, _ = adapter._prepare_task(request("two"), "yeoman")
    assert duplicate["id"] == first["id"]
    assert len(dispatched) == 1
    assert conflict["status"]["state"] == protocol.STATE_REJECTED
    assert conflict["artifacts"][0]["parts"][0]["data"]["error"]["code"] == "IDEMPOTENCY_CONFLICT"


def test_a2a_skill_call_uses_strict_profile_client(monkeypatch):
    monkeypatch.setattr(tools, "_resolve_peer", lambda _agent: {"url": "http://peer", "auth": {}})
    monkeypatch.setattr(tools, "_send_profile_invocation", lambda *args: {"result": {"status": "completed"}})
    assert '"completed"' in tools.a2a_skill_call({"agent": "yeoman", "skill": "conversation", "input": {"text": "hi"}})
    assert "required" in tools.a2a_skill_call({"agent": "yeoman", "skill": "conversation", "input": "hi"})
