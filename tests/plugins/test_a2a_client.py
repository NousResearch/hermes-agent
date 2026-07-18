import asyncio
import json
import threading

import httpx
import pytest
from google.protobuf.json_format import MessageToDict

from plugins.platforms.a2a import client, client_state, config, setup
from plugins.platforms.a2a.server import build_agent_card


def test_card_validation_requires_exact_jsonrpc_interface_and_safe_capabilities():
    card = build_agent_card("https://peer.example/a2a")
    client._validate_card(card, "https://peer.example/a2a")
    card.supported_interfaces[0].url = "https://attacker.example/a2a"
    with pytest.raises(client.A2AClientError, match="does not match"):
        client._validate_card(card, "https://peer.example/a2a")


def test_card_validation_rejects_streaming_and_non_text_modes():
    card = build_agent_card("https://peer.example/a2a")
    card.capabilities.streaming = True
    with pytest.raises(client.A2AClientError, match="capabilities"):
        client._validate_card(card, "https://peer.example/a2a")

    card = build_agent_card("https://peer.example/a2a")
    extra = card.supported_interfaces.add()
    extra.url = "https://attacker.example/a2a"
    extra.protocol_binding = "JSONRPC"
    extra.protocol_version = "1.0"
    with pytest.raises(client.A2AClientError, match="does not match"):
        client._validate_card(card, "https://peer.example/a2a")


def test_named_peer_only_and_strict_text(monkeypatch):
    monkeypatch.setattr(client.config, "load_a2a_settings", lambda: type("S", (), {"peers": {}})())
    with pytest.raises(client.A2AClientError, match="not configured"):
        client._peer("unknown")


@pytest.fixture
def peer_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    setup.ensure_a2a_platform_config(public_url="https://self.example/a2a")
    setup.add_peer("peer", url="http://127.0.0.1:9999/a2a", token="t" * 40)
    return tmp_path


@pytest.mark.asyncio
async def test_official_client_wire_auth_context_and_task_operations(peer_home):
    card = MessageToDict(build_agent_card("http://127.0.0.1:9999/a2a"))
    calls = []
    card_paths = []
    send_count = 0

    async def transport(request):
        nonlocal send_count
        if request.method == "GET":
            card_paths.append(str(request.url))
            return httpx.Response(200, json=card)
        body = json.loads(request.content)
        calls.append((request.headers.get("authorization"), body))
        method = body["method"]
        if method == "SendMessage":
            send_count += 1
            context_id = body["params"]["message"].get("contextId") or f"ctx-{send_count}"
            result = {"task": {"id": f"task-{send_count}", "contextId": context_id, "status": {"state": "TASK_STATE_COMPLETED"}, "artifacts": [{"artifactId": "a", "parts": [{"text": "answer"}]}]}}
        elif method == "GetTask":
            result = {"id": body["params"]["id"], "contextId": "ctx", "status": {"state": "TASK_STATE_COMPLETED"}}
        elif method == "ListTasks":
            result = {"tasks": []}
        else:
            result = {"id": body["params"]["id"], "contextId": "ctx", "status": {"state": "TASK_STATE_CANCELED"}}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    api = client.NamedPeerClient(transport=httpx.MockTransport(transport))
    first, texts = await api.ask("peer", "hello")
    second, _ = await api.ask("peer", "again")
    third, _ = await api.ask("peer", "fresh", new_context=True)
    got = await api.get_task("peer", first.id)
    listed = await api.list_tasks("peer")
    canceled = await api.cancel("peer", second.id)

    assert texts == ["answer"]
    assert set(card_paths) == {"http://127.0.0.1:9999/.well-known/agent-card.json"}
    assert all(item[0] == "Bearer " + "t" * 40 for item in calls)
    assert "contextId" not in calls[0][1]["params"]["message"]
    assert calls[0][1]["params"]["configuration"] == {}
    assert calls[1][1]["params"]["message"]["contextId"] == first.context_id
    assert "contextId" not in calls[2][1]["params"]["message"]
    assert [item[1]["method"] for item in calls[3:]] == ["GetTask", "ListTasks", "CancelTask"]
    assert calls[3][1]["params"] == {"id": first.id}
    assert calls[4][1]["params"] == {}
    assert calls[5][1]["params"] == {"id": second.id}
    assert got.id == first.id and list(listed.tasks) == []
    assert canceled.status.state != 0 and third.context_id != first.context_id


@pytest.mark.asyncio
async def test_redirect_and_transport_errors_are_rejected_and_sanitized(peer_home):
    async def redirect(_request):
        return httpx.Response(302, headers={"location": "https://attacker.example/card"})

    with pytest.raises(client.A2AClientError) as redirected:
        await client.NamedPeerClient(transport=httpx.MockTransport(redirect)).fetch_card("peer")
    assert "attacker" not in str(redirected.value)

    async def timeout(request):
        raise httpx.ReadTimeout("secret-url-and-token", request=request)

    with pytest.raises(client.A2AClientError) as failed:
        await client.NamedPeerClient(transport=httpx.MockTransport(timeout)).fetch_card("peer")
    assert "secret" not in str(failed.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "   ", "/restart"])
async def test_ask_rejects_empty_or_command_text_before_network(peer_home, text):
    async def must_not_run(_request):
        pytest.fail("network should not be called")

    with pytest.raises(ValueError):
        await client.NamedPeerClient(transport=httpx.MockTransport(must_not_run)).ask("peer", text)


@pytest.mark.asyncio
async def test_ask_rejects_new_context_with_explicit_context_before_network(peer_home):
    async def must_not_run(_request):
        pytest.fail("network should not be called")

    with pytest.raises(ValueError, match="context"):
        await client.NamedPeerClient(transport=httpx.MockTransport(must_not_run)).ask(
            "peer", "hello", new_context=True, context_id="ctx-explicit"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        "TASK_STATE_FAILED",
        "TASK_STATE_CANCELED",
        "TASK_STATE_REJECTED",
        "TASK_STATE_INPUT_REQUIRED",
        "TASK_STATE_AUTH_REQUIRED",
        "TASK_STATE_WORKING",
    ],
)
async def test_ask_accepts_only_completed_terminal_task(peer_home, state):
    card = MessageToDict(build_agent_card("http://127.0.0.1:9999/a2a"))

    async def transport(request):
        if request.method == "GET":
            return httpx.Response(200, json=card)
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "task": {
                        "id": "task",
                        "contextId": "context",
                        "status": {"state": state},
                    }
                },
            },
        )

    with pytest.raises(client.A2AClientError, match="did not complete"):
        await client.NamedPeerClient(transport=httpx.MockTransport(transport)).ask("peer", "hello")


@pytest.mark.asyncio
async def test_ask_rejects_nontext_artifact_and_oversized_response(peer_home):
    card = MessageToDict(build_agent_card("http://127.0.0.1:9999/a2a"))

    async def nontext(request):
        if request.method == "GET":
            return httpx.Response(200, json=card)
        body = json.loads(request.content)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"task": {"id": "task", "contextId": "context", "status": {"state": "TASK_STATE_COMPLETED"}, "artifacts": [{"artifactId": "a", "parts": [{"raw": "AA=="}]}]}}})

    with pytest.raises(client.A2AClientError, match="non-text"):
        await client.NamedPeerClient(transport=httpx.MockTransport(nontext)).ask("peer", "hello")

    async def oversized(_request):
        return httpx.Response(200, headers={"content-length": str(client._MAX_BODY_BYTES + 1)}, content=b"x")

    with pytest.raises(client.A2AClientError):
        await client.NamedPeerClient(transport=httpx.MockTransport(oversized)).fetch_card("peer")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "artifacts",
    [[], [{"artifactId": "a", "parts": []}], [{"artifactId": "a", "parts": [{"text": "   "}]}]],
)
async def test_completed_task_requires_nonempty_text_output(peer_home, artifacts):
    card = MessageToDict(build_agent_card("http://127.0.0.1:9999/a2a"))

    async def transport(request):
        if request.method == "GET":
            return httpx.Response(200, json=card)
        body = json.loads(request.content)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"task": {"id": "task", "contextId": "context", "status": {"state": "TASK_STATE_COMPLETED"}, "artifacts": artifacts}}})

    with pytest.raises(client.A2AClientError):
        await client.NamedPeerClient(transport=httpx.MockTransport(transport)).ask("peer", "hello")


@pytest.mark.asyncio
async def test_cancellation_before_client_tuple_assignment_closes_owned_http(peer_home):
    entered = asyncio.Event()
    closed = asyncio.Event()
    release = asyncio.Event()

    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()

        async def aclose(self):
            closed.set()
            release.set()

    task = asyncio.create_task(client.NamedPeerClient(transport=Transport()).ask("peer", "hello"))
    await entered.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert closed.is_set()


@pytest.mark.asyncio
async def test_close_attempts_both_resources_and_bounds_resistant_child(monkeypatch):
    monkeypatch.setattr(client, "_CLOSE_TIMEOUT", 0.01)
    http_called = asyncio.Event()
    release = asyncio.Event()

    class SDK:
        async def close(self):
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()

    class HTTP:
        async def aclose(self):
            http_called.set()
            raise RuntimeError("close error")

    api = client.NamedPeerClient()
    await asyncio.wait_for(api._close_owned(HTTP(), SDK()), timeout=0.05)
    assert http_called.is_set()
    assert api._owned_tasks
    with pytest.raises(client.A2AClientError, match="cleanup"):
        await api.fetch_card("peer")
    await asyncio.wait_for(api.aclose(), timeout=0.05)
    assert api._owned_tasks
    release.set()
    await api.aclose()
    assert http_called.is_set()
    assert not api._owned_tasks


@pytest.mark.asyncio
async def test_sdk_and_fallback_close_coalesce_without_double_transport_close(monkeypatch):
    monkeypatch.setattr(client, "_CLOSE_TIMEOUT", 0.1)
    calls = 0
    concurrent = 0
    max_concurrent = 0

    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):  # pragma: no cover
            raise AssertionError("request not expected")

        async def aclose(self):
            nonlocal calls, concurrent, max_concurrent
            calls += 1
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            await asyncio.sleep(0.06)
            concurrent -= 1

    http = client._CloseSerializedAsyncClient(transport=Transport())

    class SDK:
        async def close(self):
            await http.aclose()

    api = client.NamedPeerClient()
    await api._close_owned(http, SDK())
    await api.aclose()

    assert calls == 1
    assert max_concurrent == 1


@pytest.mark.asyncio
async def test_new_context_revision_prevents_old_inflight_restore(peer_home):
    card = MessageToDict(build_agent_card("http://127.0.0.1:9999/a2a"))
    old_entered = asyncio.Event()
    release_old = asyncio.Event()

    async def transport(request):
        if request.method == "GET":
            return httpx.Response(200, json=card)
        body = json.loads(request.content)
        text = body["params"]["message"]["parts"][0]["text"]
        if text == "old":
            old_entered.set()
            await release_old.wait()
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"task": {"id": f"task-{text}", "contextId": f"ctx-{text}", "status": {"state": "TASK_STATE_COMPLETED"}, "artifacts": [{"artifactId": "a", "parts": [{"text": text}]}]}}})

    first = asyncio.create_task(client.NamedPeerClient(transport=httpx.MockTransport(transport)).ask("peer", "old"))
    await old_entered.wait()
    reset = asyncio.create_task(
        client.NamedPeerClient(transport=httpx.MockTransport(transport)).ask(
            "peer", "new", new_context=True
        )
    )
    await asyncio.sleep(0.03)
    assert not reset.done()
    release_old.set()
    await asyncio.gather(first, reset)
    assert client_state.get_peer_state("peer")["context_id"] == "ctx-new"


@pytest.mark.asyncio
async def test_remove_readd_generation_rejects_stale_inflight_completion(peer_home):
    card = MessageToDict(build_agent_card("http://127.0.0.1:9999/a2a"))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def transport(request):
        if request.method == "GET":
            return httpx.Response(200, json=card)
        entered.set()
        await release.wait()
        body = json.loads(request.content)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"task": {"id": "old-task", "contextId": "old-context", "status": {"state": "TASK_STATE_COMPLETED"}, "artifacts": [{"artifactId": "a", "parts": [{"text": "old"}]}]}}})

    old_generation = config.load_a2a_settings().peers["peer"]["generation"]
    inflight = asyncio.create_task(client.NamedPeerClient(transport=httpx.MockTransport(transport)).ask("peer", "old"))
    await entered.wait()
    assert setup.remove_peer("peer")
    setup.add_peer("peer", url="http://127.0.0.1:9999/a2a", token="n" * 40)
    assert config.load_a2a_settings().peers["peer"]["generation"] != old_generation
    release.set()
    result = await asyncio.gather(inflight, return_exceptions=True)
    assert isinstance(result[0], client.A2AClientError)
    assert client_state.get_peer_state("peer") == {}


@pytest.mark.asyncio
async def test_peer_authority_snapshot_is_atomic_across_remove_readd(
    peer_home, monkeypatch
):
    old_url = "http://127.0.0.1:9999/a2a"
    new_url = "http://127.0.0.1:9998/a2a"
    old_token = "t" * 40
    new_token = "n" * 40
    old_generation = config.load_a2a_settings().peers["peer"]["generation"]
    original_load = config.load_a2a_settings
    config_read = threading.Event()
    release_snapshot = threading.Event()
    mutation_started = threading.Event()

    def paused_load():
        settings = original_load()
        if threading.current_thread().name == "a2a-authority-reader":
            config_read.set()
            assert release_snapshot.wait(timeout=2)
        return settings

    monkeypatch.setattr(client.config, "load_a2a_settings", paused_load)

    def replace_peer():
        mutation_started.set()
        assert setup.remove_peer("peer")
        setup.add_peer("peer", url=new_url, token=new_token)

    snapshot_result = []
    snapshot_error = []

    def named_reader():
        try:
            snapshot_result.append(client._peer("peer"))
        except BaseException as exc:  # pragma: no cover - assertion reports it
            snapshot_error.append(exc)

    authority_thread = threading.Thread(
        target=named_reader, name="a2a-authority-reader"
    )
    authority_thread.start()
    assert await asyncio.to_thread(config_read.wait, 2)
    mutation = asyncio.create_task(asyncio.to_thread(replace_peer))
    assert await asyncio.to_thread(mutation_started.wait, 2)
    await asyncio.sleep(0)
    assert not mutation.done()
    release_snapshot.set()
    await asyncio.to_thread(authority_thread.join, 2)
    await mutation

    assert not snapshot_error
    assert snapshot_result == [(old_url, old_token, old_generation)]
    current_url, current_token, current_generation = client._peer("peer")
    assert (current_url, current_token) == (new_url, new_token)
    assert current_generation != old_generation

    async def must_not_send(_request):
        pytest.fail("stale authority must fail before network access")

    api = client.NamedPeerClient(transport=httpx.MockTransport(must_not_send))
    with pytest.raises(client.A2AClientError, match="authority changed"):
        await api._client("peer", old_generation)


@pytest.mark.asyncio
async def test_fetch_card_rechecks_generation_before_network(peer_home, monkeypatch):
    old = ("http://127.0.0.1:9999/a2a", "t" * 40, "old-generation")
    new = ("http://127.0.0.1:9998/a2a", "n" * 40, "new-generation")
    snapshots = iter((old, new))
    monkeypatch.setattr(client, "_peer", lambda _name: next(snapshots))

    async def must_not_send(_request):
        pytest.fail("stale card authority must fail before network access")

    api = client.NamedPeerClient(transport=httpx.MockTransport(must_not_send))
    with pytest.raises(client.A2AClientError, match="authority changed"):
        await api.fetch_card("peer")


@pytest.mark.asyncio
async def test_expired_lease_successor_wins_and_stale_completion_fails(peer_home):
    card = MessageToDict(build_agent_card("http://127.0.0.1:9999/a2a"))
    old_entered = asyncio.Event()
    release_old = asyncio.Event()

    async def transport(request):
        if request.method == "GET":
            return httpx.Response(200, json=card)
        body = json.loads(request.content)
        message = body["params"]["message"]
        text = message["parts"][0]["text"]
        if text == "old":
            old_entered.set()
            await release_old.wait()
        context_id = message.get("contextId") or f"ctx-{text}"
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"task": {"id": f"task-{text}", "contextId": context_id, "status": {"state": "TASK_STATE_COMPLETED"}, "artifacts": [{"artifactId": "a", "parts": [{"text": text}]}]}}})

    old = asyncio.create_task(
        client.NamedPeerClient(transport=httpx.MockTransport(transport)).ask("peer", "old")
    )
    await old_entered.wait()
    with client_state._state_lock() as directory_fd:
        data = client_state._load_unlocked(directory_fd)
        data["peers"]["peer"]["lease_expires_at"] = 0
        client_state._save_unlocked(data, directory_fd)

    successor, _ = await client.NamedPeerClient(
        transport=httpx.MockTransport(transport)
    ).ask("peer", "new")
    release_old.set()
    result = await asyncio.gather(old, return_exceptions=True)

    assert successor.id == "task-new"
    assert isinstance(result[0], client.A2AClientError)
    state = client_state.get_peer_state("peer")
    assert state["context_id"] == "ctx-new"
    assert state["task_id"] == "task-new"


@pytest.mark.asyncio
async def test_two_client_instances_queue_and_later_reuses_earlier_context(peer_home):
    card = MessageToDict(build_agent_card("http://127.0.0.1:9999/a2a"))
    release = asyncio.Event()
    order = []
    messages = []

    async def transport(request):
        if request.method == "GET":
            return httpx.Response(200, json=card)
        body = json.loads(request.content)
        text = body["params"]["message"]["parts"][0]["text"]
        order.append(text)
        messages.append(body["params"]["message"])
        if text == "first":
            await release.wait()
        context_id = body["params"]["message"].get("contextId") or "ctx-first"
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"task": {"id": f"task-{text}", "contextId": context_id, "status": {"state": "TASK_STATE_COMPLETED"}, "artifacts": [{"artifactId": "a", "parts": [{"text": text}]}]}}})

    first = asyncio.create_task(client.NamedPeerClient(transport=httpx.MockTransport(transport)).ask("peer", "first"))
    second = asyncio.create_task(client.NamedPeerClient(transport=httpx.MockTransport(transport)).ask("peer", "second"))
    await asyncio.sleep(0.03)
    assert order == ["first"]
    release.set()
    await asyncio.gather(first, second)
    assert order == ["first", "second"]
    assert messages[1]["contextId"] == "ctx-first"
    assert client_state.get_peer_state("peer")["context_id"] == "ctx-first"


@pytest.mark.asyncio
async def test_failed_earlier_request_releases_lease_for_waiting_client(peer_home):
    card = MessageToDict(build_agent_card("http://127.0.0.1:9999/a2a"))
    first_entered = asyncio.Event()
    release_failure = asyncio.Event()

    async def transport(request):
        if request.method == "GET":
            return httpx.Response(200, json=card)
        body = json.loads(request.content)
        text = body["params"]["message"]["parts"][0]["text"]
        if text == "fail":
            first_entered.set()
            await release_failure.wait()
            return httpx.Response(500, json={"error": "unsafe detail"})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": {"task": {"id": "task-ok", "contextId": "ctx-ok", "status": {"state": "TASK_STATE_COMPLETED"}, "artifacts": [{"artifactId": "a", "parts": [{"text": "ok"}]}]}}})

    failing = asyncio.create_task(client.NamedPeerClient(transport=httpx.MockTransport(transport)).ask("peer", "fail"))
    await first_entered.wait()
    waiting = asyncio.create_task(client.NamedPeerClient(transport=httpx.MockTransport(transport)).ask("peer", "ok"))
    await asyncio.sleep(0.03)
    assert not waiting.done()
    release_failure.set()
    results = await asyncio.gather(failing, waiting, return_exceptions=True)
    assert isinstance(results[0], client.A2AClientError)
    assert results[1][0].id == "task-ok"
