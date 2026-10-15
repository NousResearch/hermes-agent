"""Auxiliary Relay calls must label the codec by the body they hand to Relay.

Every auxiliary attempt reaches Relay as a Chat Completions request (``prepare_chat_messages`` and a
``chat.completions.create`` callback; the Codex adapter translates to Responses inside that callback).
A recovery attempt planned for a Codex destination used to label it ``codex_responses``: Relay's
Responses codec then refused the chat body before dispatch ("request is missing input"), so the
fallback that should have rescued a failed compression chunk never reached the provider.
"""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("nemo_relay")

import hermes_yaml as yaml
from agent import auxiliary_client, relay_runtime
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

CODEX_URL = "https://chatgpt.com/backend-api/codex"
MODEL = "gpt-6.1-sol"
CHAT_BODY = {"model": MODEL, "max_tokens": 64,
             "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]}


@pytest.fixture()
def managed_relay_turn(tmp_path, monkeypatch):
    """A live turn whose Relay host runs managed execution (shared metrics enabled)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(), session_id="session-1", platform="cli")
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(lease, turn_id="turn-1", task_id="task-1")
    lease.host.retain_managed_execution("test.codec-label")
    try:
        yield turn
    finally:
        lease.host.release_managed_execution("test.codec-label")
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


def _answer(text):
    return SimpleNamespace(model=MODEL, choices=[SimpleNamespace(
        message=SimpleNamespace(content=text), finish_reason="stop")])


def _client(seen):
    def create(**kwargs):
        seen.append(kwargs)
        return _answer("reached")
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_sync_attempt_labelled_for_codex_reaches_the_provider(managed_relay_turn):
    seen = []

    @auxiliary_client._relay_auxiliary_call
    def run(task):
        auxiliary_client._set_relay_auxiliary_route("openai-codex", MODEL, "chat_completions")
        return auxiliary_client._relay_sync_completion(
            _client(seen), dict(CHAT_BODY), provider="openai-codex", api_mode="codex_responses",
            create=lambda request: _client(seen).chat.completions.create(**request))

    assert run("compression").choices[0].message.content == "reached"
    assert [call["messages"] for call in seen] == [CHAT_BODY["messages"]]


def test_async_attempt_labelled_for_codex_reaches_the_provider(managed_relay_turn):
    seen = []

    async def create(request):
        return _client(seen).chat.completions.create(**request)

    @auxiliary_client._relay_auxiliary_call_async
    async def run(task):
        auxiliary_client._set_relay_auxiliary_route("openai-codex", MODEL, "chat_completions")
        return await auxiliary_client._relay_async_completion(
            _client(seen), dict(CHAT_BODY), provider="openai-codex", api_mode="codex_responses", create=create)

    assert asyncio.run(run("compression")).choices[0].message.content == "reached"
    assert [call["messages"] for call in seen] == [CHAT_BODY["messages"]]


def test_stream_attempt_labelled_for_codex_reaches_the_provider(managed_relay_turn):
    seen = []

    @auxiliary_client._relay_auxiliary_call
    def run(task):
        auxiliary_client._set_relay_auxiliary_route("openai-codex", MODEL, "codex_responses")
        return auxiliary_client._relay_sync_stream(
            _client(seen), {**CHAT_BODY, "stream": True}, provider="openai-codex")

    assert run("moa_aggregator").choices[0].message.content == "reached"
    assert [call["messages"] for call in seen] == [CHAT_BODY["messages"]]


def test_chat_label_is_unchanged(managed_relay_turn):
    seen = []

    @auxiliary_client._relay_auxiliary_call
    def run(task):
        auxiliary_client._set_relay_auxiliary_route("openrouter", "m", "chat_completions")
        return auxiliary_client._relay_sync_completion(
            _client(seen), dict(CHAT_BODY), create=lambda request: _client(seen).chat.completions.create(**request))

    assert run("compression").choices[0].message.content == "reached"
    assert len(seen) == 1


# ── End to end: the live failure (session core3e-06-p0, 4th summary chunk) ────────────────────────


def _jwt(account_id="acct-test"):
    payload = json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}).encode()
    return "e30." + base64.urlsafe_b64encode(payload).rstrip(b"=").decode() + ".synthetic-signature"


def _ok_stream(text):
    item = {"type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}
    response = {"id": "resp_test", "object": "response", "created_at": 0, "status": "completed",
                "model": MODEL, "output": [item],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}
    events = [{"type": "response.output_item.done", "output_index": 0, "item": item},
              {"type": "response.completed", "response": response}]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body)


def _error_stream(code):
    body = "data: " + json.dumps({"error": {"type": code, "code": code, "message": f"synthetic {code}"}}) + "\n\n"
    return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body)


@pytest.fixture()
def codex_profile(tmp_path, monkeypatch):
    """The bench profile shape: Codex main model, the same Codex model as configured fallback.

    Same directory as ``managed_relay_turn``'s HERMES_HOME so the turn and the config share a profile.
    """
    home = tmp_path / "profile"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": MODEL, "provider": "openai-codex", "base_url": CODEX_URL,
                  "api_mode": "codex_responses"},
        "fallback_providers": [{"provider": "openai-codex", "model": MODEL}],
        "auxiliary": {"compression": {"provider": "auto", "timeout": 120}, "transient_retries": 2},
    }), encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    token = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("code", ["server_error", "server_is_overloaded"])
def test_compression_fallback_after_a_stream_error_reaches_the_transport(codex_profile, managed_relay_turn,
                                                                         monkeypatch, code):
    from hermes_cli import auth

    requests = []

    def respond(request):
        requests.append(request.url.path)
        return _error_stream(code) if len(requests) == 1 else _ok_stream("SUMMARY-OK")

    def http_client(*, async_mode=False):
        return (httpx.AsyncClient if async_mode else httpx.Client)(transport=httpx.MockTransport(respond))

    monkeypatch.setattr(auxiliary_client, "_openai_http_client_kwargs",
                        lambda _url, *, async_mode=False: {"http_client": http_client(async_mode=async_mode)})
    monkeypatch.setattr(auxiliary_client, "_read_codex_singleton_token", lambda: _jwt())
    monkeypatch.setattr(auth, "resolve_codex_runtime_credentials",
                        lambda *a, **k: {"api_key": _jwt("acct-refreshed"), "base_url": CODEX_URL,
                                         "provider": "openai-codex"}, raising=False)

    response = auxiliary_client.call_llm(
        task="compression",
        main_runtime={"model": MODEL, "provider": "openai-codex", "base_url": CODEX_URL,
                      "api_key": _jwt(), "api_mode": "codex_responses"},
        messages=[{"role": "user", "content": "Summarize: synthetic transcript line."}])

    assert response.choices[0].message.content == "SUMMARY-OK"
    assert len(requests) == 2
