"""Offline wire-level regression for the silent paid aux fallback in #136687."""
import asyncio
import contextvars
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agent import auxiliary_client as ac
from agent.status_output import StatusOutputMixin
from hermes_constants import get_hermes_home


class _Session(StatusOutputMixin):
    def __init__(self, session_id):
        self.session_id = session_id
        self.statuses = []
        self.printed = []
        self.status_callback = lambda kind, text: self.statuses.append((kind, text))
        self._print_fn = self.printed.append
        self.log_prefix = ""


@pytest.fixture
def wire(monkeypatch, tmp_path):
    """Real config, resolver, SDK and recovery ladder; only the HTTP transport is offline."""
    from agent import anthropic_adapter
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(anthropic_adapter, "_get_claude_code_version", lambda: "test")
    ac.clear_runtime_main()
    ac._reset_aux_unhealthy_cache()
    attempts = []
    state = SimpleNamespace(primary="sk-ant-oat01-test", fallback="sk-ant-api03-test", fail=False)
    state.notices_at_paid = []
    def send(_client, request, **kwargs):
        key = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ")
        attempts.append(key)
        if key == state.fallback and hasattr(state, "session"):
            state.notices_at_paid.append(len(state.session.statuses))
        primary_failed = key == state.primary and (
            not getattr(state, "same_provider", False) or json.loads(request.content)["model"] == "claude-sonnet-4-5")
        if primary_failed or state.fail:
            return httpx.Response(429, request=request, json={"type": "error", "error": {
                "type": "rate_limit_error", "message": "Rate limit exceeded"}})
        message = {
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "recovered"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        if json.loads(request.content).get("stream"):
            events = [
                {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "recovered"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
                {"type": "message_stop"},
            ]
            content = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
            return httpx.Response(200, request=request, content=content, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, request=request, json=message)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", send)
    # Any accidental async HTTP path also fails locally, rather than accessing a real account.
    async def no_network(*args, **kwargs):
        raise AssertionError("Unexpected async network transport")
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
    def configure():
        config = {
            "model": {"provider": "subscription", "default": "claude-sonnet-4-5"},
            "providers": {
                name: {"base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages", "api_key": key}
                for name, key in (("subscription", state.primary), ("paid", state.fallback))
            },
            "fallback_providers": [{"provider": "subscription" if getattr(state, "same_provider", False) else "paid",
                                    "model": "claude-haiku-4-5"}],
        }
        (get_hermes_home() / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    yield state, attempts, configure
    ac.clear_runtime_main()
    ac._reset_aux_unhealthy_cache()


def _invoke(session, async_mode, context_only=False):
    from run_agent import AIAgent
    # Exercise the facade's explicit runtime and the same warning sink the CLI/gateway uses.
    runtime = AIAgent._current_main_runtime(session)
    kwargs = dict(task="title_generation", main_runtime=runtime, messages=[{"role": "user", "content": "Title"}])
    if context_only:
        kwargs.pop("main_runtime")
    return asyncio.run(ac.async_call_llm(**kwargs)) if async_mode else ac.call_llm(**kwargs)


@pytest.mark.parametrize("async_mode,fails", [
    pytest.param(False, False, id="sync-success"),
    pytest.param(True, False, id="async-success"),
    pytest.param(False, True, id="failed-request-and-ui"),
])
def test_paid_fallback_is_visible_even_when_it_fails(wire, async_mode, fails):
    state, attempts, configure = wire
    state.fail = fails
    configure()
    session = _Session("session-a")
    state.session = session
    if fails:
        def broken_ui(kind, text):
            session.statuses.append((kind, text))
            raise RuntimeError("offline UI disconnected")
        session.status_callback = broken_ui
    if fails:
        with pytest.raises(Exception, match="Rate limit exceeded"):
            _invoke(session, async_mode)
    else:
        assert _invoke(session, async_mode).choices[0].message.content == "recovered"
    assert state.fallback in attempts
    assert len(session.statuses) == 1
    kind, message = session.statuses[0]
    assert kind == "warn" and "additional charges" in message and "subscription" in message
    assert "paid" in message
    assert state.primary not in message and state.fallback not in message
    assert session.printed == [message]
    assert state.notices_at_paid and all(count == 1 for count in state.notices_at_paid)


def test_notice_is_scoped_to_the_live_session(wire):
    _state, _attempts, configure = wire
    configure()
    first, second = _Session("first"), _Session("second")
    from agent.turn_context import _publish_runtime_main
    contexts = {}
    for owner in (first, second):
        _publish_runtime_main(owner)
        contexts[owner] = contextvars.copy_context()
    for owner in (first, first, second):
        ac._reset_aux_unhealthy_cache()
        assert contexts[owner].run(_invoke, owner, False, True).choices[0].message.content == "recovered"
    assert len(first.statuses) == len(second.statuses) == 1
    first.session_id = "new-session"
    ac._reset_aux_unhealthy_cache()
    _invoke(first, False)
    assert len(first.statuses) == 2


def test_same_provider_model_fallback_is_silent(wire):
    state, attempts, configure = wire
    state.same_provider = True
    configure()
    session = _Session("same-provider")
    assert _invoke(session, False).choices[0].message.content == "recovered"
    assert attempts and set(attempts) == {state.primary}
    assert session.statuses == []


def test_custom_destinations_deduplicate_without_disclosing_urls(wire, caplog):
    session = _Session("custom")
    source, _ = ac.resolve_provider_client(
        "custom", "test-model", explicit_base_url="https://source.internal/v1", explicit_api_key="offline-key")
    runtime = {"billing_warning_callback": session._warn_auxiliary_billing}
    caplog.set_level("INFO", logger="agent.auxiliary_billing")
    for url in ("https://secret-user:secret-pass@one.internal/v1?key=secret-query",
                "https://one.internal/v1?key=rotated-secret", "https://two.internal/v1"):
        target, _ = ac.resolve_provider_client(
            "custom", "test-model", explicit_base_url=url, explicit_api_key="offline-key")
        ac.warn_service_fallback(source, target, source_provider="custom", target_provider="custom", runtime=runtime)
    assert len(session.statuses) == 2
    notices = " ".join(message for _, message in session.statuses)
    route_logs = [r.message for r in caplog.records if r.name == "agent.auxiliary_billing"]
    assert len(route_logs) == 3
    for text in [notices, *route_logs]:
        assert all(secret not in text for secret in ("secret", "internal", "offline-key"))
    assert all("additional charges" in text for _, text in session.statuses)
