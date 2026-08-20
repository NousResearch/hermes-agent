"""Fault-injection tests for provider HTTP failure modes.

Companion to #54964: these tests drive REAL sockets through the REAL
``openai`` SDK client construction path
(``agent.auxiliary_client.resolve_provider_client``) — no client mock, no
patched transport. Each test proves how a genuine failure mode actually
surfaces, which is the primitive Hermes' own retry/fallback code depends
on getting right.
"""

from __future__ import annotations

import openai
import pytest

from agent.auxiliary_client import resolve_provider_client
from tests.fault_lab.fake_provider_server import FakeProviderServer


def test_429_surfaces_real_rate_limit_error_with_reason_intact():
    """A real 429 must raise with the original reason, not a swallowed generic one."""
    with FakeProviderServer() as server:
        server.script(
            429,
            {"error": {"message": "Rate limit exceeded: 10 req/min",
                       "type": "rate_limit_error"}},
        )
        client, _ = resolve_provider_client(
            "custom", explicit_base_url=server.base_url, explicit_api_key="fake-key"
        )
        with pytest.raises(openai.RateLimitError) as exc_info:
            client.chat.completions.create(
                model="fault-lab-model",
                messages=[{"role": "user", "content": "hi"}],
            )
        assert "Rate limit exceeded: 10 req/min" in str(exc_info.value)
        assert len(server.requests) == 1


def test_5xx_surfaces_real_internal_server_error():
    with FakeProviderServer() as server:
        server.script(500, {"error": {"message": "upstream exploded", "type": "server_error"}})
        client, _ = resolve_provider_client(
            "custom", explicit_base_url=server.base_url, explicit_api_key="fake-key"
        )
        with pytest.raises(openai.InternalServerError) as exc_info:
            client.chat.completions.create(
                model="fault-lab-model",
                messages=[{"role": "user", "content": "hi"}],
            )
        assert "upstream exploded" in str(exc_info.value)


def test_truncated_stream_ends_silently_without_finish_reason_stop():
    """A mid-stream connection close ends the SDK's iteration WITHOUT raising.

    This is the real fault: silent truncation is indistinguishable from a
    normal end-of-stream at the SDK level unless the caller explicitly
    checks that a ``finish_reason`` of ``"stop"`` was actually observed.
    Any Hermes streaming path that assumes "loop ended => complete" is
    exposed to exactly this failure mode.
    """
    with FakeProviderServer() as server:
        server.script_truncated_stream(["Hello", " world", " this is"])
        client, _ = resolve_provider_client(
            "custom", explicit_base_url=server.base_url, explicit_api_key="fake-key"
        )
        stream = client.chat.completions.create(
            model="fault-lab-model",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        collected = ""
        saw_stop = False
        for event in stream:
            choice = event.choices[0]
            collected += choice.delta.content or ""
            if choice.finish_reason == "stop":
                saw_stop = True

        # The stream ended (no exception), content arrived, but the
        # completion was never actually signaled — this is the fault.
        assert collected == "Hello world this is"
        assert saw_stop is False


def test_no_scripted_response_left_returns_fault_lab_marker_not_a_hang():
    """An un-scripted call gets a deterministic 500, never a silent hang."""
    with FakeProviderServer() as server:
        client, _ = resolve_provider_client(
            "custom", explicit_base_url=server.base_url, explicit_api_key="fake-key"
        )
        with pytest.raises(openai.InternalServerError) as exc_info:
            client.chat.completions.create(
                model="fault-lab-model",
                messages=[{"role": "user", "content": "hi"}],
            )
        assert "fault_lab: no scripted response left" in str(exc_info.value)
