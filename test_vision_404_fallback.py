#!/usr/bin/env python3
"""E2E test: vision_analyze 404 must fall through to a working aux backend.

Reproduces the user's exact failure -- a LiteLLM router proxying only TEXT
models returns 404 "No endpoints found that support image input" on any
image payload -- and asserts the patched call_llm/async_call_llm fall back
to the next available backend instead of raising.

This validates the fix landed in agent/auxiliary_client.py (session 20260723):
`_is_model_not_found_error` now matches capability-404s ("no endpoints found
that support image input", "does not support vision", ...) and `call_llm`
includes it in `should_fallback`, so the fallback chain engages.

No network: we stub the vision client to raise the router's 404 and stub the
fallback chain to return a sentinel "working" client.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import agent.auxiliary_client as ac


class Router404(Exception):
    status_code = 404

    def __str__(self):
        return ('litellm.NotFoundError: OpenrouterException - '
                '{"error":{"message":"No endpoints found that support image input","code":404}}')


def _msg(content):
    return SimpleNamespace(content=content, reasoning_content=None, refusal=None)


def _choices(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=_msg(content), finish_reason="stop", index=0)], usage=None)


def _boom_client():
    """A fake OpenAI client whose create() raises the router vision-404."""
    class _Create:
        def __call__(self, **kw):
            raise Router404()

    class _Completions:
        create = _Create()

    class _Chat:
        completions = _Completions()

    class _Client:
        base_url = SimpleNamespace(host="router")
        api_key = "fake-router-key"
        chat = _Chat()

    return _Client()


def _ok_sync_client():
    """A fake SYNC OpenAI client whose create() returns FALLBACK_OK."""
    class _Create:
        def __call__(self, **kw):
            return _choices("FALLBACK_OK")

    class _Completions:
        create = _Create()

    class _Chat:
        completions = _Completions()

    class _Client:
        base_url = SimpleNamespace(host="fallback")
        api_key = "fake-fallback-key"
        chat = _Chat()

    return _Client()


def _ok_async_client():
    """A fake ASYNC OpenAI client whose create() returns FALLBACK_OK."""
    class _Create:
        async def __call__(self, **kw):
            return _choices("FALLBACK_OK")

    class _Completions:
        create = _Create()

    class _Chat:
        completions = _Completions()

    class _Client:
        base_url = SimpleNamespace(host="fallback")
        api_key = "fake-fallback-key"
        chat = _Chat()

    return _Client()


def _make_fake_vision_provider_client(provider=None, model=None, *a, **k):
    """Stand-in for resolve_vision_provider_client: returns the boom client."""
    return (provider or "auto"), _boom_client(), (model or "auto")


def _fake_payment_fallback(failed_provider, task=None, reason=None):
    return _ok_sync_client(), "fallback-model", "openrouter"


def _fake_main_fallback_chain(task, failed_provider="", reason=None):
    return None, None, ""


def _fake_configured_fallback_chain(task, failed_provider="", reason=None):
    return None, None, ""


VISION_MSGS = [{"role": "user", "content": [
    {"type": "text", "text": "describe"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}}]}]


def _run_sync():
    with patch.object(ac, "resolve_vision_provider_client", _make_fake_vision_provider_client) as p1, \
         patch.object(ac, "_try_payment_fallback", _fake_payment_fallback) as p2, \
         patch.object(ac, "_try_main_fallback_chain", _fake_main_fallback_chain) as p3, \
         patch.object(ac, "_try_configured_fallback_chain", _fake_configured_fallback_chain) as p4:
        resp = ac.call_llm(
            task="vision", messages=VISION_MSGS, max_tokens=200,
            main_runtime={"provider": "custom:litellm-router", "model": "auto"})
    content = getattr(getattr(resp, "choices", [None])[0].message, "content", None)
    assert content == "FALLBACK_OK", f"unexpected content: {content!r}"
    print("SYNC E2E PASS -> fallback returned:", content)


def _run_async():
    # async_call_llm wraps the fallback sync client into an async client via
    # _to_async_client (which builds a real AsyncOpenAI and would hit the net).
    # Stub it to return our offline async fake so the test stays network-free
    # while still exercising the vision-404 -> fallback decision in call_llm.
    with patch.object(ac, "resolve_vision_provider_client", _make_fake_vision_provider_client) as p1, \
         patch.object(ac, "_try_payment_fallback", _fake_payment_fallback) as p2, \
         patch.object(ac, "_try_main_fallback_chain", _fake_main_fallback_chain) as p3, \
         patch.object(ac, "_try_configured_fallback_chain", _fake_configured_fallback_chain) as p4, \
         patch.object(ac, "_to_async_client", lambda c, m, **k: (_ok_async_client(), m)) as p5:
        resp = asyncio.run(
            ac.async_call_llm(
                task="vision", messages=VISION_MSGS, max_tokens=200,
                main_runtime={"provider": "custom:litellm-router", "model": "auto"}))
    content = getattr(getattr(resp, "choices", [None])[0].message, "content", None)
    assert content == "FALLBACK_OK", f"unexpected content: {content!r}"
    print("ASYNC E2E PASS -> fallback returned:", content)


def test_sync():
    _run_sync()


def test_async():
    _run_async()


if __name__ == "__main__":
    _run_sync()
    _run_async()
    print("\nALL E2E TESTS PASSED")
