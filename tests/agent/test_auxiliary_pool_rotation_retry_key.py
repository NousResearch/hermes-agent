"""A post-rotation aux retry must send the rotated key, not the main session's pinned one.

Aux calls on the main provider pin the main session's key. When that key hits a 429 the pool
rotates, but the retry used to re-resolve the main route and re-send the benched key; its second
429 was then charged to the healthy entry the pool had just rotated to, emptying a two-seat pool.
"""
import json
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import httpx
from openai import RateLimitError

from agent import anthropic_credentials as ac
from agent import auxiliary_client as aux
from agent.credential_pool import load_pool

MODEL = "claude-opus-4-8"
MAIN_RUNTIME = {"provider": "anthropic", "model": MODEL, "base_url": "https://api.anthropic.com",
                "api_key": "key-a"}


def _entry(entry_id, token, priority):
    return {"id": entry_id, "label": entry_id, "source": "manual:hermes_pkce", "auth_type": "oauth",
            "access_token": token, "refresh_token": f"{token}-refresh", "priority": priority,
            "expires_at_ms": int(time.time() * 1000) + 3_600_000}


def _seed_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(ac.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(ac, "_first_env", lambda *names: "")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [
        _entry("seat-a", "key-a", 0), _entry("seat-b", "key-b", 1),
    ]}}))


def _rate_limit():
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    return RateLimitError("This request would exceed your account's rate limit.", response=response, body={})


def _route():
    return aux._LadderRoute(
        client=SimpleNamespace(api_key="key-a", base_url="https://api.anthropic.com"), task="approval", tag="",
        async_mode=False,
        base_info="https://api.anthropic.com", resolved_provider="auto", resolved_model=MODEL,
        resolved_base_url=None, resolved_api_key=None, resolved_api_mode=None, final_model=MODEL,
        main_runtime=MAIN_RUNTIME, route_info=None, timeout=None)


def _retry_kwargs():
    return dict(task="approval", messages=[{"role": "user", "content": "ok?"}], temperature=None,
                max_tokens=16, tools=None, effective_timeout=30.0, effective_extra_body={},
                reasoning_config=None, resolved_base_url=None, resolved_api_key=None,
                resolved_api_mode=None, main_runtime=MAIN_RUNTIME, final_model=MODEL, extra_headers=None)


def _drive_to_pool_retry(ladder):
    assert next(ladder).kind == "call"  # same-key retry the ladder grants a plain 429
    step = ladder.throw(_rate_limit())
    assert step.kind == "retry_same_provider"
    return step


def test_retry_after_pool_rotation_sends_the_rotated_key(tmp_path, monkeypatch):
    _seed_pool(tmp_path, monkeypatch)
    ladder = aux._ladder_credential_rungs(_rate_limit(), _route(), {"model": MODEL}, False)
    step = _drive_to_pool_retry(ladder)

    kind, _args, kw = aux._ladder_step_call(step, cast(Any, None), _retry_kwargs(), {})
    retry_client, _ = aux._prepare_same_provider_retry(async_mode=False, **kw)

    assert kind == "retry"
    assert retry_client.api_key == "key-b"
    assert MAIN_RUNTIME["api_key"] == "key-a"  # the session's published runtime is not mutated
    ladder.close()


def test_retry_failure_is_charged_to_the_key_the_retry_sent(tmp_path, monkeypatch):
    _seed_pool(tmp_path, monkeypatch)
    ladder = aux._ladder_credential_rungs(_rate_limit(), _route(), {"model": MODEL}, False)
    _drive_to_pool_retry(ladder)

    with patch.object(aux, "_recover_provider_pool", wraps=aux._recover_provider_pool) as recover:
        try:
            ladder.throw(_rate_limit())
        except StopIteration as done:
            assert done.value[0] is None
    assert recover.call_args.kwargs["failed_api_key"] == "key-b"
    statuses = {e.label: e.last_status for e in load_pool("anthropic").entries()}
    assert statuses["seat-a"] == "exhausted"
