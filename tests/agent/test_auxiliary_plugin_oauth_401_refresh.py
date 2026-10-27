"""Auxiliary calls on a plugin OAuth provider (Solstice) refresh the pooled sign-in after a 401.

Title generation and approval inherit the main route as ``resolved_provider == "auto"``; vision and a
pinned ``auxiliary.<task>.provider`` resolve to ``solstice``. Every one of them used to raise the raw
401 without spending the refresh token: the auto route found no host row for
generativelanguage.googleapis.com (shared with the Gemini API-key provider), and the named route asked
the profile's generic ``refresh_runtime_credentials``, which cannot reach a pooled row. The pinned
route then benched the only credential, and the next call reported that 401 as a 429 cooldown.

Real ``call_llm`` path; only Google's HTTP and the token broker are faked.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

import agent.auxiliary_client as aux
from agent.auxiliary_unavailable import pool_cooldown_message
from hermes_cli import auth_oauth_pkce_plugin as pkce

GOOGLE = "generativelanguage.googleapis.com"
BASE_URL = "https://generativelanguage.googleapis.com/v1alpha"
OK_BODY = {"candidates": [{"content": {"role": "model", "parts": [{"text": "A Title"}]}, "finishReason": "STOP"}],
           "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2, "totalTokenCount": 3}}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _row(access, expires_at_ms, **extra):
    return {"id": "sol1", "label": "solstice", "auth_type": "oauth", "priority": 0, "source": pkce.POOL_SOURCE,
            "access_token": access, "refresh_token": "1//rt-0", "expires_at_ms": expires_at_ms, **extra}


@pytest.fixture
def env(monkeypatch, tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    (home / "config.yaml").write_text("model:\n  provider: solstice\n  default: gemini-3.1-pro\n", encoding="utf-8")
    grants, seen = [], []

    def fake_post_token(provider, cfg, data, *, code):
        grants.append(dict(data))
        return {"access_token": f"ya29.fresh-{len(grants)}", "refresh_token": "1//rt-0",
                "expires_at_ms": _now_ms() + 3_600_000, "last_refresh": "now"}

    monkeypatch.setattr(pkce, "_post_token", fake_post_token)

    def respond(url, headers):
        bearer = headers.get("Authorization", "")
        seen.append(bearer)
        req = httpx.Request("POST", str(url))
        if GOOGLE in str(url) and "fresh" in bearer:
            return httpx.Response(200, json=OK_BODY, request=req)
        return httpx.Response(401, request=req, json={"error": {
            "code": 401, "status": "UNAUTHENTICATED", "message": "Request had invalid authentication credentials."}})

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kw: respond(url, kw.get("headers") or {}))
    monkeypatch.setattr(aux, "_client_cache", {})
    return home, grants, seen


def _write(home, row):
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"solstice": [row]}}),
                                    encoding="utf-8")


def _disk_row(home):
    return json.loads((home / "auth.json").read_text(encoding="utf-8-sig"))["credential_pool"]["solstice"][0]


def _text(resp):
    return resp.choices[0].message.content


def _title(runtime):
    return aux.call_llm(task="title_generation", main_runtime=runtime,
                        messages=[{"role": "user", "content": "name this chat"}], max_tokens=20)


@pytest.mark.parametrize("session_key", [None, "ya29.revoked"])
def test_auto_route_401_refreshes_the_pooled_sign_in_and_retries(env, session_key):
    """Title generation inherits the main Solstice route ("auto"); a 401 on a token the pool still
    believes valid refreshes that row and retries, whether or not the session's key is pinned."""
    home, grants, seen = env
    _write(home, _row("ya29.revoked", _now_ms() + 3_000_000))
    runtime = {"provider": "solstice", "model": "gemini-3.1-pro", "base_url": BASE_URL, "api_mode": "chat_completions"}
    if session_key:
        runtime["api_key"] = session_key

    assert _text(_title(runtime)) == "A Title"
    assert len(grants) == 1
    assert seen == ["Bearer ya29.revoked", "Bearer ya29.fresh-1"]
    row = _disk_row(home)
    assert row["access_token"] == "ya29.fresh-1"
    assert row.get("last_status") != "exhausted"


def test_session_key_the_pool_knows_is_expired_is_not_sent(env):
    """A pinned session key whose pooled row is past its stored expiry is leased (refreshed) from the
    pool instead of being sent to Google for a guaranteed 401."""
    home, grants, seen = env
    _write(home, _row("ya29.expired", _now_ms() - 1000))
    runtime = {"provider": "solstice", "model": "gemini-3.1-pro", "api_key": "ya29.expired",
               "base_url": BASE_URL, "api_mode": "chat_completions"}

    assert _text(_title(runtime)) == "A Title"
    assert len(grants) == 1
    assert "Bearer ya29.expired" not in seen


def test_pinned_aux_provider_401_refreshes_instead_of_benching(env):
    """``auxiliary.title_generation.provider: solstice`` used to bench the only credential on a 401
    (no refresh), so the next call failed with a cooldown message. It refreshes and keeps working."""
    home, grants, _seen = env
    (home / "config.yaml").write_text(
        "model:\n  provider: solstice\n  default: gemini-3.1-pro\n"
        "auxiliary:\n  title_generation:\n    provider: solstice\n    model: gemini-3.1-pro\n", encoding="utf-8")
    _write(home, _row("ya29.revoked", _now_ms() + 3_000_000))

    assert _text(_title(None)) == "A Title"
    assert _disk_row(home).get("last_status") != "exhausted"
    assert pool_cooldown_message("solstice") is None
    assert _text(_title(None)) == "A Title"
    assert len(grants) == 1


@pytest.mark.parametrize("task", ["vision", "approval"])
def test_vision_and_approval_401_refresh(env, task):
    home, grants, _seen = env
    _write(home, _row("ya29.revoked", _now_ms() + 3_000_000))
    content = ([{"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]
               if task == "vision" else "approve?")

    resp = aux.call_llm(task=task, messages=[{"role": "user", "content": content}], max_tokens=20)

    assert _text(resp) == "A Title"
    assert len(grants) == 1
    assert _disk_row(home).get("last_status") != "exhausted"


@pytest.mark.parametrize(("code", "expected", "absent"), [
    (401, "rejected its sign-in (401)", "429"),
    (403, "rejected its sign-in (403)", "429"),
    (402, "billing error (402)", "429"),
    (429, "rate limit / quota error (429)", "sign-in"),
])
def test_cooldown_message_names_the_recorded_error(env, code, expected, absent):
    """A benched credential's message comes from its own ``last_error_code``: a sign-in the provider
    rejected used to be reported as "cooling down after a rate limit / quota error (429)"."""
    home, _, _ = env
    _write(home, _row("ya29.dead", _now_ms() + 3_000_000, last_status="exhausted",
                      last_status_at=time.time(), last_error_code=code))

    message = pool_cooldown_message("solstice")

    assert message is not None and "cooling down" in message
    assert expected in message
    assert absent not in message
    if code in (401, 403):
        assert "Sign in again with `hermes auth add solstice`" in message
