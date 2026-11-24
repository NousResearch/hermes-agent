"""PromptCacheWarmer keep-alives must not go out on an expired Nous Portal agent key.

Nous Portal agent keys are short-lived JWTs (about an hour), while a 1-hour prompt-cache keep-alive
fires about 50 minutes after the request it keeps warm. ``_Run`` snapshots ``run.client`` when the
warmer arms, so without a refresh the keep-alive is sent with the key the turn started with; once
that key has expired the refresh fails with HTTP 401 and warming is disabled for the session.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

from agent.prompt_cache_lifecycle.warmer import PromptCacheWarmer, _Budget, _Run


class _AuthError(Exception):
    status_code = 401


class _FakeCompletions:
    def __init__(self, label: str, *, fail_401: bool = False):
        self.label = label
        self.fail_401 = fail_401
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_401:
            raise _AuthError("Error code: 401 - Your API key is invalid")
        return SimpleNamespace(id=f"ok-{self.label}", choices=[], usage=None)


class _FakeClient:
    def __init__(self, label: str, *, fail_401: bool = False):
        self.label = label
        self.chat = SimpleNamespace(completions=_FakeCompletions(label, fail_401=fail_401))

    def with_options(self, **kwargs):
        return self


def _make_run(client, *, provider: str = "nous", api_mode: str = "chat_completions") -> _Run:
    return _Run(
        api_kwargs={"model": "hermes-4-405b", "messages": [{"role": "user", "content": "hi"}]},
        client=client,
        model="hermes-4-405b",
        cached_tokens=200_000,
        uncached_tokens=1_000,
        ttl_s=3600.0,
        deadline_at=9999999999.0,
        is_current=lambda: True,
        api_mode=api_mode,
        witness=("sess-1", 0, 1, ("nous", "https://inference-api.nousresearch.com/v1", "hermes-4-405b")),
        billing={
            "session_id": "sess-1",
            "source": "cli",
            "provider": provider,
            "base_url": "https://inference-api.nousresearch.com/v1",
            "model": "hermes-4-405b",
            "api_mode": api_mode,
            "api_key": "stale-jwt",
            "reset_epoch": 0,
        },
        idle_ok=True,
        budget=_Budget(turn=1),
    )


def _agent(client, *, provider: str = "nous", **extra):
    return SimpleNamespace(provider=provider, api_mode="chat_completions", api_key="stale-jwt", client=client,
                           log_prefix="[test] ", **extra)


def test_keepalive_adopts_a_fresh_nous_key_before_expiry():
    stale = _FakeClient("stale", fail_401=True)
    fresh = _FakeClient("fresh")
    agent = _agent(stale)
    adopted = []

    def _adopt():
        adopted.append(True)
        agent.client = fresh
        agent.api_key = "fresh-jwt"
        return True

    agent._adopt_nous_key_before_expiry = _adopt
    run = _make_run(stale)

    reply = PromptCacheWarmer(agent)._send(run)

    assert adopted == [True]
    assert reply.id == "ok-fresh"
    assert run.client is fresh
    assert run.billing["api_key"] == "fresh-jwt"
    assert stale.chat.completions.calls == []
    assert len(fresh.chat.completions.calls) == 1


def test_keepalive_retries_once_after_401_with_a_forced_nous_refresh():
    stale = _FakeClient("stale", fail_401=True)
    fresh = _FakeClient("fresh")
    agent = _agent(stale, _adopt_nous_key_before_expiry=lambda: False)
    forced = []

    def _force_refresh(*, force=False, **kwargs):
        forced.append(force)
        agent.client = fresh
        agent.api_key = "fresh-jwt"
        return True

    agent._try_refresh_nous_client_credentials = _force_refresh
    run = _make_run(stale)

    reply = PromptCacheWarmer(agent)._send(run)

    assert forced == [True]
    assert reply.id == "ok-fresh"
    assert run.client is fresh
    assert len(stale.chat.completions.calls) == 1
    assert len(fresh.chat.completions.calls) == 1


def test_keepalive_401_is_raised_when_the_forced_refresh_yields_no_new_client():
    stale = _FakeClient("stale", fail_401=True)
    agent = _agent(stale, _adopt_nous_key_before_expiry=lambda: False,
                   _try_refresh_nous_client_credentials=lambda **_kw: False)
    run = _make_run(stale)

    try:
        PromptCacheWarmer(agent)._send(run)
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 401
    else:
        raise AssertionError("expected the 401 to propagate")
    assert len(stale.chat.completions.calls) == 1  # exactly one retry budget: no second send on the same key


def test_a_failing_refresh_is_logged_and_the_keepalive_still_goes_out(caplog):
    client = _FakeClient("same")
    agent = _agent(client)

    def _boom():
        raise RuntimeError("auth store locked")

    agent._adopt_nous_key_before_expiry = _boom
    run = _make_run(client)

    with caplog.at_level(logging.DEBUG, logger="agent.prompt_cache_lifecycle.warmer"):
        reply = PromptCacheWarmer(agent)._send(run)

    assert reply.id == "ok-same"
    assert any("refresh" in r.getMessage().lower() for r in caplog.records), [r.getMessage() for r in caplog.records]


def test_keepalive_for_another_provider_never_touches_nous_credentials():
    client = _FakeClient("other")
    called = []
    agent = _agent(client, provider="openrouter", _adopt_nous_key_before_expiry=lambda: called.append(True) or True)
    run = _make_run(client, provider="openrouter")

    reply = PromptCacheWarmer(agent)._send(run)

    assert reply.id == "ok-other"
    assert called == []
