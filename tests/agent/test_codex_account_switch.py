"""Turn-boundary adoption of a Codex/xAI singleton account switch (#136685).

A cached gateway agent keeps the access token it was built with: after ``hermes auth add``
switches Hermes to a different account, the agent kept spending the OLD account's quota until
that stale token expired, because the 401-path refresh refuses account swaps on purpose (a pool
or explicit credential must not be silently re-attributed). ``_adopt_codex_singleton_account_
switch`` registers an agent as a singleton follower only when its JWT ``sub`` matches the
singleton's, then adopts the re-authed account at the turn boundary — never a pool fallback,
never across a different base_url, never for an unregistered credential.
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

from agent.client_lifecycle import ClientLifecycleMixin

CODEX_BASE = "https://chatgpt.com/backend-api/codex"
OPAQUE_KEY = "opaque-" + "not-a-jwt"


def _jwt(sub: str) -> str:
    def _part(payload: dict) -> str:
        return (
            base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8"))
            .decode("ascii")
            .rstrip("=")
        )

    return f"{_part({'alg': 'none'})}.{_part({'sub': sub})}.{_part({})}"


def _singleton(
    sub: str,
    *,
    key: str | None = None,
    source: str = "hermes-auth-store",
    base_url: str = CODEX_BASE,
) -> dict:
    return {
        "api_key": key if key is not None else _jwt(sub),
        "base_url": base_url,
        "source": source,
    }


def _agent(
    api_key: str,
    *,
    provider: str = "openai-codex",
    api_mode: str = "codex_responses",
    base_url: str = CODEX_BASE,
) -> SimpleNamespace:
    adopted = []

    def _record_adopt(api_key, base_url, *, reason):
        adopted.append((api_key, base_url, reason))
        return True

    return SimpleNamespace(
        provider=provider,
        api_mode=api_mode,
        api_key=api_key,
        base_url=base_url,
        _fallback_activated=False,
        _adopt_openai_credentials=_record_adopt,
        _adopted=adopted,
    )


def _patch_codex_resolver(monkeypatch, value_or_exc, calls=None):
    from hermes_cli import auth as _auth

    def _resolve(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        if isinstance(value_or_exc, Exception):
            raise value_or_exc
        return value_or_exc

    monkeypatch.setattr(_auth, "resolve_codex_runtime_credentials", _resolve)


def _run(agent) -> bool:
    return ClientLifecycleMixin._adopt_codex_singleton_account_switch(agent)


def test_registers_follower_only_when_agent_runs_the_singleton_account(monkeypatch):
    calls = []
    _patch_codex_resolver(monkeypatch, _singleton("account-a"), calls)
    matching = _agent(_jwt("account-a"))
    assert _run(matching) is False and matching._adopted == []
    assert matching._codex_singleton_followed_sub == "account-a"

    # A pool/explicit credential for a different account never registers.
    foreign = _agent(_jwt("account-b"))
    assert _run(foreign) is False
    assert (
        not hasattr(foreign, "_codex_singleton_followed_sub")
        or foreign._codex_singleton_followed_sub is None
    )


def test_adopts_reauthed_account_at_turn_boundary(monkeypatch):
    agent = _agent(_jwt("account-a"))
    _patch_codex_resolver(monkeypatch, _singleton("account-a"))
    _run(agent)  # registers the followed account

    _patch_codex_resolver(monkeypatch, _singleton("account-b"))
    assert _run(agent) is True
    assert agent._adopted == [(_jwt("account-b"), CODEX_BASE, "codex_account_switch")]
    assert agent._codex_singleton_followed_sub == "account-b"

    # Idempotent: the same singleton account never adopts twice.
    assert _run(agent) is False and agent._adopted == [
        (_jwt("account-b"), CODEX_BASE, "codex_account_switch")
    ]


def test_unregistered_credential_is_never_adopted(monkeypatch):
    # The agent's key never matched the singleton account, so a later switch must not touch it.
    agent = _agent(_jwt("explicit-key-account"))
    _patch_codex_resolver(monkeypatch, _singleton("account-a"))
    _run(agent)

    _patch_codex_resolver(monkeypatch, _singleton("account-b"))
    assert _run(agent) is False and agent._adopted == []


def test_pool_fallback_is_never_followed(monkeypatch):
    # Empty singleton store + pool fallback resolves a token with source="credential_pool" —
    # that is not a user re-auth.
    agent = _agent(_jwt("account-a"))
    _patch_codex_resolver(
        monkeypatch, _singleton("account-pool", source="credential_pool")
    )
    assert _run(agent) is False
    assert (
        not hasattr(agent, "_codex_singleton_followed_sub")
        or agent._codex_singleton_followed_sub is None
    )


def test_route_mismatch_blocks_adoption(monkeypatch):
    # A custom base_url is a deliberate routing choice; only the account may move, not the route.
    agent = _agent(_jwt("account-a"), base_url="https://proxy.example/codex")
    _patch_codex_resolver(
        monkeypatch, _singleton("account-a", base_url="https://proxy.example/codex")
    )
    _run(agent)

    _patch_codex_resolver(
        monkeypatch, _singleton("account-b", base_url="https://proxy.example/codex")
    )
    assert _run(agent) is True  # same route: switch adopted

    agent2 = _agent(_jwt("account-a"), base_url="https://proxy.example/codex")
    _patch_codex_resolver(
        monkeypatch, _singleton("account-a", base_url="https://proxy.example/codex")
    )
    _run(agent2)
    _patch_codex_resolver(monkeypatch, _singleton("account-b"))  # default route differs
    assert _run(agent2) is False and agent2._adopted == []


def test_non_jwt_singleton_never_registers(monkeypatch):
    agent = _agent(OPAQUE_KEY)
    _patch_codex_resolver(monkeypatch, _singleton("opaque", key=OPAQUE_KEY))
    assert _run(agent) is False
    assert not hasattr(agent, "_codex_singleton_followed_sub")


def test_provider_and_fallback_gates_skip_the_store(monkeypatch):
    calls = []
    _patch_codex_resolver(monkeypatch, _singleton("account-a"), calls)
    assert _run(_agent(_jwt("account-a"), provider="openai")) is False
    assert _run(_agent(_jwt("account-a"), api_mode="chat_completions")) is False
    failing_over = _agent(_jwt("account-a"))
    failing_over._fallback_activated = True
    assert _run(failing_over) is False
    assert calls == []  # gated out before any store read


def test_resolver_failure_is_nonfatal(monkeypatch):
    agent = _agent(_jwt("account-a"))
    _patch_codex_resolver(monkeypatch, RuntimeError("store unreadable"))
    assert _run(agent) is False and agent._adopted == []


def test_xai_follows_its_own_resolver(monkeypatch):
    from hermes_cli import auth as _auth

    def _resolve_xai(**kwargs):
        return {
            "api_key": _jwt("xai-account-a"),
            "base_url": "https://api.x.ai/v1",
            "source": "hermes-auth-store",
        }

    monkeypatch.setattr(_auth, "resolve_xai_oauth_runtime_credentials", _resolve_xai)
    agent = _agent(
        _jwt("xai-account-a"), provider="xai-oauth", base_url="https://api.x.ai/v1"
    )
    assert _run(agent) is False  # registers
    assert agent._codex_singleton_followed_sub == "xai-account-a"

    monkeypatch.setattr(
        _auth,
        "resolve_xai_oauth_runtime_credentials",
        lambda **kwargs: {
            "api_key": _jwt("xai-account-b"),
            "base_url": "https://api.x.ai/v1",
            "source": "hermes-auth-store",
        },
    )
    assert _run(agent) is True
    assert agent._adopted == [
        (_jwt("xai-account-b"), "https://api.x.ai/v1", "codex_account_switch")
    ]


def test_failed_adoption_keeps_following_the_old_account(monkeypatch):
    # A client rebuild failure must not advance the followed account: retry next turn.
    agent = _agent(_jwt("account-a"))
    agent._adopt_openai_credentials = lambda api_key, base_url, *, reason: False
    _patch_codex_resolver(monkeypatch, _singleton("account-a"))
    _run(agent)

    _patch_codex_resolver(monkeypatch, _singleton("account-b"))
    assert _run(agent) is False
    assert agent._codex_singleton_followed_sub == "account-a"
