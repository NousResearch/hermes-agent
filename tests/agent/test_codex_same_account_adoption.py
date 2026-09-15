"""Regression tests: same-account singleton adoption in ``_try_refresh_codex_client_credentials``.

Bug (observed on a long-lived gateway, 2026-09-11): a cached gateway agent built its
OpenAI client with the xai-oauth singleton token, a sibling process rotated the grant,
and every later turn sent the dead bearer. The one-shot refresh COULD have healed it —
``_try_refresh_codex_client_credentials(force=True)`` resolves the singleton — but the
anti-swap guard compared raw token strings, saw ``singleton != active``, and skipped the
refresh even though both JWTs belong to the SAME account (identical ``sub``). The turn
then aborted as non-retryable while the auth store held a working grant.

The fix: when both JWTs carry the same ``sub`` claim, adopting the singleton's newer
token is not an account swap — it is a rotation catch-up. The conservative no-swap path
stays for different-account (or unparseable-sub) pairs; these tests pin that contract.

v2 (2026-09-15): a same-account candidate that is ALREADY EXPIRED must not be adopted —
adopting it burns the one recovery retry on a bearer that cannot validate (root store
kept a dead chain while a sibling profile's shadow held the live one). Fall through to a
forced refresh instead.
"""

import base64
import json
import time
import unittest.mock
from types import SimpleNamespace

from agent.client_lifecycle import _jwt_expires_within, _same_account_jwt_subject


def _b64(obj: dict) -> str:
    raw = json.dumps(obj).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _jwt(sub: str, jti: str = "jti-1", exp: float | None = None) -> str:
    payload = {"sub": sub, "jti": jti}
    if exp is not None:
        payload["exp"] = exp
    return f"{_b64({'alg': 'none'})}.{_b64(payload)}."


class TestSameAccountJwtSubject:
    def test_same_subject_true(self):
        assert _same_account_jwt_subject(_jwt("user-a"), _jwt("user-a", jti="new")) is True

    def test_different_subject_false(self):
        assert _same_account_jwt_subject(_jwt("user-a"), _jwt("user-b")) is False

    def test_missing_subject_false(self):
        """A JWT without ``sub`` (or a non-JWT string) cannot prove same-account."""
        assert _same_account_jwt_subject("not-a-jwt", _jwt("user-a")) is False
        assert _same_account_jwt_subject(_jwt("user-a"), "not-a-jwt") is False

    def test_empty_inputs_false(self):
        assert _same_account_jwt_subject("", "") is False


class TestJwtExpiresWithin:
    def test_expired_true(self):
        assert _jwt_expires_within(_jwt("a", exp=time.time() - 60)) is True

    def test_future_exp_false(self):
        assert _jwt_expires_within(_jwt("a", exp=time.time() + 3600)) is False

    def test_non_jwt_false(self):
        """Non-JWT strings cannot be proven dead — candidate policy unchanged."""
        assert _jwt_expires_within("sk-not-a-jwt") is False
        assert _jwt_expires_within("") is False


def _fake_agent(api_key: str) -> SimpleNamespace:
    """Minimal AIAgent stand-in exposing what _try_refresh_codex_client_credentials touches."""
    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="xai-oauth",
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        _client_replacements=[],
    )

    def try_refresh(*, force: bool = True) -> bool:
        from agent import client_lifecycle as CL

        # Re-implement nothing: call the REAL method on the real class via __new__ trickery
        # is overkill — instead exec the unbound function with our namespace.
        return CL.ClientLifecycleMixin._try_refresh_codex_client_credentials(
            agent, force=force
        )

    agent.try_refresh = try_refresh
    agent._adopt_openai_credentials = lambda key, url, reason="": (
        setattr(agent, "api_key", key),
        setattr(agent, "base_url", url),
        agent._client_replacements.append(reason),
    ) and True
    agent._sync_client_kwargs_credentials = lambda: None
    return agent


class TestAdoptionContract:
    def test_expired_candidate_falls_through_to_forced_refresh(self):
        """Same-sub candidate that is already expired: no adoption, forced refresh instead."""
        now = time.time()
        old_jwt = _jwt("acct", jti="old", exp=now - 3600)
        cand_jwt = _jwt("acct", jti="cand", exp=now - 60)   # same sub, also dead
        fresh_jwt = _jwt("acct", jti="fresh", exp=now + 3600)

        resolve_calls = []

        def fake_resolve(**kwargs):
            resolve_calls.append(kwargs)
            if kwargs.get("refresh_if_expiring") is False:
                return {"api_key": cand_jwt, "base_url": "https://api.x.ai/v1"}
            return {"api_key": fresh_jwt, "base_url": "https://api.x.ai/v1"}

        agent = _fake_agent(old_jwt)
        import hermes_cli.auth as A
        with unittest.mock.patch.object(
            A, "resolve_xai_oauth_runtime_credentials", fake_resolve
        ):
            ok = agent.try_refresh(force=True)

        assert ok is True
        assert agent.api_key == fresh_jwt
        assert len(resolve_calls) == 2
        assert all("adoption" not in r for r in agent._client_replacements), (
            "expired candidate must NOT be adopted"
        )

    def test_live_candidate_is_adopted(self):
        """Same-sub candidate with future exp: adopted without a forced refresh."""
        now = time.time()
        old_jwt = _jwt("acct", jti="old", exp=now - 3600)
        cand_jwt = _jwt("acct", jti="cand", exp=now + 3600)

        def fake_resolve(**kwargs):
            return {"api_key": cand_jwt, "base_url": "https://api.x.ai/v1"}

        agent = _fake_agent(old_jwt)
        import hermes_cli.auth as A
        with unittest.mock.patch.object(
            A, "resolve_xai_oauth_runtime_credentials", fake_resolve
        ):
            ok = agent.try_refresh(force=True)

        assert ok is True
        assert agent.api_key == cand_jwt
        assert agent._client_replacements == ["xai-oauth_same_account_adoption"]
