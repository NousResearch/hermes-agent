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
"""

import base64
import json

from agent.client_lifecycle import _same_account_jwt_subject


def _jwt(sub: str, jti: str = "jti-1") -> str:
    def _b64(obj: dict) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{_b64({'alg': 'none'})}.{_b64({'sub': sub, 'jti': jti})}."


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
