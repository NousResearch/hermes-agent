"""``token_expires_at``: when a pool entry's access token expires, from its own fields, without I/O.

``hermes usage`` and the model picker use it to explain an account whose usage read was refused.
"""

from __future__ import annotations

import base64
import json

from agent.credential_pool import PooledCredential
from agent.credential_pool_timestamps import token_expires_at


def _jwt(claims: dict) -> str:
    def part(data: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")

    return f"{part({'alg': 'none'})}.{part(claims)}.signature"


def _entry(access_token: str, **fields) -> PooledCredential:
    return PooledCredential.from_dict("openai-codex", {
        "id": "e1", "label": "e1", "auth_type": "oauth", "priority": 0, "source": "manual",
        "access_token": access_token, **fields})


def test_a_jwt_access_token_expires_at_its_exp_claim():
    assert token_expires_at(_entry(_jwt({"exp": 1_800_000_000}))) == 1_800_000_000


def test_the_rows_own_expiry_wins_over_the_token():
    entry = _entry(_jwt({"exp": 1_800_000_000}), expires_at_ms=1_700_000_000_000)

    assert token_expires_at(entry) == 1_700_000_000


def test_an_opaque_token_without_an_expiry_field_is_unknown():
    assert token_expires_at(_entry("opaque-access-token")) is None
