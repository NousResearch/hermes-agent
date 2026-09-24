"""Aux pool recovery must size the bench by what actually failed. (#119533)

``_recover_provider_pool`` used to mark every payment/quota-shaped aux error
as a bare 402 with no ``failure_reason``. For errors whose billing nature is
only inferred from the body (non-402 status, or no status at all) that turned
one transient/OpenRouter-side response into a full hour bench of the sole
pool entry — and every later turn died at client init with ``No LLM provider
configured``. True HTTP 402 keeps the hour bench (genuine depletion
re-fails); inferred billing cools down briefly like the main loop's
``billing_unverified`` path.
"""

from __future__ import annotations

import json
import time

_TEST_KEY = "sk-or-test-key-1"


def _write_auth_store(tmp_path, payload: dict) -> None:
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps(payload), encoding="utf-8")


def _pool_payload() -> dict:
    return {"version": 1, "credential_pool": {"openrouter": [{
        "id": "cred-1",
        "label": "cred-1",
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": _TEST_KEY,
        "base_url": "https://openrouter.ai/api/v1",
    }]}}


class _AuxError(Exception):
    def __init__(self, message: str, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _recover(monkeypatch, tmp_path, message: str, status_code):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _write_auth_store(tmp_path, _pool_payload())
    from agent.auxiliary_client import _recover_provider_pool
    _recover_provider_pool(
        "openrouter", _AuxError(message, status_code), failed_api_key=_TEST_KEY,
    )
    from agent.credential_pool import load_pool
    return load_pool("openrouter")


def test_aux_unverified_payment_error_benches_sole_entry_briefly(tmp_path, monkeypatch):
    # No HTTP status, only a quota-shaped body: billing is inferred, the
    # credential may be healthy — must not sit out a full hour.
    pool = _recover(
        monkeypatch, tmp_path,
        "quota exceeded: too many tokens per day, retry later", None,
    )
    assert not pool.has_available()
    wait = pool.next_available_at() - time.time()
    assert wait <= 300, f"sole entry benched {wait:.0f}s on unverified quota error"


def test_aux_true_402_keeps_full_bench(tmp_path, monkeypatch):
    pool = _recover(
        monkeypatch, tmp_path,
        "402 Payment Required: insufficient credits", 402,
    )
    assert not pool.has_available()
    wait = pool.next_available_at() - time.time()
    assert wait > 300, f"genuine 402 should keep the hour bench, got {wait:.0f}s"


def test_aux_403_quota_body_stays_transient(tmp_path, monkeypatch):
    # Guard: a quota-bodied 403 must never be upgraded to a billing bench.
    pool = _recover(
        monkeypatch, tmp_path,
        "key limit exceeded: quota exceeded for this key", 403,
    )
    assert not pool.has_available()
    wait = pool.next_available_at() - time.time()
    assert wait <= 300, f"sole entry benched {wait:.0f}s on 403 quota error"
