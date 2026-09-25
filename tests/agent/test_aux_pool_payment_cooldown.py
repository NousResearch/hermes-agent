"""Aux pool recovery must size the bench by what actually failed. (#119533)

``_recover_provider_pool`` used to mark every payment/quota-shaped aux error as a bare 402
with no ``failure_reason``. For errors whose billing nature is only inferred from the body
(non-402 status, or no status at all) that turned one transient/OpenRouter-side response into
a full hour bench of the failed entry. True HTTP 402 keeps the hour bench (genuine depletion
re-fails); inferred billing cools down briefly like the main loop's ``billing_unverified``
path.

A *sole* credential never reaches the bench from the auxiliary path at all
(``test_auxiliary_pool_isolation.py``), so these pin the cooldown *sizing* on a pool that has
a rotation target.
"""

from __future__ import annotations

import json
import time

_FAILED_KEY = "sk-or-test-key-1"
_SIBLING_KEY = "sk-or-test-key-2"


class _AuxError(Exception):
    def __init__(self, message: str, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _write_auth_store(tmp_path) -> None:
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"openrouter": [
            {
                "id": f"cred-{index}",
                "label": f"cred-{index}",
                "auth_type": "api_key",
                "priority": index,
                "source": "manual",
                "access_token": key,
                "base_url": "https://openrouter.ai/api/v1",
            }
            for index, key in enumerate((_FAILED_KEY, _SIBLING_KEY))
        ]}}),
        encoding="utf-8",
    )


def _bench_seconds(monkeypatch, tmp_path, message: str, status_code) -> float:
    """Drive the real aux recovery path; return how long the failed entry sits out."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _write_auth_store(tmp_path)
    from agent.auxiliary_client import _recover_provider_pool
    _recover_provider_pool(
        "openrouter", _AuxError(message, status_code), failed_api_key=_FAILED_KEY,
    )
    from agent.credential_pool import _exhausted_until, load_pool
    entry = next(e for e in load_pool("openrouter").entries() if e.runtime_api_key == _FAILED_KEY)
    until = _exhausted_until(entry, sole_credential=False)
    assert until is not None, "failed entry was not benched at all"
    return until - time.time()


def test_aux_unverified_payment_error_benches_briefly(tmp_path, monkeypatch):
    # No HTTP status, only a quota-shaped body: billing is inferred, the credential may be
    # healthy — must not sit out a full hour.
    wait = _bench_seconds(
        monkeypatch, tmp_path, "quota exceeded: too many tokens per day, retry later", None,
    )
    assert wait <= 300, f"benched {wait:.0f}s on an unverified quota error"


def test_aux_true_402_keeps_full_bench(tmp_path, monkeypatch):
    wait = _bench_seconds(
        monkeypatch, tmp_path, "402 Payment Required: insufficient credits", 402,
    )
    assert wait > 300, f"genuine 402 should keep the hour bench, got {wait:.0f}s"


def test_aux_403_quota_body_stays_transient(tmp_path, monkeypatch):
    # Guard: a quota-bodied 403 must never be upgraded to a billing bench.
    wait = _bench_seconds(
        monkeypatch, tmp_path, "key limit exceeded: quota exceeded for this key", 403,
    )
    assert wait <= 300, f"benched {wait:.0f}s on a 403 quota error"
