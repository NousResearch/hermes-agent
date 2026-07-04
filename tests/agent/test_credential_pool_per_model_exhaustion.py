"""Per-model exhaustion scoping for shared OAuth credentials (issue #47986).

The ChatGPT-account Codex endpoint serves multiple models (``gpt-5.5`` and
``gpt-5.3-codex-spark``) from a SINGLE OAuth credential, but each model has its
own independent usage-limit window.  Previously, when ``gpt-5.5`` hit its
usage limit Hermes marked the whole pool credential ``exhausted``, which also
locked out Spark even though Spark still had quota.  These tests lock in the
scoped behavior: a block recorded for one model must not hide the credential
from a sibling model, while unscoped callers keep the historical
provider-wide block.
"""

from __future__ import annotations

import json
import time

import pytest


def _write_auth_store(tmp_path, payload: dict) -> None:
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps(payload))


def _single_codex_entry(**overrides) -> dict:
    entry = {
        "id": "codex-1",
        "label": "chatgpt",
        "auth_type": "oauth",
        "priority": 0,
        "source": "manual",
        "access_token": "codex-oauth-token",
    }
    entry.update(overrides)
    return {
        "version": 1,
        "credential_pool": {"openai-codex": [entry]},
    }


def test_sibling_model_selectable_when_other_model_exhausted(tmp_path, monkeypatch):
    """gpt-5.5 usage-limit block must NOT hide the credential from Spark."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        _single_codex_entry(
            last_status="exhausted",
            last_status_at=time.time(),
            last_error_code=429,
            last_error_reason="usage_limit_reached",
            last_error_model="gpt-5.5",
        ),
    )

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")

    # Same model that hit the limit → still blocked.
    assert pool.select(requested_model="gpt-5.5") is None
    # Sibling model on the same credential → available.
    spark = pool.select(requested_model="gpt-5.3-codex-spark")
    assert spark is not None
    assert spark.id == "codex-1"


def test_unscoped_select_preserves_provider_wide_block(tmp_path, monkeypatch):
    """A caller that supplies no model must see the historical (safe) block."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        _single_codex_entry(
            last_status="exhausted",
            last_status_at=time.time(),
            last_error_code=429,
            last_error_reason="usage_limit_reached",
            last_error_model="gpt-5.5",
        ),
    )

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    assert pool.select() is None


def test_scoped_block_does_not_clear_exhausted_flag(tmp_path, monkeypatch):
    """Serving Spark must not wipe the genuine gpt-5.5 cooldown on the entry."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    reset_at = time.time() + 3600
    _write_auth_store(
        tmp_path,
        _single_codex_entry(
            last_status="exhausted",
            last_status_at=time.time(),
            last_error_code=429,
            last_error_reason="usage_limit_reached",
            last_error_reset_at=reset_at,
            last_error_model="gpt-5.5",
        ),
    )

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    spark = pool.select(requested_model="gpt-5.3-codex-spark")
    assert spark is not None
    # The gpt-5.5 exhaustion is untouched — it is still blocked.
    assert pool.select(requested_model="gpt-5.5") is None


def test_mark_exhausted_and_rotate_records_model(tmp_path, monkeypatch):
    """The write path must persist which model actually hit the limit."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, _single_codex_entry())

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"reason": "usage_limit_reached"},
        model_id="gpt-5.5",
    )

    persisted = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entry = persisted["credential_pool"]["openai-codex"][0]
    assert entry["last_status"] == "exhausted"
    assert entry["last_error_model"] == "gpt-5.5"

    # Reload and confirm the scoping survives a round-trip.
    pool2 = load_pool("openai-codex")
    assert pool2.select(requested_model="gpt-5.3-codex-spark") is not None
    assert pool2.select(requested_model="gpt-5.5") is None


def test_two_distinct_models_exhausted_widens_to_provider_wide(tmp_path, monkeypatch):
    """When a SECOND, different model is also exhausted we can only track one,
    so the block must conservatively widen back to provider-wide."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, _single_codex_entry())

    from agent.credential_pool import load_pool

    pool = load_pool("openai-codex")
    pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"reason": "usage_limit_reached"},
        model_id="gpt-5.5",
    )
    # Spark still works at this point.
    assert pool.select(requested_model="gpt-5.3-codex-spark") is not None

    # Now Spark ALSO hits its limit on the same credential.
    pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"reason": "usage_limit_reached"},
        model_id="gpt-5.3-codex-spark",
    )

    persisted = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    entry = persisted["credential_pool"]["openai-codex"][0]
    # Widened: no single model can be recorded, so fall back to provider-wide.
    assert entry.get("last_error_model") in (None, "")

    pool2 = load_pool("openai-codex")
    assert pool2.select(requested_model="gpt-5.5") is None
    assert pool2.select(requested_model="gpt-5.3-codex-spark") is None
