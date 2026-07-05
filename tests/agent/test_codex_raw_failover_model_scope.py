"""Regression for the raw-Codex failover path skipping a sibling model.

Issue #47986 scoped credential exhaustion per model (``last_error_model``), so a
shared ChatGPT-account OAuth credential that hit ``gpt-5.5``'s usage limit should
still serve ``gpt-5.3-codex-spark`` (independent quota window).

The pool/adapter paths were fixed, but the *raw* Codex path used by the main
agent loop (``resolve_provider_client(..., raw_codex=True)`` ->
``_read_codex_access_token()``) still selected the pool entry WITHOUT passing the
requested model.  So during failover the whole ``openai-codex`` provider was
skipped and the chain fell through to the next provider (e.g. Grok), even though
Spark was usable.  These tests pin the raw path to the per-model scoping.
"""
import json

import agent.auxiliary_client as ac


def _write_scoped_exhausted_pool(hermes_home, blocked_model="gpt-5.5"):
    """auth.json with a single openai-codex pool entry exhausted for one model."""
    import time as _time

    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_store = {
        "version": 1,
        "providers": {},  # no singleton — force the pool path
        "credential_pool": {
            "openai-codex": [
                {
                    "source": "device_code",
                    "access_token": "shared-oauth-token",
                    "auth_type": "oauth",
                    "last_status": "exhausted",
                    "last_error_code": 429,
                    "last_error_reason": "usage_limit_reached",
                    "last_error_model": blocked_model,
                    "last_error_reset_at": _time.time() + 3600,  # 1h cooldown
                },
            ],
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(auth_store))


def test_raw_codex_failover_serves_sibling_when_scoped_exhausted(tmp_path, monkeypatch):
    """Spark is reachable via the raw path when the block is scoped to gpt-5.5."""
    hermes_home = tmp_path / "hermes"
    _write_scoped_exhausted_pool(hermes_home, blocked_model="gpt-5.5")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    client, resolved = ac.resolve_provider_client(
        "openai-codex", model="gpt-5.3-codex-spark", raw_codex=True,
    )

    assert client is not None, "Spark should be usable while only gpt-5.5 is exhausted"
    assert resolved and "spark" in resolved.lower()


def test_raw_codex_failover_still_blocks_the_exhausted_model(tmp_path, monkeypatch):
    """The requested model that actually hit its limit stays blocked."""
    hermes_home = tmp_path / "hermes"
    _write_scoped_exhausted_pool(hermes_home, blocked_model="gpt-5.5")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    client, resolved = ac.resolve_provider_client(
        "openai-codex", model="gpt-5.5", raw_codex=True,
    )

    assert client is None, "gpt-5.5 must remain blocked while exhausted"
    assert resolved is None


def test_read_codex_access_token_threads_requested_model(tmp_path, monkeypatch):
    """Direct unit check: the token reader honors the requested model."""
    hermes_home = tmp_path / "hermes"
    _write_scoped_exhausted_pool(hermes_home, blocked_model="gpt-5.5")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Sibling model → token surfaces from the scoped-exhausted credential.
    assert ac._read_codex_access_token("gpt-5.3-codex-spark") == "shared-oauth-token"
    # The exhausted model → no token (pool blocked, no singleton fallback).
    assert ac._read_codex_access_token("gpt-5.5") is None
