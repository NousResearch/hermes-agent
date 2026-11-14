"""The Codex auxiliary-availability warning must name the real reason.

A pool whose only Codex credential sits in a 429/quota cooldown yields no client exactly like
absent credentials do, so the branch emitted one constant message: "no Codex OAuth token found
(run: hermes model)". Re-authenticating cannot lift a usage-limit cooldown, and the message hid
the facts that are actionable (the reset time, or adding a second credential). These tests pin
the reason per credential state through the real pool reader against a temp ``HERMES_HOME``, and
pin that the missing-credential case keeps its original wording.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pytest

import hermes_cli.auth as auth_mod
from agent.auxiliary_client import _ResolveRequest, _resolve_openai_codex_branch

CODEX_LOGGER = "agent.auxiliary_client"


@pytest.fixture
def codex_warnings():
    """Capture WARNING records from the auxiliary client's logger."""
    logger = logging.getLogger(CODEX_LOGGER)
    previous_level = logger.level
    seen: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
            seen.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        yield seen
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


@pytest.fixture(autouse=True)
def _no_upstream_quota_probe(monkeypatch):
    """Pin the Codex usage-endpoint probe off: these tests assert wording, not recovery."""
    monkeypatch.setattr(auth_mod, "_probe_codex_quota_restored", lambda *a, **k: False)


def _write_auth_store(entries: list[dict]) -> None:
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    store = {"version": 1, "providers": {}, "credential_pool": {"openai-codex": entries}}
    (home / "auth.json").write_text(json.dumps(store), encoding="utf-8")


def _pool_entry(**overrides) -> dict:
    now = time.time()
    entry = {
        "id": "cred-1",
        "label": "codex-1",
        "auth_type": "oauth",
        "priority": 0,
        "source": "manual:device_code",
        "access_token": "tok-1",
        "refresh_token": "refresh-1",
        "last_status": "ok",
        "last_status_at": now,
    }
    entry.update(overrides)
    return entry


def _quota_cooldown_entry(reset_in: float = 2 * 24 * 3600) -> dict:
    now = time.time()
    return _pool_entry(
        last_status="exhausted",
        last_status_at=now,
        last_error_code=429,
        last_error_reason="usage_limit_reached",
        last_error_message="The usage limit has been reached",
        last_error_reset_at=now + reset_in,
    )


def _codex_request(*, raw_codex: bool = False) -> _ResolveRequest:
    return _ResolveRequest(
        provider="openai-codex",
        original_provider="openai-codex",
        model="codex-mini-latest",
        async_mode=False,
        raw_codex=raw_codex,
        explicit_base_url=None,
        explicit_api_key=None,
        api_mode=None,
        main_runtime=None,
        is_vision=False,
        task=None,
    )


@pytest.mark.parametrize("raw_codex", [False, True])
def test_quota_cooldown_names_the_cooldown_instead_of_a_missing_login(codex_warnings, raw_codex):
    """A 429-quota credential is not an absent credential: name the cooldown and its reset."""
    _write_auth_store([_quota_cooldown_entry()])

    assert _resolve_openai_codex_branch(_codex_request(raw_codex=raw_codex)) == (None, None)

    assert len(codex_warnings) == 1, codex_warnings
    warning = codex_warnings[0]
    assert "cooling down" in warning, warning
    assert "no Codex OAuth token found" not in warning, warning


def test_missing_credentials_keep_the_re_authentication_advice(codex_warnings):
    """Genuinely absent credentials keep the pre-existing, actionable wording."""
    _write_auth_store([])

    assert _resolve_openai_codex_branch(_codex_request()) == (None, None)

    assert len(codex_warnings) == 1, codex_warnings
    assert "no Codex OAuth token found (run: hermes model)" in codex_warnings[0], codex_warnings


def test_a_permanently_failed_credential_is_not_called_a_cooldown(codex_warnings):
    """A revoked grant must keep pointing at re-authentication — time cannot clear it."""
    _write_auth_store([_pool_entry(
        last_status="dead",
        last_status_at=time.time(),
        last_error_code=401,
        last_error_reason="invalid_grant",
        last_error_message="The refresh token was revoked",
    )])

    assert _resolve_openai_codex_branch(_codex_request()) == (None, None)

    assert len(codex_warnings) == 1, codex_warnings
    assert "no Codex OAuth token found (run: hermes model)" in codex_warnings[0], codex_warnings
