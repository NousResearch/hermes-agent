"""Regression tests: `hermes auth status openrouter` agrees with inference.

OpenRouter is absent from PROVIDER_REGISTRY on purpose, so get_auth_status()
fell through to ``{"logged_in": False}`` even with a valid credential — while
inference used it fine via _openrouter_auto_detected(). A bespoke status builder
(get_openrouter_auth_status) now mirrors those detection rungs.
"""

import uuid

import pytest


@pytest.fixture(autouse=True)
def _clean_inference_env(monkeypatch):
    """Strip credential-shaped env vars so each source is tested in isolation."""
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def _seed_openrouter_pool(token: str = "sk-or-FAKEKEY123") -> None:
    """Mimic `hermes auth add openrouter <token>` — a manual pool entry."""
    from agent.credential_pool import (
        AUTH_TYPE_API_KEY,
        SOURCE_MANUAL,
        PooledCredential,
        load_pool,
    )

    pool = load_pool("openrouter")
    pool.add_entry(
        PooledCredential(
            provider="openrouter",
            id=uuid.uuid4().hex[:6],
            label="api-key-1",
            auth_type=AUTH_TYPE_API_KEY,
            priority=0,
            source=SOURCE_MANUAL,
            access_token=token,
            base_url="https://openrouter.ai/api/v1",
        )
    )


def test_status_logged_in_from_pool(tmp_path, monkeypatch):
    """The reported bug: pooled key showed `auth list` but `auth status` said logged out."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)
    _seed_openrouter_pool()

    from hermes_cli.auth import get_auth_status

    status = get_auth_status("openrouter")
    assert status["logged_in"] is True
    assert status["key_source"] == "credential_pool:openrouter"


def test_status_logged_out_with_nothing(tmp_path, monkeypatch):
    """No env key, no pool entry — still logged out (no false positive)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir(parents=True, exist_ok=True)

    from hermes_cli.auth import get_auth_status

    assert get_auth_status("openrouter")["logged_in"] is False


def test_status_logged_in_from_env(monkeypatch):
    """OPENROUTER_API_KEY alone reports logged in with the env source."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-ENVKEY123")

    from hermes_cli.auth import get_auth_status

    status = get_auth_status("openrouter")
    assert status["logged_in"] is True
    assert status["key_source"] == "OPENROUTER_API_KEY"


def test_status_logged_in_from_legacy_openai_key(monkeypatch):
    """An sk-or- key in OPENAI_API_KEY (legacy home) counts; a real OpenAI key doesn't."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-or-v1-LEGACY123")

    from hermes_cli.auth import get_auth_status

    assert get_auth_status("openrouter")["logged_in"] is True

    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-REALOPENAIKEY")
    assert get_auth_status("openrouter")["logged_in"] is False
