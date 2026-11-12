"""Credential order for explicit custom endpoints (regression for #145452).

A route's own credential — the same-origin main key or the ``providers.<id>`` entry owning the
URL — must beat an ambient ``OPENAI_API_KEY``, which stays only the documented last resort for a
URL with no credential of its own, and then only on an OpenAI host or the exact ``OPENAI_BASE_URL``
it is paired with (#91308). Real config.yaml under a temp ``HERMES_HOME``, resolved through the
task route; HTTP is faked.
"""

from unittest.mock import MagicMock, patch

import pytest

import hermes_yaml as yaml

from agent.auxiliary_client import resolve_provider_client


def test_main_credential_beats_ambient_openai_key_on_the_same_origin(tmp_path, monkeypatch):
    """A live OPENAI_API_KEY is not the route's credential. The main endpoint's key for the
    same origin wins, so a self-hosted proxy never receives an OpenAI key (regression for
    #145452)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    base_url = "https://proxy.example.test/v1"
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "base_url": base_url,
                  "api_key": "sk-main-route-key", "default": "main-model"},
    }))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-openai")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    captured: dict = {}

    def _capture_create(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    import agent.auxiliary_client as ac

    with patch.object(ac, "_create_openai_client", side_effect=_capture_create):
        resolve_provider_client(
            "custom", model="main-model", explicit_base_url=base_url, explicit_api_key=None,
        )

    assert captured.get("api_key") == "sk-main-route-key", (
        "the route's own same-origin credential must beat the ambient OPENAI_API_KEY, got: "
        + repr(captured.get("api_key"))
    )


def test_declared_provider_key_env_beats_ambient_openai_key(tmp_path, monkeypatch):
    """A ``providers.<id>`` block owning the explicit base_url supplies the credential even
    when the main model lives at another origin and OPENAI_API_KEY is exported. The OpenAI
    key must never be sent to the third-party endpoint (regression for #145452)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    base_url = "https://proxy.example.test/v1"
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "litellm", "base_url": base_url, "default": "my-combo"},
        "providers": {
            "litellm": {
                "name": "my litellm", "base_url": base_url, "key_env": "ROUTE_KEY",
                "models": {"my-combo": {"context_length": 500000}},
            },
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("ROUTE_KEY", "sk-route-credential")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-openai")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    captured: dict = {}

    def _capture_create(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    import agent.auxiliary_client as ac

    with patch.object(ac, "_create_openai_client", side_effect=_capture_create):
        resolve_provider_client(
            "custom", model="my-combo", explicit_base_url=base_url, explicit_api_key=None,
        )

    assert captured.get("api_key") == "sk-route-credential"


@pytest.mark.parametrize(("paired_base", "expected"), [
    ("https://other.example.net/v1", "ambient-openai-key"),
    (None, "no-key-required"),
])
def test_route_without_credential_falls_back_to_openai_key_never_the_main_key(
    tmp_path, monkeypatch, paired_base, expected,
):
    """A cross-origin base_url with no credential of its own gets the documented
    OPENAI_API_KEY last resort only when paired with it via OPENAI_BASE_URL (#91308) —
    never the main route's key (regression for #145452)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "base_url": "https://main.example.test/v1",
                  "api_key": "main-route-key", "default": "main-model"},
    }))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key")
    if paired_base:
        monkeypatch.setenv("OPENAI_BASE_URL", paired_base)
    else:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    captured: dict = {}

    def _capture_create(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    import agent.auxiliary_client as ac

    with patch.object(ac, "_create_openai_client", side_effect=_capture_create):
        resolve_provider_client(
            "custom", model="other-model",
            explicit_base_url="https://other.example.net/v1", explicit_api_key=None,
        )

    assert captured.get("api_key") == expected
