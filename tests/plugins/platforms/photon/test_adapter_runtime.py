"""Runtime helper tests for PhotonAdapter."""
from __future__ import annotations

from pathlib import Path

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon import adapter as adapter_mod
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    monkeypatch.delenv("PHOTON_WEBHOOK_SECRET", raising=False)
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


def test_active_hermes_home_label_uses_current_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "hermes-profile"
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert adapter_mod._active_hermes_home_label() == str(home)


def test_managed_tunnel_autostart_skips_user_owned_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PHOTON_WEBHOOK_PUBLIC_URL",
        "https://example.com/photon/webhook",
    )
    adapter = _make_adapter(monkeypatch)

    assert adapter._should_autostart_tunnel() is False


def test_managed_tunnel_autostart_allows_trycloudflare_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PHOTON_WEBHOOK_PUBLIC_URL",
        "https://current.trycloudflare.com/photon/webhook",
    )
    adapter = _make_adapter(monkeypatch)

    assert adapter._should_autostart_tunnel() is True


def test_delete_stale_managed_webhooks_keeps_current_and_manual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)
    deleted: list[str] = []

    def fake_delete_webhook(
        project_id: str,
        project_secret: str,
        *,
        webhook_id: str,
    ) -> None:
        assert project_id == "test-project-id"
        assert project_secret == "test-project-secret"
        deleted.append(webhook_id)

    monkeypatch.setattr(adapter_mod, "delete_webhook", fake_delete_webhook)

    hooks = [
        {
            "id": "old-managed",
            "webhookUrl": "https://old.trycloudflare.com/photon/webhook",
        },
        {
            "id": "current-managed",
            "webhookUrl": "https://current.trycloudflare.com/photon/webhook",
        },
        {
            "id": "manual",
            "webhookUrl": "https://example.com/photon/webhook",
        },
    ]

    remaining = adapter._delete_stale_managed_webhooks(
        hooks,
        keep_url="https://current.trycloudflare.com/photon/webhook",
    )

    assert deleted == ["old-managed"]
    assert [hook["id"] for hook in remaining] == ["current-managed", "manual"]
