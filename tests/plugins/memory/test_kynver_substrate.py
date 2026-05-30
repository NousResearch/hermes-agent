"""Kynver substrate default-on / health gating."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from plugins.memory.kynver.substrate import kynver_explicitly_disabled, substrate_active


def test_substrate_inactive_when_explicitly_disabled(monkeypatch, tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "KYNVER_API_URL=https://example.test\nKYNVER_API_KEY=secret\nKYNVER_AGENT_OS_SLUG=forge\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("plugins.memory.kynver.agentos_bridge._active_env_path", lambda: env_path)
    monkeypatch.setenv("KYNVER_DISABLED", "1")
    assert substrate_active() is False


def test_substrate_active_when_healthy(monkeypatch, tmp_path: Path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "KYNVER_API_URL=https://example.test\nKYNVER_API_KEY=secret\nKYNVER_AGENT_OS_SLUG=forge\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("plugins.memory.kynver.agentos_bridge._active_env_path", lambda: env_path)
    monkeypatch.setattr(
        "plugins.memory.kynver.substrate.probe_agentos_health",
        lambda _client=None: True,
    )
    assert kynver_explicitly_disabled() is False
    assert substrate_active() is True
