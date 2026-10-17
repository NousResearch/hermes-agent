"""Credential-pool discovery must not activate a disabled Copilot provider."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_disabled_copilot_provider_skips_token_resolution_and_exchange(tmp_path, monkeypatch):
    """A disabled provider must not shell out or call GitHub during pool discovery."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "providers:\n  copilot:\n    enabled: false\n",
        encoding="utf-8",
    )
    (home / "auth.json").write_text(
        '{"version": 1, "credential_pool": {}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import copilot_auth

    resolve_token = MagicMock(return_value=("gho_test", "GH_TOKEN"))
    exchange_token = MagicMock(return_value=("tid=test", None))
    monkeypatch.setattr(copilot_auth, "resolve_copilot_token", resolve_token)
    monkeypatch.setattr(copilot_auth, "get_copilot_api_token", exchange_token)

    from agent.credential_pool import load_pool

    assert load_pool("copilot").entries() == []
    resolve_token.assert_not_called()
    exchange_token.assert_not_called()
