from __future__ import annotations

import pytest

from hermes_cli.dashboard_embed import (
    EmbedPolicyError,
    configured_embed_parent_origins,
    resolve_embedded_profile,
)


def _config():
    return {
        "dashboard": {
            "embed_parent_origins": ["https://console.runi.services"],
            "embed_profiles": {
                "console-hermes": "default",
                "console-wolf": "wolf",
            },
        }
    }


def test_embed_profile_is_pinned_by_server_policy():
    assert resolve_embedded_profile(_config(), "console-wolf", "wolf") == "wolf"
    assert resolve_embedded_profile(_config(), "console-hermes", None) is None

    with pytest.raises(EmbedPolicyError, match="does not permit profile"):
        resolve_embedded_profile(_config(), "console-wolf", "torkil")


def test_unknown_embed_id_fails_closed():
    with pytest.raises(EmbedPolicyError, match="not configured"):
        resolve_embedded_profile(_config(), "console-torkil", "torkil")


def test_parent_origins_are_exact_http_origins():
    config = _config()
    config["dashboard"]["embed_parent_origins"].extend(
        [
            "https://console.runi.services/path",
            "javascript:alert(1)",
            "https://evil.example%0d%0aX-Frame-Options:ALLOWALL",
            "https://evil.example;frame-ancestors*",
            "https://console.runi.services",
        ]
    )
    assert configured_embed_parent_origins(config) == (
        "https://console.runi.services",
    )


def test_websocket_embed_scope_uses_live_dashboard_config(monkeypatch):
    from hermes_cli import web_server

    monkeypatch.setattr(web_server, "load_config", _config)
    assert web_server._resolve_pty_embed_profile("console-wolf", "wolf") == "wolf"
    with pytest.raises(EmbedPolicyError):
        web_server._resolve_pty_embed_profile("console-wolf", "torkil")
