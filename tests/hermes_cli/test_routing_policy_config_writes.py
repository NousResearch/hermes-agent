import copy

import pytest


def test_save_config_rejects_denied_route_before_write(tmp_path, monkeypatch):
    from hermes_cli.config import save_config
    from hermes_cli.routing_policy import RoutingPolicyError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    original = "routing_policy:\n  enabled: true\n  deny:\n    providers: [openrouter]\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(RoutingPolicyError):
        save_config({
            "routing_policy": {"enabled": True, "deny": {"providers": ["openrouter"]}},
            "model": {"provider": "openrouter", "default": "safe-looking"},
        }, strip_defaults=False)

    assert path.read_text(encoding="utf-8") == original


def test_set_config_value_rejects_denied_route_before_write(tmp_path, monkeypatch):
    from hermes_cli.config import set_config_value
    from hermes_cli.routing_policy import RoutingPolicyError

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    original = "routing_policy:\n  enabled: true\n  deny:\n    providers: [openrouter]\nmodel:\n  provider: openai\n  default: allowed\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(RoutingPolicyError):
        set_config_value("model.provider", "openrouter")

    assert path.read_text(encoding="utf-8") == original