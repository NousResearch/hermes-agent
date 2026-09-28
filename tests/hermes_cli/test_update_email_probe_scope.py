"""Update diagnostics must see the selected home's stored email configuration."""
import os

import pytest


@pytest.mark.parametrize("credential_source", ["file", "environment", "scope", "missing"])
def test_update_build_email_probe(tmp_path, monkeypatch, capsys, credential_source):
    from hermes_constants import get_hermes_home
    from hermes_cli.source_build import build_update_products
    from gateway.config import load_gateway_config, Platform
    from gateway.platform_registry import platform_registry
    from agent.secret_scope import current_secret_scope, reset_secret_scope, set_secret_scope

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    credentials = {
        "EMAIL_ADDRESS": "agent@example.test",
        "EMAIL_PASSWORD": "fixture-password",
        "EMAIL_IMAP_HOST": "imap.example.test",
        "EMAIL_SMTP_HOST": "smtp.example.test",
    }
    for name in credentials:
        monkeypatch.delenv(name, raising=False)
    stored = credentials if credential_source == "file" else {}
    # A caller-selected scope is authoritative even when the disk is incomplete.
    (home / ".env").write_text("\n".join(f"{k}={v}" for k, v in stored.items()), encoding="utf-8")
    if credential_source == "environment":
        for name, value in credentials.items():
            monkeypatch.setenv(name, value)
    (home / "config.yaml").write_text(
        "platforms:\n  email:\n    enabled: true\n    extra:\n      address: agent@example.test\n", encoding="utf-8")
    token = set_secret_scope(credentials if credential_source == "scope" else None)
    try:
        assert Platform.EMAIL in load_gateway_config().get_connected_platforms()
        assert platform_registry.get("email") is not None
        before_scope = current_secret_scope()
        before_env = {name: os.environ.get(name) for name in credentials}
        capsys.readouterr()
        # No frontend directories: real update-build entry point, no installs or connections.
        build_update_products(tmp_path / "python-only-source", desktop=False)
        output = capsys.readouterr().out
        assert ("Email:" in output) == (credential_source == "missing")
        assert current_secret_scope() is before_scope
        assert {name: os.environ.get(name) for name in credentials} == before_env
    finally:
        reset_secret_scope(token)


@pytest.mark.parametrize("outcome", [True, False, RuntimeError("fixture probe failure")])
def test_probe_scope_restored_and_dependency_results_preserved(tmp_path, monkeypatch, outcome):
    from agent.secret_scope import current_secret_scope, get_secret
    from gateway import config
    from gateway.platform_registry import PlatformEntry, PlatformRegistry
    import gateway.platform_registry as registry_module
    from hermes_cli.main_install_repair import _configured_features_missing_deps
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text("PROBE_TOKEN=fixture-value\n", encoding="utf-8")
    seen = []

    def probe():
        seen.append(get_secret("PROBE_TOKEN"))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    registry = PlatformRegistry()
    registry.register(PlatformEntry(
        name="email", label="Fixture SDK", adapter_factory=lambda _: None,
        check_fn=probe, install_hint="fixture installation hint", source="builtin"))
    monkeypatch.setattr(registry_module, "platform_registry", registry)
    monkeypatch.setattr(config, "load_gateway_config", lambda: config.GatewayConfig(
        platforms={config.Platform.EMAIL: config.PlatformConfig(enabled=True, token="fixture")}))
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    other = tmp_path / "other-home"
    other.mkdir()
    (other / ".env").write_text("PROBE_TOKEN=other-value\n", encoding="utf-8")
    before = current_secret_scope()
    previous_mode = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for selected in (home, other, home):
            home_token = set_hermes_home_override(str(selected))
            stored = (selected / ".env").read_bytes()
            try:
                result = _configured_features_missing_deps()
                assert result == ([("Fixture SDK", "fixture installation hint")] if outcome is False else [])
                assert current_secret_scope() is before
                assert (selected / ".env").read_bytes() == stored
            finally:
                reset_hermes_home_override(home_token)
    finally:
        set_multiplex_active(previous_mode)
    assert seen == ["fixture-value", "other-value", "fixture-value"]
    assert "PROBE_TOKEN" not in os.environ
