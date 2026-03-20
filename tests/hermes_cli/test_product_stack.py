from pathlib import Path
from unittest.mock import patch

import yaml

from hermes_cli.product_config import load_product_config
from hermes_cli.product_stack import (
    KANIDM_IMAGE,
    ensure_kanidm_certificates,
    ensure_product_stack_started,
    get_kanidm_compose_path,
    get_kanidm_data_root,
    get_kanidm_server_config_path,
    initialize_product_stack,
    resolve_product_urls,
)


def test_resolve_product_urls_uses_public_host_when_bind_is_wildcard(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = load_product_config()
    config["network"]["bind_host"] = "0.0.0.0"
    config["network"]["public_host"] = "officebox.local"
    config["network"]["app_port"] = 18086
    config["network"]["kanidm_port"] = 18443

    urls = resolve_product_urls(config)

    assert urls == {
        "public_host": "officebox.local",
        "app_base_url": "http://officebox.local:18086",
        "issuer_url": "https://officebox.local:18443",
        "oidc_callback_url": "http://officebox.local:18086/api/auth/oidc/callback",
    }


def test_resolve_product_urls_rejects_raw_ip_public_host(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = load_product_config()
    config["network"]["public_host"] = "192.168.1.27"

    try:
        resolve_product_urls(config)
    except ValueError as exc:
        assert "hostname or domain" in str(exc)
    else:
        raise AssertionError("expected raw IP public_host to be rejected")


def test_initialize_product_stack_generates_files_and_bootstraps_product_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = load_product_config()
    config["network"]["public_host"] = "hermes.local"
    config["network"]["kanidm_port"] = 19443

    with (
        patch("hermes_cli.product_stack._runtime_user_spec", return_value="1000:1000"),
        patch("hermes_cli.product_stack.save_env_value_secure") as mock_save_env,
    ):
        mock_save_env.return_value = {"success": True}
        bootstrapped = initialize_product_stack(config)

    assert bootstrapped["auth"]["issuer_url"] == "https://hermes.local:19443"
    assert bootstrapped["services"]["kanidm"]["image"] == KANIDM_IMAGE
    mock_save_env.assert_called_once()

    server_toml = get_kanidm_server_config_path().read_text(encoding="utf-8")
    assert 'domain = "hermes.local"' in server_toml
    assert 'origin = "https://hermes.local:19443"' in server_toml

    compose = yaml.safe_load(get_kanidm_compose_path().read_text(encoding="utf-8"))
    service = compose["services"]["kanidm"]
    assert service["image"] == KANIDM_IMAGE
    assert "0.0.0.0:19443:8443" in service["ports"]
    assert f"{get_kanidm_data_root().as_posix()}:/data" in service["volumes"]
    assert "user" in service

    reloaded = load_product_config()
    assert reloaded["network"]["public_host"] == "hermes.local"
    assert reloaded["auth"]["issuer_url"] == "https://hermes.local:19443"


def test_initialize_product_stack_reuses_existing_client_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = load_product_config()

    with (
        patch("hermes_cli.product_stack._runtime_user_spec", return_value="1000:1000"),
        patch("hermes_cli.product_stack.get_env_value", return_value="existing-secret"),
        patch("hermes_cli.product_stack.save_env_value_secure") as mock_save_env,
    ):
        initialize_product_stack(config)

    mock_save_env.assert_not_called()


def test_ensure_product_stack_started_uses_generated_compose_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with (
        patch("hermes_cli.product_stack._runtime_user_spec", return_value="1000:1000"),
        patch("hermes_cli.product_stack.get_env_value", return_value="existing-secret"),
        patch("hermes_cli.product_stack.subprocess.run") as mock_run,
    ):
        ensure_product_stack_started()

    cert_command = mock_run.call_args_list[0].args[0]
    compose_command = mock_run.call_args_list[1].args[0]
    assert cert_command[:3] == ["docker", "run", "--rm"]
    assert "--user" in cert_command
    assert f"{get_kanidm_data_root().as_posix()}:/data" in cert_command
    assert compose_command[:4] == ["docker", "compose", "-f", str(get_kanidm_compose_path())]
    assert compose_command[-2:] == ["up", "-d"]
    assert Path(compose_command[3]) == get_kanidm_compose_path()


def test_ensure_kanidm_certificates_skips_when_files_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = load_product_config()
    with (
        patch("hermes_cli.product_stack._runtime_user_spec", return_value="1000:1000"),
        patch("hermes_cli.product_stack.get_env_value", return_value="existing-secret"),
    ):
        initialize_product_stack(config)

    data_root = get_kanidm_data_root()
    (data_root / "key.pem").write_text("key", encoding="utf-8")
    (data_root / "chain.pem").write_text("chain", encoding="utf-8")

    with patch("hermes_cli.product_stack.subprocess.run") as mock_run:
        ensure_kanidm_certificates(config)

    mock_run.assert_not_called()
