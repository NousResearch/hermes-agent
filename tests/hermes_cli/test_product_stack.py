import json
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import yaml

from hermes_cli.product_config import load_product_config
from hermes_cli.product_stack import (
    POCKET_ID_IMAGE,
    bootstrap_first_admin_enrollment,
    bootstrap_product_oidc_client,
    ensure_product_stack_started,
    get_first_admin_enrollment_state_path,
    get_pocket_id_compose_path,
    get_pocket_id_data_root,
    get_pocket_id_env_path,
    initialize_product_stack,
    resolve_product_urls,
)


def test_resolve_product_urls_uses_public_host_when_bind_is_wildcard(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = load_product_config()
    config["network"]["bind_host"] = "0.0.0.0"
    config["network"]["public_host"] = "officebox.local"
    config["network"]["app_port"] = 18086
    config["network"]["pocket_id_port"] = 19111

    urls = resolve_product_urls(config)

    assert urls == {
        "public_host": "officebox.local",
        "app_base_url": "http://officebox.local:18086",
        "issuer_url": "http://officebox.local:19111",
        "oidc_callback_url": "http://officebox.local:18086/api/auth/oidc/callback",
        "pocket_id_setup_url": "http://officebox.local:19111/setup",
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
    config["network"]["pocket_id_port"] = 19411

    with (
        patch("hermes_cli.product_stack._runtime_user_spec", return_value="1000:1000"),
        patch("hermes_cli.product_stack.save_env_value_secure") as mock_save_env,
    ):
        mock_save_env.return_value = {"success": True}
        bootstrapped = initialize_product_stack(config)

    assert bootstrapped["auth"]["issuer_url"] == "http://hermes.local:19411"
    assert bootstrapped["services"]["pocket_id"]["image"] == POCKET_ID_IMAGE
    assert mock_save_env.call_count == 3

    env_text = get_pocket_id_env_path().read_text(encoding="utf-8")
    assert "APP_URL=http://hermes.local:19411" in env_text
    assert "STATIC_API_KEY=" in env_text
    assert "ENCRYPTION_KEY=" in env_text

    compose = yaml.safe_load(get_pocket_id_compose_path().read_text(encoding="utf-8"))
    service = compose["services"]["pocket-id"]
    assert service["image"] == POCKET_ID_IMAGE
    assert "0.0.0.0:19411:1411" in service["ports"]
    assert f"{get_pocket_id_data_root().as_posix()}:/app/data" in service["volumes"]
    assert "user" in service

    reloaded = load_product_config()
    assert reloaded["network"]["public_host"] == "hermes.local"
    assert reloaded["auth"]["issuer_url"] == "http://hermes.local:19411"


def test_initialize_product_stack_reuses_existing_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = load_product_config()

    def _existing_env_value(key):
        existing = {
            "HERMES_PRODUCT_OIDC_CLIENT_SECRET": "client-secret",
            "HERMES_POCKET_ID_STATIC_API_KEY": "static-api-key",
            "HERMES_POCKET_ID_ENCRYPTION_KEY": "enc-key",
        }
        return existing.get(key, "")

    with (
        patch("hermes_cli.product_stack._runtime_user_spec", return_value="1000:1000"),
        patch("hermes_cli.product_stack.get_env_value", side_effect=_existing_env_value),
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

    compose_command = mock_run.call_args.args[0]
    assert compose_command[:4] == ["docker", "compose", "-f", str(get_pocket_id_compose_path())]
    assert compose_command[-4:] == ["up", "-d", "--wait", "--force-recreate"]
    assert Path(compose_command[3]) == get_pocket_id_compose_path()


class _Response:
    def __init__(self, status_code, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.content = b"{}" if json_data is not None else b""

    def json(self):
        return self._json_data


class _ClientStub:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, path):
        response = self.responses[("GET", path)]
        self.requests.append(("GET", path, None))
        return response

    def request(self, method, path, **kwargs):
        response = self.responses[(method, path)]
        self.requests.append((method, path, kwargs))
        return response


def test_bootstrap_product_oidc_client_creates_client_and_rotates_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = load_product_config()
    config["network"]["public_host"] = "officebox.local"
    initialize_product_stack(config)

    stub = _ClientStub(
        {
            ("GET", "/api/oidc/clients/hermes-core"): _Response(404, text="missing"),
            ("POST", "/api/oidc/clients"): _Response(201, {"id": "hermes-core"}),
            ("POST", "/api/oidc/clients/hermes-core/secret"): _Response(200, {"secret": "new-client-secret"}),
        }
    )

    with (
        patch("hermes_cli.product_stack.ensure_product_stack_started"),
        patch("hermes_cli.product_stack._wait_for_pocket_id_ready"),
        patch("hermes_cli.product_stack.httpx.Client", return_value=stub),
        patch("hermes_cli.product_stack.get_env_value", return_value="setup-static-key"),
        patch("hermes_cli.product_stack.save_env_value_secure") as mock_save_env,
        patch(
            "hermes_cli.product_stack.load_product_oidc_client_settings",
            return_value=type(
                "_Settings",
                (),
                {
                    "issuer_url": "http://officebox.local:1411",
                },
            )(),
        ),
        patch(
            "hermes_cli.product_stack.discover_product_oidc_provider_metadata",
            return_value=type(
                "_Metadata",
                (),
                {
                    "authorization_endpoint": "http://officebox.local:1411/authorize",
                    "token_endpoint": "http://officebox.local:1411/token",
                },
            )(),
        ),
    ):
        state = bootstrap_product_oidc_client(config)

    assert state["client_id"] == "hermes-core"
    assert state["issuer_url"] == "http://officebox.local:1411"
    assert state["authorization_endpoint"] == "http://officebox.local:1411/authorize"
    assert state["token_endpoint"] == "http://officebox.local:1411/token"
    assert any(req[0] == "POST" and req[1] == "/api/oidc/clients" for req in stub.requests)
    mock_save_env.assert_called_once_with("HERMES_PRODUCT_OIDC_CLIENT_SECRET", "new-client-secret")


def test_bootstrap_first_admin_enrollment_stores_native_setup_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = load_product_config()
    config["network"]["public_host"] = "officebox.local"
    config["bootstrap"]["first_admin_username"] = "supplier-admin"
    config["bootstrap"]["first_admin_display_name"] = "Supplier Admin"
    config["bootstrap"]["first_admin_email"] = "admin@example.com"

    with patch(
        "hermes_cli.product_stack.bootstrap_product_oidc_client",
        return_value={"client_id": "hermes-core", "issuer_url": "http://officebox.local:1411"},
    ):
        state = bootstrap_first_admin_enrollment(config)

    assert state == {
        "username": "supplier-admin",
        "display_name": "Supplier Admin",
        "email": "admin@example.com",
        "auth_mode": "passkey",
        "setup_url": "http://officebox.local:1411/setup",
        "oidc_client_id": "hermes-core",
    }

    saved = json.loads(get_first_admin_enrollment_state_path().read_text(encoding="utf-8"))
    assert saved == state


def test_bootstrap_first_admin_enrollment_reuses_existing_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config = load_product_config()
    initialize_product_stack(config)
    existing = {
        "username": "admin",
        "display_name": "Administrator",
        "email": "",
        "auth_mode": "passkey",
        "setup_url": "http://localhost:1411/setup",
        "oidc_client_id": "hermes-core",
    }
    get_first_admin_enrollment_state_path().write_text(json.dumps(existing), encoding="utf-8")

    with patch(
        "hermes_cli.product_stack.bootstrap_product_oidc_client",
        return_value={"client_id": "hermes-core", "issuer_url": "http://localhost:1411"},
    ):
        state = bootstrap_first_admin_enrollment(config)

    assert state == existing
