"""Bundled product service generation for the hermes-core distribution."""

from __future__ import annotations

import ipaddress
import json
import os
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import httpx

from hermes_cli.config import _secure_dir, _secure_file, get_env_value, save_env_value_secure
from hermes_cli.product_oidc import (
    discover_product_oidc_provider_metadata,
    load_product_oidc_client_settings,
)
from hermes_cli.product_config import (
    ensure_product_home,
    get_product_storage_root,
    load_product_config,
    save_product_config,
)
from utils import atomic_json_write, atomic_yaml_write


POCKET_ID_IMAGE = "ghcr.io/pocket-id/pocket-id:v2"
_READY_TIMEOUT_SECONDS = 45.0


def get_product_services_root() -> Path:
    return get_product_storage_root() / "services"


def get_pocket_id_service_root() -> Path:
    return get_product_services_root() / "pocket-id"


def get_pocket_id_data_root() -> Path:
    return get_pocket_id_service_root() / "data"


def get_product_bootstrap_root() -> Path:
    return get_product_storage_root() / "bootstrap"


def get_first_admin_enrollment_state_path() -> Path:
    return get_product_bootstrap_root() / "first_admin_enrollment.json"


def get_pocket_id_compose_path() -> Path:
    return get_pocket_id_service_root() / "compose.yaml"


def get_pocket_id_env_path() -> Path:
    return get_pocket_id_service_root() / ".env"


def _secure_tree(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        _secure_dir(path)


def _public_host(config: Dict[str, Any]) -> str:
    network = config.get("network", {})
    host = str(network.get("public_host", "")).strip()
    if host:
        return host
    bind_host = str(network.get("bind_host", "")).strip()
    if bind_host and bind_host not in {"0.0.0.0", "::", "[::]"}:
        return bind_host
    return "localhost"


def _validate_public_host(host: str) -> None:
    candidate = (host or "").strip()
    if not candidate:
        raise ValueError("product network.public_host must not be empty")
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return
    raise ValueError(
        "product network.public_host must be a hostname or domain, not a raw IP address"
    )


def _runtime_user_spec() -> str:
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None or getgid is None:
        return ""
    try:
        return f"{getuid()}:{getgid()}"
    except OSError:
        return ""


def resolve_product_urls(config: Dict[str, Any] | None = None) -> Dict[str, str]:
    product_config = config or load_product_config()
    network = product_config.get("network", {})
    public_host = _public_host(product_config)
    _validate_public_host(public_host)
    app_port = int(network.get("app_port", 8086))
    pocket_id_port = int(network.get("pocket_id_port", 1411))
    app_base_url = f"http://{public_host}:{app_port}"
    issuer_url = f"http://{public_host}:{pocket_id_port}"
    return {
        "public_host": public_host,
        "app_base_url": app_base_url,
        "issuer_url": issuer_url,
        "oidc_callback_url": f"{app_base_url}/api/auth/oidc/callback",
        "pocket_id_setup_url": f"{issuer_url}/setup",
    }


def _required_secret(config: Dict[str, Any], env_key: str) -> str:
    current = (get_env_value(env_key) or "").strip()
    if current:
        return current
    generated = secrets.token_urlsafe(48)
    save_env_value_secure(env_key, generated)
    return generated


def _ensure_client_secret(config: Dict[str, Any]) -> str:
    env_key = str(config.get("auth", {}).get("client_secret_ref", "")).strip()
    if not env_key:
        raise ValueError("auth.client_secret_ref must be configured in product.yaml")
    return _required_secret(config, env_key)


def _ensure_static_api_key(config: Dict[str, Any]) -> str:
    env_key = str(
        config.get("services", {}).get("pocket_id", {}).get("static_api_key_ref", "")
    ).strip()
    if not env_key:
        raise ValueError("services.pocket_id.static_api_key_ref must be configured in product.yaml")
    return _required_secret(config, env_key)


def _ensure_encryption_key(config: Dict[str, Any]) -> str:
    env_key = str(
        config.get("services", {}).get("pocket_id", {}).get("encryption_key_ref", "")
    ).strip()
    if not env_key:
        raise ValueError("services.pocket_id.encryption_key_ref must be configured in product.yaml")
    current = (get_env_value(env_key) or "").strip()
    if current:
        return current
    generated = secrets.token_urlsafe(32)
    save_env_value_secure(env_key, generated)
    return generated


def _build_env_file(config: Dict[str, Any]) -> str:
    network = config.get("network", {})
    services_cfg = config.get("services", {}).get("pocket_id", {})
    return "\n".join(
        [
            f"APP_URL={resolve_product_urls(config)['issuer_url']}",
            f"ENCRYPTION_KEY={_ensure_encryption_key(config)}",
            f"STATIC_API_KEY={_ensure_static_api_key(config)}",
            f"PUID={services_cfg.get('puid', 1000)}",
            f"PGID={services_cfg.get('pgid', 1000)}",
            f"HOST={network.get('bind_host', '0.0.0.0')}",
            "PORT=1411",
            "",
        ]
    )


def _build_compose_spec(config: Dict[str, Any]) -> Dict[str, Any]:
    network = config.get("network", {})
    services_cfg = config.get("services", {}).get("pocket_id", {})
    bind_host = str(network.get("bind_host", "0.0.0.0")).strip() or "0.0.0.0"
    pocket_id_port = int(network.get("pocket_id_port", 1411))
    container_name = (
        str(services_cfg.get("container_name", "hermes-pocket-id")).strip() or "hermes-pocket-id"
    )
    data_root = get_pocket_id_data_root().as_posix()
    service: Dict[str, Any] = {
        "image": str(services_cfg.get("image", POCKET_ID_IMAGE) or POCKET_ID_IMAGE),
        "container_name": container_name,
        "restart": "unless-stopped",
        "env_file": [get_pocket_id_env_path().as_posix()],
        "ports": [f"{bind_host}:{pocket_id_port}:1411"],
        "volumes": [f"{data_root}:/app/data"],
        "healthcheck": {
            "test": ["CMD", "/app/pocket-id", "healthcheck"],
            "interval": "90s",
            "timeout": "5s",
            "retries": 2,
            "start_period": "10s",
        },
    }
    runtime_user = str(services_cfg.get("user", "")).strip()
    if runtime_user:
        service["user"] = runtime_user
    return {"services": {"pocket-id": service}}


def initialize_product_stack(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    product_config = config or load_product_config()
    ensure_product_home()
    _secure_tree(
        get_product_services_root(),
        get_pocket_id_service_root(),
        get_pocket_id_data_root(),
        get_product_bootstrap_root(),
    )

    urls = resolve_product_urls(product_config)
    product_config.setdefault("network", {})["public_host"] = urls["public_host"]
    product_config.setdefault("auth", {})["provider"] = "pocket-id"
    product_config["auth"]["issuer_url"] = urls["issuer_url"]
    product_config.setdefault("services", {}).setdefault("pocket_id", {})
    product_config["services"]["pocket_id"].setdefault("mode", "docker")
    product_config["services"]["pocket_id"].setdefault("container_name", "hermes-pocket-id")
    product_config["services"]["pocket_id"].setdefault("image", POCKET_ID_IMAGE)
    product_config["services"]["pocket_id"].setdefault("puid", 1000)
    product_config["services"]["pocket_id"].setdefault("pgid", 1000)
    runtime_user = _runtime_user_spec()
    if runtime_user:
        product_config["services"]["pocket_id"].setdefault("user", runtime_user)

    _ensure_client_secret(product_config)

    env_path = get_pocket_id_env_path()
    env_path.write_text(_build_env_file(product_config), encoding="utf-8")
    _secure_file(env_path)

    compose_path = get_pocket_id_compose_path()
    atomic_yaml_write(compose_path, _build_compose_spec(product_config))
    _secure_file(compose_path)

    save_product_config(product_config)
    return product_config


def ensure_product_stack_started(config: Dict[str, Any] | None = None) -> subprocess.CompletedProcess[str]:
    product_config = config or initialize_product_stack()
    compose_path = get_pocket_id_compose_path()
    return subprocess.run(
        ["docker", "compose", "-f", str(compose_path), "up", "-d", "--wait", "--force-recreate"],
        check=True,
        capture_output=True,
        text=True,
    )


def _wait_for_pocket_id_ready(config: Dict[str, Any], timeout_seconds: float = _READY_TIMEOUT_SECONDS) -> None:
    health_url = resolve_product_urls(config)["issuer_url"] + "/.well-known/openid-configuration"
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(health_url, timeout=5.0)
            if response.status_code == 200:
                return
            last_error = RuntimeError(f"Pocket ID health endpoint returned {response.status_code}")
        except Exception as exc:  # pragma: no cover - exercised via retry path
            last_error = exc
        time.sleep(1.0)
    raise RuntimeError(f"Pocket ID did not become ready at {health_url}: {last_error}")


def _api_headers(config: Dict[str, Any]) -> Dict[str, str]:
    return {"X-API-Key": _ensure_static_api_key(config)}


def _oidc_client_payload(config: Dict[str, Any]) -> Dict[str, Any]:
    urls = resolve_product_urls(config)
    brand_name = str(config.get("product", {}).get("brand", {}).get("name", "Hermes Core")).strip() or "Hermes Core"
    return {
        "id": str(config.get("auth", {}).get("client_id", "hermes-core")).strip() or "hermes-core",
        "name": brand_name,
        "callbackURLs": [urls["oidc_callback_url"]],
        "logoutCallbackURLs": [urls["app_base_url"]],
        "isPublic": False,
        "pkceEnabled": True,
        "requiresReauthentication": False,
        "credentials": {"federatedIdentities": []},
        "launchURL": urls["app_base_url"],
        "hasLogo": False,
        "hasDarkLogo": False,
        "isGroupRestricted": False,
    }


def _request_json(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    expected_status: int,
    **kwargs: Any,
) -> Dict[str, Any]:
    response = client.request(method, url, **kwargs)
    if response.status_code != expected_status:
        raise RuntimeError(f"{method} {url} failed with {response.status_code}: {response.text}")
    return response.json() if response.content else {}


def bootstrap_product_oidc_client(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    product_config = initialize_product_stack(config or load_product_config())
    ensure_product_stack_started(product_config)
    _wait_for_pocket_id_ready(product_config)

    urls = resolve_product_urls(product_config)
    client_payload = _oidc_client_payload(product_config)
    client_id = client_payload["id"]
    base_url = urls["issuer_url"]
    headers = _api_headers(product_config)

    with httpx.Client(base_url=base_url, headers=headers, timeout=10.0) as client:
        get_response = client.get(f"/api/oidc/clients/{client_id}")
        if get_response.status_code == 404:
            _request_json(client, "POST", "/api/oidc/clients", expected_status=201, json=client_payload)
        elif get_response.status_code == 200:
            _request_json(
                client,
                "PUT",
                f"/api/oidc/clients/{client_id}",
                expected_status=200,
                json={key: value for key, value in client_payload.items() if key != "id"},
            )
        else:
            raise RuntimeError(
                f"GET {base_url}/api/oidc/clients/{client_id} failed with "
                f"{get_response.status_code}: {get_response.text}"
            )

        secret_response = _request_json(
            client,
            "POST",
            f"/api/oidc/clients/{client_id}/secret",
            expected_status=200,
        )

    client_secret = str(secret_response.get("secret", "")).strip()
    if not client_secret:
        raise RuntimeError("Pocket ID did not return an OIDC client secret")
    save_env_value_secure(str(product_config["auth"]["client_secret_ref"]), client_secret)
    settings = load_product_oidc_client_settings(product_config)
    metadata = discover_product_oidc_provider_metadata(settings)
    return {
        "client_id": client_id,
        "issuer_url": settings.issuer_url,
        "callback_url": urls["oidc_callback_url"],
        "authorization_endpoint": metadata.authorization_endpoint,
        "token_endpoint": metadata.token_endpoint,
    }


def load_first_admin_enrollment_state() -> Dict[str, Any] | None:
    state_path = get_first_admin_enrollment_state_path()
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8"))


def bootstrap_first_admin_enrollment(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    product_config = initialize_product_stack(config or load_product_config())
    oidc_state = bootstrap_product_oidc_client(product_config)
    existing_state = load_first_admin_enrollment_state()

    username = str(product_config.get("bootstrap", {}).get("first_admin_username", "admin")).strip() or "admin"
    display_name = str(
        product_config.get("bootstrap", {}).get("first_admin_display_name", "Administrator")
    ).strip() or "Administrator"
    email = str(product_config.get("bootstrap", {}).get("first_admin_email", "")).strip()
    state = {
        "username": username,
        "display_name": display_name,
        "email": email,
        "auth_mode": str(product_config.get("auth", {}).get("mode", "passkey")).strip() or "passkey",
        "setup_url": resolve_product_urls(product_config)["pocket_id_setup_url"],
        "oidc_client_id": oidc_state["client_id"],
    }
    if existing_state == state:
        return existing_state

    state_path = get_first_admin_enrollment_state_path()
    atomic_json_write(state_path, state)
    _secure_file(state_path)
    return state
