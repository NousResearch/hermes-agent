"""Bundled product service generation for the hermes-core distribution."""

from __future__ import annotations

import ipaddress
import os
import re
import secrets
import subprocess
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

from hermes_cli.config import _secure_dir, _secure_file, get_env_value, save_env_value_secure
from hermes_cli.product_config import (
    ensure_product_home,
    get_product_storage_root,
    load_product_config,
    save_product_config,
)
from utils import atomic_json_write, atomic_yaml_write


KANIDM_IMAGE = "kanidm/server:latest"
IDM_ADMIN_NAME = "idm_admin"
_RECOVER_PASSWORD_RE = re.compile(r'new_password:\s*"([^"]+)"')


def get_product_services_root() -> Path:
    return get_product_storage_root() / "services"


def get_kanidm_service_root() -> Path:
    return get_product_services_root() / "kanidm"


def get_kanidm_data_root() -> Path:
    return get_kanidm_service_root() / "data"


def get_product_bootstrap_root() -> Path:
    return get_product_storage_root() / "bootstrap"


def get_first_admin_state_path() -> Path:
    return get_product_bootstrap_root() / "first_admin.json"


def get_kanidm_compose_path() -> Path:
    return get_kanidm_service_root() / "compose.yaml"


def get_kanidm_server_config_path() -> Path:
    return get_kanidm_data_root() / "server.toml"


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
    kanidm_port = int(network.get("kanidm_port", 8443))
    app_base_url = f"http://{public_host}:{app_port}"
    issuer_url = f"https://{public_host}:{kanidm_port}"
    return {
        "public_host": public_host,
        "app_base_url": app_base_url,
        "issuer_url": issuer_url,
        "oidc_callback_url": f"{app_base_url}/api/auth/oidc/callback",
    }


def _ensure_client_secret(config: Dict[str, Any]) -> str:
    env_key = str(config.get("auth", {}).get("client_secret_ref", "")).strip()
    if not env_key:
        raise ValueError("auth.client_secret_ref must be configured in product.yaml")
    current = (get_env_value(env_key) or "").strip()
    if current:
        return current
    generated = secrets.token_urlsafe(48)
    save_env_value_secure(env_key, generated)
    return generated


def _build_server_config(config: Dict[str, Any]) -> str:
    urls = resolve_product_urls(config)
    kanidm_port = int(config.get("network", {}).get("kanidm_port", 8443))
    return (
        'version = "2"\n'
        f'domain = "{urls["public_host"]}"\n'
        f'origin = "{urls["issuer_url"]}"\n'
        f'bindaddress = "0.0.0.0:{kanidm_port}"\n'
        'db_path = "/data/kanidm.db"\n'
        'tls_chain = "/data/chain.pem"\n'
        'tls_key = "/data/key.pem"\n'
    )


def _build_compose_spec(config: Dict[str, Any]) -> Dict[str, Any]:
    network = config.get("network", {})
    services_cfg = config.get("services", {}).get("kanidm", {})
    bind_host = str(network.get("bind_host", "0.0.0.0")).strip() or "0.0.0.0"
    kanidm_port = int(network.get("kanidm_port", 8443))
    container_name = str(services_cfg.get("container_name", "hermes-kanidm")).strip() or "hermes-kanidm"
    data_root = get_kanidm_data_root().as_posix()
    service: Dict[str, Any] = {
        "image": str(services_cfg.get("image", KANIDM_IMAGE) or KANIDM_IMAGE),
        "container_name": container_name,
        "restart": "unless-stopped",
        "ports": [f"{bind_host}:{kanidm_port}:8443"],
        "volumes": [f"{data_root}:/data"],
        "command": [
            "/sbin/kanidmd",
            "server",
            "-c",
            "/data/server.toml",
        ],
    }
    runtime_user = str(services_cfg.get("user", "")).strip()
    if runtime_user:
        service["user"] = runtime_user
    return {
        "services": {
            "kanidm": service
        }
    }


def initialize_product_stack(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    product_config = config or load_product_config()
    ensure_product_home()
    _secure_tree(
        get_product_services_root(),
        get_kanidm_service_root(),
        get_kanidm_data_root(),
        get_product_bootstrap_root(),
    )

    urls = resolve_product_urls(product_config)
    product_config.setdefault("network", {})["public_host"] = urls["public_host"]
    product_config.setdefault("auth", {})["provider"] = "kanidm"
    product_config["auth"]["issuer_url"] = urls["issuer_url"]
    product_config.setdefault("services", {}).setdefault("kanidm", {})
    product_config["services"]["kanidm"].setdefault("mode", "docker")
    product_config["services"]["kanidm"].setdefault("container_name", "hermes-kanidm")
    product_config["services"]["kanidm"].setdefault("image", KANIDM_IMAGE)
    runtime_user = _runtime_user_spec()
    if runtime_user:
        product_config["services"]["kanidm"].setdefault("user", runtime_user)

    _ensure_client_secret(product_config)

    server_config_path = get_kanidm_server_config_path()
    server_config_path.write_text(_build_server_config(product_config), encoding="utf-8")
    _secure_file(server_config_path)

    atomic_yaml_write(get_kanidm_compose_path(), _build_compose_spec(product_config))
    _secure_file(get_kanidm_compose_path())

    save_product_config(product_config)
    return product_config


def ensure_kanidm_certificates(config: Dict[str, Any] | None = None) -> None:
    product_config = config or load_product_config()
    data_root = get_kanidm_data_root()
    key_path = data_root / "key.pem"
    chain_path = data_root / "chain.pem"
    if key_path.exists() and chain_path.exists():
        return

    image = str(product_config.get("services", {}).get("kanidm", {}).get("image", KANIDM_IMAGE) or KANIDM_IMAGE)
    command = [
        "docker",
        "run",
        "--rm",
    ]
    runtime_user = str(product_config.get("services", {}).get("kanidm", {}).get("user", "")).strip()
    if runtime_user:
        command.extend(["--user", runtime_user])
    command.extend(
        [
            "-v",
            f"{data_root.as_posix()}:/data",
            image,
            "/sbin/kanidmd",
            "cert-generate",
            "-c",
            "/data/server.toml",
        ]
    )
    subprocess.run(command, check=True, capture_output=True, text=True)
    for path in (key_path, chain_path, data_root / "cert.pem", data_root / "ca.pem", data_root / "cakey.pem"):
        if path.exists():
            _secure_file(path)


def _docker_exec_command(
    config: Dict[str, Any],
    *args: str,
    exec_user: str | None = None,
) -> list[str]:
    container_name = str(
        config.get("services", {}).get("kanidm", {}).get("container_name", "hermes-kanidm")
    ).strip() or "hermes-kanidm"
    command = ["docker", "exec", "-w", "/"]
    runtime_user = str(config.get("services", {}).get("kanidm", {}).get("user", "")).strip()
    selected_user = runtime_user if exec_user is None else exec_user
    if selected_user:
        command.extend(["-u", selected_user])
    command.extend([container_name, *args])
    return command


def _recover_account_password(config: Dict[str, Any], account_name: str) -> str:
    result = subprocess.run(
        _docker_exec_command(
            config,
            "/sbin/kanidmd",
            "-c",
            "/data/server.toml",
            "recover-account",
            account_name,
            exec_user="0:0",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    match = _RECOVER_PASSWORD_RE.search(result.stdout + "\n" + result.stderr)
    if not match:
        raise RuntimeError(f"Could not parse recovery password for {account_name} from Kanidm output")
    return match.group(1)


def _kanidm_module():
    try:
        return import_module("kanidm")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'kanidm' Python package is required for bundled product bootstrap. "
            "Install hermes-agent with the updated product dependencies."
        ) from exc


def _authenticated_kanidm_admin_client(config: Dict[str, Any]):
    kanidm = _kanidm_module()
    return kanidm, kanidm.KanidmClient(
        uri=resolve_product_urls(config)["issuer_url"],
        verify_hostnames=False,
        ca_path=str(get_kanidm_data_root() / "ca.pem"),
    )


async def _bootstrap_first_admin_remote(
    config: Dict[str, Any],
    password: str,
    username: str,
    display_name: str,
    ttl_seconds: int,
) -> None:
    kanidm, client = _authenticated_kanidm_admin_client(config)
    try:
        await client.authenticate_password(
            username=IDM_ADMIN_NAME,
            password=password,
            update_internal_auth_token=True,
        )
        try:
            existing_person = await client.person_account_get(username)
        except (kanidm.NoMatchingEntries, AttributeError):
            existing_person = None
        if existing_person is None:
            await client.person_account_create(username, display_name)
    finally:
        await client.openapi_client.close()


def load_first_admin_state() -> Dict[str, Any] | None:
    state_path = get_first_admin_state_path()
    if not state_path.exists():
        return None
    import json

    return json.loads(state_path.read_text(encoding="utf-8"))


def bootstrap_first_admin(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    product_config = initialize_product_stack(config or load_product_config())
    username = str(product_config.get("bootstrap", {}).get("first_admin_username", "admin")).strip() or "admin"
    display_name = str(
        product_config.get("bootstrap", {}).get("first_admin_display_name", "Administrator")
    ).strip() or "Administrator"
    ttl_seconds = int(product_config.get("bootstrap", {}).get("first_admin_reset_ttl_seconds", 86400))

    existing_state = load_first_admin_state()
    if existing_state and existing_state.get("username") == username and existing_state.get("temporary_password"):
        return existing_state

    ensure_product_stack_started(product_config)
    password = _recover_account_password(product_config, IDM_ADMIN_NAME)
    import asyncio

    asyncio.run(
        _bootstrap_first_admin_remote(
            product_config,
            password,
            username,
            display_name,
            ttl_seconds,
        )
    )
    temporary_password = _recover_account_password(product_config, username)

    state = {
        "username": username,
        "display_name": display_name,
        "temporary_password": temporary_password,
        "auth_mode": str(product_config.get("auth", {}).get("mode", "passkey")).strip() or "passkey",
    }
    state_path = get_first_admin_state_path()
    atomic_json_write(state_path, state)
    _secure_file(state_path)
    return state


def ensure_product_stack_started(config: Dict[str, Any] | None = None) -> subprocess.CompletedProcess[str]:
    product_config = config or initialize_product_stack()
    ensure_kanidm_certificates(product_config)
    compose_path = get_kanidm_compose_path()
    return subprocess.run(
        ["docker", "compose", "-f", str(compose_path), "up", "-d", "--wait", "--force-recreate"],
        check=True,
        capture_output=True,
        text=True,
    )
