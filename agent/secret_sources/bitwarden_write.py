"""In-process Bitwarden Secrets Manager writes for masked secret capture.

The ``bws`` CLI accepts secret values only as argv parameters. This module uses
Bitwarden's SDK instead so captured values never enter a child process command line.
"""

from __future__ import annotations

import importlib
from urllib.parse import urlparse
from uuid import UUID

from agent.secret_scope import get_secret
from hermes_cli.config import load_config
from hermes_constants import get_hermes_home


def _sdk_urls(server_url: str) -> tuple[str, str]:
    base = server_url.strip().rstrip("/")
    if not base:
        return "https://api.bitwarden.com", "https://identity.bitwarden.com"
    parsed = urlparse(base)
    if parsed.hostname in {"bitwarden.com", "vault.bitwarden.com"}:
        return "https://api.bitwarden.com", "https://identity.bitwarden.com"
    if parsed.hostname in {"bitwarden.eu", "vault.bitwarden.eu"}:
        return "https://api.bitwarden.eu", "https://identity.bitwarden.eu"
    return f"{base}/api", f"{base}/identity"


def _bitwarden_config() -> dict:
    config = load_config()
    secrets = config.get("secrets") if isinstance(config, dict) else None
    bitwarden = secrets.get("bitwarden") if isinstance(secrets, dict) else None
    return bitwarden if isinstance(bitwarden, dict) else {}


def _load_sdk():
    from tools.lazy_deps import ensure

    ensure("secret.bitwarden", prompt=False)
    importlib.invalidate_caches()
    return importlib.import_module("bitwarden_sdk")


def _require_success(response, action: str):
    if not getattr(response, "success", False) or getattr(response, "data", None) is None:
        raise RuntimeError(f"Bitwarden {action} failed")
    return response.data


def store_bitwarden_secret(name: str, value: str) -> dict:
    """Create or replace *name* in the configured project; return metadata only."""
    cfg = _bitwarden_config()
    if not cfg.get("enabled"):
        raise RuntimeError("Bitwarden Secrets Manager is not enabled")
    project_id = UUID(str(cfg.get("project_id") or ""))
    token_env = str(cfg.get("access_token_env") or "BWS_ACCESS_TOKEN")
    access_token = str(get_secret(token_env, "") or "").strip()
    if not access_token:
        raise RuntimeError("Bitwarden access token is unavailable")

    sdk = _load_sdk()
    api_url, identity_url = _sdk_urls(str(cfg.get("server_url") or ""))
    client = sdk.BitwardenClient(sdk.client_settings_from_dict({
        "apiUrl": api_url,
        "identityUrl": identity_url,
        "userAgent": "Hermes Agent secret capture",
        "deviceType": sdk.DeviceType.SDK,
    }))
    _require_success(client.auth().login_access_token(access_token), "authentication")
    project = _require_success(client.projects().get(str(project_id)), "project lookup")
    synced = _require_success(client.secrets().sync(str(project.organization_id), None), "secret lookup")
    matches = [
        item for item in (synced.secrets or [])
        if item.key == name and item.project_id == project_id
    ]
    if len(matches) > 1:
        raise RuntimeError("Bitwarden project contains duplicate secret names")
    if matches:
        current = matches[0]
        response = client.secrets().update(
            current.organization_id,
            str(current.id),
            name,
            value,
            current.note,
            [project_id],
        )
        operation = "updated"
    else:
        response = client.secrets().create(
            project.organization_id,
            name,
            value,
            "Stored by Hermes masked secret capture",
            [project_id],
        )
        operation = "created"
    _require_success(response, f"secret {operation}")

    # A subsequent source fetch must see the remote write instead of an old L1/L2 cache entry.
    from agent.secret_sources.bitwarden import clear_caches

    clear_caches(get_hermes_home())
    return {
        "success": True,
        "stored_as": name,
        "validated": True,
        "operation": operation,
    }
