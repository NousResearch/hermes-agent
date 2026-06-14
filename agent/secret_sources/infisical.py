"""Infisical Universal Auth secret source integration.

Hermes pulls API keys from Infisical at process startup so they don't
have to live in plaintext in ``~/.hermes/.env``.

Design summary
--------------

* The bootstrap credentials are a Machine Identity client ID and client
  secret read from environment variables (``INFISICAL_CLIENT_ID`` and
  ``INFISICAL_CLIENT_SECRET`` by default).  They live in ``.env`` or the
  parent shell; they are never stored in ``config.yaml``.
* Hermes exchanges those credentials for a short-lived access token via
  Infisical Universal Auth, then calls the v4 secrets list endpoint for
  the configured project/environment/path.
* Returned ``secretKey`` / ``secretValue`` pairs are applied to
  ``os.environ`` using the same non-destructive semantics as other
  secret sources: existing values win unless ``override_existing`` is
  enabled.
* Failures never block Hermes startup.  Missing credentials, auth
  failures, network errors, and malformed responses are surfaced as a
  one-line warning by the caller while Hermes continues with whatever
  credentials were already present.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Infisical's latest documented endpoint versions are not uniform:
# Universal Auth login is v1, while the secrets list API is v4.
DEFAULT_API_URL = "https://us.infisical.com"
UNIVERSAL_AUTH_LOGIN_PATH = "/api/v1/auth/universal-auth/login"
SECRETS_LIST_PATH = "/api/v4/secrets"
_HTTP_TIMEOUT = 30

_CacheKey = Tuple[str, str, str, str, str, str, str, str, str, str]
_CACHE: Dict[_CacheKey, "_CachedFetch"] = {}


@dataclass
class _CachedFetch:
    secrets: Dict[str, str]
    fetched_at: float

    def is_fresh(self, ttl_seconds: float) -> bool:
        if ttl_seconds <= 0:
            return False
        return (time.time() - self.fetched_at) < ttl_seconds


@dataclass
class FetchResult:
    """Outcome of a single Infisical pull."""

    secrets: Dict[str, str] = field(default_factory=dict)
    applied: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _fingerprint(value: str) -> str:
    """SHA-256 prefix used only for cache keys."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _normalize_api_url(api_url: str) -> str:
    api_url = (api_url or DEFAULT_API_URL).strip().rstrip("/")
    if not api_url:
        return DEFAULT_API_URL
    if not api_url.startswith(("http://", "https://")):
        raise RuntimeError(
            "secrets.infisical.api_url must start with http:// or https://"
        )
    return api_url


def _normalize_secret_path(secret_path: str) -> str:
    secret_path = (secret_path or "/").strip()
    if not secret_path:
        return "/"
    if not secret_path.startswith("/"):
        secret_path = "/" + secret_path
    return secret_path


def _bool_param(value: bool) -> str:
    return "true" if value else "false"


def _http_json(
    method: str,
    url: str,
    *,
    body: Optional[dict[str, Any]] = None,
    token: str = "",
    params: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Make a JSON Infisical API request using stdlib urllib."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    data = None
    headers = {
        "Accept": "application/json",
        "User-Agent": "hermes-agent",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Infisical API returned HTTP {exc.code}: {detail[:300]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Infisical API request failed: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Infisical API request failed: {exc}") from exc

    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Infisical API returned non-JSON output: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Infisical API returned unexpected shape: {type(payload).__name__}"
        )
    return payload


def login_universal_auth(
    *,
    client_id: str,
    client_secret: str,
    api_url: str = DEFAULT_API_URL,
    organization_slug: str = "",
) -> tuple[str, int]:
    """Exchange Universal Auth credentials for an Infisical access token."""
    if not client_id:
        raise RuntimeError("Infisical client ID is empty")
    if not client_secret:
        raise RuntimeError("Infisical client secret is empty")

    api_url = _normalize_api_url(api_url)
    body: dict[str, Any] = {
        "clientId": client_id,
        "clientSecret": client_secret,
    }
    if organization_slug:
        body["organizationSlug"] = organization_slug

    payload = _http_json(
        "POST",
        f"{api_url}{UNIVERSAL_AUTH_LOGIN_PATH}",
        body=body,
    )
    token = payload.get("accessToken")
    if not isinstance(token, str) or not token:
        raise RuntimeError("Infisical login response did not include accessToken")
    expires_in = payload.get("expiresIn")
    if not isinstance(expires_in, int):
        expires_in = 0
    return token, expires_in


def fetch_infisical_secrets(
    *,
    client_id: str,
    client_secret: str,
    project_id: str,
    environment: str = "prod",
    secret_path: str = "/",
    api_url: str = DEFAULT_API_URL,
    organization_slug: str = "",
    cache_ttl_seconds: float = 300,
    use_cache: bool = True,
    recursive: bool = False,
    include_imports: bool = True,
    expand_secret_references: bool = True,
) -> tuple[Dict[str, str], List[str]]:
    """Fetch secrets from Infisical for one project/environment/path."""
    if not client_id:
        raise RuntimeError("Infisical client ID is empty")
    if not client_secret:
        raise RuntimeError("Infisical client secret is empty")
    if not project_id:
        raise RuntimeError("Infisical project_id is empty")
    if not environment:
        raise RuntimeError("Infisical environment is empty")

    api_url = _normalize_api_url(api_url)
    secret_path = _normalize_secret_path(secret_path)
    cache_key: _CacheKey = (
        _fingerprint(client_id),
        _fingerprint(client_secret),
        api_url,
        organization_slug or "",
        project_id,
        environment,
        secret_path,
        _bool_param(recursive),
        _bool_param(include_imports),
        _bool_param(expand_secret_references),
    )
    if use_cache:
        cached = _CACHE.get(cache_key)
        if cached and cached.is_fresh(cache_ttl_seconds):
            return dict(cached.secrets), []

    token, _expires_in = login_universal_auth(
        client_id=client_id,
        client_secret=client_secret,
        api_url=api_url,
        organization_slug=organization_slug,
    )
    payload = _http_json(
        "GET",
        f"{api_url}{SECRETS_LIST_PATH}",
        token=token,
        params={
            "projectId": project_id,
            "environment": environment,
            "secretPath": secret_path,
            "viewSecretValue": "true",
            "expandSecretReferences": _bool_param(expand_secret_references),
            "recursive": _bool_param(recursive),
            "includeImports": _bool_param(include_imports),
        },
    )
    secrets, warnings = _extract_secret_values(payload, include_imports=include_imports)
    if use_cache:
        _CACHE[cache_key] = _CachedFetch(secrets=dict(secrets), fetched_at=time.time())
    return secrets, warnings


def _extract_secret_values(
    payload: dict[str, Any],
    *,
    include_imports: bool,
) -> tuple[Dict[str, str], List[str]]:
    """Extract env-var shaped secrets from Infisical's v4 list response."""
    secrets: Dict[str, str] = {}
    warnings: List[str] = []

    # Imported secrets are lower precedence than secrets in the requested path.
    if include_imports:
        imports = payload.get("imports")
        if isinstance(imports, list):
            for imported in imports:
                if not isinstance(imported, dict):
                    continue
                _merge_secret_items(
                    imported.get("secrets"),
                    secrets,
                    warnings,
                    source="import",
                )

    _merge_secret_items(payload.get("secrets"), secrets, warnings, source="secret")
    return secrets, warnings


def _merge_secret_items(
    items: Any,
    secrets: Dict[str, str],
    warnings: List[str],
    *,
    source: str,
) -> None:
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        key = item.get("secretKey")
        value = item.get("secretValue")
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if not _is_valid_env_name(key):
            warnings.append(
                f"Skipping {source} {key!r}: not a valid env-var name"
            )
            continue
        if key in secrets:
            warnings.append(
                f"Duplicate secret {key!r}: later value overwrote earlier one"
            )
        secrets[key] = value


def _is_valid_env_name(name: str) -> bool:
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


def apply_infisical_secrets(
    *,
    enabled: bool,
    client_id_env: str = "INFISICAL_CLIENT_ID",
    client_secret_env: str = "INFISICAL_CLIENT_SECRET",
    project_id: str = "",
    project_id_env: str = "INFISICAL_PROJECT_ID",
    environment: str = "prod",
    secret_path: str = "/",
    api_url: str = DEFAULT_API_URL,
    organization_slug: str = "",
    override_existing: bool = True,
    cache_ttl_seconds: float = 300,
    recursive: bool = False,
    include_imports: bool = True,
    expand_secret_references: bool = True,
) -> FetchResult:
    """Pull Infisical secrets and set them on ``os.environ``."""
    result = FetchResult()

    if not enabled:
        return result

    client_id = os.environ.get(client_id_env, "").strip()
    if not client_id:
        result.error = (
            f"secrets.infisical.enabled is true but {client_id_env} is not set.  "
            "Run `hermes secrets infisical setup`."
        )
        return result

    client_secret = os.environ.get(client_secret_env, "").strip()
    if not client_secret:
        result.error = (
            f"secrets.infisical.enabled is true but {client_secret_env} is not set.  "
            "Run `hermes secrets infisical setup`."
        )
        return result

    resolved_project_id = (project_id or "").strip()
    if not resolved_project_id:
        resolved_project_id = os.environ.get(project_id_env, "").strip()
    if not resolved_project_id:
        result.error = (
            "secrets.infisical.project_id is empty and "
            f"{project_id_env} is not set.  Run `hermes secrets infisical setup`."
        )
        return result

    try:
        secrets, warnings = fetch_infisical_secrets(
            client_id=client_id,
            client_secret=client_secret,
            project_id=resolved_project_id,
            environment=environment,
            secret_path=secret_path,
            api_url=api_url,
            organization_slug=organization_slug,
            cache_ttl_seconds=cache_ttl_seconds,
            recursive=recursive,
            include_imports=include_imports,
            expand_secret_references=expand_secret_references,
        )
    except RuntimeError as exc:
        result.error = str(exc)
        return result

    result.secrets = secrets
    result.warnings.extend(warnings)

    bootstrap_names = {client_id_env, client_secret_env, project_id_env}
    for key, value in secrets.items():
        if key in bootstrap_names:
            result.skipped.append(key)
            continue
        if not override_existing and os.environ.get(key):
            result.skipped.append(key)
            continue
        os.environ[key] = value
        result.applied.append(key)

    return result


def _reset_cache_for_tests() -> None:
    """Clear the in-process fetch cache."""
    _CACHE.clear()
