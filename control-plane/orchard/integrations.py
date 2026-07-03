"""Integration catalog + per-tenant configuration status.

An "integration" (GitLab, Jira, Wiki...) groups one or more fields:
  - secret fields (tokens)      -> entered via the one-time form, never shown
  - non-secret fields (URLs...) -> shown filled-in, editable

The catalog is admin-curated (integrations.yaml). Per-tenant status is derived
from the same per-tenant SecretStore, so nothing new stores values.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .config import Settings


def load_catalog(settings: Settings) -> list[dict]:
    path = settings.integrations.catalog_file
    if not path or not Path(path).is_file():
        return []
    try:
        data = yaml.safe_load(Path(path).read_text()) or {}
    except Exception:
        return []
    out = []
    for it in data.get("integrations", []):
        if it.get("id") and it.get("fields"):
            out.append(it)
    return out


def get(settings: Settings, integration_id: str) -> dict | None:
    return next((i for i in load_catalog(settings) if i["id"] == integration_id), None)


def _field_secret(f: dict) -> bool:
    return bool(f.get("secret", True))


def shared_config(settings: Settings) -> dict[str, str]:
    """Org-wide, non-secret config (URLs, emails, ...) for ALL workers, sourced
    from the catalog. These are common to everyone — NOT stored per-employee.
    (Future: a runtime, UI-editable shared store can override these.)"""
    out: dict[str, str] = {}
    for it in load_catalog(settings):
        for f in it["fields"]:
            if not _field_secret(f) and f.get("value"):
                out[f["env"]] = str(f["value"])
    return out


def fetch_allowlist(settings: Settings) -> dict:
    """Domain -> auth config for `orchard-fetch`. Domains come from each
    integration's non-secret URL fields (e.g. GITHUB_API_URL) and any
    data_sources. Written into each tenant home as integrations.json."""
    from urllib.parse import urlparse
    domains: dict[str, dict] = {}

    def add(url, entry):
        host = urlparse(str(url)).hostname if url else None
        if host:
            domains[host.lower()] = entry

    for it in load_catalog(settings):
        token_env = next((f["env"] for f in it["fields"] if _field_secret(f)), None)
        entry = {
            "token_env": token_env,
            "auth": it.get("auth", "bearer"),
            "accept": it.get("accept", "application/json"),
            "user_env": it.get("user_env", ""),
        }
        # non-secret URL fields (the integration's endpoints)
        for f in it["fields"]:
            if not _field_secret(f) and str(f.get("value", "")).startswith("http"):
                add(f["value"], entry)
        # explicit data_sources, if any
        for ds in it.get("data_sources", []):
            add(ds.get("url") if isinstance(ds, dict) else ds, entry)
    return {"domains": domains}


def secret_fields(settings: Settings, integration_id: str) -> list[dict]:
    """Only the token fields of an integration — the ONLY thing an employee enters."""
    it = get(settings, integration_id)
    if not it:
        return []
    return [{"env": f["env"], "label": f.get("label", f["env"])}
            for f in it["fields"] if _field_secret(f)]


def status(settings: Settings, store, tenant_id: str) -> list[dict]:
    """Per integration: per-employee TOKEN status + the shared (org) config
    values. Secret values are NEVER returned; non-secret shared values ARE (so
    the UI shows them). `configured` depends only on the employee's tokens."""
    have = store.all(tenant_id)  # per-tenant secrets = tokens ONLY
    out = []
    for it in load_catalog(settings):
        fields = []
        tokens_set = True
        for f in it["fields"]:
            env = f["env"]
            if _field_secret(f):
                is_set = env in have and have[env] != ""
                tokens_set = tokens_set and is_set
                fields.append({"env": env, "label": f.get("label", env),
                               "secret": True, "set": is_set})
            else:
                val = str(f.get("value", ""))     # common/org value
                fields.append({"env": env, "label": f.get("label", env),
                               "secret": False, "shared": True,
                               "value": val, "set": bool(val)})
        out.append({
            "id": it["id"],
            "name": it.get("name", it["id"]),
            "icon": it.get("icon", "🔌"),
            "docs_url": it.get("docs_url", ""),
            "configured": tokens_set,   # only the employee's tokens matter
            "fields": fields,
        })
    return out


def delete(settings: Settings, store, tenant_id: str, integration_id: str) -> int:
    """Remove the employee's TOKENS for this integration (shared config stays)."""
    it = get(settings, integration_id)
    if not it:
        return 0
    n = 0
    for f in it["fields"]:
        if _field_secret(f) and store.delete(tenant_id, f["env"]):
            n += 1
    return n
