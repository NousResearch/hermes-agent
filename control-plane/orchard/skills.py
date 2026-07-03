"""Read which secrets a tenant's skills require.

A custom skill declares its data source + required token(s) in SKILL.md
frontmatter:

    metadata:
      orchard:
        data_sources:
          - name: jira
            url: https://jira.corp/api
        secrets:
          - env: JIRA_TOKEN
            label: "Jira API token"
            required: true
            docs_url: https://...

We scan both the tenant's own skills (writable, where they extend/add) and the
shared base library (read-only, curated by the admin).
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .config import Settings


def _parse_frontmatter(path: Path) -> dict:
    try:
        text = path.read_text()
    except Exception:
        return {}
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        return yaml.safe_load(parts[1]) or {}
    except Exception:
        return {}


def _skill_dirs(settings: Settings, tenant_id: str) -> list[Path]:
    dirs = [settings.paths.home_for(tenant_id) / "skills"]
    if settings.skills.shared_dir:
        dirs.append(Path(settings.skills.shared_dir))
    return [d for d in dirs if d.is_dir()]


def required_secrets(settings: Settings, tenant_id: str) -> list[dict]:
    """All secrets required across the tenant's skills, deduped by env name.
    Each: {env, label, required, docs_url, skills: [names]}."""
    by_env: dict[str, dict] = {}
    for base in _skill_dirs(settings, tenant_id):
        for skill_md in base.rglob("SKILL.md"):
            fm = _parse_frontmatter(skill_md)
            name = (fm.get("name") or skill_md.parent.name)
            orchard = ((fm.get("metadata") or {}).get("orchard") or {})
            for sec in (orchard.get("secrets") or []):
                env = sec.get("env")
                if not env:
                    continue
                entry = by_env.setdefault(env, {
                    "env": env,
                    "label": sec.get("label", env),
                    "required": bool(sec.get("required", True)),
                    "docs_url": sec.get("docs_url", ""),
                    "skills": [],
                })
                if name not in entry["skills"]:
                    entry["skills"].append(name)
    return list(by_env.values())


def secret_status(settings: Settings, store, tenant_id: str) -> list[dict]:
    """required_secrets annotated with whether each is currently set (no values)."""
    have = set(store.names(tenant_id))
    out = []
    for s in required_secrets(settings, tenant_id):
        out.append({**s, "set": s["env"] in have})
    return out
