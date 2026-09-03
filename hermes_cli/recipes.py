"""Hermes Recipes — shareable setup bundles (export / preview / install).

Inspired by Poke Recipes (poke.com/docs/creating-recipes, released
2026-03-19): one shareable file that bundles a starter prompt, the
automations, skills, and MCP integrations that make a workflow, so another
user can adopt the whole setup in one command.

Design constraints (deliberate deltas from Poke's hosted marketplace):

- **Secrets never travel.** Export strips API keys, headers, env blocks, and
  anything secret-shaped from MCP entries; the recipe records only the *names*
  of secrets the installer must supply (``required_secrets``).
- **Consent-first install.** ``hermes recipe install`` previews everything the
  recipe would add and asks before writing. Cron jobs are installed *paused*
  by default (enable with ``--enable`` or ``hermes cron resume``).
- **No code execution from recipes.** stdio MCP servers (``command:``) and
  cron ``script`` / ``monitor_script`` fields are refused on install — a
  recipe is data, not a program. Remote MCP URLs are validated through the
  SSRF guard before being written to config.
- **File or URL.** Recipes are plain YAML; share them as gists, repo files,
  or any https URL. No marketplace, no payouts — the sharing half only.

Zero model-tool footprint: this is a CLI command + docs, per the footprint
ladder in AGENTS.md.
"""

from __future__ import annotations

import copy
import datetime as _dt
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

RECIPE_VERSION = 1

# Fields on a cron job that are safe to carry in a recipe. Everything else
# (run history, streaks, origin chat ids, scripts) is host-local state.
_JOB_EXPORT_FIELDS = (
    "name",
    "prompt",
    "schedule",
    "repeat",
    "deliver",
    "skills",
    "enabled_toolsets",
)

# Keys inside an MCP server entry that must never be exported, matched
# case-insensitively as substrings. ``headers`` and ``env`` are dropped
# wholesale (they are where credentials live).
_SECRET_KEY_PATTERN = re.compile(
    r"(api[_-]?key|token|secret|password|credential|authorization|bearer)",
    re.IGNORECASE,
)
_MCP_DROP_KEYS = {"headers", "env", "auth", "oauth"}
# MCP entry keys allowed into a recipe (allowlist beats blocklist for
# a secret boundary).
_MCP_EXPORT_FIELDS = ("transport", "url", "description", "enabled")

_MAX_RECIPE_BYTES = 256 * 1024  # a recipe is text; 256 KiB is generous


class RecipeError(Exception):
    """User-facing recipe failure (bad format, unsafe content, IO)."""


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _sanitize_mcp_entry(name: str, entry: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Return (sanitized entry, required secret names) for one MCP server.

    stdio servers (``command:``) are refused — a recipe must not carry
    executable configuration.
    """
    if not isinstance(entry, dict):
        raise RecipeError(f"MCP server '{name}' has a non-dict config; cannot export")
    if entry.get("command"):
        raise RecipeError(
            f"MCP server '{name}' is a stdio server (command:). Recipes only "
            "carry remote (http/sse) servers — stdio config is executable and "
            "host-specific."
        )
    if not entry.get("url"):
        raise RecipeError(f"MCP server '{name}' has no url; cannot export")

    required: List[str] = []
    for key in entry:
        if key in _MCP_DROP_KEYS and entry.get(key):
            if key == "headers" and isinstance(entry[key], dict):
                required.extend(sorted(entry[key].keys()))
            else:
                required.append(key)
        elif _SECRET_KEY_PATTERN.search(key) and entry.get(key):
            required.append(key)

    clean = {k: copy.deepcopy(entry[k]) for k in _MCP_EXPORT_FIELDS if k in entry}
    return clean, required


def _sanitize_job(job: Dict[str, Any]) -> Dict[str, Any]:
    if job.get("no_agent") or job.get("script") or job.get("monitor_script"):
        raise RecipeError(
            f"cron job '{job.get('name') or job.get('id')}' uses a local script; "
            "script-backed jobs are host-specific and cannot be exported"
        )
    clean: Dict[str, Any] = {}
    for key in _JOB_EXPORT_FIELDS:
        value = job.get(key)
        if value not in (None, "", []):
            clean[key] = copy.deepcopy(value)
    schedule = clean.get("schedule")
    if isinstance(schedule, dict):
        # jobs.json stores the parsed schedule dict; flatten back to the
        # portable string form so the recipe stays human-editable.
        flat = (
            schedule.get("raw")
            or schedule.get("display")
            or schedule.get("expr")
        )
        if not flat:
            raise RecipeError(
                f"cron job '{job.get('name') or job.get('id')}' has a "
                "schedule that cannot be exported as a string"
            )
        clean["schedule"] = flat
    repeat = clean.get("repeat")
    if isinstance(repeat, dict):
        # stored as {"times": N|null, "completed": M}; only the target count
        # is portable.
        times = repeat.get("times")
        if times:
            clean["repeat"] = int(times)
        else:
            clean.pop("repeat", None)
    if not clean.get("prompt"):
        raise RecipeError(
            f"cron job '{job.get('name') or job.get('id')}' has no prompt; "
            "only prompt-based jobs are exportable"
        )
    if not clean.get("schedule"):
        raise RecipeError(f"cron job '{job.get('name') or job.get('id')}' has no schedule")
    # Delivery targets like specific chat ids are host-local; keep only
    # generic targets.
    if clean.get("deliver") not in (None, "local", "origin", "log"):
        clean["deliver"] = "local"
    return clean


def build_recipe(
    *,
    name: str,
    description: str = "",
    author: str = "",
    starter_prompt: str = "",
    job_ids: Optional[List[str]] = None,
    mcp_names: Optional[List[str]] = None,
    skills: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble a recipe dict from the current install's state."""
    from cron import jobs as cron_jobs
    from hermes_cli.config import load_config

    recipe: Dict[str, Any] = {
        "recipe": RECIPE_VERSION,
        "name": name,
        "description": description,
        "author": author,
        "created_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if starter_prompt:
        recipe["starter_prompt"] = starter_prompt
    if skills:
        recipe["skills"] = list(skills)

    if job_ids:
        exported = []
        all_jobs = {j.get("id"): j for j in cron_jobs.load_jobs()}
        for jid in job_ids:
            job = all_jobs.get(jid)
            if job is None:
                # allow matching by name too
                matches = [j for j in all_jobs.values() if j.get("name") == jid]
                if len(matches) == 1:
                    job = matches[0]
            if job is None:
                raise RecipeError(f"cron job '{jid}' not found")
            exported.append(_sanitize_job(job))
        if exported:
            recipe["cron_jobs"] = exported

    if mcp_names:
        config = load_config()
        servers = config.get("mcp_servers") or {}
        out: Dict[str, Any] = {}
        all_required: Dict[str, List[str]] = {}
        for sname in mcp_names:
            entry = servers.get(sname)
            if entry is None:
                raise RecipeError(f"MCP server '{sname}' not found in config.yaml")
            clean, required = _sanitize_mcp_entry(sname, entry)
            out[sname] = clean
            if required:
                all_required[sname] = required
        if out:
            recipe["mcp_servers"] = out
        if all_required:
            recipe["required_secrets"] = all_required

    return recipe


def dump_recipe(recipe: Dict[str, Any]) -> str:
    buf = io.StringIO()
    yaml.safe_dump(recipe, buf, sort_keys=False, allow_unicode=True, width=100)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Load / validate
# ---------------------------------------------------------------------------

def _fetch_url(url: str) -> str:
    from tools.url_safety import create_ssrf_safe_client, is_safe_url

    if not is_safe_url(url):
        raise RecipeError(f"unsafe recipe URL refused: {url}")
    with create_ssrf_safe_client(timeout=20, follow_redirects=False) as client:
        resp = client.get(url)
    if resp.status_code != 200:
        raise RecipeError(f"fetching recipe failed: HTTP {resp.status_code}")
    if len(resp.content) > _MAX_RECIPE_BYTES:
        raise RecipeError("recipe too large (limit 256 KiB)")
    return resp.text


def load_recipe(source: str) -> Dict[str, Any]:
    """Load and validate a recipe from a local path or an http(s) URL."""
    if source.startswith(("http://", "https://")):
        text = _fetch_url(source)
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise RecipeError(f"recipe file not found: {source}")
        if path.stat().st_size > _MAX_RECIPE_BYTES:
            raise RecipeError("recipe too large (limit 256 KiB)")
        text = path.read_text(encoding="utf-8")

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RecipeError(f"invalid recipe YAML: {exc}") from exc
    return validate_recipe(data)


def validate_recipe(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise RecipeError("recipe must be a YAML mapping")
    version = data.get("recipe")
    if not isinstance(version, int) or version < 1:
        raise RecipeError("missing/invalid 'recipe' version field (expected: recipe: 1)")
    if version > RECIPE_VERSION:
        raise RecipeError(
            f"recipe version {version} is newer than this Hermes understands "
            f"({RECIPE_VERSION}); update Hermes first"
        )
    if not isinstance(data.get("name"), str) or not data["name"].strip():
        raise RecipeError("recipe needs a non-empty 'name'")

    jobs = data.get("cron_jobs") or []
    if not isinstance(jobs, list):
        raise RecipeError("'cron_jobs' must be a list")
    for job in jobs:
        if not isinstance(job, dict):
            raise RecipeError("each cron job must be a mapping")
        for banned in ("script", "monitor_script", "no_agent", "workdir", "command"):
            if job.get(banned):
                raise RecipeError(
                    f"cron job '{job.get('name', '?')}' carries '{banned}' — "
                    "recipes must not contain executable/host-specific config"
                )
        if not job.get("prompt") or not job.get("schedule"):
            raise RecipeError("each recipe cron job needs 'prompt' and 'schedule'")
        if not isinstance(job.get("schedule"), str):
            raise RecipeError(
                f"cron job '{job.get('name', '?')}' schedule must be a string "
                "(e.g. 'every 2h' or '0 8 * * *')"
            )
        if job.get("repeat") is not None and not isinstance(job["repeat"], int):
            raise RecipeError(
                f"cron job '{job.get('name', '?')}' repeat must be an integer"
            )

    servers = data.get("mcp_servers") or {}
    if not isinstance(servers, dict):
        raise RecipeError("'mcp_servers' must be a mapping")
    from tools.url_safety import is_safe_url

    for sname, entry in servers.items():
        if not isinstance(entry, dict):
            raise RecipeError(f"MCP server '{sname}' must be a mapping")
        if entry.get("command"):
            raise RecipeError(
                f"MCP server '{sname}' is a stdio server (command:) — refused; "
                "recipes may only reference remote http/sse servers"
            )
        url = entry.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise RecipeError(f"MCP server '{sname}' needs an http(s) 'url'")
        for key in entry:
            if key in _MCP_DROP_KEYS or _SECRET_KEY_PATTERN.search(key):
                raise RecipeError(
                    f"MCP server '{sname}' carries a secret-shaped field "
                    f"('{key}') — recipes must not contain credentials"
                )
        if not is_safe_url(url):
            raise RecipeError(f"MCP server '{sname}' URL refused by the SSRF guard: {url}")

    skills = data.get("skills") or []
    if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
        raise RecipeError("'skills' must be a list of skill identifiers")

    return data


# ---------------------------------------------------------------------------
# Preview / install
# ---------------------------------------------------------------------------

def describe_recipe(recipe: Dict[str, Any]) -> str:
    """Human-readable preview of what installing the recipe would add."""
    lines: List[str] = []
    lines.append(f"Recipe: {recipe['name']}")
    if recipe.get("author"):
        lines.append(f"Author: {recipe['author']}")
    if recipe.get("description"):
        lines.append(f"  {recipe['description']}")
    jobs = recipe.get("cron_jobs") or []
    if jobs:
        lines.append(f"\nCron jobs ({len(jobs)}) — installed PAUSED unless --enable:")
        for job in jobs:
            sched = job.get("schedule")
            if isinstance(sched, dict):
                sched = sched.get("raw") or json.dumps(sched)
            lines.append(f"  • {job.get('name') or '(unnamed)'}  [{sched}]")
            prompt = str(job.get("prompt", "")).strip().splitlines()[0]
            lines.append(f"      {prompt[:100]}{'…' if len(prompt) > 100 else ''}")
    servers = recipe.get("mcp_servers") or {}
    if servers:
        lines.append(f"\nMCP servers ({len(servers)}):")
        for sname, entry in servers.items():
            lines.append(f"  • {sname}: {entry.get('url')}")
    required = recipe.get("required_secrets") or {}
    if required:
        lines.append("\nSecrets you must supply after install (never shipped in recipes):")
        for sname, keys in required.items():
            lines.append(f"  • {sname}: {', '.join(keys)}")
    skills = recipe.get("skills") or []
    if skills:
        lines.append(f"\nSkills to install ({len(skills)}):")
        for s in skills:
            lines.append(f"  • {s}")
    if recipe.get("starter_prompt"):
        lines.append("\nStarter prompt:")
        lines.append(f"  {recipe['starter_prompt'][:200]}")
    return "\n".join(lines)


def install_recipe(
    recipe: Dict[str, Any],
    *,
    enable_jobs: bool = False,
) -> Dict[str, Any]:
    """Apply a validated recipe. Returns a summary dict of what was created.

    Cron jobs are created paused unless ``enable_jobs``. MCP servers are
    merged into config.yaml (existing entries with the same name are NOT
    overwritten). Skills are reported for manual install (the hub flow owns
    quarantine/consent) rather than auto-installed.
    """
    from cron import jobs as cron_jobs
    from hermes_cli.config import load_config, save_config

    summary: Dict[str, Any] = {"cron_jobs": [], "mcp_servers": [], "mcp_skipped": [],
                               "skills": list(recipe.get("skills") or [])}

    for job in recipe.get("cron_jobs") or []:
        created = cron_jobs.create_job(
            prompt=job["prompt"],
            schedule=str(job["schedule"]),
            name=job.get("name"),
            repeat=job.get("repeat"),
            deliver=job.get("deliver") or "local",
            skills=job.get("skills"),
            enabled_toolsets=job.get("enabled_toolsets"),
        )
        if not enable_jobs:
            cron_jobs.update_job(created["id"], {"enabled": False})
        summary["cron_jobs"].append({"id": created["id"], "name": created.get("name")})

    servers = recipe.get("mcp_servers") or {}
    if servers:
        config = load_config()
        existing = config.get("mcp_servers")
        if not isinstance(existing, dict):
            existing = {}
        changed = False
        for sname, entry in servers.items():
            if sname in existing:
                summary["mcp_skipped"].append(sname)
                continue
            existing[sname] = copy.deepcopy(entry)
            summary["mcp_servers"].append(sname)
            changed = True
        if changed:
            config["mcp_servers"] = existing
            save_config(config, merge_existing=True)

    return summary
