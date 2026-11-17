"""Shared library for the educator profile's cron wake-gate scripts.

Lives next to ``educator_release_watch.py`` and ``educator_explore.py`` in
``<profile>/scripts/`` (the only directory cron runs scripts from). No LLM,
no writes to any outside system: Notion and Jira are only read (GET, plus
Notion's read-only ``…/query`` POST), and the two git checkouts are only
fetched and reset.

How the scheduler runs these scripts (cron/scheduler.py ``_run_job_script``):

- interpreter: the gateway's own ``sys.executable``, so ``cron.*`` /
  ``agent.*`` imports resolve against the installed Hermes;
- cwd: the job's ``workdir`` (the workspace);
- env: the owning profile's scoped environment (multiplex fix 4f), then
  ``_sanitize_subprocess_env`` — Notion / Jira / GitHub / Zendesk variables
  pass, **every ``SLACK_*`` token is stripped**. So nothing here can post to
  Slack, and the veto ``run-due`` drivers (which need the Slack token to read
  ❌ reactions and to post) are run by the agent's own tools instead: the
  script only *detects* due veto actions and wakes the agent.

Output contract (``cron/scheduler.py::_parse_wake_gate``): one compact JSON
line on stdout. ``{"wakeAgent": false}`` skips the agent run entirely (no
LLM cost); any other JSON object wakes it and is injected into the prompt as
``## Script Output``.

State lives in ``<profile>/cron/educator-state/<job>.json`` — under ``cron/``,
which the profile's own agent cannot write (``agent_write_guard``), so a
prompt-injected agent cannot suppress its own wake-ups.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2025-09-03"
HTTP_TIMEOUT = 30
GIT_TIMEOUT = 300
JIRA_KEY = re.compile(r"\b[A-Z][A-Z0-9_]+-\d+\b")
CHECKOUTS = ("product-skills", "ea-monorepo")

# Veto drivers, by autonomy action: the tool call that executes due ones.
VETO_TOOL_CALLS = {
    "post_release_update": "slack_post_thread(run_due=true)",
    "zendesk_create_draft": "zendesk_create_draft(run_due=true)",
}


class SourceError(RuntimeError):
    """An outside system could not be read; reported once per message."""

    def __init__(self, source: str, message: str):
        super().__init__(message)
        self.source = source


# ── paths, config, time ──────────────────────────────────────────────────

def hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:  # pragma: no cover - outside an installed Hermes
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def workspace() -> Path:
    return hermes_home() / "workspace"


def load_profile_config() -> dict:
    """The profile's config.yaml (read-only). The scripts take every data
    source id and property name from it, so they never drift from the
    write guard's own settings."""
    import yaml
    path = hermes_home() / "config.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def notion_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def parse_ts(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


# ── state ────────────────────────────────────────────────────────────────

def state_path(job: str) -> Path:
    return hermes_home() / "cron" / "educator-state" / f"{job}.json"


def load_state(job: str) -> Optional[dict]:
    try:
        data = json.loads(state_path(job).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def save_state(job: str, state: dict) -> None:
    path = state_path(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1, sort_keys=True, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── HTTP (stdlib; replaced in tests) ─────────────────────────────────────

def http_json(method: str, url: str, headers: dict, body: Any = None,
              timeout: int = HTTP_TIMEOUT) -> tuple[int, dict]:
    """One request → ``(status, parsed JSON object)``. Honors the standard
    proxy environment variables (urllib does)."""
    data = None
    headers = dict(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    try:
        parsed = json.loads(raw.decode("utf-8") or "{}")
    except (UnicodeDecodeError, ValueError):
        parsed = {}
    return status, parsed if isinstance(parsed, dict) else {}


def _call(source: str, method: str, url: str, headers: dict, body: Any = None) -> tuple[int, dict]:
    try:
        return http_json(method, url, headers, body)
    except Exception as e:  # network error, timeout, TLS …
        raise SourceError(source, f"{type(e).__name__}: {e}") from e


# ── Notion (read-only) ───────────────────────────────────────────────────

def _notion_headers() -> dict:
    token = env("NOTION_API_KEY") or env("NOTION_API_TOKEN")
    if not token:
        raise SourceError("notion", "NOTION_API_KEY is not set in the profile's .env")
    return {"Authorization": f"Bearer {token}", "Notion-Version": NOTION_VERSION}


def _notion_error(status: int, data: dict) -> SourceError:
    return SourceError("notion", f"HTTP {status}: {str(data.get('message') or '')[:200]}")


def notion_get(path: str) -> dict:
    status, data = _call("notion", "GET", f"{NOTION_API}/{path}", _notion_headers())
    if status != 200:
        raise _notion_error(status, data)
    return data


def notion_query(data_source_id: str, filter_: dict, limit_pages: int = 20) -> list[dict]:
    """All rows of a data source matching ``filter_`` (paginated, bounded)."""
    rows: list[dict] = []
    body: dict = {"filter": filter_, "page_size": 100}
    for _ in range(limit_pages):
        status, data = _call("notion", "POST", f"{NOTION_API}/data_sources/{data_source_id}/query",
                             _notion_headers(), body)
        if status != 200:
            raise _notion_error(status, data)
        rows.extend(r for r in data.get("results") or [] if isinstance(r, dict))
        if not data.get("has_more") or not data.get("next_cursor"):
            break
        body = {**body, "start_cursor": data["next_cursor"]}
    return rows


def prop_text(prop: Any) -> str:
    """Plain text of any Notion property value (title, rich_text, url,
    select, status, multi_select, people …)."""
    if not isinstance(prop, dict):
        return ""
    kind = prop.get("type")
    val = prop.get(kind) if kind else None
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return str(val.get("name") or val.get("plain_text") or "")
    if isinstance(val, list):
        parts = []
        for item in val:
            if isinstance(item, dict):
                parts.append(str(item.get("plain_text") or item.get("name") or ""))
        return " ".join(p for p in parts if p)
    return ""


def prop_names(prop: Any) -> list[str]:
    if isinstance(prop, dict) and isinstance(prop.get(prop.get("type") or ""), list):
        return [str(i.get("name")) for i in prop[prop["type"]] if isinstance(i, dict) and i.get("name")]
    return []


def page_title(page: dict) -> str:
    for prop in (page.get("properties") or {}).values():
        if isinstance(prop, dict) and prop.get("type") == "title":
            return prop_text(prop)
    return ""


def jira_keys(text: str) -> list[str]:
    return list(dict.fromkeys(JIRA_KEY.findall(text or "")))


# ── Jira (read-only) ─────────────────────────────────────────────────────

def _jira_auth() -> tuple[str, dict]:
    base = env("JIRA_BASE_URL").rstrip("/")
    email, token = env("JIRA_EMAIL"), env("JIRA_API_TOKEN")
    if not (base and email and token):
        raise SourceError("jira", "JIRA_BASE_URL / JIRA_EMAIL / JIRA_API_TOKEN are not set in the profile's .env")
    basic = base64.b64encode(f"{email}:{token}".encode()).decode()
    return base, {"Authorization": f"Basic {basic}", "Accept": "application/json"}


def jira_statuses(keys: list[str]) -> dict[str, Optional[str]]:
    """``{key: status name}``; a key Jira does not know maps to None.

    One ``search/jql`` call per 50 keys; a 400 (JQL with an unknown key
    fails as a whole) falls back to one read per key."""
    out: dict[str, Optional[str]] = {}
    if not keys:
        return out
    base, headers = _jira_auth()
    for i in range(0, len(keys), 50):
        chunk = keys[i:i + 50]
        q = urllib.parse.urlencode({"jql": f"key in ({','.join(chunk)})", "fields": "status",
                                    "maxResults": 100})
        status, data = _call("jira", "GET", f"{base}/rest/api/3/search/jql?{q}", headers)
        if status == 200:
            found = {str(it.get("key")): str(((it.get("fields") or {}).get("status") or {}).get("name") or "")
                     for it in data.get("issues") or [] if isinstance(it, dict)}
            for key in chunk:
                out[key] = found.get(key)
            continue
        if status != 400:
            raise SourceError("jira", f"HTTP {status}: {str(data.get('errorMessages') or '')[:200]}")
        for key in chunk:
            s, d = _call("jira", "GET", f"{base}/rest/api/3/issue/{key}?fields=status", headers)
            if s == 404:
                out[key] = None
            elif s == 200:
                out[key] = str(((d.get("fields") or {}).get("status") or {}).get("name") or "")
            else:
                raise SourceError("jira", f"HTTP {s} reading {key}")
    return out


# ── git checkouts (fetch + reset only; replaced in tests) ────────────────

def run_git(args: list[str], cwd: str, env: dict) -> tuple[int, str]:
    try:
        proc = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True,
                              text=True, timeout=GIT_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, f"{type(e).__name__}: {e}"
    return proc.returncode, (proc.stderr or proc.stdout or "").strip()


def _git_env() -> dict:
    e = dict(os.environ)
    e["GIT_TERMINAL_PROMPT"] = "0"
    token = env("GITHUB_PR_TOKEN")
    if token:
        # Passed through git's env-config, never argv, so it is not visible
        # in the process list. Fine-grained PATs authenticate as Basic
        # x-access-token:<token>.
        cred = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        e["GIT_CONFIG_COUNT"] = "1"
        e["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
        e["GIT_CONFIG_VALUE_0"] = f"Authorization: Basic {cred}"
    return e


def refresh_checkouts(root: Optional[Path] = None) -> dict[str, Optional[str]]:
    """Bring each read-only checkout to its upstream tip. Returns
    ``{repo: error or None}``. ``reset --hard`` (not ``pull``) so a stray
    local change can never wedge the update; nothing is ever pushed."""
    root = root or workspace() / "checkouts"
    results: dict[str, Optional[str]] = {}
    genv = _git_env()
    for repo in CHECKOUTS:
        path = root / repo
        if not path.is_dir():
            results[repo] = f"checkout missing at {path} — clone it (ROLLOUT.md step 3)"
            continue
        fetch = ["fetch", "--quiet", "--prune", "origin"]
        if (path / ".git" / "shallow").exists():
            fetch.insert(2, "--depth=1")
        err = None
        for args in (fetch, ["reset", "--quiet", "--hard", "@{upstream}"]):
            rc, out = run_git(args, str(path), genv)
            if rc != 0:
                err = f"git {args[0]} failed (exit {rc}): {out[:300]}"
                break
        results[repo] = err
    return results


# ── cron ledger, job lookup, veto (replaced in tests) ────────────────────

def find_job_id(script_name: str) -> Optional[str]:
    """The pre-run script is not told its job id: find the job whose
    ``script`` is this file."""
    try:
        from cron.jobs import list_jobs
        for job in list_jobs(include_disabled=True):
            if Path(str(job.get("script") or "")).name == script_name:
                return str(job.get("id"))
    except Exception:
        return None
    return None


def list_executions(job_id: str) -> list[dict]:
    """Newest-first execution rows for the job (cron/executions.py)."""
    from cron.executions import list_executions as _list
    return _list(job_id=job_id, limit=500)


def due_pending_ids(action: str) -> list[str]:
    """Veto actions whose waiting window has passed (agent/autonomy.py) —
    a local file read; executing them needs the agent's Slack token."""
    try:
        from agent import autonomy
        return autonomy.due_pending_ids(action, utcnow())
    except Exception:
        return []


# ── shared checks ────────────────────────────────────────────────────────

def previous_woken_run(job_id: Optional[str], state: dict) -> Optional[dict]:
    """The finished execution in which this script last woke the agent."""
    at = parse_ts((state.get("last_wake") or {}).get("at"))
    if not job_id or at is None:
        return None
    try:
        rows = list_executions(job_id)
    except Exception:
        return None
    for row in rows:
        if row.get("status") in ("claimed", "running"):
            continue
        claimed = parse_ts(row.get("claimed_at"))
        if claimed is not None and claimed <= at:
            return row
    return None


def notebook_info() -> tuple[Optional[float], int]:
    nb = workspace() / "notebook.md"
    try:
        st = nb.stat()
    except OSError:
        return None, 0
    return st.st_mtime, st.st_size


def notebook_missed(run: Optional[dict]) -> bool:
    """True when the woken run finished without rewriting the notebook."""
    if not run:
        return False
    started = parse_ts(run.get("started_at")) or parse_ts(run.get("claimed_at"))
    if started is None:
        return False
    mtime, _size = notebook_info()
    return mtime is None or mtime < started.timestamp()


def emit(payload: dict) -> dict:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str))
    return payload
