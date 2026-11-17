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


# Data sources whose query hit the page cap this run (results incomplete);
# the caller reports them. Reset by the caller.
TRUNCATED_QUERIES: list[str] = []


def notion_query(data_source_id: str, filter_: dict, limit_pages: int = 20) -> list[dict]:
    """All rows of a data source matching ``filter_`` (paginated, bounded at
    ``limit_pages`` × 100 rows; hitting the bound is recorded in
    ``TRUNCATED_QUERIES``, never silent)."""
    rows: list[dict] = []
    body: dict = {"filter": filter_, "page_size": 100}
    for _ in range(limit_pages):
        status, data = _call("notion", "POST", f"{NOTION_API}/data_sources/{data_source_id}/query",
                             _notion_headers(), body)
        if status != 200:
            raise _notion_error(status, data)
        rows.extend(r for r in data.get("results") or [] if isinstance(r, dict))
        if not data.get("has_more") or not data.get("next_cursor"):
            return rows
        body = {**body, "start_cursor": data["next_cursor"]}
    TRUNCATED_QUERIES.append(data_source_id)
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


# ── approval measurement (product-skills learn-from-approval contract) ───
#
# The agent has no code execution and must not be the author of its own
# track record, so the script measures: Notion's markdown of the approved
# row vs the script's own Draft-time baseline, whitespace collapsed and the
# header line dropped, Levenshtein distance in characters.

EDIT_DISTANCE_CELL_BUDGET = 40_000_000
_HEADER = re.compile(r"\A\s*<!--.*?-->[^\S\n]*\n?", re.DOTALL)


def normalise_markdown(text: str) -> str:
    return re.sub(r"\s+", " ", _HEADER.sub("", text or "", count=1)).strip()


def _banded_levenshtein(a: str, b: str, k: int) -> Optional[int]:
    """Exact distance if it is <= k, else None. O(len(a) * k)."""
    n, m = len(a), len(b)
    if abs(n - m) > k:
        return None
    inf = k + 1
    width = 2 * k + 1
    # row i holds D[i][j] for j = i-k .. i+k at index t = j - i + k
    prev = [j - k if 0 <= j - k <= m else inf for j in range(width)]
    for i in range(1, n + 1):
        ca = a[i - 1]
        cur = [inf] * width
        best = inf
        for t in range(width):
            j = i - k + t
            if j < 0 or j > m:
                continue
            if j == 0:
                v = i
            else:
                v = prev[t] + (ca != b[j - 1])                 # D[i-1][j-1]
                if t + 1 < width and prev[t + 1] + 1 < v:      # D[i-1][j]
                    v = prev[t + 1] + 1
                if t > 0 and cur[t - 1] + 1 < v:               # D[i][j-1]
                    v = cur[t - 1] + 1
            cur[t] = v if v < inf else inf
            if cur[t] < best:
                best = cur[t]
        if best > k:
            return None
        prev = cur
    d = prev[m - n + k]
    return d if d <= k else None


def edit_distance(a: str, b: str) -> tuple[int, bool]:
    """``(distance, exact)``. Exact Levenshtein while the work stays under
    ``EDIT_DISTANCE_CELL_BUDGET``; beyond that (a near-total rewrite of a
    long document) an upper-bound estimate from a line diff."""
    p = 0
    while p < len(a) and p < len(b) and a[p] == b[p]:
        p += 1
    a, b = a[p:], b[p:]
    s = 0
    while s < len(a) and s < len(b) and a[-1 - s] == b[-1 - s]:
        s += 1
    if s:
        a, b = a[:-s], b[:-s]
    if not a or not b:
        return max(len(a), len(b)), True
    k = max(16, abs(len(a) - len(b)))
    while True:
        if len(a) * (2 * k + 1) > EDIT_DISTANCE_CELL_BUDGET:
            break
        d = _banded_levenshtein(a, b, k)
        if d is not None:
            return d, True
        if k >= max(len(a), len(b)):
            break
        k *= 2
    import difflib
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    approx = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    return max(approx, abs(len(a) - len(b))), False


def page_markdown(page_id: str) -> dict:
    return notion_get(f"pages/{page_id}/markdown")


# Script-owned evidence, under workspace/metrics/ (agent_write_guard
# extra_paths — the agent reads it, never writes it):
#   baselines/<row-id>.md   the row as the script last saw it in Draft (the
#                           agent's final draft, re-drafts included); taken
#                           on first sight past Draft if never seen in Draft
#   approved/<row-id>.md    the row when approved (or edited after approval)
#   edit-rate.csv           one line per approved row, replaced on re-edit
# Graduation evidence reads edit-rate.csv. The agent's own classification of
# the edits goes in workspace/learning/edit-classifications.csv.

EDIT_RATE_HEADER = ("approved_at", "row_id", "asset_type", "destination", "product",
                    "draft_chars", "approved_chars", "chars_changed", "edit_rate",
                    "measured_by", "baseline_status", "baseline_taken_at")


def metrics_dir() -> Path:
    return workspace() / "metrics"


def _row_file(kind: str, row_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f-]{32,36}", row_id or ""):
        raise ValueError(f"not a Notion page id: {row_id!r}")
    return metrics_dir() / kind / f"{row_id}.md"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _fetch_markdown(row_id: str) -> str:
    page = page_markdown(row_id)
    if page.get("truncated"):
        raise SourceError("notion", f"Notion returned truncated markdown for {row_id}")
    return str(page.get("markdown") or "")


def take_baseline(row_id: str, asset_type: str, status: str, now: datetime) -> None:
    """Snapshot the row body as the baseline. Raises SourceError."""
    body = _fetch_markdown(row_id)
    _atomic_text(_row_file("baselines", row_id),
                 f"<!-- baseline row={row_id} asset_type={asset_type} status={status} "
                 f"taken={iso(now)} -->\n{body}\n")


def has_baseline(row_id: str) -> bool:
    return _row_file("baselines", row_id).is_file()


def _baseline_meta(text: str) -> dict:
    m = re.match(r"<!-- baseline (.*?) -->", text)
    return dict(kv.split("=", 1) for kv in (m.group(1).split() if m else []) if "=" in kv)


def upsert_edit_rate(record: dict) -> None:
    import csv
    import io
    path = metrics_dir() / "edit-rate.csv"
    rows = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("row_id") != record["row_id"]]
    rows.append(record)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EDIT_RATE_HEADER, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, buf.getvalue())


def measure_approval(info: dict, now: datetime) -> dict:
    """Measure an approved row against its baseline, save the approved copy,
    write its edit-rate.csv line. Returns the fields added to the
    ``asset_approved`` / ``asset_edited_after_approval`` event."""
    row_id = info["row_id"]
    try:
        approved = _fetch_markdown(row_id)
    except SourceError as e:
        return {"measure_error": f"could not read the approved row: {e}"}
    approved_path = _row_file("approved", row_id)
    _atomic_text(approved_path, f"<!-- approved row={row_id} asset_type={info.get('asset_type')} "
                                f"taken={iso(now)} -->\n{approved}\n")
    approved_n = normalise_markdown(approved)
    out: dict = {"approved_chars": len(approved_n), "approved_path": str(approved_path)}
    record = {"approved_at": iso(now), "row_id": row_id, "asset_type": info.get("asset_type", ""),
              "destination": info.get("destination", ""), "product": info.get("product", ""),
              "approved_chars": len(approved_n)}
    base_path = _row_file("baselines", row_id)
    if not base_path.is_file():
        out.update(snapshot_missing=True, measured_by="none")
        record.update(measured_by="none")
    else:
        text = base_path.read_text(encoding="utf-8")
        meta = _baseline_meta(text)
        draft_n = normalise_markdown(text)
        dist, exact = edit_distance(draft_n, approved_n)
        rate = f"{dist / max(len(draft_n), 1):.4f}"
        measured_by = "script" if exact else "estimate"
        out.update(snapshot_missing=False, draft_chars=len(draft_n), chars_changed=dist,
                   chars_changed_exact=exact, edit_rate=float(rate), measured_by=measured_by,
                   baseline_path=str(base_path), baseline_status=meta.get("status", ""))
        record.update(draft_chars=len(draft_n), chars_changed=dist, edit_rate=rate,
                      measured_by=measured_by, baseline_status=meta.get("status", ""),
                      baseline_taken_at=meta.get("taken", ""))
    upsert_edit_rate(record)
    out["edit_rate_csv"] = str(metrics_dir() / "edit-rate.csv")
    return out


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


# ── git checkouts (verified fetch + reset only; run_git replaced in tests) ──
#
# The checkouts feed skills.external_dirs, so whoever controls their content
# controls the agent's skills, and whoever controls .git/config can make git
# run a program (core.fsmonitor, hooks, credential helpers, filters …) as
# the gateway user with this profile's secrets. The agent cannot write them
# (agent_write_guard.extra_paths: workspace/checkouts); this code does not
# trust them either:
#
# - .git must be a real directory and .git/config must contain only the keys
#   a plain `git clone` writes (parsed here, never by git), with
#   remote.origin.url equal to the expected repo — anything else refuses the
#   refresh and is reported, so an admin re-clones;
# - every git call runs with system/global config off and the dangerous keys
#   forced off on the command line;
# - fetch names the expected URL and an explicit refspec (never the
#   configured remote), and reset targets refs/remotes/origin/<default branch>
#   looked up from the remote (never @{upstream}, which local config decides).

GIT_HARDENING = (
    "-c", "core.fsmonitor=false",
    "-c", "core.hooksPath=/dev/null",
    "-c", "credential.helper=",
    "-c", "protocol.file.allow=never",
    "-c", "protocol.ext.allow=never",
    "-c", "protocol.ssh.allow=never",
    "-c", "core.askPass=",
)
_GIT_CONFIG_ALLOWED = {
    "core": {"repositoryformatversion", "filemode", "bare", "logallrefupdates",
             "ignorecase", "precomposeunicode", "symlinks"},
    "extensions": {"objectformat"},
    "remote.origin": {"url", "fetch"},
}
_BRANCH_KEYS = {"remote", "merge"}
_SECTION = re.compile(r'^\[\s*([A-Za-z0-9.-]+)(?:\s+"((?:[^"\\\n]|\\.)*)")?\s*\]\s*$')
_KEYVAL = re.compile(r"^([A-Za-z][A-Za-z0-9-]*)\s*(?:=\s*(.*?))?\s*$")


def expected_checkouts(cfg: Optional[dict] = None) -> dict[str, str]:
    """``{dir name: https clone URL}`` from ``github_pr.repos`` (the repos
    the profile's token is scoped to), falling back to the two designed."""
    cfg = load_profile_config() if cfg is None else cfg
    gh = cfg.get("github_pr") if isinstance(cfg.get("github_pr"), dict) else {}
    repos = gh.get("repos") if isinstance(gh.get("repos"), dict) else {}
    out = {}
    for name in CHECKOUTS:
        full = next((str(k) for k in repos if str(k).lower().endswith("/" + name)),
                    f"everafterai/{name}")
        out[name] = f"https://github.com/{full}.git"
    return out


def git_config_problem(repo_path: Path, expected_url: str) -> Optional[str]:
    """Why this checkout's git metadata cannot be trusted, or None."""
    git_dir = repo_path / ".git"
    if git_dir.is_symlink() or not git_dir.is_dir():
        return ".git is not a plain directory"
    cfg_path = git_dir / "config"
    if cfg_path.is_symlink() or not cfg_path.is_file():
        return ".git/config is missing or a symlink"
    for extra in ("config.worktree", "commondir"):
        if (git_dir / extra).exists():
            return f".git/{extra} exists"
    section = None
    url = None
    for raw in cfg_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        m = _SECTION.match(line)
        if m:
            name, sub = m.group(1).lower(), m.group(2)
            section = f"{name}.{sub}" if sub is not None else name
            if name == "branch" and sub is not None:
                continue
            if section not in _GIT_CONFIG_ALLOWED:
                return f"unexpected section [{section}] in .git/config"
            continue
        m = _KEYVAL.match(line)
        if not m or section is None or line.endswith("\\"):
            return f"unparseable line in .git/config: {line[:80]!r}"
        key = m.group(1).lower()
        allowed = _BRANCH_KEYS if section.startswith("branch.") else _GIT_CONFIG_ALLOWED[section]
        if key not in allowed:
            return f"unexpected key {section}.{key} in .git/config"
        if section == "remote.origin" and key == "url":
            url = (m.group(2) or "").strip().strip('"')
    if url != expected_url:
        return f"remote.origin.url is {url!r}, expected {expected_url!r}"
    return None


def run_git(args: list[str], cwd: str, env: dict) -> tuple[int, str]:
    try:
        proc = subprocess.run(["git", *GIT_HARDENING, *args], cwd=cwd, env=env,
                              capture_output=True, text=True, timeout=GIT_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, f"{type(e).__name__}: {e}"
    return proc.returncode, (proc.stderr or proc.stdout or "").strip()


def _git_env() -> dict:
    e = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    e.update(GIT_TERMINAL_PROMPT="0", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
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


def _default_branch(url: str, path: Path, genv: dict) -> tuple[Optional[str], Optional[str]]:
    rc, out = run_git(["ls-remote", "--symref", url, "HEAD"], str(path), genv)
    if rc != 0:
        return None, f"git ls-remote failed (exit {rc}): {out[:300]}"
    m = re.search(r"^ref:\s*refs/heads/([A-Za-z0-9._/-]+)\s+HEAD$", out, re.MULTILINE)
    if not m or ".." in m.group(1):
        return None, "could not read the remote's default branch"
    return m.group(1), None


def refresh_checkouts(root: Optional[Path] = None,
                      expected: Optional[dict[str, str]] = None) -> dict[str, Optional[str]]:
    """Bring each read-only checkout to its remote's default-branch tip.
    Returns ``{repo: error or None}``; a checkout whose metadata fails
    ``git_config_problem`` is refused before any git command runs."""
    root = root or workspace() / "checkouts"
    expected = expected or expected_checkouts()
    results: dict[str, Optional[str]] = {}
    genv = _git_env()
    for repo, url in expected.items():
        path = root / repo
        if not path.is_dir():
            results[repo] = f"checkout missing at {path} — clone it (ROLLOUT.md step 4)"
            continue
        problem = git_config_problem(path, url)
        if problem:
            results[repo] = (f"checkout refused, nothing run: {problem} — an admin must delete "
                             f"{path} and re-clone it (ROLLOUT.md step 4)")
            continue
        branch, err = _default_branch(url, path, genv)
        if err is None:
            fetch = ["fetch", "--quiet", "--no-tags", url,
                     f"+refs/heads/{branch}:refs/remotes/origin/{branch}"]
            if (path / ".git" / "shallow").exists():
                fetch.insert(2, "--depth=1")
            for args in (fetch, ["reset", "--quiet", "--hard", f"refs/remotes/origin/{branch}"]):
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
