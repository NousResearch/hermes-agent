#!/usr/bin/env python3
"""Launch a Codex create-PR task from a stored Sentry action packet.

This helper is intentionally conservative: it validates the local repo mapping,
writes a complete prompt/log bundle under ~/.hermes/sentry-actions/<id>/, and
starts a detached tmux session. Codex is instructed to commit, push, AND create
PR, and never merge.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

DEFAULT_REPO_MAP = {
    "incremnt": "/home/colm/onemore",
    "onemore": "/home/colm/onemore",
    "scenr": "/home/colm/scenr",
    "happenings": "/home/colm/scenr",
}


def _slug(value: str, fallback: str = "sentry") -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._").lower()
    return value[:80] or fallback


def _deep_get(data: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        cur: Any = data
        ok = True
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur not in (None, ""):
            return cur
    return None


def _load_repo_map() -> dict[str, str]:
    mapping = dict(DEFAULT_REPO_MAP)
    cfg = get_hermes_home() / "sentry_repo_map.json"
    if cfg.exists():
        loaded = json.loads(cfg.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            mapping.update({str(k).lower(): str(v) for k, v in loaded.items()})
    return mapping


def _resolve_repo(payload: dict[str, Any]) -> tuple[str, Path]:
    project = str(
        _deep_get(payload, "project", "data.project", "event.project", "issue.project.slug", "issue.project")
        or ""
    )
    slug = _slug(project)
    repo_map = _load_repo_map()
    repo_path = repo_map.get(slug)
    if not repo_path:
        raise SystemExit(
            f"No repo mapping for Sentry project '{project or slug}'. Add it to "
            f"{get_hermes_home() / 'sentry_repo_map.json'}"
        )
    path = Path(repo_path).expanduser().resolve()
    if not (path / ".git").exists():
        raise SystemExit(f"Mapped repo path is not a git repo: {path}")
    return slug, path


def build_codex_prompt(action_packet: dict[str, Any]) -> str:
    payload = action_packet.get("payload") if isinstance(action_packet.get("payload"), dict) else {}
    project_slug, repo_path = _resolve_repo(payload)
    issue_url = _deep_get(payload, "url", "issue.url", "web_url", "data.url") or ""
    issue_id = _deep_get(payload, "id", "issue.id", "issue.shortId", "data.issue.id") or action_packet.get("id")
    title = _deep_get(payload, "title", "issue.title", "message", "data.title") or "Sentry alert"
    environment = _deep_get(payload, "environment", "event.environment", "data.environment") or "unknown"
    release = _deep_get(payload, "release", "event.release", "data.release") or "unknown"

    raw = json.dumps(payload, indent=2, sort_keys=True)[:12000]
    return f"""Fix Sentry issue and create a PR.

Sentry context:
- Project: {project_slug}
- Issue: {issue_id}
- Title: {title}
- Environment: {environment}
- Release: {release}
- URL: {issue_url}

Repo:
- Path: {repo_path}

Raw Sentry webhook payload, possibly truncated:
```json
{raw}
```

Instructions:
1. Investigate the root cause from the Sentry payload, stack trace, tags, release, and local code.
2. Make the smallest safe code fix.
3. Add or update a regression test where practical.
4. Run the relevant tests.
5. Commit, push, AND create PR.
6. Do not merge the PR.
7. Report the PR URL and a concise root-cause summary.
"""


def launch(action_path: Path) -> Path:
    packet = json.loads(action_path.read_text(encoding="utf-8"))
    payload = packet.get("payload") if isinstance(packet.get("payload"), dict) else {}
    project_slug, repo_path = _resolve_repo(payload)
    action_id = _slug(str(packet.get("id") or action_path.stem), "sentry-action")
    run_dir = get_hermes_home() / "sentry-actions" / action_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_codex_prompt(packet)
    prompt_path = run_dir / "prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    log_path = run_dir / "codex.log"

    session = _slug(f"sentry-{project_slug}-{action_id}")[:80]
    cmd = (
        f"cd {shlex.quote(str(repo_path))} && "
        f"codex exec --full-auto {shlex.quote(prompt)} "
        f"> {shlex.quote(str(log_path))} 2>&1; "
        f"printf '\\nDONE rc=%s at {datetime.now(timezone.utc).isoformat()}\\n' \"$?\" >> {shlex.quote(str(log_path))}"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session, cmd], check=True)
    (run_dir / "session.txt").write_text(session + "\n", encoding="utf-8")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action_packet", type=Path)
    parser.add_argument("--print-prompt", action="store_true")
    args = parser.parse_args()
    packet = json.loads(args.action_packet.read_text(encoding="utf-8"))
    if args.print_prompt:
        print(build_codex_prompt(packet))
        return 0
    run_dir = launch(args.action_packet)
    print(f"Launched Sentry Create PR action. Logs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
