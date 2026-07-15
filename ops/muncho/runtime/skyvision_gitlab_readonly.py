#!/usr/bin/env python3
"""Bounded SkyVision GitLab helper for Muncho local worker.

The helper intentionally keeps token handling local:
- reads the token from ~/.hermes/secrets/skyvision_gitlab_group_ops.env;
- never prints/stores token values;
- exposes read-only project, branch, MR, pipeline, tree, and file metadata;
- does not expose repository file content or mutations in this build.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

SECRET_ENV = Path.home() / ".hermes" / "secrets" / "skyvision_gitlab_group_ops.env"
DEFAULT_GROUP = "skyvision.bg"
DEFAULT_TIMEOUT = 20

SECRET_KEY_RE = re.compile(r"(token|secret|password|passwd|authorization|private[_-]?key|api[_-]?key|credential)", re.I)
SECRET_VALUE_RE = re.compile(r"(glpat_[A-Za-z0-9_-]+|Bearer\s+[A-Za-z0-9._-]+|PRIVATE-TOKEN\s*[:=])", re.I)
DENIED_PATH_PARTS = {
    ".env",
    ".ssh",
    "storage",
    "vendor",
    "node_modules",
    "secrets",
    "private",
    "dump",
    "dumps",
    "backup",
    "backups",
}
DENIED_EXTENSIONS = {".sql", ".dump", ".bak", ".key", ".pem", ".p12", ".pfx", ".crt", ".log"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if SECRET_KEY_RE.search(str(key)):
                out[key] = item if isinstance(item, bool) else "[REDACTED]"
            else:
                out[key] = sanitize(item)
        return out
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        if SECRET_VALUE_RE.search(value):
            return "[REDACTED]"
        return value
    return value


def response(status: str, **kwargs: Any) -> dict[str, Any]:
    payload = {
        "status": status,
        "generated_at_utc": utc_now(),
        "capability": "skyvision_gitlab_ops",
        "secrets_returned": False,
        "mutation_performed": False,
        **kwargs,
    }
    return sanitize(payload)


def die(status: str, code: int = 2, **kwargs: Any) -> None:
    print(json.dumps(response(status, **kwargs), ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(code)


def load_env() -> dict[str, str]:
    if not SECRET_ENV.exists():
        die("blocked_missing_secret_env", secret_env_path=str(SECRET_ENV), token_value_returned=False)
    env: dict[str, str] = {}
    for line in SECRET_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key.strip()] = val.strip().strip("'\"")
    base_url = env.get("GITLAB_BASE_URL", "").rstrip("/")
    token = env.get("GITLAB_TOKEN", "")
    if not base_url or not token:
        die("blocked_missing_gitlab_base_url_or_token", secret_env_path=str(SECRET_ENV), token_present=bool(token))
    env["GITLAB_BASE_URL"] = base_url
    return env


def quote_path(path: str) -> str:
    return urllib.parse.quote(path, safe="")


def validate_repo_path(path: str | None, *, allow_empty: bool = True) -> str:
    if path is None:
        return ""
    normalized = path.strip().lstrip("/")
    if not normalized:
        if allow_empty:
            return ""
        die("blocked_missing_path")
    if ".." in normalized.split("/"):
        die("blocked_path_traversal", path=normalized)
    parts = {part.lower() for part in normalized.split("/") if part}
    if parts & DENIED_PATH_PARTS:
        die("blocked_denied_path_part", path=normalized, denied=sorted(parts & DENIED_PATH_PARTS))
    suffix = Path(normalized).suffix.lower()
    if suffix in DENIED_EXTENSIONS:
        die("blocked_denied_path_extension", path=normalized, extension=suffix)
    return normalized


class GitLab:
    def __init__(self, env: dict[str, str]) -> None:
        self.base_url = env["GITLAB_BASE_URL"]
        self.token = env["GITLAB_TOKEN"]
        self.token_label = env.get("GITLAB_TOKEN_LABEL", "")
        self.token_scope = env.get("GITLAB_TOKEN_SCOPE", "")

    def request(self, method: str, api_path: str, params: dict[str, Any] | None = None) -> tuple[Any, dict[str, str]]:
        query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
        url = f"{self.base_url}/api/v4{api_path}"
        if query:
            url = f"{url}?{query}"
        req = urllib.request.Request(url, method=method)
        req.add_header("PRIVATE-TOKEN", self.token)
        req.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
                headers = {k.lower(): v for k, v in resp.headers.items()}
                if method == "HEAD":
                    return {}, headers
                raw = resp.read(4_000_000)
                if not raw:
                    return {}, headers
                return json.loads(raw.decode("utf-8", errors="replace")), headers
        except urllib.error.HTTPError as exc:
            body = exc.read(1000).decode("utf-8", errors="replace") if exc.fp else ""
            die("gitlab_http_error", http_status=exc.code, reason=body[:200], api_path=api_path)
        except Exception as exc:
            die("gitlab_request_failed", error_type=type(exc).__name__, api_path=api_path)

    def project_id(self, project_path: str) -> str:
        if project_path.isdigit():
            return project_path
        return quote_path(project_path)


def short_sha(value: str | None) -> str:
    return (value or "")[:12]


def cmd_status(gl: GitLab, _args: argparse.Namespace) -> dict[str, Any]:
    return response(
        "ok",
        base_url=gl.base_url,
        secret_env_path=str(SECRET_ENV),
        token_present=True,
        token_label=gl.token_label,
        token_scope=gl.token_scope,
        allowed_operations=[
            "status",
            "me",
            "group-projects",
            "project",
            "branches",
            "mrs",
            "pipelines",
            "tree",
            "file-stat",
        ],
        mutations_enabled=False,
    )


def cmd_me(gl: GitLab, _args: argparse.Namespace) -> dict[str, Any]:
    data, _ = gl.request("GET", "/user")
    return response(
        "ok",
        user={
            "id": data.get("id"),
            "username": data.get("username"),
            "name": data.get("name"),
            "state": data.get("state"),
            "web_url": data.get("web_url"),
        },
    )


def cmd_group_projects(gl: GitLab, args: argparse.Namespace) -> dict[str, Any]:
    group = args.group or DEFAULT_GROUP
    data, _ = gl.request(
        "GET",
        f"/groups/{quote_path(group)}/projects",
        {
            "include_subgroups": "true",
            "per_page": min(args.limit, 100),
            "simple": "true",
            "order_by": "last_activity_at",
            "sort": "desc",
        },
    )
    projects = [
        {
            "id": p.get("id"),
            "name": p.get("name"),
            "path_with_namespace": p.get("path_with_namespace"),
            "default_branch": p.get("default_branch"),
            "visibility": p.get("visibility"),
            "archived": p.get("archived"),
            "last_activity_at": p.get("last_activity_at"),
            "web_url": p.get("web_url"),
        }
        for p in (data if isinstance(data, list) else [])
    ]
    return response("ok", group=group, count=len(projects), projects=projects)


def cmd_project(gl: GitLab, args: argparse.Namespace) -> dict[str, Any]:
    data, _ = gl.request("GET", f"/projects/{gl.project_id(args.project)}")
    return response(
        "ok",
        project={
            "id": data.get("id"),
            "name": data.get("name"),
            "path_with_namespace": data.get("path_with_namespace"),
            "default_branch": data.get("default_branch"),
            "visibility": data.get("visibility"),
            "archived": data.get("archived"),
            "last_activity_at": data.get("last_activity_at"),
            "web_url": data.get("web_url"),
            "open_issues_count": data.get("open_issues_count"),
        },
    )


def cmd_branches(gl: GitLab, args: argparse.Namespace) -> dict[str, Any]:
    data, _ = gl.request("GET", f"/projects/{gl.project_id(args.project)}/repository/branches", {"per_page": min(args.limit, 100)})
    branches = [
        {
            "name": b.get("name"),
            "default": b.get("default"),
            "merged": b.get("merged"),
            "protected": b.get("protected"),
            "commit": {
                "id": short_sha((b.get("commit") or {}).get("id")),
                "title": (b.get("commit") or {}).get("title"),
                "created_at": (b.get("commit") or {}).get("created_at"),
            },
        }
        for b in (data if isinstance(data, list) else [])
    ]
    return response("ok", project=args.project, count=len(branches), branches=branches)


def cmd_mrs(gl: GitLab, args: argparse.Namespace) -> dict[str, Any]:
    data, _ = gl.request(
        "GET",
        f"/projects/{gl.project_id(args.project)}/merge_requests",
        {"state": args.state, "per_page": min(args.limit, 100), "order_by": "updated_at", "sort": "desc"},
    )
    mrs = [
        {
            "iid": mr.get("iid"),
            "title": mr.get("title"),
            "state": mr.get("state"),
            "draft": mr.get("draft") or mr.get("work_in_progress"),
            "source_branch": mr.get("source_branch"),
            "target_branch": mr.get("target_branch"),
            "merge_status": mr.get("merge_status"),
            "detailed_merge_status": mr.get("detailed_merge_status"),
            "updated_at": mr.get("updated_at"),
            "web_url": mr.get("web_url"),
        }
        for mr in (data if isinstance(data, list) else [])
    ]
    return response("ok", project=args.project, state=args.state, count=len(mrs), merge_requests=mrs)


def cmd_pipelines(gl: GitLab, args: argparse.Namespace) -> dict[str, Any]:
    data, _ = gl.request("GET", f"/projects/{gl.project_id(args.project)}/pipelines", {"per_page": min(args.limit, 100), "ref": args.ref})
    pipelines = [
        {
            "id": p.get("id"),
            "iid": p.get("iid"),
            "status": p.get("status"),
            "ref": p.get("ref"),
            "sha": short_sha(p.get("sha")),
            "created_at": p.get("created_at"),
            "updated_at": p.get("updated_at"),
            "web_url": p.get("web_url"),
        }
        for p in (data if isinstance(data, list) else [])
    ]
    return response("ok", project=args.project, count=len(pipelines), pipelines=pipelines)


def cmd_tree(gl: GitLab, args: argparse.Namespace) -> dict[str, Any]:
    path = validate_repo_path(args.path)
    data, _ = gl.request(
        "GET",
        f"/projects/{gl.project_id(args.project)}/repository/tree",
        {"path": path or None, "ref": args.ref, "per_page": min(args.limit, 100)},
    )
    entries = [
        {
            "name": item.get("name"),
            "path": item.get("path"),
            "type": item.get("type"),
            "mode": item.get("mode"),
            "id": short_sha(item.get("id")),
        }
        for item in (data if isinstance(data, list) else [])
    ]
    return response("ok", project=args.project, ref=args.ref, path=path, count=len(entries), entries=entries)


def cmd_file_stat(gl: GitLab, args: argparse.Namespace) -> dict[str, Any]:
    path = validate_repo_path(args.path, allow_empty=False)
    _, headers = gl.request(
        "HEAD",
        f"/projects/{gl.project_id(args.project)}/repository/files/{quote_path(path)}",
        {"ref": args.ref},
    )
    interesting = {
        "x-gitlab-blob-id": headers.get("x-gitlab-blob-id"),
        "x-gitlab-commit-id": short_sha(headers.get("x-gitlab-commit-id")),
        "x-gitlab-last-commit-id": short_sha(headers.get("x-gitlab-last-commit-id")),
        "x-gitlab-content-sha256": headers.get("x-gitlab-content-sha256"),
        "x-gitlab-size": headers.get("x-gitlab-size"),
        "content-type": headers.get("content-type"),
    }
    return response("ok", project=args.project, ref=args.ref, path=path, file_metadata=interesting, file_content_returned=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded SkyVision GitLab helper")
    parser.add_argument("command", choices=["status", "me", "group-projects", "project", "branches", "mrs", "pipelines", "tree", "file-stat"])
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--project", default="")
    parser.add_argument("--path", default="")
    parser.add_argument("--ref", default=None)
    parser.add_argument("--state", default="opened", choices=["opened", "closed", "merged", "all"])
    parser.add_argument("--limit", type=int, default=30)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    env = load_env()
    gl = GitLab(env)
    if args.command in {"project", "branches", "mrs", "pipelines", "tree", "file-stat"} and not args.project:
        die("blocked_missing_project")
    handlers = {
        "status": cmd_status,
        "me": cmd_me,
        "group-projects": cmd_group_projects,
        "project": cmd_project,
        "branches": cmd_branches,
        "mrs": cmd_mrs,
        "pipelines": cmd_pipelines,
        "tree": cmd_tree,
        "file-stat": cmd_file_stat,
    }
    out = handlers[args.command](gl, args)
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
