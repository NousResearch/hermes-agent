#!/usr/bin/env python3
"""Search and install skills from the Clawbus API."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_URL = "https://www.clawbus.com/api"


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")).expanduser()


def default_skills_dir() -> Path:
    return hermes_home() / "skills"


def api_get(path: str, params: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{BASE_URL}{path}?{query}"
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "hermes-clawbus-skill"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"ERROR: Clawbus API returned HTTP {exc.code}: {detail}")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"ERROR: Clawbus API request failed: {exc}")

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: Clawbus API returned invalid JSON: {exc}")


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))


def safe_file_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe file path from API: {relative}")
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    if root_resolved not in [resolved, *resolved.parents]:
        raise ValueError(f"file path escapes skill directory: {relative}")
    return resolved


def write_skill_files(destination: Path, files: list[dict[str, Any]]) -> list[str]:
    written: list[str] = []
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    for item in files:
        rel_path = item.get("path")
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise SystemExit("ERROR: install response included a file without path.")
        content = item.get("content")
        if not isinstance(content, str):
            raise SystemExit(f"ERROR: file {rel_path} did not include text content.")
        try:
            target = safe_file_path(destination, rel_path)
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(rel_path)
    return written


def search(args: argparse.Namespace) -> int:
    data = api_get("/skills/search", {"q": args.query, "limit": args.limit})
    print_json(data)
    return 0


def trending(args: argparse.Namespace) -> int:
    data = api_get(
        "/skills/trending",
        {"period": args.period, "limit": args.limit},
    )
    print_json(data)
    return 0


def install(args: argparse.Namespace) -> int:
    data = api_get("/skills/install", {"slug": args.slug, "mode": "files"})
    skill = data.get("skill") if isinstance(data.get("skill"), dict) else {}
    slug = skill.get("slug") or args.slug
    if not isinstance(slug, str) or not slug:
        raise SystemExit("ERROR: install response did not include a skill slug.")

    files = data.get("files")
    if not files:
        content = data.get("content")
        if isinstance(content, str) and content.strip():
            files = [{"path": "SKILL.md", "content": content}]
        else:
            raise SystemExit(f"ERROR: no files returned for skill {args.slug}.")
    if not isinstance(files, list):
        raise SystemExit("ERROR: install response files field was not a list.")

    skills_dir = Path(args.skills_dir).expanduser() if args.skills_dir else default_skills_dir()
    destination = skills_dir / slug
    written = write_skill_files(destination, files)

    meta = {
        "source": "clawbus",
        "slug": slug,
        "installedAt": datetime.now(timezone.utc).isoformat(),
        "api": BASE_URL,
    }
    (destination / "_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print_json(
        {
            "installed": slug,
            "path": str(destination),
            "files": written,
            "skillPage": f"https://www.clawbus.com/skills/{slug}",
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    search_parser = sub.add_parser("search", help="Search Clawbus skills.")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.set_defaults(func=search)

    trending_parser = sub.add_parser("trending", help="List trending skills.")
    trending_parser.add_argument("--period", default="week")
    trending_parser.add_argument("--limit", type=int, default=10)
    trending_parser.set_defaults(func=trending)

    install_parser = sub.add_parser("install", help="Install a Clawbus skill.")
    install_parser.add_argument("slug")
    install_parser.add_argument("--skills-dir", default="")
    install_parser.set_defaults(func=install)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
