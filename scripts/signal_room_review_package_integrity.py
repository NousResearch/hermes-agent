#!/usr/bin/env python3
"""Validate Signal Room review package links and manifest artifacts."""
from __future__ import annotations

import argparse
import html.parser
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_name = "href" if tag == "a" else "src" if tag == "img" else None
        if attr_name is None:
            return
        for key, value in attrs:
            if key == attr_name and value:
                self.paths.append(value)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_relative_path(path: str) -> bool:
    parsed = urlparse(path)
    if parsed.scheme or parsed.netloc or path.startswith("/"):
        return False
    return ".." not in Path(path).parts


def _hub_paths(package_dir: Path) -> list[str]:
    hub_path = package_dir / "REVIEW_HUB.html"
    if not hub_path.exists():
        return []
    parser = LinkParser()
    parser.feed(hub_path.read_text())
    return parser.paths


def evaluate_package_integrity(package_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    checked_paths: set[str] = set()
    manifest_path = package_dir / "handoff_manifest.json"
    if not manifest_path.exists():
        return {
            "passed": False,
            "errors": ["missing handoff_manifest.json"],
            "checked_paths": [],
        }

    manifest = read_json(manifest_path)
    for rel_path in manifest.get("artifacts", []):
        checked_paths.add(str(rel_path))
        if not (package_dir / str(rel_path)).exists():
            errors.append(f"missing manifest artifact: {rel_path}")

    for rel_path in manifest.get("primary_review_files", {}).values():
        checked_paths.add(str(rel_path))
        if not _safe_relative_path(str(rel_path)):
            errors.append(f"unsafe primary path: {rel_path}")
        elif not (package_dir / str(rel_path)).exists():
            errors.append(f"missing primary review file: {rel_path}")

    for rel_path in _hub_paths(package_dir):
        checked_paths.add(str(rel_path))
        if not _safe_relative_path(str(rel_path)):
            errors.append(f"unsafe hub path: {rel_path}")
        elif not (package_dir / str(rel_path)).exists():
            errors.append(f"missing hub asset: {rel_path}")

    return {
        "passed": not errors,
        "errors": errors,
        "package_dir": str(package_dir),
        "checked_paths": sorted(checked_paths),
    }


def write_integrity_scorecard(package_dir: Path, out: Path) -> dict[str, Any]:
    result = evaluate_package_integrity(package_dir)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = write_integrity_scorecard(args.package_dir, args.out)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
