#!/usr/bin/env python3
"""Classify Elixir analytics Hermes changes into release lanes."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


PackageRule = dict[str, Any]


LOCAL_ARTIFACTS = {
    ".hermes-bootstrap-complete",
}


def _matches_prefix(prefixes: tuple[str, ...]) -> Callable[[str], bool]:
    return lambda path: path.startswith(prefixes)


PACKAGE_RULES: list[PackageRule] = [
    {
        "id": "profile-distribution",
        "title": "Profile Distribution",
        "description": (
            "Elixir analytics profile config, skill, roadmap, soul, and "
            "profile-owned runner plugin."
        ),
        "matches": _matches_prefix(("profile-distributions/elixir-analytics/",)),
    },
    {
        "id": "hermes-runtime",
        "title": "Hermes Runtime",
        "description": (
            "Core runtime changes needed to discover profile-owned plugins and "
            "expose the analytics runner toolset."
        ),
        "matches": lambda path: path
        in {
            "hermes_cli/plugins.py",
            "hermes_cli/profile_distribution.py",
            "toolsets.py",
            "tests/hermes_cli/test_elixir_analytics_profile_distribution.py",
            "tests/hermes_cli/test_elixir_analytics_runner_plugin.py",
            "tests/hermes_cli/test_elixir_analytics_release_packaging.py",
            "tests/plugins/test_disk_cleanup_plugin.py",
            "tests/test_toolsets.py",
            "scripts/check_elixir_analytics_release_packaging.py",
        },
    },
    {
        "id": "slack-gateway",
        "title": "Slack Gateway",
        "description": (
            "Slack Socket Mode recovery logging and tests that prove gateway "
            "health after reconnects."
        ),
        "matches": lambda path: path
        in {
            "gateway/platforms/slack.py",
            "tests/gateway/test_slack.py",
        },
    },
]


def _git_status_files() -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    files: list[str] = []
    for line in output.splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[1]
        files.append(path)
    return files


def classify_release_files(files: list[str]) -> dict[str, Any]:
    packages_by_id: dict[str, dict[str, Any]] = {}
    local_artifacts: list[str] = []
    unclassified: list[str] = []

    for file in files:
        if file in LOCAL_ARTIFACTS:
            local_artifacts.append(file)
            continue

        rule = next((candidate for candidate in PACKAGE_RULES if candidate["matches"](file)), None)
        if rule is None:
            unclassified.append(file)
            continue

        package = packages_by_id.setdefault(
            rule["id"],
            {
                "id": rule["id"],
                "title": rule["title"],
                "description": rule["description"],
                "files": [],
            },
        )
        package["files"].append(file)

    return {
        "packages": [
            packages_by_id[rule["id"]]
            for rule in PACKAGE_RULES
            if rule["id"] in packages_by_id
        ],
        "local_artifacts": local_artifacts,
        "unclassified": unclassified,
    }


def _render_text(result: dict[str, Any]) -> str:
    lines: list[str] = []
    for package in result["packages"]:
        lines.append(f"{package['id']} — {package['title']}")
        lines.append(package["description"])
        for file in package["files"]:
            lines.append(f"  - {file}")
        lines.append("")

    if result["local_artifacts"]:
        lines.append("local-artifacts")
        for file in result["local_artifacts"]:
            lines.append(f"  - {file}")
        lines.append("")

    if result["unclassified"]:
        lines.append("unclassified")
        for file in result["unclassified"]:
            lines.append(f"  - {file}")
        lines.append("")

    return "\n".join(lines).rstrip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when changed files are unclassified.",
    )
    parser.add_argument("--files", nargs="*", help="Files to classify; defaults to git status.")
    args = parser.parse_args()

    files = args.files if args.files is not None else _git_status_files()
    result = classify_release_files(files)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_render_text(result))

    return 1 if args.strict and result["unclassified"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
