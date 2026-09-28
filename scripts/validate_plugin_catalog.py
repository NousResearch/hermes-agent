#!/usr/bin/env python3
"""Standalone structural validator for plugin-catalog entry files.

Validates ``plugin-catalog/*.yaml`` catalog entries and
``plugin-catalog/removed.yaml`` against the catalog contract schema, using
only stdlib + PyYAML so the admission CI (and third-party repos) can run it
WITHOUT installing hermes-agent.

NOTE: this script intentionally duplicates the schema rules instead of
importing ``hermes_cli`` — the whole point is the no-install requirement for
cheap cross-repo CI use. The runtime twin of this schema lives in
``hermes_cli/plugin_catalog.py``; if the contract changes there, update the
rules here in lockstep.

Usage:
    python3 scripts/validate_plugin_catalog.py plugin-catalog/
    python3 scripts/validate_plugin_catalog.py entry.yaml removed.yaml
    python3 scripts/validate_plugin_catalog.py --json plugin-catalog/

Exit codes: 0 = all files valid (warnings allowed), 1 = at least one error.
Human output is one ``<file>: ERROR: ...`` / ``<file>: warning: ...`` line
per finding; ``--json`` emits a machine-readable report on stdout instead.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

try:
    import yaml
except ImportError:  # pragma: no cover - dependency guidance only
    print(
        "ERROR: PyYAML is required (pip install pyyaml)",
        file=sys.stderr,
    )
    sys.exit(2)

NAME_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TIERS = ("official", "community")
CATEGORIES = ("desktop", "memory", "platform", "web", "tools", "voice", "automation", "models", "general")
PLATFORMS = ("linux", "macos", "windows")
CAPABILITY_KEYS = (
    "provides_tools",
    "provides_hooks",
    "provides_middleware",
    "requires_env",
)
# Top-level keys the contract knows about. Unknown keys WARN (forward
# compatibility: newer catalogs must stay valid under older validators).
KNOWN_KEYS = {
    "name",
    "repo",
    "sha",
    "subdir",
    "description",
    "maintainer",
    "tier",
    "category",
    "requires_hermes",
    "docs_url",
    "version",
    "image",
    "screenshots",
    "readme",
    "platforms",
    "capabilities",
    "title",
    "onboarding",
}
# Cosmetic labels attached to the pin. ``version`` is never parsed; ``image`` and ``screenshots``
# may only point at GitHub so the Desktop catalog browser and the docs site never fetch from
# third-party hosts and a raw URL pinned to the entry's commit stays as immutable as the sha.
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,31}$")
IMAGE_HOSTS = ("raw.githubusercontent.com", "github.com")
IMAGE_HOST_SUFFIX = ".githubusercontent.com"
MAX_SCREENSHOTS = 6
# ``readme: true`` makes the docs site render the README from the pinned commit; the build knows
# the raw-file URL scheme of these forges only.
README_REPO_HOSTS = ("github.com", "gitlab.com")
REQUIRED_KEYS = ("name", "repo", "sha", "description", "maintainer")

# One comparator clause of a requires_hermes spec, e.g. ">=0.19" or "!=1.2.3".
_COMPARATOR_RE = re.compile(r"^(>=|<=|==|!=|>|<)\s*\d+(\.\d+)*$")


def _is_allowed_image_url(url: str) -> bool:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    return parts.scheme == "https" and bool(host) and (host in IMAGE_HOSTS or host.endswith(IMAGE_HOST_SUFFIX))


def _is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _repo_host(repo: object) -> str:
    return (urlsplit(repo).hostname or "").lower() if isinstance(repo, str) else ""


def _check_page_fields(data: dict, errors: list[str]) -> None:
    """``screenshots`` and ``readme`` feed the entry's page at /docs/plugins/<name>; both optional (README on by default)."""
    shots = data.get("screenshots")
    if shots is not None:
        if not isinstance(shots, list) or not all(isinstance(s, str) and _is_allowed_image_url(s) for s in shots):
            errors.append(
                f"screenshots must be a list of https URLs on {list(IMAGE_HOSTS)} or *{IMAGE_HOST_SUFFIX}"
            )
        elif len(shots) > MAX_SCREENSHOTS:
            errors.append(f"screenshots lists {len(shots)} URLs; at most {MAX_SCREENSHOTS} are shown")
    readme = data.get("readme")
    if readme is not None:
        if not isinstance(readme, bool):
            errors.append(f"readme must be true or false, got {readme!r}")
        elif readme and _repo_host(data.get("repo")) not in README_REPO_HOSTS:
            errors.append(f"readme: true needs a repo on {list(README_REPO_HOSTS)} (the site fetches it from the pinned commit); omit it for other forges")


def _check_requires_hermes(spec: object, errors: list[str]) -> None:
    if not isinstance(spec, str):
        errors.append(f"requires_hermes must be a string, got {type(spec).__name__}")
        return
    if spec.strip() == "":
        return  # empty = no constraint
    for clause in spec.split(","):
        if not _COMPARATOR_RE.match(clause.strip()):
            errors.append(
                f"requires_hermes clause {clause.strip()!r} is not a valid "
                "comparator spec (expected e.g. '>=0.19')"
            )


def validate_entry(data: object) -> tuple[list[str], list[str]]:
    """Validate one catalog entry document. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict):
        return ["top-level document must be a YAML mapping"], warnings

    for key in REQUIRED_KEYS:
        if key not in data:
            errors.append(f"missing required key: {key}")

    for key in sorted(set(data) - KNOWN_KEYS):
        warnings.append(f"unknown top-level key {key!r} (ignored by this validator)")

    name = data.get("name")
    if "name" in data and (not isinstance(name, str) or not NAME_RE.match(name)):
        errors.append(f"name {name!r} must match [a-z0-9_-]{{1,64}}")

    repo = data.get("repo")
    if "repo" in data and (
        not isinstance(repo, str) or not repo.startswith("https://")
    ):
        errors.append(f"repo {repo!r} must be an https:// URL")

    sha = data.get("sha")
    if "sha" in data and (not isinstance(sha, str) or not SHA_RE.match(sha)):
        errors.append(f"sha {sha!r} must be exactly 40 lowercase hex characters")

    for key in ("description", "maintainer"):
        if key in data and not _is_nonempty_str(data[key]):
            errors.append(f"{key} must be a non-empty string")

    tier = data.get("tier", "community")
    if tier not in TIERS:
        errors.append(f"tier {tier!r} must be one of {list(TIERS)}")

    category = data.get("category", "desktop")
    if category not in CATEGORIES:
        errors.append(f"category {category!r} must be one of {list(CATEGORIES)}")

    if "requires_hermes" in data:
        _check_requires_hermes(data["requires_hermes"], errors)

    version = data.get("version")
    if version is not None and (not isinstance(version, str) or not VERSION_RE.match(version)):
        errors.append(f"version {version!r} must be 1-32 chars of [A-Za-z0-9._+-] (quote it in YAML)")

    image = data.get("image")
    if image is not None and (not isinstance(image, str) or not _is_allowed_image_url(image)):
        errors.append(f"image {image!r} must be an https URL on {list(IMAGE_HOSTS)} or *{IMAGE_HOST_SUFFIX}")

    _check_page_fields(data, errors)

    platforms = data.get("platforms", [])
    if platforms is None:
        platforms = []
    if not isinstance(platforms, list):
        errors.append("platforms must be a list")
    else:
        bad = [p for p in platforms if p not in PLATFORMS]
        if bad:
            errors.append(f"platforms {bad!r} not in allowed set {list(PLATFORMS)}")

    caps = data.get("capabilities", {})
    if caps is None:
        caps = {}
    if not isinstance(caps, dict):
        errors.append("capabilities must be a mapping")
    else:
        for key in sorted(set(caps) - set(CAPABILITY_KEYS)):
            warnings.append(f"unknown capabilities key {key!r}")
        for key in CAPABILITY_KEYS:
            if key not in caps:
                continue
            value = caps[key]
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                errors.append(f"capabilities.{key} must be a list of strings")

    return errors, warnings


def validate_removed(data: object) -> tuple[list[str], list[str]]:
    """Validate the removed.yaml document. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict):
        return ["top-level document must be a YAML mapping"], warnings

    removed = data.get("removed")
    if removed is None:
        errors.append("missing required key: removed")
        return errors, warnings
    if not isinstance(removed, list):
        errors.append("removed must be a list")
        return errors, warnings

    for i, item in enumerate(removed):
        if not isinstance(item, dict):
            errors.append(f"removed[{i}] must be a mapping")
            continue
        if not _is_nonempty_str(item.get("name")):
            errors.append(f"removed[{i}] missing non-empty 'name'")
        for key in ("repo", "reason", "date"):
            if key in item and not isinstance(item[key], str):
                errors.append(f"removed[{i}].{key} must be a string")

    return errors, warnings


def validate_file(path: Path) -> tuple[list[str], list[str]]:
    """Validate one YAML file (dispatching on filename). Returns (errors, warnings)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except OSError as exc:
        return [f"cannot read file: {exc}"], []
    except yaml.YAMLError as exc:
        return [f"invalid YAML: {exc}"], []

    if path.name == "removed.yaml":
        return validate_removed(data)
    return validate_entry(data)


def collect_paths(args: list[str]) -> list[Path]:
    paths: list[Path] = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.yaml")))
            paths.extend(sorted(p.glob("*.yml")))
        else:
            paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Standalone structural validator for plugin-catalog entry files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="catalog entry files, removed.yaml, or a directory of them",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable JSON report on stdout",
    )
    opts = parser.parse_args(argv)

    files = collect_paths(opts.paths)
    if not files:
        print("ERROR: no YAML files found", file=sys.stderr)
        return 1

    report = []
    any_errors = False
    for path in files:
        errors, warnings = validate_file(path)
        any_errors = any_errors or bool(errors)
        report.append(
            {
                "path": str(path),
                "ok": not errors,
                "errors": errors,
                "warnings": warnings,
            }
        )

    if opts.json:
        print(json.dumps({"ok": not any_errors, "files": report}, indent=2))
    else:
        for entry in report:
            for err in entry["errors"]:
                print(f"{entry['path']}: ERROR: {err}")
            for warn in entry["warnings"]:
                print(f"{entry['path']}: warning: {warn}")
        checked = len(report)
        bad = sum(1 for e in report if not e["ok"])
        if any_errors:
            print(f"FAIL: {bad}/{checked} file(s) invalid")
        else:
            print(f"OK: {checked} file(s) valid")

    return 1 if any_errors else 0


if __name__ == "__main__":
    sys.exit(main())
