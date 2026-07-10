"""Fork-delta computation and manifest coverage checks."""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import gitops


DEFAULT_MANIFEST = Path("docs/sync/fork-features.json")


@dataclass(frozen=True)
class ForkFeature:
    feature: str
    tests: tuple[str, ...]
    paths: tuple[str, ...]
    why: str


@dataclass(frozen=True)
class ForkDeltaReport:
    base: str
    fork_ref: str
    changed_paths: tuple[str, ...]
    covered_paths: tuple[str, ...]
    uncovered_paths: tuple[str, ...]
    covered_features: tuple[str, ...]


def load_manifest(path: Path) -> list[ForkFeature]:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    features: list[ForkFeature] = []
    for entry in raw:
        features.append(
            ForkFeature(
                feature=str(entry["feature"]),
                tests=tuple(str(item) for item in entry.get("tests", [])),
                paths=tuple(str(item) for item in entry.get("paths", [])),
                why=str(entry.get("why", "")),
            )
        )
    return features


def _matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, pattern.rstrip("/") + "/**")


def compute_fork_delta(
    repo: Path,
    *,
    base: str,
    fork_ref: str = "fork/main",
    manifest_path: Path | None = None,
    touched_paths: set[str] | None = None,
) -> ForkDeltaReport:
    manifest = load_manifest(repo / (manifest_path or DEFAULT_MANIFEST))
    changed_set = set(gitops.changed_files(repo, base, fork_ref))
    if touched_paths is not None:
        changed_set &= touched_paths
    changed = tuple(sorted(changed_set))
    covered: set[str] = set()
    covered_features: set[str] = set()
    for path in changed:
        for feature in manifest:
            if any(_matches(path, pattern) for pattern in feature.paths):
                covered.add(path)
                covered_features.add(feature.feature)
    uncovered = tuple(path for path in changed if path not in covered)
    return ForkDeltaReport(
        base=base,
        fork_ref=fork_ref,
        changed_paths=changed,
        covered_paths=tuple(sorted(covered)),
        uncovered_paths=uncovered,
        covered_features=tuple(sorted(covered_features)),
    )


def manifest_nodeids(path: Path) -> list[str]:
    return [node for feature in load_manifest(path) for node in feature.tests]


def covered_path(path: str, features: list[ForkFeature]) -> bool:
    return any(_matches(path, pattern) for feature in features for pattern in feature.paths)


def manifest_as_jsonable(features: list[ForkFeature]) -> list[dict[str, Any]]:
    return [
        {
            "feature": feature.feature,
            "tests": list(feature.tests),
            "paths": list(feature.paths),
            "why": feature.why,
        }
        for feature in features
    ]
