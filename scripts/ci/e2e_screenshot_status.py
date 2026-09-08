#!/usr/bin/env python3
"""Select Desktop E2E visual evidence and build its CI review status."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SOURCE = "playwright e2e"
EVIDENCE_START = "<!-- hermes-e2e-evidence:start -->"
EVIDENCE_END = "<!-- hermes-e2e-evidence:end -->"


def _files(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.rglob(pattern) if path.is_file()) if root.exists() else []


def _is_explicit_screenshot(path: Path) -> bool:
    """Exclude Playwright's automatic and visual-comparator PNG outputs."""
    return not (
        path.name.startswith(("test-finished-", "test-failed-"))
        or path.name.endswith(("-actual.png", "-expected.png", "-diff.png"))
    )


def build_manifest(results_dir: Path) -> dict:
    """Record stable screenshot names from one E2E run for main/PR comparison."""
    screenshots = [path for path in _files(results_dir, "*.png") if _is_explicit_screenshot(path)]
    return {"version": 1, "screenshot_names": sorted({path.name for path in screenshots})}


def _base_screenshot_names(path: Path | None) -> set[str] | None:
    """Return ``None`` when main evidence is unavailable (never guess newness)."""
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    names = data.get("screenshot_names", []) if isinstance(data, dict) else []
    if not isinstance(data, dict) or not isinstance(names, list):
        return None
    return {name for name in names if isinstance(name, str)}


def _stage_name(kind: str, path: Path, results_dir: Path) -> str:
    relative = path.relative_to(results_dir).as_posix()
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
    return f"{kind}-{digest}-{path.name}"


def select_evidence(results_dir: Path, base_manifest: Path | None = None) -> dict:
    """Select only screenshots new to main, plus every generated visual diff."""
    base_names = _base_screenshot_names(base_manifest)
    screenshots = [] if base_names is None else [
        path for path in _files(results_dir, "*.png")
        if _is_explicit_screenshot(path) and path.name not in base_names
    ]
    diffs: list[dict[str, Path]] = []
    for diff in _files(results_dir, "*-diff.png"):
        stem = diff.with_name(diff.name.removesuffix("-diff.png"))
        entry = {"diff": diff}
        for kind in ("actual", "expected"):
            candidate = stem.with_name(f"{stem.name}-{kind}.png")
            if candidate.is_file():
                entry[kind] = candidate
        diffs.append(entry)
    return {"screenshots": screenshots, "diffs": diffs}


def stage_evidence(results_dir: Path, evidence_dir: Path, selection: dict) -> dict:
    """Copy selected PNGs into a flat, path-safe evidence artifact."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[Path, str] = {}

    def stage(kind: str, path: Path) -> str:
        if path in staged:
            return staged[path]
        name = _stage_name(kind, path, results_dir)
        shutil.copyfile(path, evidence_dir / name)
        staged[path] = name
        return name

    manifest = {"version": 1, "screenshots": [], "diffs": []}
    for screenshot in selection["screenshots"]:
        manifest["screenshots"].append({
            "name": screenshot.name,
            "file": stage("screenshot", screenshot),
        })
    for diff in selection["diffs"]:
        entry = {"name": diff["diff"].name.removesuffix("-diff.png"), "diff": stage("diff", diff["diff"])}
        for kind in ("actual", "expected"):
            if kind in diff:
                entry[kind] = stage(kind, diff[kind])
        manifest["diffs"].append(entry)

    (evidence_dir / "e2e-evidence.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_status(selection: dict, artifact_url: str = "") -> list[dict]:
    """Return the review status. The trusted publisher replaces its marker."""
    screenshots = selection["screenshots"]
    diffs = selection["diffs"]
    if not screenshots and not diffs:
        return []

    summary_parts = []
    if screenshots:
        summary_parts.append(
            f"{len(screenshots)} new screenshot{'s' if len(screenshots) != 1 else ''} vs main"
        )
    if diffs:
        summary_parts.append(f"{len(diffs)} visual diff{'s' if len(diffs) != 1 else ''}")

    result: dict[str, str] = {
        "kind": "info",
        "title": "Desktop E2E visual evidence",
        "summary": "; ".join(summary_parts) + ".",
        "detail": "\n".join((EVIDENCE_START, "<sub>inline evidence is publishing...</sub>", EVIDENCE_END)),
    }
    if artifact_url:
        result["link"] = artifact_url
        result["link_label"] = "View test artifacts"
    return [{"source": SOURCE, "results": [result]}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--artifact-url", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.manifest_output.write_text(
        json.dumps(build_manifest(args.results_dir), sort_keys=True) + "\n", encoding="utf-8"
    )
    selection = select_evidence(args.results_dir, args.base_manifest)
    stage_evidence(args.results_dir, args.evidence_dir, selection)
    args.output.write_text(
        json.dumps(build_status(selection, args.artifact_url)) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
