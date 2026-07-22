#!/usr/bin/env python3
"""Build a review_status entry from Playwright screenshot artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCE = "playwright e2e"


def _files(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.rglob(pattern) if path.is_file()) if root.exists() else []


def _is_explicit_screenshot(path: Path) -> bool:
    """Exclude Playwright's automatic and visual-comparator PNG outputs."""
    return not (
        path.name.startswith(("test-finished-", "test-failed-"))
        or path.name.endswith(("-actual.png", "-expected.png", "-diff.png"))
    )


def build_status(results_dir: Path, artifact_url: str = "") -> list[dict]:
    """Return the E2E screenshot review status in the shared nested format."""
    screenshots = [path for path in _files(results_dir, "*.png") if _is_explicit_screenshot(path)]
    diffs = _files(results_dir, "*-diff.png")
    if not screenshots and not diffs:
        return []

    result: dict[str, str] = {
        "kind": "info",
        "title": "Desktop E2E screenshots",
        "summary": (
            f"{len(screenshots)} screenshot{'s' if len(screenshots) != 1 else ''} captured; "
            f"{len(diffs)} visual diff{'s' if len(diffs) != 1 else ''}."
        ),
    }
    if artifact_url:
        result["link"] = artifact_url
        result["link_label"] = "View screenshots"

    if screenshots:
        lines = ["<details>", f"<summary>{len(screenshots)} captured screenshot{'s' if len(screenshots) != 1 else ''}</summary>", ""]
        for path in screenshots:
            lines.append(f"- [`{path.name}`]({artifact_url})" if artifact_url else f"- `{path.name}`")
        lines.extend(["", "</details>"])
        result["detail"] = "\n".join(lines)

    return [{"source": SOURCE, "results": [result]}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--artifact-url", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.write_text(json.dumps(build_status(args.results_dir, args.artifact_url)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())