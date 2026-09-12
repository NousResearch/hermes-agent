"""Repository hygiene tests for bundled Desktop plugin entry files."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPO_ROOT / "apps" / "desktop" / "src" / "plugins"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_bundled_plugin_js_entries_are_sources_not_ts_compiler_artifacts() -> None:
    """Only source-only plugin.js entries may be tracked or unignored.

    A plugin.js next to plugin.tsx is TypeScript output. Tracking or unignoring
    it lets the generated file shadow the TSX source and makes ``tsc --clean``
    dirty the checkout by deleting a tracked file.
    """
    checkout = _git("rev-parse", "--is-inside-work-tree")
    if checkout.returncode != 0:
        pytest.skip("repository hygiene requires a Git checkout")

    tracked = _git("ls-files", "apps/desktop/src/plugins/*/plugin.js")
    assert tracked.returncode == 0, tracked.stderr

    tracked_entries = {
        REPO_ROOT / line for line in tracked.stdout.splitlines() if line.strip()
    }
    tsx_entries = sorted(PLUGIN_ROOT.glob("*/plugin.tsx"))
    generated_entries = {entry.with_suffix(".js") for entry in tsx_entries}

    tracked_generated = sorted(
        path.relative_to(REPO_ROOT) for path in tracked_entries & generated_entries
    )
    assert not tracked_generated, (
        "generated Desktop plugin entries are tracked alongside plugin.tsx: "
        + ", ".join(map(str, tracked_generated))
    )

    for generated in generated_entries:
        relative = generated.relative_to(REPO_ROOT)
        ignored = _git("check-ignore", "--no-index", "-q", str(relative))
        assert ignored.returncode == 0, (
            f"{relative} is generated from plugin.tsx but is not gitignored"
        )

    source_only_entries = tracked_entries - generated_entries
    for source in source_only_entries:
        relative = source.relative_to(REPO_ROOT)
        ignored = _git("check-ignore", "--no-index", "-q", str(relative))
        assert ignored.returncode == 1, (
            f"{relative} is a source-only plugin.js entry but is gitignored"
        )
