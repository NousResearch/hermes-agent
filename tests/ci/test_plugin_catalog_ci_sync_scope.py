"""Regression coverage for #138499's upstream-sync scope exception."""

import os
from pathlib import Path

import hermes_yaml as yaml

from tests.ci.test_plugin_catalog_ci_confinement import WORKFLOW, _run, _step


def test_labeled_upstream_sync_keeps_changed_entries_for_pinned_validation(tmp_path):
    """The label relaxes only the scope check, not the changed-entry output."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert workflow["jobs"]["pinned-source-validate"]["env"]["UPSTREAM_SYNC"] == (
        "${{ contains(github.event.pull_request.labels.*.name, 'upstream-sync') }}")

    repo = tmp_path / "pr"
    repo.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text("""#!/bin/sh
case \"$*\" in
  *"rev-parse -q --verify HEAD^2"*) exit 0 ;;
  *"--diff-filter=AMRT"*) printf '%s\\n' plugin-catalog/new.yaml ;;
  *"diff --name-only HEAD^1 HEAD"*) printf '%s\\n' plugin-catalog/new.yaml scripts/inherited.py ;;
  *) exit 1 ;;
esac
""", encoding="utf-8")
    git.chmod(0o755)
    output = tmp_path / "gh-output.txt"
    output.touch()

    result = _run(_step("Find changed catalog"), tmp_path, repo, {
        **os.environ, "GITHUB_OUTPUT": str(output), "UPSTREAM_SYNC": "true",
        "PATH": f"{bin_dir}:{os.environ['PATH']}"})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "plugin-catalog/new.yaml" in output.read_text(encoding="utf-8")
