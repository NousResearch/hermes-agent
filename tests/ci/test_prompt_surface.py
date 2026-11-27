"""The committed prompt-surface snapshot is exactly what the code renders.

``tests/fixtures/prompt_surface/`` holds the system prompt and ``tools[]`` array a stock install
sends on each surface (see ``scripts/ci/prompt_surface.py``). A code change that alters either
must carry the regenerated snapshot, so the change shows up in review as real prompt text and
real schemas, and the snapshot path needs a hermes-agent-core approval to merge.

Regenerate with ``scripts/run-in-hermes-env python scripts/ci/prompt_surface.py render``.
"""

from __future__ import annotations

import filecmp
import importlib.util
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("prompt_surface", REPO / "scripts" / "ci" / "prompt_surface.py")
assert _spec is not None and _spec.loader is not None
prompt_surface = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prompt_surface)


def _tree_differences(left: Path, right: Path) -> list[str]:
    cmp = filecmp.dircmp(left, right)
    out = [f"only in committed: {n}" for n in cmp.left_only] + [f"not committed: {n}" for n in cmp.right_only]
    _, mismatch, errors = filecmp.cmpfiles(left, right, cmp.common_files, shallow=False)
    out += [f"stale: {n}" for n in mismatch + errors]
    for sub in cmp.common_dirs:
        out += [f"{sub}/{d}" for d in _tree_differences(left / sub, right / sub)]
    return out


def test_committed_prompt_surface_is_current(tmp_path):
    rendered = tmp_path / "prompt_surface"
    assert prompt_surface.render(rendered) == 0
    differences = _tree_differences(prompt_surface.SNAPSHOT, rendered)
    assert not differences, (
        "The system prompt or tool schemas changed. Regenerate the snapshot and commit it with the "
        "change (merging it needs a hermes-agent-core approval):\n"
        "    scripts/run-in-hermes-env python scripts/ci/prompt_surface.py render\n"
        + "\n".join(differences[:20])
    )


def test_render_ignores_workspaces_above_the_throwaway_home(tmp_path, monkeypatch):
    outer = tmp_path / "project"
    temp = outer / "tmp"
    temp.mkdir(parents=True)
    (outer / "pyproject.toml").write_text('[project]\nname = "outer"\n', encoding="utf-8")
    (outer / ".git").mkdir()
    (outer / "outer.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(tempfile, "tempdir", str(temp))

    rendered = tmp_path / "snapshot"
    assert prompt_surface.render(rendered) == 0
    assert _tree_differences(prompt_surface.SNAPSHOT, rendered) == []


# Not "posix" yet: the macOS lane would run it, and the render dies at the sys.platform pin there.
@pytest.mark.platforms("linux")
def test_render_is_unchanged_under_a_symlinked_temp_dir(tmp_path, monkeypatch):
    """macOS temp dirs sit behind a symlink (/tmp, /var/folders); the unresolved path must not reach the snapshot."""
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "link"))

    rendered = tmp_path / "snapshot"
    assert prompt_surface.render(rendered) == 0
    assert _tree_differences(prompt_surface.SNAPSHOT, rendered) == []


def test_snapshot_is_checked_out_with_the_lf_that_render_writes():
    """render() writes LF on every host and the check above compares bytes, so a Windows clone with
    core.autocrlf=true must not check the committed snapshot out as CRLF."""
    files = "\n".join(p.relative_to(REPO).as_posix() for p in prompt_surface.SNAPSHOT.rglob("*") if p.is_file())
    attrs = subprocess.run(["git", "check-attr", "--stdin", "eol"], cwd=REPO, input=files,
                           capture_output=True, text=True, check=True).stdout.splitlines()
    assert attrs and [line for line in attrs if not line.endswith(": eol: lf")] == []


def test_summary_names_the_surface_and_tool_that_changed(tmp_path):
    base = prompt_surface.SNAPSHOT
    head = tmp_path / "head"
    shutil.copytree(base, head)
    prompt = head / "prompts" / "cli.claude.txt"
    text = prompt.read_text(encoding="utf-8-sig")
    prompt.write_text(text.replace("# Finishing the job", "# Finish every job"), encoding="utf-8")
    tool = head / "tools" / "write_file.json"
    text = tool.read_text(encoding="utf-8-sig")
    tool.write_text(text.replace('"description": "', '"description": "Prefer patch. ', 1), encoding="utf-8")

    summary = prompt_surface.summarize(base, head)

    assert "| cli.claude |" in summary and "+# Finish every job" in summary
    assert "`write_file`: description +" in summary
    assert prompt_surface.summarize(base, base) == ""
