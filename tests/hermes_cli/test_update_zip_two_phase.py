"""Tests for the two-phase ZIP replace and the shared venv-layout helpers.

``_atomic_replace_dir`` (#49145) made each *individual* directory swap safe,
but the ZIP update replaced ~70 top-level entries in a loop with no atomicity
across iterations. An interruption partway left some entries at the new
version and the rest at the old one -- every file valid Python, the
combination unbootable. That is the mechanism behind the ``ImportError`` in
#76091 and the field report in #63717.

Reference: issues #76104 (ZIP atomicity) and #76105 (venv-helper duplication).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import update_cmd
from hermes_constants import venv_bin_dir, venv_python_path


# ---------------------------------------------------------------------------
# Two-phase replace
# ---------------------------------------------------------------------------

def _live_tree(root: Path, names: dict[str, str]) -> None:
    for name, marker in names.items():
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "version.txt").write_text(marker)


def _stage_all(root: Path, new: Path, names: list[str]) -> list[tuple[str, str]]:
    return [
        (
            update_cmd._stage_replacement(str(new / n), str(root / n)),
            str(root / n),
        )
        for n in names
    ]


def test_staging_touches_nothing_live(tmp_path):
    """Phase 1 must not modify the install -- a failure there is a no-op."""
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old", "tools": "old"})
    _live_tree(new, {"agent": "new", "tools": "new"})

    _stage_all(live, new, ["agent", "tools"])

    assert (live / "agent" / "version.txt").read_text() == "old"
    assert (live / "tools" / "version.txt").read_text() == "old"


def test_commit_swaps_every_entry(tmp_path):
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old", "tools": "old"})
    _live_tree(new, {"agent": "new", "tools": "new"})

    update_cmd._commit_staged_replacements(_stage_all(live, new, ["agent", "tools"]))

    assert (live / "agent" / "version.txt").read_text() == "new"
    assert (live / "tools" / "version.txt").read_text() == "new"
    # No staging/backup litter left behind.
    assert not [p for p in os.listdir(live) if "hermes-update" in p]


def test_failed_swap_rolls_back_every_earlier_swap(tmp_path, monkeypatch):
    """The regression: a mid-loop failure must not leave a mixed-version tree.

    Before the two-phase split this produced `agent/` new + `tools/` stale --
    the exact shape that yields
    `ImportError: cannot import name 'TODO_INJECTION_HEADER'`.
    """
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old", "tools": "old"})
    _live_tree(new, {"agent": "new", "tools": "new"})
    staged = _stage_all(live, new, ["agent", "tools"])

    real_rename = os.rename
    calls = {"n": 0}

    def flaky_rename(src, dst):
        calls["n"] += 1
        # Let the first entry swap fully (2 renames), then break the second.
        if calls["n"] == 4:
            raise OSError("simulated AV interference")
        return real_rename(src, dst)

    monkeypatch.setattr(update_cmd.os, "rename", flaky_rename)

    with pytest.raises(OSError):
        update_cmd._commit_staged_replacements(staged)

    monkeypatch.undo()
    # Both entries must be back at the OLD version -- not one new, one old.
    versions = {
        n: (live / n / "version.txt").read_text() for n in ("agent", "tools")
    }
    assert versions == {"agent": "old", "tools": "old"}, (
        f"mixed-version tree after rollback: {versions}"
    )


def test_commit_handles_entries_absent_from_the_install(tmp_path):
    """A brand-new top-level dir has no live counterpart to move aside."""
    live, new = tmp_path / "live", tmp_path / "new"
    live.mkdir()
    _live_tree(new, {"brand_new": "new"})

    update_cmd._commit_staged_replacements(_stage_all(live, new, ["brand_new"]))

    assert (live / "brand_new" / "version.txt").read_text() == "new"


def test_staging_clears_leftovers_from_an_interrupted_run(tmp_path):
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old"})
    _live_tree(new, {"agent": "new"})
    stale = Path(f"{live / 'agent'}.hermes-update-staging")
    stale.mkdir()
    (stale / "junk.txt").write_text("from a previous crash")

    update_cmd._commit_staged_replacements(_stage_all(live, new, ["agent"]))

    assert (live / "agent" / "version.txt").read_text() == "new"
    assert not (live / "agent" / "junk.txt").exists()


# ---------------------------------------------------------------------------
# Shared venv helpers (#76105)
# ---------------------------------------------------------------------------

def test_venv_helpers_agree_with_each_other():
    v = Path("/opt/proj/venv")
    assert venv_python_path(v).parent == venv_bin_dir(v)


def test_venv_helpers_accept_str_and_path():
    assert venv_python_path("/opt/x/venv") == venv_python_path(Path("/opt/x/venv"))


def test_venv_helpers_are_platform_consistent():
    """Whatever the platform, the two halves must not disagree."""
    v = Path("/opt/proj/venv")
    bin_name = venv_bin_dir(v).name
    exe_name = venv_python_path(v).name
    assert (bin_name, exe_name) in {("Scripts", "python.exe"), ("bin", "python")}


def test_managed_uv_helper_delegates_to_the_shared_one():
    from hermes_cli.managed_uv import _venv_python

    v = Path("/opt/proj/venv")
    assert _venv_python(v) == venv_python_path(v)


def test_no_open_coded_venv_layout_remains_in_hermes_cli():
    """Fails if a new call site hand-rolls Scripts/bin again (#76105)."""
    import hermes_cli

    pkg = Path(hermes_cli.__file__).parent
    offenders = []
    for py in pkg.rglob("*.py"):
        for lineno, line in enumerate(
            py.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if '"Scripts"' not in line:
                continue
            # Comments and docstrings referencing the path are fine.
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith(("'", '"', "-")):
                continue
            if "if" in line or "/" in line:
                offenders.append(f"{py.relative_to(pkg)}:{lineno}: {stripped}")
    assert not offenders, "open-coded venv layout found:\n" + "\n".join(offenders)
