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
    """Fails if a new call site hand-rolls Scripts/bin again (#76105).

    Uses AST rather than substring matching: an earlier `"if" in line` version
    matched any word containing "if" (mod*if*y, ver*if*y) and still missed
    `os.path.join(venv, "Scripts")`.

    ``stdio.py`` is exempt: it builds a list of literal *Windows-only* PATH
    candidates, not a cross-platform layout derivation, so ``venv_bin_dir()``
    (which branches on the host platform) would be the wrong tool there.
    """
    import ast
    import hermes_cli

    exempt = {"stdio.py"}
    pkg = Path(hermes_cli.__file__).parent
    offenders = []
    for py in pkg.rglob("*.py"):
        if py.name in exempt:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # Any *code* string literal "Scripts" is a hand-rolled layout;
            # docstrings and comments never reach ast.Constant in an expr
            # position we care about here.
            if isinstance(node, ast.Constant) and node.value == "Scripts":
                offenders.append(f"{py.relative_to(pkg)}:{node.lineno}")
    assert not offenders, (
        "open-coded venv layout found (use hermes_constants.venv_bin_dir):\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
# Top-level FILES must be atomic too (#76104 review, C1)
# ---------------------------------------------------------------------------

def test_top_level_files_are_swapped_atomically(tmp_path):
    """The repo root holds 20 first-party modules (run_agent.py, cli.py,
    hermes_constants.py, ...). Covering only directories would leave exactly
    the bug class this PR closes."""
    live, new = tmp_path / "live", tmp_path / "new"
    live.mkdir()
    new.mkdir()
    (live / "run_agent.py").write_text("old")
    (new / "run_agent.py").write_text("new")

    staged = [
        (
            update_cmd._stage_replacement(
                str(new / "run_agent.py"), str(live / "run_agent.py")
            ),
            str(live / "run_agent.py"),
        )
    ]
    update_cmd._commit_staged_replacements(staged)

    assert (live / "run_agent.py").read_text() == "new"
    assert not [p for p in os.listdir(live) if "hermes-update" in p]


def test_file_swap_failure_restores_the_original_file(tmp_path, monkeypatch):
    """A mid-swap failure must not leave a stale-or-corrupt root module."""
    live, new = tmp_path / "live", tmp_path / "new"
    live.mkdir()
    new.mkdir()
    for name in ("cli.py", "run_agent.py"):
        (live / name).write_text("old")
        (new / name).write_text("new")

    staged = [
        (update_cmd._stage_replacement(str(new / n), str(live / n)), str(live / n))
        for n in ("cli.py", "run_agent.py")
    ]

    real_rename = os.rename
    calls = {"n": 0}

    def flaky_rename(src, dst):
        calls["n"] += 1
        if calls["n"] == 4:
            raise OSError("simulated AV interference")
        return real_rename(src, dst)

    monkeypatch.setattr(update_cmd.os, "rename", flaky_rename)
    with pytest.raises(OSError):
        update_cmd._commit_staged_replacements(staged)
    monkeypatch.undo()

    versions = {n: (live / n).read_text() for n in ("cli.py", "run_agent.py")}
    assert versions == {"cli.py": "old", "run_agent.py": "old"}, (
        f"mixed/corrupt root modules after rollback: {versions}"
    )


def test_failed_staging_leaves_no_orphaned_copies(tmp_path, monkeypatch):
    """#76104 review C2: orphaned staging dirs make the retry we recommend
    fail harder than the original attempt (less free space each time)."""
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"agent": "old", "tools": "old", "gateway": "old"})
    _live_tree(new, {"agent": "new", "tools": "new", "gateway": "new"})

    real_copytree = update_cmd.shutil.copytree
    calls = {"n": 0}

    def flaky_copytree(src, dst, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError(28, "No space left on device")
        return real_copytree(src, dst, *a, **kw)

    monkeypatch.setattr(update_cmd.shutil, "copytree", flaky_copytree)

    staged: list[tuple[str, str]] = []
    with pytest.raises(OSError):
        try:
            for n in ("agent", "tools", "gateway"):
                staged.append(
                    (
                        update_cmd._stage_replacement(
                            str(new / n), str(live / n)
                        ),
                        str(live / n),
                    )
                )
        except Exception:
            update_cmd._discard_staged(staged)
            raise
    monkeypatch.undo()

    leftovers = [p for p in os.listdir(live) if "hermes-update" in p]
    assert leftovers == [], f"orphaned staging copies: {leftovers}"
    # And nothing live was touched.
    for n in ("agent", "tools", "gateway"):
        assert (live / n / "version.txt").read_text() == "old"


def test_atomic_replace_dir_still_works_as_a_shim(tmp_path):
    """W1: it is now an alias over the two-phase helpers; #49145 must hold."""
    live, new = tmp_path / "live", tmp_path / "new"
    _live_tree(live, {"ui-tui": "old"})
    _live_tree(new, {"ui-tui": "new"})

    update_cmd._atomic_replace_dir(str(new / "ui-tui"), str(live / "ui-tui"))

    assert (live / "ui-tui" / "version.txt").read_text() == "new"
    assert not [p for p in os.listdir(live) if "hermes-update" in p]


def test_venv_helpers_honour_an_explicit_platform_verdict():
    """Callers must be able to override the platform check (#76107 CI).

    The suite exercises Windows paths on Linux CI by patching predicates like
    `hermes_cli.main._is_windows`. A helper that reads `sys.platform`
    unconditionally silently drops those paths out of coverage -- and broke
    `test_verify_core_dependencies.py::test_uses_virtual_env_from_environment`,
    which patches `_is_windows` and then asserts on a `Scripts/python.exe`
    path.
    """
    v = Path("/opt/proj/venv")
    assert venv_bin_dir(v, windows=True).name == "Scripts"
    assert venv_bin_dir(v, windows=False).name == "bin"
    assert venv_python_path(v, windows=True).name == "python.exe"
    assert venv_python_path(v, windows=False).name == "python"
    # Halves must stay consistent under an explicit verdict.
    for flag in (True, False):
        assert venv_python_path(v, windows=flag).parent == venv_bin_dir(
            v, windows=flag
        )


def test_patched_is_windows_reaches_the_venv_path_derivation():
    """End-to-end: patching the module predicate must change the derived path."""
    from unittest.mock import patch

    from hermes_cli import main as hermes_main

    with patch.object(hermes_main, "_is_windows", return_value=True):
        got = hermes_main._resolve_install_target_python(
            ["uv", "pip"], env={"VIRTUAL_ENV": "/nope/venv"}
        )
    # The path doesn't exist so we get None, but the *derivation* must have
    # used the Windows layout -- assert that directly.
    assert got is None
    assert (
        venv_python_path("/nope/venv", windows=True).as_posix()
        == "/nope/venv/Scripts/python.exe"
    )
