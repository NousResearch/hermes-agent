"""Editable-finder drift detection and repair (#134413).

A PEP 660 editable install resolves first-party top-level names from a
``MAPPING`` snapshotted when the install was written. ``hermes update`` pulls new
files into the tree, so a pull that adds a top-level module leaves the snapshot
behind: the file is in the tree and unimportable through the finder, and mapped
modules that import it fail too. These tests pin the two halves of the repair —
detecting exactly the shipped tops the finder cannot resolve, and rewriting only
the ``MAPPING`` literal so the file keeps setuptools' shape.
"""

from __future__ import annotations

from pathlib import Path

from hermes_cli import editable_mapping as em


def _install(tmp_path: Path, *, mapped: dict[str, bool]) -> Path:
    """A checkout with an editable finder; *mapped* selects which tops it knows."""
    root = tmp_path / "checkout"
    root.mkdir()
    for name, is_mapped in mapped.items():
        if is_mapped:
            (root / f"{name}.py").write_text("VALUE = 1\n", encoding="utf-8")
        else:
            # a top the pull added: in the tree, absent from the finder map
            (root / f"{name}.py").write_text("VALUE = 2\n", encoding="utf-8")
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "setup.py").write_text("", encoding="utf-8")
    # Not shipped tops: must never be reported as missing.
    for noise in ("tests", "evals"):
        (root / noise).mkdir()
        (root / noise / "__init__.py").write_text("", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[tool.setuptools.packages.find]\ninclude = [\n  "pkg",\n  "pkg.*",\n]\n\n[tool.other]\nx = 1\n',
        encoding="utf-8",
    )
    site = root / "venv" / "lib" / "python3.14" / "site-packages"
    site.mkdir(parents=True)
    known = [name for name, is_mapped in mapped.items() if is_mapped] + ["pkg"]
    mapping = ", ".join(f"{name!r}: {str(root / name)!r}" for name in known)
    (site / "__editable___hermes_agent_0_0_0_finder.py").write_text(
        f"from __future__ import annotations\n"
        f"MAPPING: dict[str, str] = {{{mapping}}}\n"
        f"NAMESPACES: dict[str, list[str]] = {{'plugins.kanban': [{str(root / 'plugins' / 'kanban')!r}]}}\n"
        f"PATH_PLACEHOLDER = 'x'\n",
        encoding="utf-8",
    )
    return root


def _finder(root: Path) -> Path:
    """The finder the fixture wrote, asserted present for the type checker."""
    path = em.finder_path(root)
    assert path is not None
    return path


def test_expected_tops_follow_the_packaging_config_not_the_filesystem(tmp_path):
    """``tests/`` and ``evals/`` have ``__init__.py`` but are not distributed tops."""
    root = _install(tmp_path, mapped={"alpha": True})

    tops = em.expected_tops(root)

    assert {"alpha", "pkg"} <= tops
    assert {"tests", "evals", "setup"} & tops == set()


def test_stale_tops_reports_only_the_tops_the_finder_cannot_resolve(tmp_path):
    root = _install(tmp_path, mapped={"alpha": True, "beta": False})

    assert em.stale_tops(root) == ["beta"]


def test_in_step_install_reports_nothing_and_refresh_is_a_no_op(tmp_path):
    root = _install(tmp_path, mapped={"alpha": True})
    finder = _finder(root)
    before = finder.read_text(encoding="utf-8")

    assert em.stale_tops(root) == []
    assert em.refresh_mapping(root) == []
    assert finder.read_text(encoding="utf-8") == before


def test_refresh_repairs_the_mapping_and_leaves_the_rest_of_the_file_alone(tmp_path):
    root = _install(tmp_path, mapped={"alpha": True, "beta": False})
    finder = _finder(root)
    before_lines = finder.read_text(encoding="utf-8").splitlines()

    added = em.refresh_mapping(root)

    assert added == ["beta"]
    assert em.stale_tops(root) == []
    after_lines = finder.read_text(encoding="utf-8").splitlines()
    # Only the MAPPING literal changed; every other line is byte-identical.
    assert [line for line in after_lines if not line.startswith("MAPPING")] == \
           [line for line in before_lines if not line.startswith("MAPPING")]
    assert str(root / "beta") in "\n".join(after_lines)
    # The rewritten literal still parses (checked in-module, but assert the shape here too).
    compile(finder.read_text(encoding="utf-8"), str(finder), "exec")


def test_no_finder_means_no_work(tmp_path):
    """A wheel / pipx / developer venv has no editable finder: nothing to repair."""
    root = tmp_path / "plain"
    root.mkdir()
    (root / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")

    assert em.finder_path(root) is None
    assert em.stale_tops(root) == []
    assert em.refresh_mapping(root) == []
