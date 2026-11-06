"""Regression tests for #137483: pm/workspace.py must inject ``[project].version`` on virtual members.

The workspace member writer at ``pm/workspace.py:_workspace_member`` injects a
``[project].name`` for any plugin whose ``pyproject.toml`` lacks a build
backend (a "virtual" / metadata-only member — the only kind the runner
carries for the bundled LCM / QMS / funding-bot catalog plugins, whose
pyproject is a lint config with no ``[project]`` section at all). PEP
621 mandates that a ``[project]`` table declare either a static
``version`` or a ``dynamic`` list that names ``"version"``; without one
of these the package installer rejects the metadata with
``the required project.version field is neither set nor present`` and
the plugin is auto-disabled on every ``hermes update``.
"""
from __future__ import annotations

import tomllib

import pytest


def _read_member_pyproject(member_dir) -> dict:
    return tomllib.loads((member_dir / "pyproject.toml").read_text(encoding="utf-8"))


def test_virtual_member_with_empty_pyproject_gets_injected_name_and_version(tmp_path):
    """A plugin whose pyproject is purely a lint config — no ``[project]`` at all —
    must still produce a PEP 621-conformant ``[project]`` table after workspace
    rewriting, with both ``name`` and ``version`` set (#137483).
    """
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "hermes-lcm"
    plugin.mkdir(parents=True)
    # Lint-only pyproject: tool.ruff config, no [project], no [build-system].
    (plugin / "pyproject.toml").write_text(
        '[tool.ruff]\nline-length = 120\n',
        encoding="utf-8",
    )
    root = tmp_path / "gen"
    root.mkdir()

    member = _workspace_member(plugin, root, identity=plugin)
    document = _read_member_pyproject(member)

    project = document.get("project", {})
    assert project.get("name"), f"injected project.name missing in {project!r}"
    assert project.get("name", "").startswith("hermes-plugin-"), project
    assert project.get("version"), (
        f"injected [project] table is missing a version (PEP 621 requires either "
        f"a static version or a dynamic list naming 'version') — #137483. Got: {project!r}"
    )


def test_virtual_member_with_partial_project_gets_only_missing_version(tmp_path):
    """A plugin whose pyproject already declares ``[project].name`` (no version)
    must have its missing version filled in by the workspace writer, not the
    existing name rewritten.
    """
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "adverse-event-bot"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(
        '[project]\nname = "adverse-event-bot"\ndependencies = ["requests>=2.31"]\n',
        encoding="utf-8",
    )
    root = tmp_path / "gen"
    root.mkdir()

    member = _workspace_member(plugin, root, identity=plugin)
    document = _read_member_pyproject(member)
    project = document.get("project", {})

    # The original name may or may not survive the rename-to-unique-key pass
    # (the workspace writer names metadata-only members by their unique key to
    # avoid cross-profile ``uv lock`` collisions); what matters is the version.
    assert project.get("version"), (
        f"missing project.version in metadata-only member — #137483. Got: {project!r}"
    )


def test_buildable_member_unchanged(tmp_path):
    """A buildable member (one with ``[build-system]``) must keep its own
    pyproject verbatim — uv verifies the declared ``[project].name`` against
    the package metadata its backend produces, so the writer must not touch
    it. This guards against the version-injector accidentally rewriting a
    buildable member's pyproject.
    """
    from pm.workspace import _workspace_member

    plugin = tmp_path / "home" / "plugins" / "replay"
    plugin.mkdir(parents=True)
    original = (
        '[project]\nname = "replay-plugin"\nversion = "1.0"\n'
        '[build-system]\nrequires = []\nbuild-backend = "backend"\n'
    )
    (plugin / "pyproject.toml").write_text(original, encoding="utf-8")
    root = tmp_path / "gen"
    root.mkdir()

    member = _workspace_member(plugin, root, identity=plugin)
    assert (member / "pyproject.toml").read_text(encoding="utf-8") == original
