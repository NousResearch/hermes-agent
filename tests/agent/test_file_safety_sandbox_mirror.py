"""Tests for the sandbox-mirror write guard in agent/file_safety.

The guard fires when a tool tries to write into the per-task mirror
directory created by a non-local terminal backend (Docker, Daytona, etc.).
Those paths look like ``…/sandboxes/<backend>/<task>/home/.hermes/…`` and
they accumulate divergent copies of authoritative profile state (SOUL.md,
config.yaml, memories/*.md) because the host Hermes process never reads
them. Soft guard — defense in depth, NOT a security boundary.

Reference: #32049 — under ``terminal.backend: docker``, the agent's
``write_file`` / ``patch`` calls landed on the sandbox mirror of SOUL.md
while the host process kept loading the untouched authoritative file.
The agent reported success; the rule never took effect.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# classify_sandbox_mirror_target — pure path-shape detection
# ---------------------------------------------------------------------------


class TestClassifySandboxMirrorTarget:
    def test_docker_mirror_soul_md_classified(self, tmp_path):
        """The exact path shape reported in #32049."""
        from agent.file_safety import classify_sandbox_mirror_target

        target = (
            tmp_path
            / "profiles" / "group1"
            / "sandboxes" / "docker" / "default" / "home" / ".hermes"
            / "profiles" / "group1" / "SOUL.md"
        )
        target.parent.mkdir(parents=True)
        target.write_text("# mirror copy\n")

        result = classify_sandbox_mirror_target(str(target))
        assert result is not None
        assert result["target_path"] == str(target.resolve())
        assert result["mirror_root"].endswith(
            "sandboxes/docker/default/home/.hermes"
        )
        assert result["inner_path"] == "profiles/group1/SOUL.md"

    @pytest.mark.parametrize(
        "backend,inner",
        [
            ("docker", "profiles/coder/memories/MEMORY.md"),
            ("daytona", "profiles/default/cron/jobs.json"),
            ("podman", ".env"),
        ],
    )
    def test_other_backends_and_inner_files_match(self, tmp_path, backend, inner):
        """The detector is backend-agnostic — sandbox-mirror shape is what matters."""
        from agent.file_safety import classify_sandbox_mirror_target

        target = (
            tmp_path
            / "sandboxes" / backend / "task-42" / "home" / ".hermes"
            / Path(inner)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")

        result = classify_sandbox_mirror_target(str(target))
        assert result is not None
        assert result["inner_path"] == inner
        assert backend in result["mirror_root"]







# ---------------------------------------------------------------------------
# get_sandbox_mirror_warning — the model-facing string
# ---------------------------------------------------------------------------


class TestGetSandboxMirrorWarning:
    def test_non_mirror_returns_none(self, tmp_path):
        from agent.file_safety import get_sandbox_mirror_warning

        target = tmp_path / ".hermes" / "profiles" / "group1" / "SOUL.md"
        target.parent.mkdir(parents=True)
        target.write_text("# real SOUL\n")

        assert get_sandbox_mirror_warning(str(target)) is None

    def test_mirror_warning_names_mirror_root_and_inner_path(self, tmp_path):
        from agent.file_safety import get_sandbox_mirror_warning

        target = (
            tmp_path
            / "profiles" / "group1"
            / "sandboxes" / "docker" / "default" / "home" / ".hermes"
            / "profiles" / "group1" / "SOUL.md"
        )
        target.parent.mkdir(parents=True)
        target.write_text("# mirror copy\n")

        warn = get_sandbox_mirror_warning(str(target))
        assert warn is not None
        # Must name the mirror root so the user can locate the sandbox.
        assert "sandboxes/docker/default/home/.hermes" in warn
        # Must hint at what the agent likely meant.
        assert "profiles/group1/SOUL.md" in warn
        # Must name the bypass kwarg shared with the cross-profile guard.
        assert "cross_profile=True" in warn

    def test_warning_is_defense_in_depth_not_boundary(self, tmp_path):
        from agent.file_safety import get_sandbox_mirror_warning

        target = (
            tmp_path
            / "sandboxes" / "docker" / "t" / "home" / ".hermes"
            / "profiles" / "g" / "SOUL.md"
        )
        target.parent.mkdir(parents=True)
        target.write_text("x")

        warn = get_sandbox_mirror_warning(str(target))
        # Must self-document as defense-in-depth so future reviewers
        # don't promote it to a hard block (matches the existing
        # cross-profile guard's contract).
        assert "not a security boundary" in warn.lower()


# ---------------------------------------------------------------------------
# Independence from cross-profile classifier
# ---------------------------------------------------------------------------


class TestSandboxMirrorIsOrthogonalToCrossProfile:
    """The sandbox-mirror guard must fire even when the inner path is
    in-profile from the host's view — the bug is the mirror, not the
    profile mismatch."""

    def test_same_profile_mirror_still_flagged(self, tmp_path, monkeypatch):
        import agent.file_safety as fs
        monkeypatch.setattr(fs, "_hermes_root_path", lambda: tmp_path)
        monkeypatch.setattr(fs, "_hermes_home_path", lambda: tmp_path / "profiles" / "group1")

        target = (
            tmp_path
            / "profiles" / "group1"
            / "sandboxes" / "docker" / "default" / "home" / ".hermes"
            / "profiles" / "group1" / "SOUL.md"
        )
        target.parent.mkdir(parents=True)
        target.write_text("x")

        # cross-profile classifier: active profile == target's inner-mirror
        # profile name; on the existing detector the path's parts[2] is
        # ``sandboxes``, not a scoped area, so it returns None.
        assert fs.classify_cross_profile_target(str(target)) is None
        # sandbox-mirror classifier: fires unconditionally on the shape.
        assert fs.classify_sandbox_mirror_target(str(target)) is not None
