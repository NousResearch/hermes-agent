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
        assert result["mirror_root"].replace("\\", "/").endswith(
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

    def test_inner_path_uses_posix_separators(self, tmp_path):
        """``inner_path`` is a logical HERMES_HOME-relative identifier (surfaced
        to the model as "the authoritative file is likely <inner_path>"), not an
        on-disk path, so it must be POSIX-style on every platform — matching the
        ``classify_container_mirror_target`` sibling. Regression guard: on Windows
        ``str(Path(...))`` leaked native separators (``profiles\\group1\\...``),
        diverging from the sibling and mangling the hint.
        """
        from agent.file_safety import classify_sandbox_mirror_target

        target = (
            tmp_path
            / "sandboxes" / "docker" / "task-1" / "home" / ".hermes"
            / "profiles" / "group1" / "memories" / "MEMORY.md"
        )
        target.parent.mkdir(parents=True)
        target.write_text("x")

        result = classify_sandbox_mirror_target(str(target))
        assert result is not None
        assert result["inner_path"] == "profiles/group1/memories/MEMORY.md"
        assert "\\" not in result["inner_path"]

    def test_path_outside_sandbox_returns_none(self, tmp_path):
        """A plain Hermes path is not a mirror."""
        from agent.file_safety import classify_sandbox_mirror_target

        target = tmp_path / ".hermes" / "profiles" / "group1" / "SOUL.md"
        target.parent.mkdir(parents=True)
        target.write_text("# real SOUL\n")

        assert classify_sandbox_mirror_target(str(target)) is None

    def test_sandboxes_segment_without_home_hermes_returns_none(self, tmp_path):
        """A ``sandboxes/`` directory unrelated to Hermes-state mirroring (e.g.
        the sandbox workspace itself) is not flagged."""
        from agent.file_safety import classify_sandbox_mirror_target

        target = (
            tmp_path
            / "sandboxes" / "docker" / "task-42" / "workspace" / "main.py"
        )
        target.parent.mkdir(parents=True)
        target.write_text("print('hi')\n")

        assert classify_sandbox_mirror_target(str(target)) is None

    def test_sandboxes_segment_with_home_but_no_hermes_returns_none(self, tmp_path):
        """``sandboxes/<backend>/<task>/home/anything-not-hermes`` is not a mirror."""
        from agent.file_safety import classify_sandbox_mirror_target

        target = (
            tmp_path
            / "sandboxes" / "docker" / "task-42" / "home" / ".bashrc"
        )
        target.parent.mkdir(parents=True)
        target.write_text("alias ll='ls -la'\n")

        assert classify_sandbox_mirror_target(str(target)) is None

    def test_truncated_sandbox_path_returns_none(self, tmp_path):
        """``…/sandboxes/<backend>/<task>`` without ``home/.hermes/<thing>`` is not a mirror."""
        from agent.file_safety import classify_sandbox_mirror_target

        target = tmp_path / "sandboxes" / "docker" / "task-42"
        target.mkdir(parents=True)

        assert classify_sandbox_mirror_target(str(target)) is None

    def test_non_existent_path_still_classifies_by_shape(self, tmp_path):
        """Detection is path-shape only — it must not require the file to exist
        (the agent is about to CREATE the mirror file, that's the bug)."""
        from agent.file_safety import classify_sandbox_mirror_target

        target = (
            tmp_path
            / "profiles" / "group1"
            / "sandboxes" / "docker" / "default" / "home" / ".hermes"
            / "profiles" / "group1" / "SOUL.md"
        )
        # Parent directory exists so .resolve() doesn't strip the tail
        # under strict mode, but the file itself does NOT exist.
        target.parent.mkdir(parents=True)
        assert not target.exists()

        result = classify_sandbox_mirror_target(str(target))
        assert result is not None
        assert result["inner_path"] == "profiles/group1/SOUL.md"


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
        # ``mirror_root`` is a native host path (backslashes on Windows), so
        # normalize separators before the substring check — same pattern the
        # container-mirror tests use. ``inner_path`` below stays POSIX-exact.
        assert "sandboxes/docker/default/home/.hermes" in warn.replace("\\", "/")
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
