"""Tests for backend-visible skill path mapping (hermes-agent#41541, #73842).

On remote terminal backends the agent must see skill paths that resolve
inside the sandbox (e.g. ``/root/.hermes/skills/...`` for Docker), never
host paths.  On local/unknown backends behavior must be byte-identical to
before (host paths everywhere).
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.skill_commands import _build_skill_message
from agent.skill_path_mapping import map_skill_dir_for_backend
from agent.skill_preprocessing import substitute_template_vars


def _mounts(skills_dir: Path, external_dir: Path | None = None, container_base: str = "/root/.hermes") -> list[dict]:
    mounts = [
        {
            "host_path": str(skills_dir),
            "source_path": str(skills_dir),
            "container_path": f"{container_base}/skills",
        }
    ]
    if external_dir is not None:
        mounts.append(
            {
                "host_path": str(external_dir),
                "source_path": str(external_dir),
                "container_path": f"{container_base}/external_skills/0",
            }
        )
    return mounts


def _make_skill(base: Path, name: str, body: str = "# Skill") -> Path:
    skill_dir = base / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(body)
    return skill_dir


class TestMapSkillDirForBackend:
    def test_each_container_backend_maps(self, tmp_path):
        for backend in ("docker", "singularity", "modal"):
            skills_dir = _make_skill(tmp_path, f"probe-{backend}")
            with (
                patch.dict(os.environ, {"TERMINAL_ENV": backend}),
                patch(
                    "tools.credential_files.get_skills_directory_mount",
                    return_value=_mounts(skills_dir),
                ),
            ):
                assert map_skill_dir_for_backend(skills_dir / "probe") == (
                    f"/root/.hermes/skills/probe"
                )

    def test_ssh_live_env_maps_to_remote_home(self, tmp_path):
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        skill_dir = _make_skill(skills_root, "ssh-skill")

        class FakeSSH:
            _remote_home = "/home/remotebox"

        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "ssh"}),
            # Patch at the lazy-import boundary: map_skill_dir_for_backend
            # resolves tools.terminal_tool.get_active_env at call time, so
            # this works regardless of module-copy resolution in the suite.
            # A non-None task_id is required — _active_terminal_env returns
            # early when task_id is falsy.
            patch("tools.terminal_tool.get_active_env", return_value=FakeSSH()),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=_mounts(skills_root, container_base="/home/remotebox/.hermes"),
            ),
        ):
            assert map_skill_dir_for_backend(skill_dir, task_id="task-1") == (
                "/home/remotebox/.hermes/skills/ssh-skill"
            )

    def test_ssh_without_live_env_falls_back_to_host_path(self, tmp_path):
        skills_dir = _make_skill(tmp_path, "ssh-skill")
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "ssh"}),
            patch("agent.skill_path_mapping._active_terminal_env", return_value=None),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=_mounts(skills_dir, container_base="/home/remotebox/.hermes"),
            ),
        ):
            host = str(skills_dir / "ssh-skill")
            assert map_skill_dir_for_backend(skills_dir / "ssh-skill") == host

    def test_local_backend_preserves_host_path(self, tmp_path):
        skills_dir = _make_skill(tmp_path, "local-skill")
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "local"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=_mounts(skills_dir),
            ),
        ):
            host = str(skills_dir / "local-skill")
            assert map_skill_dir_for_backend(skills_dir / "local-skill") == host

    def test_unknown_backend_falls_back_to_host_path(self, tmp_path):
        skills_dir = _make_skill(tmp_path, "weird-skill")
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "banana"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=_mounts(skills_dir),
            ),
        ):
            host = str(skills_dir / "weird-skill")
            assert map_skill_dir_for_backend(skills_dir / "weird-skill") == host

    def test_dir_outside_any_mount_falls_back_to_host_path(self, tmp_path):
        skills_dir = _make_skill(tmp_path, "mounted")
        stray = tmp_path / "elsewhere" / "stray"
        stray.mkdir(parents=True)
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "docker"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=_mounts(skills_dir),
            ),
        ):
            assert map_skill_dir_for_backend(stray) == str(stray)

    def test_external_skill_dir_maps_via_mount_table(self, tmp_path):
        skills_dir = _make_skill(tmp_path, "mounted")
        external = _make_skill(tmp_path, "external-skill")
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "docker"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=_mounts(skills_dir, external_dir=external),
            ),
        ):
            assert map_skill_dir_for_backend(external / "external-skill") == (
                "/root/.hermes/external_skills/0/external-skill"
            )

    def test_sanitized_mount_source_maps_to_container_path(self, tmp_path):
        # When the skills tree contains symlinks the mount table's host_path
        # is a sanitized copy, not the real skills dir.  The agent-visible
        # skill dir is still the canonical path, so the mapper must match the
        # mount's source_path (canonical tree) — not only the bind source.
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        skill_dir = _make_skill(skills_root, "mounted")
        sanitized_copy = tmp_path / "hermes-skills-safe-xyz123"
        sanitized_copy.mkdir()
        (sanitized_copy / "mounted").mkdir()
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "docker"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=[
                    {
                        "host_path": str(sanitized_copy),
                        "source_path": str(skills_root),
                        "container_path": "/root/.hermes/skills",
                    }
                ],
            ),
        ):
            # Input is the canonical host skill dir (what skill loading
            # passes in), NOT the sanitized copy path.
            assert map_skill_dir_for_backend(skill_dir) == (
                "/root/.hermes/skills/mounted"
            )

    def test_mount_without_source_path_still_matches_host_path(self, tmp_path):
        # Backward compatibility: mount tables without the source_path key
        # (older mocks / consumers) still map via host_path.
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        skill_dir = _make_skill(skills_root, "mounted")
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "docker"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=[
                    {
                        "host_path": str(skills_root),
                        "container_path": "/root/.hermes/skills",
                    }
                ],
            ),
        ):
            assert map_skill_dir_for_backend(skill_dir) == (
                "/root/.hermes/skills/mounted"
            )

    def test_windows_case_insensitive_match(self, tmp_path, monkeypatch):
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        skill_dir = _make_skill(skills_root, "mounted")
        monkeypatch.setattr(os, "name", "nt")
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "docker"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=_mounts(skills_root),
            ),
        ):
            # Host path casing differs from the mount source casing.
            assert map_skill_dir_for_backend(
                tmp_path / "SKILLS" / "mounted"
            ) == "/root/.hermes/skills/mounted"

    def test_exact_mount_root_maps_to_container_root(self, tmp_path):
        skills_dir = _make_skill(tmp_path, "mounted")
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "docker"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=_mounts(skills_dir),
            ),
        ):
            assert map_skill_dir_for_backend(skills_dir) == "/root/.hermes/skills"

    def test_none_skill_dir_returns_empty(self):
        assert map_skill_dir_for_backend(None) == ""


class TestTemplateVarSubstitution:
    def test_docker_substitutes_container_path(self, tmp_path):
        skills_dir = _make_skill(tmp_path, "templated")
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "docker"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=_mounts(skills_dir),
            ),
        ):
            out = substitute_template_vars(
                "Run: node ${HERMES_SKILL_DIR}/scripts/foo.js",
                skills_dir / "templated",
                session_id=None,
            )
        assert out == "Run: node /root/.hermes/skills/templated/scripts/foo.js"

    def test_local_keeps_host_path(self, tmp_path):
        skills_dir = _make_skill(tmp_path, "templated")
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "local"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=_mounts(skills_dir),
            ),
        ):
            out = substitute_template_vars(
                "Run: node ${HERMES_SKILL_DIR}/scripts/foo.js",
                skills_dir / "templated",
                session_id=None,
            )
        assert out == f"Run: node {skills_dir / 'templated'}/scripts/foo.js"


class TestBuildSkillMessage:
    def _message(self, skill_dir: Path, *, linked_files: dict | None = None) -> str:
        loaded_skill = {
            "content": "# Skill\n\nRun the bundled script.",
            "linked_files": linked_files or {},
            "setup_needed": False,
        }
        return _build_skill_message(
            loaded_skill=loaded_skill,
            skill_dir=skill_dir,
            activation_note="[IMPORTANT: The user has invoked the \"x\" skill. The full skill content is loaded below.]",
        )

    def test_docker_header_and_hints_use_container_path(self, tmp_path):
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        skill_dir = _make_skill(skills_root, "scripted")
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / "run.js").write_text("console.log('hi')")
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "docker"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=_mounts(skills_root),
            ),
        ):
            msg = self._message(skill_dir)

        assert "[Skill directory: /root/.hermes/skills/scripted]" in msg
        assert "- scripts/run.js  ->  /root/.hermes/skills/scripted/scripts/run.js" in msg
        assert "node /root/.hermes/skills/scripted/scripts/foo.js" in msg
        # The host path must not leak into any agent-visible surface.
        assert str(skills_root) not in msg

    def test_local_header_and_hints_keep_host_path(self, tmp_path):
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        skill_dir = _make_skill(skills_root, "scripted")
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / "run.js").write_text("console.log('hi')")
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "local"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=_mounts(skills_root),
            ),
        ):
            msg = self._message(skill_dir)

        assert f"[Skill directory: {skill_dir}]" in msg
        assert f"- scripts/run.js  ->  {skill_dir / 'scripts' / 'run.js'}" in msg
        assert f"node {skill_dir}/scripts/foo.js" in msg

    def test_windows_supporting_file_renders_posix_against_container_hint(self, tmp_path):
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        skill_dir = _make_skill(skills_root, "win-skill")
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "docker"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=_mounts(skills_root),
            ),
        ):
            msg = self._message(
                skill_dir,
                linked_files={"scripts": ["scripts\\run.js"]},
            )

        assert "- scripts\\run.js  ->  /root/.hermes/skills/win-skill/scripts/run.js" in msg
        assert "\\root\\" not in msg

    def test_leading_slash_supporting_file_does_not_reset_hint_base(self, tmp_path):
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        skill_dir = _make_skill(skills_root, "slash-skill")
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "docker"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=_mounts(skills_root),
            ),
        ):
            msg = self._message(
                skill_dir,
                linked_files={"scripts": ["/scripts/run.js"]},
            )

        # The leading slash must not reset the join base to the filesystem
        # root (PurePosixPath division discards the base for absolute right
        # operands).
        assert (
            "- /scripts/run.js  ->  /root/.hermes/skills/slash-skill/scripts/run.js"
            in msg
        )
        assert "[Skill directory: /root/.hermes/skills/slash-skill]" in msg

    def test_docker_substitutes_skill_dir_in_content_through_message(self, tmp_path):
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        skill_dir = _make_skill(skills_root, "templated")
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "docker"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=_mounts(skills_root),
            ),
        ):
            # Verify the ${HERMES_SKILL_DIR} substitution path through
            # _build_skill_message with template content:
            loaded_skill = {
                "content": "Run: node ${HERMES_SKILL_DIR}/scripts/foo.js",
                "linked_files": {},
                "setup_needed": False,
            }
            msg = _build_skill_message(
                loaded_skill=loaded_skill,
                skill_dir=skill_dir,
                activation_note="[IMPORTANT: The user has invoked the \"x\" skill. The full skill content is loaded below.]",
            )

        assert "node /root/.hermes/skills/templated/scripts/foo.js" in msg
        assert "${HERMES_SKILL_DIR}" not in msg
        assert str(skills_root) not in msg

    def test_docker_without_mount_table_falls_back_to_host_path(self, tmp_path):
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        skill_dir = _make_skill(skills_root, "nomount")
        with (
            patch.dict(os.environ, {"TERMINAL_ENV": "docker"}),
            patch(
                "tools.credential_files.get_skills_directory_mount",
                return_value=[],
            ),
        ):
            msg = self._message(skill_dir)

        assert f"[Skill directory: {skill_dir}]" in msg
