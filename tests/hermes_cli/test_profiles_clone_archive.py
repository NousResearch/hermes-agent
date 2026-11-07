"""#145929: a cloned profile does not inherit the source's curator archive.

``skills/.archive/`` is the SOURCE profile's retired-skill history, and the deepest tree in a
real profile: copying it into a clone reproduces those paths under a staging directory that is
~30 characters longer, which is what killed profile/bot creation on Windows with
``LongPathsEnabled=0`` (`[WinError 206] The filename or extension is too long`).
"""

import os
from pathlib import Path

import pytest

from hermes_cli.profiles import create_profile


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Isolated ``~/.hermes`` for profile tests: the HOME and HERMES_HOME both point at tmp."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def _windows_long_paths_enabled() -> bool:
    """True when this Windows host opted into long paths (``LongPathsEnabled=1``).

    With it on, a copy past MAX_PATH succeeds whatever the clone does, so a long-path test
    would pass vacuously; it skips instead.
    """
    if os.name != "nt":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            return bool(winreg.QueryValueEx(key, "LongPathsEnabled")[0])
    except OSError:
        return False


def _seed_source_with_archive(default_home):
    """A source profile with one live skill and one curator-archived skill."""
    (default_home / "config.yaml").write_text("model: test", encoding="utf-8")
    live = default_home / "skills" / "live-skill"
    live.mkdir(parents=True)
    (live / "SKILL.md").write_text("live", encoding="utf-8")
    (live / "references").mkdir()
    (live / "references" / "note.md").write_text("keep me", encoding="utf-8")
    archived = default_home / "skills" / ".archive" / "retired-skill"
    archived.mkdir(parents=True)
    (archived / "SKILL.md").write_text("retired", encoding="utf-8")


class TestCloneDropsTheCuratorArchive:
    def test_clone_does_not_inherit_the_curators_archive(self, profile_env):
        """``--clone`` (and the dashboard's "New bot") keeps active skills, not retirements."""
        default_home = profile_env / ".hermes"
        _seed_source_with_archive(default_home)

        profile_dir = create_profile("coder", clone_config=True, no_alias=True)

        assert not (profile_dir / "skills" / ".archive").exists()
        assert (profile_dir / "skills" / "live-skill" / "SKILL.md").read_text(encoding="utf-8") == "live"
        assert (profile_dir / "skills" / "live-skill" / "references" / "note.md").is_file()
        # The source keeps its archive — the clone must not move or consume it.
        assert (default_home / "skills" / ".archive" / "retired-skill" / "SKILL.md").is_file()

    def test_clone_all_does_not_inherit_the_curators_archive(self, profile_env):
        """Same rule on ``--clone-all``, which copies the whole profile root."""
        default_home = profile_env / ".hermes"
        _seed_source_with_archive(default_home)
        (default_home / "memories").mkdir(exist_ok=True)
        (default_home / "memories" / "MEMORY.md").write_text("identity", encoding="utf-8")
        (default_home / "workspace").mkdir(exist_ok=True)
        (default_home / "workspace" / "notes.md").write_text("work", encoding="utf-8")

        profile_dir = create_profile("coder", clone_all=True, no_alias=True)

        assert not (profile_dir / "skills" / ".archive").exists()
        assert (profile_dir / "skills" / "live-skill" / "SKILL.md").is_file()
        # Everything else a clone-all carries still travels.
        assert (profile_dir / "memories" / "MEMORY.md").read_text(encoding="utf-8") == "identity"
        assert (profile_dir / "workspace" / "notes.md").read_text(encoding="utf-8") == "work"
        assert (default_home / "skills" / ".archive" / "retired-skill" / "SKILL.md").is_file()

    def test_clone_keeps_an_archive_dir_nested_inside_a_skill(self, profile_env):
        """The exclusion is scoped to the skills ROOT: a skill's own ``.archive`` travels with it."""
        default_home = profile_env / ".hermes"
        _seed_source_with_archive(default_home)
        nested = default_home / "skills" / "live-skill" / "scripts" / ".archive"
        nested.mkdir(parents=True)
        (nested / "old-script.sh").write_text("#!/bin/sh\n", encoding="utf-8")

        profile_dir = create_profile("coder", clone_config=True, no_alias=True)

        assert not (profile_dir / "skills" / ".archive").exists()
        assert (profile_dir / "skills" / "live-skill" / "scripts" / ".archive" / "old-script.sh").is_file()

    @pytest.mark.platforms("windows")
    def test_clone_survives_an_archived_path_past_max_path(self, profile_env):
        """The reported failure: the staging prefix pushes an archived path past MAX_PATH."""
        if _windows_long_paths_enabled():
            pytest.skip("host sets LongPathsEnabled=1: the copy succeeds with or without the fix")

        default_home = profile_env / ".hermes"
        _seed_source_with_archive(default_home)
        skills = default_home / "skills"
        staging_delta = (
            len(str(default_home / "profiles" / f".coder.staging-{os.getpid()}" / "skills"))
            - len(str(skills))
        )
        probe = skills / ".archive" / "retired-skill"
        depth = 0
        while len(str(probe / "note.md")) + staging_delta < 262:
            depth += 1
            candidate = probe / f"d{depth}"
            if len(str(candidate / "note.md")) > 254:
                pytest.skip("tmp_path is too deep to build a source whose clone passes MAX_PATH")
            candidate.mkdir()
            probe = candidate
        (probe / "note.md").write_text("deep", encoding="utf-8")

        profile_dir = create_profile("coder", clone_config=True, no_alias=True)

        assert not (profile_dir / "skills" / ".archive").exists()
        assert (profile_dir / "skills" / "live-skill" / "SKILL.md").is_file()
