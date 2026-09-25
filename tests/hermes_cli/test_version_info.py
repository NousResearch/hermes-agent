import json
import os
from pathlib import Path
import subprocess

import pytest

from hermes_cli.version_info import (
    VersionInfo,
    _derived_version,
    _reset_version_info_cache,
    _resolve_stamp_file,
    _stamp_version_info,
    get_version_info,
)


def setup_function():
    _reset_version_info_cache()


def test_derived_version_shows_plus_question_for_dirty_unknown_distance():
    assert _derived_version("0.19.0", None, dirty=True) == "0.19.0+?"
    assert _derived_version("0.19.0", None, dirty=False) == "0.19.0"
    assert _derived_version("0.19.0", 5, dirty=True) == "0.19.0+5"
    assert _derived_version("0.19.0", 0, dirty=True) == "0.19.0"


def test_display_version_names_the_distance_without_the_commit():
    """Labels say how far past the release an install is; the commit is shown
    beside them where there is room (`hermes --version` keeps `.g<sha>`)."""
    ahead = VersionInfo("0.21.5", "0.21.5+1913.gf83a9e9", 1913, "f83a9e9" + "0" * 33, "main", "git")
    assert ahead.display_version == "0.21.5+1913"
    assert VersionInfo("0.21.5", "0.21.5", 0, None, None, "git").display_version == "0.21.5"


def test_stamp_version_info_reads_nix_stamp(tmp_path, monkeypatch):
    stamp = {
        "schemaVersion": 2,
        "commit": "a" * 40,
        "branch": "feature/version",
        "baseVersion": "0.19.0",
        "displayVersion": "0.19.0+3",
        "distance": 3,
        "dirty": False,
        "source": "nix",
        "distribution": "nix",
        "updateMechanism": "external",
    }
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp))
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)

    info = get_version_info()

    assert info == VersionInfo("0.19.0", "0.19.0+3", 3, "a" * 40, "feature/version", "nix", distribution="nix")


def test_stamp_version_info_preserves_ci_provenance_and_docker_distribution(tmp_path, monkeypatch):
    stamp = {"commit": "d" * 40, "source": "ci", "distribution": "docker", "updateMechanism": "external"}
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp))
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)

    info = get_version_info()

    assert info.source == "ci"
    assert info.distribution == "docker"


def test_stamp_version_info_preserves_missing_branch(tmp_path, monkeypatch):
    stamp = {
        "schemaVersion": 2,
        "commit": "b" * 40,
        "branch": None,
        "baseVersion": "0.19.0",
        "displayVersion": "0.19.0+?",
        "distance": None,
        "dirty": True,
        "source": "docker",
        "updateMechanism": "external",
    }
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp))
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)

    info = get_version_info()

    assert info == VersionInfo("0.19.0", "0.19.0+?", None, "b" * 40, None, "docker", True)


def test_stamp_version_info_ignores_fallback_commit(tmp_path, monkeypatch):
    """All-zero commit means the stamp has no real SHA — skip it."""
    stamp = {
        "schemaVersion": 2,
        "commit": "0" * 40,
        "branch": "main",
        "source": "fallback",
        "updateMechanism": "self",
    }
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp))
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: None)

    info = get_version_info()

    assert info.source == "unknown"
    assert info.commit is None


def test_stamp_version_info_returns_none_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: None)
    assert _stamp_version_info() is None


def test_get_version_info_unknown_when_no_stamp_and_no_git(monkeypatch):
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: None)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: None)

    info = get_version_info()

    assert info.base_version == "unknown"
    assert info.derived_version == "unknown"
    assert info.distance is None
    assert info.commit is None
    assert info.source == "unknown"


def test_get_version_info_derives_identity_from_reachable_release_tag(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
            env={"HOME": str(tmp_path), "PATH": __import__("os").environ["PATH"]},
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Hermes Test")
    git("config", "user.email", "hermes@example.invalid")
    (repo / "tracked").write_text("release\n", encoding="utf-8")
    git("add", "tracked")
    git("commit", "-qm", "release")
    git("tag", "v0.21.4")
    git("tag", "v2026.9.21")
    (repo / "tracked").write_text("next\n", encoding="utf-8")
    git("commit", "-qam", "next")

    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: None)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: repo)

    info = get_version_info()

    assert info.base_version == "0.21.4"
    assert info.derived_version == f"0.21.4+1.g{git('rev-parse', '--short=7', 'HEAD')}"
    assert info.distance == 1
    assert info.commit == git("rev-parse", "HEAD")
    assert info.source == "git"


def test_get_version_info_takes_the_version_a_calver_only_release_shipped(tmp_path, monkeypatch):
    """Releases tagged only vYYYY.M.D resolve to their pyproject version, not "unknown"."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
            env={"HOME": str(tmp_path), "PATH": __import__("os").environ["PATH"]},
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Hermes Test")
    git("config", "user.email", "hermes@example.invalid")
    (repo / "pyproject.toml").write_text('[project]\nname = "hermes-agent"\nversion = "0.21.4"\n', encoding="utf-8")
    git("add", "pyproject.toml")
    git("commit", "-qm", "release")
    git("tag", "v2026.9.21")
    (repo / "pyproject.toml").write_text('[project]\nname = "hermes-agent"\nversion = "0.0.0"\n', encoding="utf-8")
    git("commit", "-qam", "next")

    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: None)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: repo)

    info = get_version_info()

    assert info.base_version == "0.21.4"
    assert info.distance == 1
    assert info.derived_version == f"0.21.4+1.g{git('rev-parse', '--short=7', 'HEAD')}"


def test_stale_unknown_stamp_defers_to_live_checkout(tmp_path, monkeypatch):
    """A stamp naming the live commit but no resolvable version must not win.

    ``write_source_stamp`` published this checkout when its only merged tags
    were CalVer (``v2026.9.24``) and a canary, so the semver walk both sides
    share recorded ``baseVersion: unknown``. The tree can resolve a version as
    soon as a semver release tag becomes reachable (or a CalVer tag's
    pyproject carries one) — a placeholder that pins "unknown" forever while
    the git walk right behind it would return 0.21.5 is a lost fact, not a
    provenance decision. The stamp is skipped; git answers.
    """

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
            env={"HOME": str(tmp_path), "PATH": __import__("os").environ["PATH"]},
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Hermes Test")
    git("config", "user.email", "hermes@example.invalid")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "hermes-agent"\nversion = "0.21.5"\n', encoding="utf-8"
    )
    git("add", "pyproject.toml")
    git("commit", "-qm", "release")
    git("tag", "v2026.9.24")

    # The stamp this checkout was left with: written while the pyproject at
    # the CalVer tag still read 0.0.0, so the version walk could not answer.
    (repo / "install-stamp.json").write_text(
        json.dumps(
            {
                "commit": git("rev-parse", "HEAD"),
                "source": "git",
                "updateMechanism": "self",
                "baseVersion": "unknown",
                "displayVersion": f"git.{git('rev-parse', '--short=7', 'HEAD')}",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: repo / "install-stamp.json")
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: repo)

    info = get_version_info()

    assert info.base_version == "0.21.5"
    assert info.distance == 0
    assert info.derived_version == "0.21.5"
    assert info.commit == git("rev-parse", "HEAD")


def test_stamped_unknown_version_still_wins_without_git(tmp_path, monkeypatch):
    """A sealed install's stamp is authoritative even when it says unknown.

    The fallback through to git exists only for a stamp that sits beside the
    ``.git`` it describes. A Docker/Nix stamp in a stamp-only tree has no live
    checkout to defer to, so the placeholder it carries is the truth.
    """
    stamp = {
        "commit": "a" * 40,
        "source": "git",
        "updateMechanism": "self",
        "baseVersion": "unknown",
        "displayVersion": "unknown",
    }
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp))
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: None)

    info = get_version_info()

    assert info.base_version == "unknown"
    assert info.derived_version == "unknown"
    assert info.commit == "a" * 40


def test_run_git_reads_git_output_as_utf8_not_the_locale_codec(tmp_path, monkeypatch):
    """Git speaks bytes; a release tag's blob may be undecodable in the locale.

    ``git show <tag>:pyproject.toml`` returns UTF-8, and real tags carry an
    em-dash (``\\x80\\x94``) that GBK cannot decode. Reading it with
    ``subprocess.run(text=True)`` hands the bytes to the console codec: on a
    cp936 install the reader thread raises ``UnicodeDecodeError``,
    ``result.stdout`` comes back None, and the CalVer fallback reports no
    version at all — so ``GET /api/health`` answers ``unknown`` even though
    ``git describe`` worked one line earlier. The interpreter the desktop app
    spawns (``-I``, which ignores ``PYTHONUTF8``) is exactly that install, on
    a default Windows console.

    Assert the decoding contract directly, so this fails on any host where
    ``_run_git`` would fall back to the locale codec.
    """
    from hermes_cli import version_info
    from hermes_cli.version_info import _run_git

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        env = {"HOME": str(tmp_path), "PATH": os.environ["PATH"]}
        result = subprocess.run(
            ["git", *args], cwd=repo, check=True, text=True, capture_output=True, env=env
        )
        return result.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Hermes Test")
    git("config", "user.email", "hermes@example.invalid")
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "hermes-agent"\nversion = "0.21.5"\ndescription = "agent — with an em-dash"\n',
        encoding="utf-8",
    )
    git("add", "pyproject.toml")
    git("commit", "-qm", "release")
    git("tag", "v2026.9.24")

    # No text= wrapper: whatever _run_git passes to subprocess.run must leave
    # decoding to it. A wrapper that asks for locale text is the regression.
    captured = {}
    real_run = subprocess.run

    def spy_run(*args, **kwargs):
        captured.update(kwargs)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(version_info.subprocess, "run", spy_run)
    _run_git(repo, "show", "v2026.9.24:pyproject.toml")

    assert "text" not in captured or captured["text"] is not True, (
        "_run_git asks subprocess for locale-decoded text; a cp936 console then "
        "loses any non-ASCII byte a release tag's pyproject carries"
    )
    assert "encoding" not in captured or captured["encoding"] in (None, "utf-8"), (
        "_run_git must not push a non-UTF-8 encoding onto git's output"
    )

    shown = _run_git(repo, "show", "v2026.9.24:pyproject.toml")
    assert shown is not None
    assert 'version = "0.21.5"' in shown
    assert "—" in shown


def test_resolve_stamp_file_honors_install_root(tmp_path, monkeypatch):
    """Sealed installs (the Nix wrapper) point HERMES_INSTALL_ROOT at the stamp dir."""
    stamp = {"commit": "e" * 40, "source": "nix", "distribution": "nix", "updateMechanism": "external"}
    (tmp_path / "install-stamp.json").write_text(json.dumps(stamp))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(tmp_path))

    assert _resolve_stamp_file() == tmp_path / "install-stamp.json"

    info = get_version_info()
    assert info.commit == "e" * 40
    assert info.source == "nix"
    assert info.distribution == "nix"


def test_resolve_stamp_file_install_root_without_stamp_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(tmp_path))
    assert _resolve_stamp_file() is None


def test_resolve_stamp_file_falls_back_to_code_root_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
    stamp = {"commit": "f" * 40, "source": "docker", "updateMechanism": "external"}
    (tmp_path / "install-stamp.json").write_text(json.dumps(stamp))
    monkeypatch.setattr("pm.paths.repo_root", lambda: tmp_path)

    assert _resolve_stamp_file() == tmp_path / "install-stamp.json"


def test_old_updater_version_stub_reads_the_same_stamp_as_version_info(tmp_path):
    """``hermes_cli.__version__`` exists only for shipped updaters that import it after the
    checkout swap (tests/compat/old_updater_surface.json). It must report the stamp's base
    version exactly as get_version_info() does, and the pre-stamp placeholder without one.
    A fresh interpreter, since the stub is evaluated when the package is imported."""
    import os
    import sys

    repo = Path(__file__).resolve().parents[2]
    probe = (
        f"import sys; sys.path.insert(0, {str(repo)!r}); import hermes_cli; "
        "from hermes_cli.version_info import get_version_info; "
        "print(hermes_cli.__version__, get_version_info().base_version)"
    )

    def read(install_root: Path) -> list[str]:
        env = {**os.environ, "HERMES_INSTALL_ROOT": str(install_root)}
        return subprocess.run(
            [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
        ).stdout.split()

    stamped = tmp_path / "stamped"
    stamped.mkdir()
    stamp = {"commit": "c" * 40, "baseVersion": "9.8.7", "source": "ci", "updateMechanism": "external"}
    (stamped / "install-stamp.json").write_text(json.dumps(stamp))
    compat, identity = read(stamped)
    assert compat == identity == stamp["baseVersion"]

    unstamped = tmp_path / "unstamped"
    unstamped.mkdir()
    assert read(unstamped)[0] == "0.0.0"
