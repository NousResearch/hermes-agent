import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from hermes_cli.version_info import (
    VersionInfo,
    _derived_version,
    _git_version_info,
    _reset_version_info_cache,
    _resolve_stamp_file,
    _run_bounded_process,
    _stamp_version_info,
    advertised_base_version,
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


def test_display_version_keeps_tagless_stamp_identity(tmp_path, monkeypatch):
    stamp = {
        "commit": "a" * 40,
        "baseVersion": "unknown",
        "displayVersion": "git.aaaaaaa",
        "source": "git",
        "payload": "bootstrap",
        "updateMechanism": "self",
    }
    stamp_file = tmp_path / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)

    info = get_version_info()

    assert info.base_version == "unknown"
    assert info.display_version == stamp["displayVersion"]
    assert info.commit == stamp["commit"]


def test_display_version_keeps_unknown_when_no_identity():
    info = VersionInfo("unknown", "unknown", None, None, None, "unknown")
    assert info.display_version == "unknown"


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


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="the blocking untracked scan uses a FIFO")
def test_a_timed_out_status_probe_never_strands_index_lock(tmp_path):
    """The update opens its receipt with this probe; killed by its 3 s timeout while refreshing
    the index, it used to leave .git/index.lock for the same run's merge to die on (#132089)."""
    def git(*args):
        subprocess.run(["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args],
                       cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    git("add", "a.txt")
    git("commit", "-qm", "a")
    (tmp_path / "junk").mkdir()
    os.mkfifo(tmp_path / "junk" / ".gitignore")  # the untracked scan blocks reading it
    (tmp_path / "junk" / "x").touch()
    (tmp_path / "a.txt").touch()  # stat-dirty: a locking status would refresh the index

    _git_version_info(tmp_path, include_untracked=True)

    assert not (tmp_path / ".git" / "index.lock").exists()


def test_bounded_process_returns_when_a_grandchild_holds_the_output(tmp_path):
    """#518: a probe whose child leaves a surviving grandchild behind must still return.

    ``capture_output`` reads a pipe, and a pipe reaches EOF only when EVERY holder of its
    write end has exited, so ``subprocess.run(timeout=...)`` blocks forever in its
    post-timeout ``communicate()`` retry while the grandchild lives. The identity probe hung
    Codex client construction that way, for minutes, inside a worker's boot.
    """
    child = (
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import threading; threading.Event().wait(30)'])\n"
        "print('version-probe-output')\n"
    )
    started = time.monotonic()
    result = _run_bounded_process([sys.executable, "-c", child], cwd=tmp_path, timeout=3)
    elapsed = time.monotonic() - started

    assert result is not None
    returncode, stdout = result
    assert returncode == 0
    assert stdout.strip() == "version-probe-output"
    assert elapsed < 10, f"the bounded probe waited {elapsed:.1f}s on a grandchild that outlived its child"


def test_bounded_process_kills_a_probe_that_overruns_its_budget(tmp_path):
    """An overrunning probe is killed and reported as a failure, never waited on."""
    child = "import threading; threading.Event().wait(30)"
    started = time.monotonic()
    assert _run_bounded_process([sys.executable, "-c", child], cwd=tmp_path, timeout=1) is None
    assert time.monotonic() - started < 10


def test_advertised_base_version_uses_the_stamp_without_a_git_probe(monkeypatch):
    """#518: request identity reads the build stamp. The stamp's live-commit check is
    provenance hygiene for ``get_version_info``, not a reason to run git while a client for a
    worker is being constructed."""
    from hermes_cli import version_info

    stamp = {"commit": "a" * 40, "baseVersion": "0.19.0", "source": "git"}
    monkeypatch.setattr(version_info, "read_install_stamp", lambda root: stamp)

    def _no_git(*args, **kwargs):
        raise AssertionError("advertised_base_version ran git")

    monkeypatch.setattr(version_info, "_run_git", _no_git)
    assert advertised_base_version() == "0.19.0"


def test_advertised_base_version_falls_back_to_a_warm_identity_then_unknown(monkeypatch):
    """No stamp: reuse the identity this process already resolved, else say unknown —
    never resolve one just to fill a User-Agent."""
    from hermes_cli import version_info

    monkeypatch.setattr(version_info, "read_install_stamp", lambda root: {})
    monkeypatch.setattr(
        version_info, "_cached_version_info",
        VersionInfo("0.21.5", "0.21.5", 0, "a" * 40, "main", "git"),
    )
    assert advertised_base_version() == "0.21.5"

    monkeypatch.setattr(version_info, "_cached_version_info", None)
    assert advertised_base_version() == "unknown"
