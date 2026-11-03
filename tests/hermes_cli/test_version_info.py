import json
import os
from pathlib import Path
import subprocess

import pytest

from hermes_cli.version_info import (
    VersionInfo,
    _derived_version,
    _git_version_info,
    _reset_version_info_cache,
    _resolve_stamp_file,
    _stamp_version_info,
    get_version_info,
)


def setup_function():
    _reset_version_info_cache()


@pytest.fixture
def live_git_checkout(tmp_path):
    """A real git repo whose only reachable tag is CalVer, plus its ``git`` helper.

    ``STABLE_TAG_RE`` rejects a 4-digit major on purpose, so a checkout in this
    shape is exactly the one ``write_source_stamp`` recorded ``baseVersion:
    unknown`` on. The git walk still resolves a version through the tag's
    ``pyproject.toml``, which is what makes a stale placeholder detectable: the
    stamp and the checkout disagree.

    Yields ``(repo, git)``; ``git`` runs a real subprocess in the repo and
    returns stripped stdout.
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
            env={"HOME": str(tmp_path), "PATH": os.environ["PATH"]},
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
    return repo, git


def _write_stamp(repo: Path, **fields: object) -> Path:
    """Write an install stamp into ``repo`` and return its path."""
    stamp = {
        "commit": None,
        "source": "git",
        "updateMechanism": "self",
        "baseVersion": "unknown",
        "displayVersion": "unknown",
    }
    stamp.update(fields)
    stamp_file = repo / "install-stamp.json"
    stamp_file.write_text(json.dumps(stamp), encoding="utf-8")
    return stamp_file


def _point_at(monkeypatch, repo: Path, stamp_file: Path) -> None:
    """Resolve the stamp and the git walk against the same temporary checkout."""
    monkeypatch.setattr("hermes_cli.version_info._resolve_stamp_file", lambda: stamp_file)
    monkeypatch.setattr("hermes_cli.version_info._resolve_repo_dir", lambda: repo)


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


@pytest.mark.parametrize("placeholder", [None, "", "unknown"])
def test_stale_unknown_stamp_defers_to_live_checkout(live_git_checkout, monkeypatch, placeholder):
    """A stamp naming the live commit but no resolvable version must not win.

    ``write_source_stamp`` published ``baseVersion: unknown`` on a checkout
    whose only reachable tags were CalVer (which ``STABLE_TAG_RE`` rejects on
    purpose). The stamp then short-circuits ``_stamp_version_info`` and the
    git walk behind it never runs, so the checkout stays "unknown" forever —
    even after a semver release becomes reachable, or after the CalVer
    fallback learns to read the tag's pyproject.

    ``baseVersion`` reaches the guard in three spellings: an absent key parsed
    as ``None``, an empty string from a template, and the literal placeholders
    the writer and older stamps use. All three describe the same missing fact,
    so all three must defer.

    The stamp is only skipped when it names HEAD and carries no version: a
    sealed install's stamp, with no ``.git`` beside it to defer to, stays
    authoritative.
    """
    repo, git = live_git_checkout
    head = git("rev-parse", "HEAD")
    stamp_file = _write_stamp(repo, commit=head, baseVersion=placeholder)
    _point_at(monkeypatch, repo, stamp_file)

    info = get_version_info()

    assert info.base_version == "0.21.5"
    assert info.distance == 0
    assert info.derived_version == "0.21.5"
    assert info.commit == head
    assert info.source == "git"


def test_valid_version_stamp_beside_a_live_checkout_stays_authoritative(live_git_checkout, monkeypatch):
    """A stamp that names a real version is not deferred, even at HEAD.

    The placeholder skip excuses only a stamp carrying no version. A stamp that
    records ``0.21.4+3`` keeps answering with that version and its provenance —
    the git walk would say ``0.21.5`` here, so the assertion tells the two
    paths apart.
    """
    repo, git = live_git_checkout
    head = git("rev-parse", "HEAD")
    stamp_file = _write_stamp(
        repo,
        commit=head,
        baseVersion="0.21.4",
        displayVersion="0.21.4+3",
        distance=3,
    )
    _point_at(monkeypatch, repo, stamp_file)

    info = get_version_info()

    assert info.base_version == "0.21.4"
    assert info.distance == 3
    assert info.derived_version == "0.21.4+3"
    assert info.commit == head


def test_stamp_naming_another_commit_defers_to_the_live_checkout(live_git_checkout, monkeypatch):
    """The pre-existing contract: a stamp describing a different commit is stale.

    This is the guard the placeholder skip sits behind. It has to keep firing
    for a stamp whose version looks perfectly valid, because that version
    describes a checkout that is no longer here — trusting it would report the
    previous build's identity against the current tree.
    """
    repo, git = live_git_checkout
    head = git("rev-parse", "HEAD")
    stamp_file = _write_stamp(
        repo,
        commit="f" * 40,
        baseVersion="0.21.4",
        displayVersion="0.21.4+3",
        distance=3,
    )
    _point_at(monkeypatch, repo, stamp_file)

    info = get_version_info()

    assert info.base_version == "0.21.5"
    assert info.distance == 0
    assert info.commit == head


def test_stamped_unknown_version_still_wins_without_git(tmp_path, monkeypatch):
    """A sealed install's stamp is authoritative even when it says unknown.

    The deferral through to git exists only for a stamp sitting beside the
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
