"""--shallow / -Shallow / HERMES_INSTALL_SHALLOW=1: a fresh install of one commit fetches only that commit.

Driven through the real repository stage against a local origin served over file:// (depth applies there).
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
POWERSHELL = next((c for c in ("pwsh", "powershell") if shutil.which(c)), None)
INSTALLERS = ["sh", pytest.param("ps1", marks=pytest.mark.skipif(POWERSHELL is None, reason="needs PowerShell"))]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


def _origin(path: Path) -> tuple[Path, list[str]]:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    # GitHub serves any reachable commit by SHA; a local upload-pack only when told to.
    _git(path, "config", "uploadpack.allowReachableSHA1InWant", "true")
    _git(path, "config", "uploadpack.allowFilter", "true")
    shas = []
    for content in ("one", "two", "three"):
        (path / "README").write_text(content)
        _git(path, "add", "README")
        _git(path, "commit", "-qm", content)
        shas.append(_git(path, "rev-parse", "HEAD"))
    return path, shas


def _stage(installer: str, tmp_path: Path, origin: Path, *, commit: str = "", shallow: bool = True,
           prelude: str = "") -> subprocess.CompletedProcess:
    install = tmp_path / "install"
    env = dict(os.environ, HOME=tmp_path.as_posix(), HERMES_HOME=(tmp_path / "home").as_posix(),
               HERMES_INSTALL_DIR=install.as_posix(), HERMES_REPO_URL=origin.as_uri())
    if shallow:
        env["HERMES_INSTALL_SHALLOW"] = "1"
    if installer == "sh":
        script = (f"source {shlex.quote((ROOT / 'scripts/install.sh').as_posix())} --manifest\nsleep() {{ :; }}\n"
                  f"{prelude}\nINSTALL_COMMIT={shlex.quote(commit)}\nstage_repository")
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60, check=False)
    if os.name != "nt":
        # pwsh on POSIX cannot run the Windows-only pinned Git: seed its PM store slot with host git.
        version = json.loads((ROOT / "pm" / "lock.json").read_text(encoding="utf-8-sig"))["packages"]["git"]["version"]
        cmd = tmp_path / "tools" / f"git-{version}-win32-x64" / "cmd"
        cmd.mkdir(parents=True, exist_ok=True)
        for name in ("git.exe", "git"):
            if not (cmd / name).exists():
                (cmd / name).symlink_to(shutil.which("git"))
        env.update(HERMES_RUNTIME_DIR=str(tmp_path / "tools"), PATH=str(cmd) + os.pathsep + env["PATH"])
    args = [POWERSHELL, "-NoProfile", "-File", str(ROOT / "scripts" / "install.ps1"), "-Stage", "repository",
            "-NonInteractive", "-InstallDir", str(install), "-HermesHome", str(tmp_path / "home")]
    if commit:
        args += ["-Commit", commit]
    return subprocess.run(args, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=240, check=False)


def _is_shallow(repo: Path) -> bool:
    return _git(repo, "rev-parse", "--is-shallow-repository") == "true"


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("installer", INSTALLERS)
def test_shallow_pin_fetches_only_the_pinned_commit(tmp_path, installer):
    origin, shas = _origin(tmp_path / "origin")
    result = _stage(installer, tmp_path, origin, commit=shas[1])
    assert result.returncode == 0, result.stdout + result.stderr
    install = tmp_path / "install"
    assert _git(install, "rev-parse", "HEAD") == shas[1]
    assert (install / "README").read_text() == "two"
    assert _is_shallow(install)
    assert _git(install, "rev-list", "--count", "HEAD") == "1"
    # A normal (unfiltered) remote that `hermes update` fetches every branch from.
    assert _git(install, "config", "--get", "remote.origin.fetch") == "+refs/heads/*:refs/remotes/origin/*"
    assert subprocess.run(["git", "-C", str(install), "config", "--get", "remote.origin.promisor"],
                          capture_output=True, check=False).returncode != 0
    assert not list(tmp_path.glob(".hermes-clone-*"))
    if installer == "ps1":
        assert _git(install, "config", "--local", "--get", "core.longpaths") == "true"


@pytest.mark.live_system_guard_bypass
def test_shallow_release_install_keeps_the_release_tag(tmp_path):
    origin, shas = _origin(tmp_path / "origin")
    _git(origin, "tag", "v1.2.3", shas[0])
    result = _stage("sh", tmp_path, origin, prelude="CHANNEL=stable\nlatest_stable_release() { printf v1.2.3; }")
    assert result.returncode == 0, result.stdout + result.stderr
    install = tmp_path / "install"
    assert _git(install, "rev-parse", "HEAD") == shas[0]
    assert _git(install, "describe", "--tags", "--exact-match") == "v1.2.3"
    assert _is_shallow(install)


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("installer", INSTALLERS)
def test_shallow_falls_back_to_the_blobless_clone(tmp_path, installer):
    """An abbreviated pin cannot be fetched by SHA: the stock blobless path installs it."""
    origin, shas = _origin(tmp_path / "origin")
    result = _stage(installer, tmp_path, origin, commit=shas[1][:12])
    assert result.returncode == 0, result.stdout + result.stderr
    install = tmp_path / "install"
    assert _git(install, "rev-parse", "HEAD") == shas[1]
    assert not _is_shallow(install)
    assert "cloning the history instead" in result.stdout + result.stderr


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("installer", INSTALLERS)
def test_without_the_option_the_clone_is_unchanged(tmp_path, installer):
    origin, shas = _origin(tmp_path / "origin")
    result = _stage(installer, tmp_path, origin, commit=shas[1], shallow=False)
    assert result.returncode == 0, result.stdout + result.stderr
    install = tmp_path / "install"
    assert _git(install, "rev-parse", "HEAD") == shas[1]
    assert not _is_shallow(install)
    assert _git(install, "rev-list", "--count", "HEAD") == "2"
