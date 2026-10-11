"""A throttled clone is retried without publishing a partial checkout."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent.parent


@pytest.mark.parametrize("materialize_fails", [False, True])
def test_clone_retries_and_publishes_only_materialized_tree(tmp_path, materialize_fails):
    origin = tmp_path / "origin"
    origin.mkdir()
    def git(*args):
        return subprocess.run(["git", "-C", str(origin), *args], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    git("init", "-b", "main")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Fixture")
    (origin / "README").write_text("complete checkout\n", encoding="utf-8")
    git("add", "README")
    git("commit", "-m", "fixture")
    dest = tmp_path / "install"
    attempts = tmp_path / "attempts"
    env = dict(os.environ, HOME=tmp_path.as_posix(), HERMES_HOME=(tmp_path / "home").as_posix(),
               HERMES_INSTALL_DIR=dest.as_posix(), HERMES_REPO_URL=origin.as_posix())
    # Inject throttling at the network boundary; the fallback and checkout use real Git.
    script = f'''source {shlex.quote((ROOT / 'scripts/install.sh').as_posix())} --manifest
sleep() {{ :; }}
git() {{
    if [ "$1" = clone ]; then
        printf '%s\\n' "$*" >> {shlex.quote(attempts.as_posix())}
        case " $* " in *" --filter=blob:none --no-checkout "*) ;; *) return 1 ;; esac
    fi
    if [ "{int(materialize_fails)}" = 1 ] && [ "${{3:-}}" = reset ]; then return 1; fi
    command git "$@"
}}
stage_repository
'''
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, check=False)
    calls = attempts.read_text(encoding="utf-8-sig").splitlines()
    assert len(calls) > 1
    assert "--no-checkout" in calls[-1]
    if materialize_fails:
        assert result.returncode != 0
        assert not dest.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert (dest / "README").read_text(encoding="utf-8-sig") == "complete checkout\n"
    assert not list(tmp_path.glob(".hermes-clone-*"))


POWERSHELL = next((c for c in ("pwsh", "powershell") if shutil.which(c)), None)


def _run_repository_stage(installer: str, tmp_path: Path, install: Path, origin_url: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, HOME=tmp_path.as_posix(), HERMES_HOME=(tmp_path / "home").as_posix(),
               HERMES_INSTALL_DIR=install.as_posix(), HERMES_REPO_URL=origin_url)
    if installer == "sh":
        script = f"source {shlex.quote((ROOT / 'scripts/install.sh').as_posix())} --manifest\nstage_repository\n"
        return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=False)
    if os.name != "nt":
        # pwsh on POSIX cannot run the Windows-only pinned Git: seed its PM store slot with host git.
        version = json.loads((ROOT / "pm" / "lock.json").read_text(encoding="utf-8-sig"))["packages"]["git"]["version"]
        cmd = tmp_path / "tools" / f"git-{version}-win32-x64" / "cmd"
        cmd.mkdir(parents=True)
        for name in ("git.exe", "git"):
            (cmd / name).symlink_to(shutil.which("git"))
        env.update(HERMES_RUNTIME_DIR=str(tmp_path / "tools"), PATH=str(cmd) + os.pathsep + env["PATH"])
    return subprocess.run([POWERSHELL, "-NoProfile", "-File", str(ROOT / "scripts" / "install.ps1"),
                           "-Stage", "repository", "-NonInteractive", "-InstallDir", str(install),
                           "-HermesHome", str(tmp_path / "home")],
                          env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240, check=False)


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("installer", [
    "sh",
    pytest.param("ps1", marks=pytest.mark.skipif(POWERSHELL is None, reason="needs PowerShell")),
])
def test_existing_treeless_checkout_is_refetched_before_update(tmp_path, installer):
    origin = tmp_path / "origin.git"
    source = tmp_path / "source"
    install = tmp_path / "install"

    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)], check=True, capture_output=True)
    def git(cwd, *args, **env):
        return subprocess.run(["git", "-C", str(cwd), *args], check=True,
                              capture_output=True, text=True, encoding="utf-8", errors="replace", env={**os.environ, **env})

    source.mkdir()
    git(source, "init", "-b", "main")
    git(source, "config", "user.email", "fixture@example.invalid")
    git(source, "config", "user.name", "Fixture")
    for index in range(5):
        (source / "history.txt").write_text(f"commit {index}\n", encoding="utf-8")
        git(source, "add", "history.txt")
        git(source, "commit", "-m", f"history {index}")
    git(source, "remote", "add", "origin", origin.as_posix())
    git(source, "push", "origin", "main")
    git(origin, "config", "uploadpack.allowFilter", "true")
    git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")

    # A file:// URL: git ignores --filter on a plain-path clone and would copy every tree.
    subprocess.run(["git", "clone", "--filter=tree:0", origin.as_uri(), install.as_posix()],
                   check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert git(install, "config", "--get", "remote.origin.partialclonefilter").stdout.strip() == "tree:0"
    head = git(install, "rev-parse", "HEAD").stdout.strip()

    result = _run_repository_stage(installer, tmp_path, install, origin.as_uri())
    assert result.returncode == 0, result.stdout + result.stderr
    assert git(install, "config", "--get", "remote.origin.partialclonefilter").stdout.strip() == "blob:none"
    assert git(install, "rev-parse", "HEAD").stdout.strip() == head, "the checkout was re-cloned, not converted"
    git(install, "rev-list", "HEAD", "--", "history.txt", GIT_NO_LAZY_FETCH="1")  # history answers offline
