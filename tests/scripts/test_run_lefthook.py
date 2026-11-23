"""The git hooks' lefthook command (``lefthook:`` in lefthook.yml, ``scripts/run-lefthook``).

lefthook's generated hook, when the binary path it baked in is gone and none is on PATH,
falls back to ``uv run lefthook``. ``uv run`` syncs the checkout's ``.venv`` and
editable-installs the checkout into it: a worktree whose ``.venv`` linked to another
checkout's venv re-pointed that checkout at the worktree's code on every commit. The
command the hook runs instead must find lefthook without uv, or skip the hooks.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.pm.activation_support import bash, posix

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.platforms("posix")


def _shim(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A git checkout carrying the repo's hook command, a `.venv` linked to another
    checkout's venv, and `uv`/`just` on PATH that record each call."""
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "lefthook.yml", root / "lefthook.yml")
    for script in ("run-lefthook", "hook-just"):
        shutil.copy2(REPO_ROOT / "scripts" / script, root / "scripts" / script)
    subprocess.run(["git", "init", "-q", str(root)], check=True, timeout=60)
    other_venv = tmp_path / "other" / ".venv"
    other_venv.mkdir(parents=True)
    (root / ".venv").symlink_to(other_venv)
    calls = tmp_path / "calls"
    _shim(tmp_path / "bin" / "uv", f'echo "uv $*" >> "{posix(calls)}"; exit 1')
    _shim(tmp_path / "bin" / "just", "exit 0")
    return root


def _hook_command(root: Path) -> str:
    """The `lefthook:` value: a double-quoted scalar without escapes, which JSON reads too."""
    lines = (root / "lefthook.yml").read_text(encoding="utf-8").splitlines()
    return json.loads(next(line for line in lines if line.startswith("lefthook:")).partition(":")[2])


def _run_hook(root: Path) -> subprocess.CompletedProcess:
    """Run the hook command the way the generated hook does: `<command> run pre-commit`."""
    path = f"{posix(root.parent / 'bin')}:/usr/bin:/bin"
    env = {**os.environ, "PATH": path}
    return subprocess.run([bash(), "-c", f'{_hook_command(root)} "$@"', "hook", "run", "pre-commit"],
                          cwd=root, env=env, capture_output=True, text=True, timeout=60, check=False)


def _package() -> str:
    os_name = platform.system().lower()
    arch = {"aarch64": "arm64", "x86_64": "x64"}.get(platform.machine(), platform.machine())
    return f"lefthook-{os_name}-{arch}"


def _native_lefthook(root: Path) -> Path:
    package = _package()
    return root / "node_modules" / ".pnpm" / f"{package}@2.2.1" / "node_modules" / package / "bin" / "lefthook"


GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]


def test_hook_runs_the_checkouts_lefthook_and_never_uv(checkout: Path):
    received = checkout.parent / "received"
    _shim(_native_lefthook(checkout), f'echo "$@" > "{posix(received)}"')

    run = _run_hook(checkout)

    assert run.returncode == 0, run.stderr
    assert received.read_text(encoding="utf-8").split() == ["run", "pre-commit"]
    assert not (checkout.parent / "calls").exists()


def test_hook_without_lefthook_skips_instead_of_reaching_uv(checkout: Path):
    run = _run_hook(checkout)

    assert run.returncode == 0, run.stderr
    assert "git hooks skipped" in run.stderr
    assert not (checkout.parent / "calls").exists()
    assert list((checkout / ".venv").iterdir()) == []


def test_hook_finds_lefthook_in_the_hoisted_layout(checkout: Path):
    """npm and `pnpm --node-linker=hoisted` put the native package straight under node_modules."""
    received = checkout.parent / "received"
    _shim(checkout / "node_modules" / _package() / "bin" / "lefthook", f'echo "$@" > "{posix(received)}"')

    run = _run_hook(checkout)

    assert run.returncode == 0, run.stderr
    assert received.read_text(encoding="utf-8").split() == ["run", "pre-commit"]


def test_linked_worktree_runs_the_main_checkouts_lefthook(checkout: Path):
    """Hooks are shared by every worktree; one without its own node_modules uses the main checkout's."""
    received = checkout.parent / "received"
    _shim(_native_lefthook(checkout), f'echo "$@" > "{posix(received)}"')
    subprocess.run([*GIT, "add", "lefthook.yml", "scripts"], cwd=checkout, check=True, timeout=60)
    subprocess.run([*GIT, "commit", "-q", "--no-verify", "-m", "init"], cwd=checkout, check=True, timeout=60)
    linked = checkout.parent / "linked"
    subprocess.run([*GIT, "worktree", "add", "-q", "--detach", str(linked)], cwd=checkout, check=True, timeout=60)

    run = _run_hook(linked)

    assert run.returncode == 0, run.stderr
    assert received.read_text(encoding="utf-8").split() == ["run", "pre-commit"]
    assert not (checkout.parent / "calls").exists()


def test_job_without_just_runs_it_in_the_checkouts_own_env(checkout: Path):
    """An unactivated shell has no `just`: a job gets it from scripts/run-in-hermes-env, never uv."""
    if shutil.which("just", path="/usr/bin:/bin"):
        pytest.skip("a system `just` would take the activated-shell branch")
    (checkout.parent / "bin" / "just").unlink()
    runner = checkout.parent / "runner"
    _shim(checkout / "scripts" / "run-in-hermes-env", f'echo "$@" > "{posix(runner)}"')

    run = subprocess.run(["sh", str(checkout / "scripts" / "hook-just"), "ruff", "a.py"], cwd=checkout,
                         env={**os.environ, "PATH": f"{posix(checkout.parent / 'bin')}:/usr/bin:/bin"},
                         capture_output=True, text=True, timeout=60, check=False)

    assert run.returncode == 0, run.stderr
    assert runner.read_text(encoding="utf-8").split() == ["just", "ruff", "a.py"]
    assert not (checkout.parent / "calls").exists()


def _pinned_lefthook() -> Path | None:
    """The pnpm-installed lefthook of this checkout or its clone's main checkout."""
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=60, check=False).stdout.strip()
    native = _native_lefthook(Path())
    for root in dict.fromkeys((REPO_ROOT, Path(common).parent if common else REPO_ROOT)):
        found = sorted((root / "node_modules" / ".pnpm").glob(
            f"{native.parts[2].partition('@')[0]}@*/node_modules/{native.parts[4]}/bin/lefthook"))
        if found:
            return found[0]
    return None


def _install_hooks(checkout: Path, tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Write the hooks with the pnpm-pinned lefthook from a copy that is then deleted: the
    state of a clone whose hooks were installed from a since-removed worktree."""
    pinned = _pinned_lefthook()
    if pinned is None:
        pytest.skip("no pnpm-installed lefthook to generate a hook with (pnpm install)")
    baked = tmp_path / "removed-worktree" / "lefthook"
    baked.parent.mkdir()
    shutil.copy2(pinned, baked)
    env = {**os.environ, "PATH": f"{posix(tmp_path / 'bin')}:/usr/bin:/bin",
           "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    env.pop("LEFTHOOK_BIN", None)
    subprocess.run([str(baked), "install", "-f"], cwd=checkout, env=env, check=True,
                   capture_output=True, timeout=60)
    shutil.rmtree(baked.parent)
    return pinned, env


def _commit(checkout: Path, env: dict[str, str], name: str) -> subprocess.CompletedProcess:
    (checkout / name).write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", name], cwd=checkout, env=env, check=True, timeout=60)
    return subprocess.run(["git", "commit", "-q", "-m", "change"], cwd=checkout, env=env,
                          capture_output=True, text=True, timeout=120, check=False)


def test_generated_hook_with_its_binary_gone_never_reaches_uv(checkout: Path, tmp_path: Path):
    """Through the hook `lefthook install` writes, once the binary it baked in is deleted."""
    _, env = _install_hooks(checkout, tmp_path)

    commit = _commit(checkout, env, "change.py")

    assert commit.returncode == 0, commit.stderr
    assert not (checkout.parent / "calls").exists()
    assert list((checkout / ".venv").iterdir()) == []
    assert "git hooks skipped" in commit.stderr


def test_commit_no_job_matches_sets_up_no_environment(checkout: Path, tmp_path: Path):
    """Without `just`, a docs-only commit still never runs the environment setup: lefthook
    filters the staged files first, and only a matched job resolves `just`."""
    pinned, env = _install_hooks(checkout, tmp_path)
    native = _native_lefthook(checkout)
    native.parent.mkdir(parents=True)
    shutil.copy2(pinned, native)
    (tmp_path / "bin" / "just").unlink()
    runner = tmp_path / "runner"
    _shim(checkout / "scripts" / "run-in-hermes-env", f'echo "$@" >> "{posix(runner)}"; exit 1')

    commit = _commit(checkout, env, "README.md")

    assert commit.returncode == 0, commit.stderr
    assert not runner.exists()
    assert not (tmp_path / "calls").exists()


def test_root_job_in_a_linked_worktree_finds_the_checkouts_hook_just(checkout: Path, tmp_path: Path):
    """A `root: apps/desktop/` job committed from a linked worktree resolves `{just}` at the
    worktree root. Git exports GIT_DIR to hooks there, and with it set `--show-toplevel`
    answers the job's cwd, so the job died on `apps/desktop/scripts/hook-just`."""
    pinned, env = _install_hooks(checkout, tmp_path)
    native = _native_lefthook(checkout)
    native.parent.mkdir(parents=True)
    shutil.copy2(pinned, native)
    received = tmp_path / "just-args"
    _shim(tmp_path / "bin" / "just", f'echo "$PWD $*" >> "{posix(received)}"')
    subprocess.run([*GIT, "add", "lefthook.yml", "scripts"], cwd=checkout, env=env, check=True, timeout=60)
    subprocess.run([*GIT, "commit", "-q", "--no-verify", "-m", "init"], cwd=checkout, env=env, check=True, timeout=60)
    linked = tmp_path / "linked"
    subprocess.run([*GIT, "worktree", "add", "-q", "--detach", str(linked)], cwd=checkout, env=env, check=True,
                   timeout=60)
    (linked / "apps" / "desktop").mkdir(parents=True)

    commit = _commit(linked, env, "apps/desktop/a.ts")

    assert commit.returncode == 0, commit.stdout + commit.stderr
    assert received.read_text(encoding="utf-8").split() == [posix(linked / "apps" / "desktop"), "eslint", "./a.ts"]
