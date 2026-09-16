"""Run the real developer installer with offline dependency executables."""

from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "setup-hermes.sh"

# These tests execute Bash and create Unix launchers, including dangling links.
# Use the native-host marker so CI discovers the supported execution lane.
pytestmark = pytest.mark.linux_only


def _run(*args: str, cwd: Path | None = None, env=None, input=None):
    return subprocess.run(
        args, cwd=cwd, env=env, input=input, capture_output=True,
        text=True, check=True, timeout=30,
    )


def _executable(path: Path, content: str):
    path.write_text("#!/bin/bash\nset -eu\n" + content, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def installer_env(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    commands = tmp_path / "commands"
    commands.mkdir()
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    # No inherited PATH, shell startup hooks, credentials, Git config or profile.
    env = {
        "HOME": str(home), "HERMES_HOME": str(home / ".hermes"),
        "PATH": str(commands), "SHELL": "/bin/bash", "TMPDIR": str(scratch),
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C",
        "CALL_LOG": str(tmp_path / "calls"),
    }
    # Only local filesystem tools are reachable; no curl/package manager/network
    # client is present. Git is real for fixtures, restricted further at setup time.
    for name in ("git", "dirname", "mkdir", "ln", "mktemp", "rmdir", "touch",
                 "tr", "grep", "cp", "chmod", "rm"):
        executable = shutil.which(name)
        assert executable, f"required test executable missing: {name}"
        (commands / name).symlink_to(executable)
    _executable(commands / "rg", "exit 0\n")
    _executable(commands / "python", '''
case "$*" in
  --version) printf 'Python dependency double\n' ;;
  "$PWD/tools/skills_sync.py") printf 'skills-sync\n' >> "$CALL_LOG" ;;
  *) printf 'Unexpected Python invocation: %s\n' "$*" >&2; exit 97 ;;
esac
''')
    _executable(commands / "hermes", "exit 0\n")
    _executable(commands / "uv", '''
printf 'uv %s\n' "$*" >> "$CALL_LOG"
case "$*" in
  --version) printf 'uv dependency double\n' ;;
  'python find 3.11') printf '%s/python\n' "$PATH" ;;
  'venv venv --python 3.11')
    mkdir -p venv/bin
    cp "$PATH/python" venv/bin/python
    cp "$PATH/hermes" venv/bin/hermes ;;
  'sync --extra all --locked') test -x venv/bin/hermes ;;
  *) printf 'Unexpected uv invocation: %s\n' "$*" >&2; exit 98 ;;
esac
''')
    return env


def _init_repo(path: Path, env) -> Path:
    path.mkdir()
    _run("git", "-c", "init.defaultBranch=main", "init", "-q", str(path), env=env)
    _run("git", "config", "user.name", "Hermes Test", cwd=path, env=env)
    _run("git", "config", "user.email", "hermes-test@example.invalid", cwd=path, env=env)
    (path / "tracked").write_text("test\n", encoding="utf-8")
    _run("git", "add", "tracked", cwd=path, env=env)
    _run("git", "commit", "-qm", "initial", cwd=path, env=env)
    return path


def _make_checkout(tmp_path: Path, env, *, linked_worktree: bool) -> Path:
    primary = _init_repo(tmp_path / "primary", env)
    if not linked_worktree:
        return primary
    checkout = tmp_path / "linked"
    _run("git", "worktree", "add", "--detach", str(checkout), cwd=primary, env=env)
    return checkout


def _run_installer(checkout: Path, env, *, existing_launcher=False, broken_launcher=False):
    # Copy the complete entrypoint unchanged, never parse/extract source fragments.
    shutil.copyfile(SETUP_SCRIPT, checkout / "setup-hermes.sh")
    (checkout / "uv.lock").touch()
    (checkout / ".env.example").write_text("# temporary fixture\n", encoding="utf-8")
    home = Path(env["HOME"])
    launcher = home / ".local/bin/hermes"
    launcher.parent.mkdir(parents=True)
    if broken_launcher:
        launcher.symlink_to(home / "missing-launcher-target")
    elif existing_launcher == "symlink":
        target = home / "canonical-hermes"
        target.write_text("canonical launcher\n", encoding="utf-8")
        launcher.symlink_to(target)
    elif existing_launcher:
        launcher.write_text("canonical launcher\n", encoding="utf-8")

    git = Path(env["PATH"]) / "git"
    real_git = git.resolve()
    git.unlink()
    _executable(git, f'''
if [ "$#" = 4 ] && [ "$1" = -C ] && [ "$3" = rev-parse ]; then
  case "$4" in
    --absolute-git-dir|--git-common-dir) exec {shlex.quote(str(real_git))} "$@" ;;
  esac
fi
printf 'Unexpected Git invocation: %s\\n' "$*" >&2
exit 99
''')
    result = _run("/bin/bash", str(checkout / "setup-hermes.sh"),
                  cwd=checkout, env=env, input="n\n")
    assert "Setup complete!" in result.stdout
    assert (checkout / ".env").read_text() == "# temporary fixture\n"
    assert (home / ".bash_profile").is_file()
    assert (Path(env["HERMES_HOME"]) / "skills").is_dir()
    assert Path(env["CALL_LOG"]).read_text().splitlines() == [
        "uv --version", "uv python find 3.11", "uv python find 3.11",
        "uv venv venv --python 3.11", "uv sync --extra all --locked", "skills-sync",
    ]
    return result, launcher, checkout / "venv/bin/hermes"


def test_setup_hermes_script_is_valid_shell():
    _run("/bin/bash", "-n", str(SETUP_SCRIPT))


@pytest.mark.parametrize("launcher_kind", ["file", "symlink"])
def test_setup_preserves_existing_launcher_from_linked_worktree(tmp_path, installer_env, launcher_kind):
    checkout = _make_checkout(tmp_path, installer_env, linked_worktree=True)
    result, launcher, _ = _run_installer(checkout, installer_env, existing_launcher=launcher_kind)
    assert "Linked Git worktree detected" in result.stdout
    assert launcher.is_symlink() == (launcher_kind == "symlink")
    if launcher_kind == "symlink":
        assert launcher.readlink() == Path(installer_env["HOME"]) / "canonical-hermes"
    assert launcher.read_text(encoding="utf-8") == "canonical launcher\n"


def test_setup_links_from_linked_worktree_when_launcher_is_absent(tmp_path, installer_env):
    checkout = _make_checkout(tmp_path, installer_env, linked_worktree=True)
    _, launcher, hermes_bin = _run_installer(checkout, installer_env)
    assert launcher.is_symlink()
    assert launcher.resolve() == hermes_bin


def test_setup_preserves_broken_launcher_from_linked_worktree(tmp_path, installer_env):
    checkout = _make_checkout(tmp_path, installer_env, linked_worktree=True)
    result, launcher, _ = _run_installer(checkout, installer_env, broken_launcher=True)
    assert "Linked Git worktree detected" in result.stdout
    assert launcher.is_symlink()
    assert launcher.readlink() == Path(installer_env["HOME"]) / "missing-launcher-target"


def test_setup_replaces_launcher_from_primary_checkout(tmp_path, installer_env):
    checkout = _make_checkout(tmp_path, installer_env, linked_worktree=False)
    _, launcher, hermes_bin = _run_installer(checkout, installer_env, existing_launcher=True)
    assert launcher.is_symlink()
    assert launcher.resolve() == hermes_bin


def test_worktree_detection_does_not_classify_submodule_as_linked_worktree(tmp_path, installer_env):
    source = _init_repo(tmp_path / "source", installer_env)
    superproject = _init_repo(tmp_path / "superproject", installer_env)
    _run("git", "-c", "protocol.file.allow=always", "submodule", "add", "-q",
         str(source), "submodule", cwd=superproject,
         # Git's submodule shell helper needs its normal Unix utilities. This
         # local-only fixture setup is separate from the restricted installer.
         env=installer_env | {"PATH": "/usr/bin:/bin", "GIT_ALLOW_PROTOCOL": "file"})
    checkout = superproject / "submodule"
    assert (checkout / ".git").is_file()
    result, launcher, hermes_bin = _run_installer(checkout, installer_env, existing_launcher=True)
    assert "Linked Git worktree detected" not in result.stdout
    assert launcher.is_symlink()
    assert launcher.resolve() == hermes_bin
