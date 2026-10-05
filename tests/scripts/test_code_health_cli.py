"""scripts/check runner, git hooks, CLI output modes and regex-rule scope, on real git repos.

Fixture repos carry a copy of this checkout's engine (scripts/check, scripts/code_health, the
guard scripts, pyproject's ruff pin) so the runner and the hooks execute exactly as they would
in a clone, against tiny trees that keep every check fast.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from scripts.code_health import cli
from scripts.code_health.config import ENFORCEMENT

REPO = Path(__file__).resolve().parents[2]
_GUARD_SCRIPTS = (
    "scripts/check-windows-footguns.py", "scripts/check_bash_shebangs.py",
    "scripts/check_no_tmp_literals.py", "scripts/check_config_yaml_writers.py",
    "scripts/ci/check_os_marker_fakes.py", "scripts/check-case-collisions.py",
    "scripts/ci/check_lazy_deps_imports.py", "scripts/ci/check_profile_archive_boundary.py",
)
_SUPPORT = ("scripts/ci/profile_scope_patterns.json", *_GUARD_SCRIPTS)
_LEGACY = "def legacy(x):\n" + "".join(f"    if x == {i}:\n        return {i}\n" for i in range(21))
_GROWN = _LEGACY + "    if x == 99:\n        return 99\n"
_ENV_COPY = "import os\n\n\ndef child_env():\n    env = os.environ.copy()\n    return env\n"
_SWITCH = "scripts/code_health/config.py"


def _env() -> dict[str, str]:
    """The caller's env minus anything that would point git at another repo or config."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return env


def _sh(cwd: Path, *argv: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), cwd=cwd, env=env or _env(), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
                          timeout=600, check=False)


def _git(repo: Path, *args: str) -> str:
    proc = _sh(repo, "git", *args)
    assert proc.returncode == 0, (args, proc.stdout, proc.stderr)
    return proc.stdout.strip()


def _write(repo: Path, files: Mapping[str, str | None]) -> None:
    for rel, text in files.items():
        path = repo / rel
        if text is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


def _commit(repo: Path, files: Mapping[str, str | None]) -> str:
    _write(repo, files)
    _git(repo, "add", "--all", "--", *files)
    _git(repo, "commit", "-q", "-m", "step")
    return _git(repo, "rev-parse", "HEAD")


def _init(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    (path / ".gitignore").write_text(".venv\n", encoding="utf-8")
    shutil.copy(REPO / "pyproject.toml", path / "pyproject.toml")
    return path


def _add_engine(repo: Path, checker: bool = True) -> list[str]:
    """What the checker's jobs read (guard scripts, policy, a tests/ dir) and, with ``checker``,
    scripts/check + scripts/code_health themselves; returns the top-level paths to stage."""
    for rel in _SUPPORT:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, repo / rel)
    _write(repo, {"tests/README.md": "os-marker-fakes scans tests/\n"})
    if checker:
        shutil.copy2(REPO / "scripts/check", repo / "scripts/check")
        shutil.copytree(REPO / "scripts/code_health", repo / "scripts/code_health",
                        ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    return ["scripts", "tests"]


def _engine_repo(tmp_path: Path) -> Path:
    """A repo whose first commit already carries the checker, with one clean module."""
    repo = _init(tmp_path / "repo")
    _write(repo, {"pkg/a.py": "def a():\n    return 1\n"})
    staged = _add_engine(repo)
    _git(repo, "add", "--all", "--", ".gitignore", "pyproject.toml", "pkg", *staged)
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _check(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return _sh(repo, sys.executable, str(repo / "scripts/check"), *args)


def _ratchet_repo(tmp_path: Path, switch: str | None = None) -> tuple[Path, str]:
    repo = _init(tmp_path / "repo")
    (repo / "scripts/ci").mkdir(parents=True)
    shutil.copy2(REPO / "scripts/ci/profile_scope_patterns.json", repo / "scripts/ci/")
    files: dict[str, str | None] = {"pkg/a.py": _LEGACY, **({_SWITCH: switch} if switch else {})}
    _write(repo, files)
    _git(repo, "add", "--all", "--", ".gitignore", "pyproject.toml", "scripts", "pkg")
    _git(repo, "commit", "-q", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


# --- F20: the staged/commit health verdict comes from the judged artifact only -----------------


@pytest.mark.parametrize("unstaged", [
    # the policy file read by the PS-P05 rule, narrowed so it no longer covers pkg/
    ("scripts/ci/profile_scope_patterns.json", lambda t: t.replace(
        '"id": "P05",', '"id": "P05",\n      "path_regex": "^nowhere/",')),
    # the engine itself, edited to exempt pkg/ from PS-P05
    ("scripts/code_health/config.py", lambda t: t.replace(
        'exclude=SINGLE_PROFILE, pattern_id="P05"',
        'exclude=SINGLE_PROFILE + ("pkg/*",), pattern_id="P05"')),
])
def test_staged_health_ignores_unstaged_policy_and_engine(tmp_path, unstaged):
    repo = _engine_repo(tmp_path)
    _write(repo, {"pkg/c.py": _ENV_COPY})
    _git(repo, "add", "--", "pkg/c.py")
    tree = _git(repo, "write-tree")
    first = _check(repo, "--staged", "--only", "health", "--base", "HEAD")
    assert first.returncode == 1 and "PS-P05" in first.stdout, first.stdout + first.stderr

    rel, edit = unstaged
    original = (repo / rel).read_text(encoding="utf-8")
    edited = edit(original)
    assert edited != original
    (repo / rel).write_text(edited, encoding="utf-8")
    again = _check(repo, "--staged", "--only", "health", "--base", "HEAD")
    assert again.returncode == 1 and "PS-P05" in again.stdout, again.stdout + again.stderr
    # the judged artifact and the user's unstaged work are both untouched
    assert _git(repo, "write-tree") == tree
    assert (repo / rel).read_text(encoding="utf-8") == edited
    assert _git(repo, "diff", "--name-only") == rel


# --- F22: selectors are validated before anything runs ----------------------------------------


@pytest.mark.parametrize("selector", ["heath", ",", "", "heath,rof", "health,rof"])
def test_unknown_or_empty_selector_is_a_usage_error(selector):
    proc = _sh(REPO, sys.executable, str(REPO / "scripts/check"), "--only", selector)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "health" in proc.stderr and "shebangs" in proc.stderr  # lists the valid names
    assert "checks, ok" not in proc.stdout


def test_valid_selector_runs_its_check_and_keeps_its_status(tmp_path):
    repo = _engine_repo(tmp_path)
    _write(repo, {"pkg/c.py": _ENV_COPY})
    _git(repo, "add", "--", "pkg/c.py")
    proc = _check(repo, "--staged", "--only", "health,shebangs", "--base", "HEAD")
    assert proc.returncode == 1 and "2 checks, FAILED: health" in proc.stdout, proc.stdout


# --- JSON output on every return path ----------------------------------------------------------


@pytest.mark.parametrize("case, switch, files, code, findings", [
    ("no measured change", None, {"README.md": "docs\n"}, 0, 0),
    ("enforcement off", 'ENFORCEMENT = "off"\n', {"pkg/a.py": _GROWN}, 0, 0),
    ("findings", None, {"pkg/a.py": _GROWN}, 1, 1),
    ("clean measured diff", None, {"pkg/b.py": "def b():\n    return 2\n"}, 0, 0),
])
def test_json_stdout_is_json_on_every_path(tmp_path, capsys, case, switch, files, code, findings):
    repo, base = _ratchet_repo(tmp_path, switch)
    head = _commit(repo, files)
    assert cli.run(repo, base, head, as_json=True) == code, case
    out, err = capsys.readouterr()
    data = json.loads(out)
    assert isinstance(data, list) and len(data) == findings, (case, out)
    if case in ("no measured change", "enforcement off"):
        assert "code health:" in err  # the human explanation moves to stderr


# --- pre-push: the pushed candidate decides, never the checked-out branch ----------------------


def _pushable(tmp_path: Path) -> tuple[Path, Path]:
    """`main` has the guards but predates scripts/check, and is published: (repo, bare remote)."""
    repo = _init(tmp_path / "repo")
    _write(repo, {"pkg/a.py": "def a():\n    return 1\n"})
    staged = _add_engine(repo, checker=False)
    _git(repo, "add", "--all", "--", ".gitignore", "pyproject.toml", "pkg", *staged)
    _git(repo, "commit", "-q", "-m", "main before scripts/check")
    (repo / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-q", "origin", "main")
    _git(repo, "fetch", "-q", "origin")
    return repo, remote


def _branch_with_checker(repo: Path, name: str, extra: dict[str, str]) -> str:
    _git(repo, "checkout", "-q", "-b", name, "main")
    _write(repo, extra)
    _git(repo, "add", "--all", "--", *_add_engine(repo), *extra)
    _git(repo, "commit", "-q", "-m", name)
    return _git(repo, "rev-parse", "HEAD")


def _remote_ref(remote: Path, branch: str) -> str:
    return _git(remote, "for-each-ref", "--format=%(objectname)", f"refs/heads/{branch}")


def test_pre_push_judges_a_checker_branch_pushed_from_an_older_checkout(tmp_path):
    repo, remote = _pushable(tmp_path)
    _branch_with_checker(repo, "feature", {"pkg/c.py": _ENV_COPY})
    good = _branch_with_checker(repo, "clean", {"pkg/d.py": "def d():\n    return 4\n"})
    assert _check(repo, "--install-hook", "pre-push").returncode == 0
    _git(repo, "checkout", "-q", "main")
    assert not (repo / "scripts/check").exists()

    push = _sh(repo, "git", "push", "origin", "feature")
    assert push.returncode != 0, push.stdout + push.stderr
    assert "PS-P05" in push.stdout + push.stderr
    assert _remote_ref(remote, "feature") == ""  # destination untouched

    # controls: a clean checker branch passes, a branch predating the checker is not judged,
    # a deletion is not judged
    push = _sh(repo, "git", "push", "origin", "clean")
    assert push.returncode == 0 and _remote_ref(remote, "clean") == good, push.stderr
    _git(repo, "checkout", "-q", "-b", "old", "main")
    old = _commit(repo, {"pkg/c.py": _ENV_COPY})
    assert _sh(repo, "git", "push", "origin", "old").returncode == 0
    assert _remote_ref(remote, "old") == old
    assert _sh(repo, "git", "push", "origin", "--delete", "clean").returncode == 0
    assert _remote_ref(remote, "clean") == ""


# --- M5: interpreter choice and the version guard ---------------------------------------------


def test_hooks_prefer_the_repo_virtualenv_over_path_python(tmp_path):
    repo = _engine_repo(tmp_path)
    assert _check(repo, "--install-hook", "pre-commit").returncode == 0
    fake = tmp_path / "fakebin"
    fake.mkdir()
    for name in ("python3", "python"):
        (fake / name).write_text("#!/bin/sh\necho 'wrong interpreter' >&2\nexit 97\n",
                                 encoding="utf-8")
        (fake / name).chmod(0o755)
    env = _env()
    env["PATH"] = f"{fake}{os.pathsep}{env['PATH']}"
    (repo / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
    _write(repo, {"pkg/b.py": "def b():\n    return 2\n"})
    _git(repo, "add", "--", "pkg/b.py")
    proc = _sh(repo, "git", "commit", "-q", "-m", "venv", env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # control: without a virtualenv the hook falls back to PATH's python3
    (repo / ".venv").unlink()
    _write(repo, {"pkg/b.py": "def b():\n    return 3\n"})
    _git(repo, "add", "--", "pkg/b.py")
    proc = _sh(repo, "git", "commit", "-q", "-m", "path", env=env)
    assert proc.returncode != 0 and "wrong interpreter" in proc.stderr


def test_check_refuses_old_python_before_importing_anything():
    # The guard reads sys.version_info; a 3.10 interpreter is emulated by overriding it before
    # scripts/check runs (tomllib, imported via ruff_runner, would otherwise crash it).
    code = ("import runpy, sys; sys.version_info = (3, 10, 12, 'final', 0); "
            f"sys.argv = ['scripts/check', '--only', 'shebangs']; "
            f"runpy.run_path({str(REPO / 'scripts/check')!r}, run_name='__main__')")
    proc = _sh(REPO, sys.executable, "-c", code)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "3.11" in proc.stderr and "Traceback" not in proc.stderr


# --- m3: the ENFORCEMENT switch ---------------------------------------------------------------


@pytest.mark.parametrize("switch, code", [
    ('ENFORCEMENT: str = "off"\n', 0),
    ('ENFORCEMENT: str = "advisory"\n', 0),
    ('ENFORCEMENT = "warn"\n', 1),  # an unknown value is not a way to relax the check
    ('ENFORCEMENT = "off"  # until the burn-down lands\n', 0),
])
def test_enforcement_switch_parses_annotated_and_unknown_values(tmp_path, capsys, switch, code):
    repo, base = _ratchet_repo(tmp_path, switch)
    head = _commit(repo, {"pkg/a.py": _GROWN})
    assert cli.run(repo, base, head) == code, capsys.readouterr().out


def test_enforcement_reads_this_trees_own_switch():
    assert cli.enforcement(REPO, "HEAD") == ENFORCEMENT
    text = (REPO / _SWITCH).read_text(encoding="utf-8")
    assert cli.parse_switch(text) == ENFORCEMENT


def _branch_behind_main(tmp_path: Path, main_switch: str, branch_files: dict[str, str]) -> Path:
    repo, base = _ratchet_repo(tmp_path, 'ENFORCEMENT = "blocking"\n')
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, branch_files)
    _git(repo, "checkout", "-q", "main")
    tip = _commit(repo, {_SWITCH: main_switch, "NEWS.md": "main moved on\n"})
    _git(repo, "update-ref", "refs/remotes/origin/main", tip)
    _git(repo, "reset", "-q", "--hard", base)  # local main is stale; only origin/main moved
    _git(repo, "checkout", "-q", "feature")
    return repo


def test_local_runs_take_the_switch_from_the_main_tip(tmp_path, monkeypatch, capsys):
    repo = _branch_behind_main(tmp_path, 'ENFORCEMENT = "advisory"\n', {"pkg/a.py": _GROWN})
    monkeypatch.chdir(repo)
    assert cli.main([]) == 0, capsys.readouterr().out  # the flip on main reaches the branch
    head = _git(repo, "rev-parse", "HEAD")
    assert cli.main(["--head", head]) == 0
    out = capsys.readouterr().out
    assert "CC 23 > 22" in out and "advisory mode" in out


def test_a_branch_cannot_relax_its_own_switch(tmp_path, monkeypatch, capsys):
    grown_and_off = {"pkg/a.py": _GROWN, _SWITCH: 'ENFORCEMENT = "off"\n'}
    repo = _branch_behind_main(tmp_path, 'ENFORCEMENT = "blocking"\n', grown_and_off)
    monkeypatch.chdir(repo)
    assert cli.main([]) == 1, capsys.readouterr().out


# --- m4: the CI guards that ran outside the lint workflow --------------------------------------


@pytest.mark.parametrize("job, files", [
    ("case-collisions", {"pkg/Notes.md": "a\n", "pkg/NOTES.md": "b\n"}),
    ("lazy-deps", {"pkg/uses.py": "import tools.lazy_deps\n"}),
    ("profile-archives", {"pkg/export.tar.gz": "not really\n"}),
])
def test_ci_only_guards_run_in_scripts_check(tmp_path, job, files):
    repo = _engine_repo(tmp_path)
    clean = _check(repo, "--staged", "--only", job, "--base", "HEAD")
    assert clean.returncode == 0 and "1 checks, ok" in clean.stdout, clean.stdout + clean.stderr
    _write(repo, files)
    _git(repo, "add", "--", *files)
    proc = _check(repo, "--staged", "--only", job, "--base", "HEAD")
    assert proc.returncode == 1 and f"FAILED: {job}" in proc.stdout, proc.stdout + proc.stderr


# --- F21: profile regex rules see executable code, not prose -----------------------------------


_FUNC = "import os\n\n\ndef build(cmd):\n{}    return cmd\n"


@pytest.mark.parametrize("body, blocks", [
    ("    # Avoid env = os.environ.copy(); use the scoped builder.\n", False),
    ('    """Avoid env = os.environ.copy(); use the scoped builder."""\n', False),
    ('    """Builder.\n\n    Never env = os.environ.copy() here.\n    """\n', False),
    ("    # don't read os.getenv(\"DISCORD_TOKEN\") directly\n", False),
    ("    cmd = [cmd]  # not os.environ.get('SLACK_TOKEN')\n", False),
    ("    'inert example: env = os.environ.copy()'\n", False),
    # positive controls: the real operations, including a literal env-key argument
    ("    env = os.environ.copy()\n    cmd = (cmd, env)\n", True),
    ("    token = os.getenv(\"DISCORD_TOKEN\")\n    cmd = (cmd, token)\n", True),
    ("    env = dict(os.environ)  # copy it\n    cmd = (cmd, env)\n", True),
])
def test_profile_regex_rules_ignore_comments_and_docstrings(tmp_path, capsys, body, blocks):
    repo, base = _ratchet_repo(tmp_path)
    head = _commit(repo, {"pkg/c.py": _FUNC.format(body)})
    code = cli.run(repo, base, head)
    out = capsys.readouterr().out
    assert code == (1 if blocks else 0), out
    assert ("PS-P05" in out or "PS-P06" in out) == blocks, out


# --- allow comments may list several rules ----------------------------------------------------


@pytest.mark.parametrize("rules", ["BLE001 S110", "BLE001, S110", "BLE001,S110"])
def test_allow_comment_lists_rules_by_comma_or_space(tmp_path, capsys, rules):
    repo, base = _ratchet_repo(tmp_path)
    swallow = ("def other():\n    try:\n        pass\n"
               f"    except Exception:  # health: allow {rules} -- boundary\n        pass\n")
    head = _commit(repo, {"pkg/b.py": swallow})
    assert cli.run(repo, base, head) == 0, capsys.readouterr().out


# --- measurement tooling failures are errors, not findings ------------------------------------


def test_tooling_failure_exits_2_without_a_traceback(tmp_path, monkeypatch, capsys):
    repo, base = _ratchet_repo(tmp_path)
    lock = {"packages": {"node_modules/typescript": {"version": "0.0.0-not-published"}}}
    _commit(repo, {"package-lock.json": json.dumps(lock), "web/a.ts": "export const a = 1;\n"})
    if not shutil.which("npm") or not shutil.which("node"):
        pytest.skip("needs node + npm to reach the npm install path")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("npm_config_offline", "true")  # a real npm failure, without the network
    monkeypatch.chdir(repo)
    assert cli.main(["--base", base, "--head", "HEAD"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("code health:") and "npm" in err
