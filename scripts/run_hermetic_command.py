#!/usr/bin/env python3
"""Run a non-pytest Hermes test command in the canonical Linux sandbox."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from run_tests_parallel import _resolve_real_home, _sandboxed_test_command


def _prepare_disposable_workspace(source: Path, destination: Path) -> Path:
    """Copy the workspace and give the copy isolated Git control files."""
    git_env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    git_env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    top_level = Path(
        subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
            text=True,
            env=git_env,
        ).strip()
    ).resolve()
    if top_level != source.resolve():
        raise RuntimeError("generic test workspace is not the resolved Git root")
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        text=True,
        env=git_env,
    ).strip()
    common_raw = subprocess.check_output(
        [
            "git",
            "-C",
            str(source),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        text=True,
        env=git_env,
    ).strip()
    common_dir = Path(common_raw).resolve()
    object_store = (common_dir / "objects").resolve()
    resolved_objects = Path(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "objects",
            ],
            text=True,
            env=git_env,
        ).strip()
    ).resolve()
    if resolved_objects != object_store or not object_store.is_dir():
        raise RuntimeError("generic test Git object store failed ownership validation")

    def ignore_sensitive(_directory: str, names: list[str]) -> set[str]:
        exact = {".git", ".venv", "venv", "__pycache__", ".npmrc", ".pypirc"}
        return {
            name
            for name in names
            if name in exact or name.startswith(".env")
        }

    shutil.copytree(
        source,
        destination,
        symlinks=True,
        ignore=ignore_sensitive,
    )
    disposable_git = destination / ".git"
    (disposable_git / "objects" / "info").mkdir(parents=True)
    (disposable_git / "hooks").mkdir()
    (disposable_git / "refs" / "heads").mkdir(parents=True)
    (disposable_git / "HEAD").write_text(head + "\n", encoding="ascii")
    (disposable_git / "config").write_text(
        "[core]\n"
        "\trepositoryformatversion = 0\n"
        "\tfilemode = true\n"
        "\tbare = false\n"
        "\tlogallrefupdates = false\n",
        encoding="ascii",
    )
    (disposable_git / "objects" / "info" / "alternates").write_text(
        str(object_store) + "\n", encoding="utf-8"
    )
    disposable_git_env = dict(git_env)
    disposable_git_env["GIT_INDEX_FILE"] = str(disposable_git / "index")
    subprocess.run(
        [
            "git",
            f"--git-dir={disposable_git}",
            f"--work-tree={destination}",
            "read-tree",
            head,
        ],
        check=True,
        env=disposable_git_env,
    )
    tracked = subprocess.check_output(
        ["git", "-C", str(source), "ls-tree", "-r", "-z", "--name-only", head],
        env=git_env,
    )
    policy_paths = []
    for raw_path in tracked.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(os.fsdecode(raw_path))
        if relative.name.startswith(".env") or relative.name in {".npmrc", ".pypirc"}:
            policy_paths.append(relative)
    for relative in policy_paths:
        result = subprocess.run(
            ["git", "-C", str(source), "show", f"{head}:{relative.as_posix()}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            env=git_env,
        )
        if result.returncode == 0:
            (destination / relative).parent.mkdir(parents=True, exist_ok=True)
            (destination / relative).write_bytes(result.stdout)
    return object_store


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", default=".")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    repo_root = Path(__file__).resolve().parent.parent
    working_directory = (repo_root / args.cwd).resolve()
    if working_directory != repo_root and repo_root not in working_directory.parents:
        parser.error("--cwd must stay inside the repository")
    real_home = _resolve_real_home(dict(os.environ))
    with tempfile.TemporaryDirectory(prefix="hermes-command-sandbox-") as raw_root:
        sandbox_root = Path(raw_root)
        home = sandbox_root / "home"
        home.mkdir()
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(home),
            "HERMES_HOME": str(home / ".hermes"),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "XDG_DATA_HOME": str(home / ".local/share"),
            "XDG_STATE_HOME": str(home / ".local/state"),
            "XDG_RUNTIME_DIR": str(sandbox_root / "run-user"),
            "TMPDIR": str(sandbox_root),
            "TMP": str(sandbox_root),
            "TEMP": str(sandbox_root),
            "CI": "true",
            "TZ": "UTC",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "HERMES_TEST_REAL_HOME": str(real_home),
            "HERMES_TEST_SANDBOX_ROOT": str(sandbox_root),
            "HERMES_TEST_REPO_ROOT": str(repo_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "RUSTUP_NO_UPDATE_CHECK": "1",
        }
        for name in ("GITHUB_SHA", "GITHUB_REF_NAME", "GITHUB_HEAD_REF"):
            value = os.environ.get(name, "")
            if value:
                env[name] = value
        (home / ".hermes").mkdir()
        (sandbox_root / "run-user").mkdir()
        cargo_home = os.environ.get("HERMES_TEST_CARGO_HOME", "").strip()
        writable_paths: tuple[Path, ...] = ()
        if cargo_home:
            cargo_path = Path(cargo_home).resolve()
            trusted_temp_raw = os.environ.get("RUNNER_TEMP", tempfile.gettempdir())
            trusted_temp = Path(trusted_temp_raw).resolve()
            if (
                not cargo_path.is_dir()
                or cargo_path == trusted_temp
                or not cargo_path.is_relative_to(trusted_temp)
            ):
                raise SystemExit(
                    "HERMES_TEST_CARGO_HOME must be a child of the trusted runner temp root"
                )
            env["CARGO_HOME"] = str(cargo_path)
            writable_paths = (cargo_path,)
        rustup_home = os.environ.get("HERMES_TEST_RUSTUP_HOME", "").strip()
        readonly_paths: list[Path] = []
        if rustup_home:
            rustup_path = Path(rustup_home).resolve()
            trusted_temp_raw = os.environ.get("RUNNER_TEMP", tempfile.gettempdir())
            trusted_temp = Path(trusted_temp_raw).resolve()
            if not rustup_path.is_dir() or not (
                rustup_path == real_home / ".rustup"
                or rustup_path.is_relative_to(trusted_temp)
            ):
                raise SystemExit("HERMES_TEST_RUSTUP_HOME is not an approved toolchain")
            env["RUSTUP_HOME"] = str(rustup_path)
            readonly_paths.append(rustup_path)
            cargo_executable = shutil.which("cargo")
            if cargo_executable:
                cargo_path = Path(cargo_executable).absolute()
                if cargo_path.is_relative_to(real_home):
                    approved_bin = real_home / ".cargo" / "bin"
                    if cargo_path.parent != approved_bin or cargo_path.name != "cargo":
                        raise SystemExit("cargo executable is in an unapproved home path")
                    readonly_paths.append(cargo_path)
        workspace = sandbox_root / "workspace"
        object_store = _prepare_disposable_workspace(repo_root, workspace)
        workspace_directory = workspace / working_directory.relative_to(repo_root)
        env["HERMES_TEST_REPO_ROOT"] = str(workspace)
        entry = workspace / "scripts" / "hermetic_command_entry.py"
        wrapped = _sandboxed_test_command(
            [sys.executable, str(entry), *command],
            env=env,
            repo_root=workspace,
            sandbox_root=sandbox_root,
            real_home=real_home,
            writable_repo=True,
            working_directory=workspace_directory,
            additional_writable_paths=writable_paths,
            additional_readonly_paths=(*readonly_paths, object_store),
        )
        return subprocess.run(wrapped, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
