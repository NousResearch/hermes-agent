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
        entry = repo_root / "scripts" / "hermetic_command_entry.py"
        wrapped = _sandboxed_test_command(
            [sys.executable, str(entry), *command],
            env=env,
            repo_root=repo_root,
            sandbox_root=sandbox_root,
            real_home=real_home,
            writable_repo=True,
            working_directory=working_directory,
            additional_writable_paths=writable_paths,
            additional_readonly_paths=tuple(readonly_paths),
        )
        return subprocess.run(wrapped, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
