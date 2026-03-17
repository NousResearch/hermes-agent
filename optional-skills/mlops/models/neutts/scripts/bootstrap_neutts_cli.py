#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
BUNDLED_CLI_DIR = SKILL_DIR / "assets" / "neutts-cli"


def _quote(path: Path) -> str:
    return shlex.quote(str(path))


def _quote_text(value: str) -> str:
    return shlex.quote(value)


def find_cli_dir() -> tuple[Path, str]:
    if BUNDLED_CLI_DIR.exists():
        return BUNDLED_CLI_DIR, "bundled"

    raise FileNotFoundError(
        "NeuTTS CLI scaffold not found in bundled skill assets."
    )


def build_commands(
    cli_dir: Path,
    install_cli: bool,
    sample_profile: bool,
    python_executable: str,
) -> list[str]:
    commands: list[str] = []
    module_runner = f"{_quote_text(python_executable)} -m neutts_cli.cli"
    if install_cli:
        commands.append(
            f"{_quote_text(python_executable)} -m pip install --no-build-isolation -e {_quote(cli_dir)}"
        )
        commands.append(f"{module_runner} doctor")
    else:
        commands.append("neutts doctor")
    if sample_profile:
        sample_audio = cli_dir / "samples" / "jo.wav"
        sample_text = cli_dir / "samples" / "jo.txt"
        if not sample_audio.exists() or not sample_text.exists():
            raise FileNotFoundError(
                "Sample profile files are missing from bundled skill assets."
            )
        commands.append(
            " ".join(
                [
                    f"{module_runner if install_cli else 'neutts'} add-voice jo-demo",
                    f"--ref-audio {_quote(sample_audio)}",
                    f"--ref-text-file {_quote(sample_text)}",
                    "--language en",
                ]
            )
        )
    return commands


def maybe_run(commands: list[str], workdir: Path, execute: bool) -> list[dict]:
    results: list[dict] = []
    for command in commands:
        if not execute:
            results.append({"command": command, "executed": False})
            continue
        completed = subprocess.run(
            shlex.split(command),
            cwd=str(workdir),
            text=True,
            capture_output=True,
            check=False,
        )
        results.append(
            {
                "command": command,
                "executed": True,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
        if completed.returncode != 0:
            break
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap the standalone NeuTTS CLI for Hermes skill usage"
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Working directory used when executing bootstrap commands",
    )
    parser.add_argument(
        "--install-cli",
        action="store_true",
        help="Install the standalone NeuTTS CLI in editable mode",
    )
    parser.add_argument(
        "--sample-profile",
        action="store_true",
        help="Add the bundled jo-demo sample profile",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Actually run the generated commands"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON output"
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    cli_dir, cli_source = find_cli_dir()
    commands = build_commands(
        cli_dir, args.install_cli, args.sample_profile, sys.executable
    )
    workdir = repo_root if repo_root.exists() else Path.cwd()
    results = maybe_run(commands, workdir, args.execute)

    payload = {
        "python_executable": sys.executable,
        "repo_root": str(repo_root),
        "workdir": str(workdir),
        "cli_dir": str(cli_dir),
        "cli_source": cli_source,
        "commands": commands,
        "results": results,
        "next_steps": [
            "Re-run with '--execute' to actually perform the bootstrap commands.",
            f"Run '{sys.executable} -m neutts_cli.cli install --all' to install the upstream NeuTTS runtime.",
            f"Run '{sys.executable} -m neutts_cli.cli list-voices' to confirm saved profiles.",
            f"Run '{sys.executable} -m neutts_cli.cli synth --voice jo-demo --text Hello from Hermes' for a smoke test.",
        ],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Repo root: {repo_root}")
        print(f"Workdir: {workdir}")
        print(f"CLI dir: {cli_dir}")
        print(f"CLI source: {cli_source}")
        for entry in results:
            print(f"- {entry['command']}")
            if entry.get("executed"):
                print(f"  rc={entry['returncode']}")
                if entry.get("stdout"):
                    print(f"  stdout: {entry['stdout']}")
                if entry.get("stderr"):
                    print(f"  stderr: {entry['stderr']}")
        for step in payload["next_steps"]:
            print(f"next: {step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
