#!/usr/bin/env python3
"""Validate and test the plugin in a disposable clone of a Hermes checkout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-root", type=Path, required=True)
    parser.add_argument(
        "--keep", type=Path, help="Keep the prepared clone here for inspection"
    )
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    plugin = (
        package
        if (package / "plugin.yaml").exists()
        else package / "plugins/camel-guard"
    )
    temporary = (
        None if args.keep else tempfile.TemporaryDirectory(prefix="camel-check-")
    )
    root = args.keep or Path(temporary.name) / "hermes"
    root = root.resolve()
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--shared",
                "--no-hardlinks",
                str(args.hermes_root.resolve()),
                str(root),
            ],
            check=True,
            timeout=120,
        )
        destination = root / "plugins/camel-guard"
        if destination.exists():
            shutil.rmtree(destination)  # disposable clone only
        destination.mkdir(parents=True)
        for filename in ("__init__.py", "policy.py", "plugin.yaml", "README.md"):
            shutil.copy2(plugin / filename, destination / filename)
        for filename in (
            "test_camel_guard_plugin.py",
            "test_camel_guard_information_flow.py",
        ):
            shutil.copy2(
                package / "tests/plugins" / filename, root / "tests/plugins" / filename
            )
        env = {**os.environ, "CAMEL_GUARD_PLUGIN_DIR": str(plugin)}
        commands = [
            [
                "bash",
                "scripts/run-in-hermes-env",
                "python",
                "-m",
                "hermes_cli.main",
                "plugins",
                "validate",
                str(plugin),
                "--install-deps",
            ],
            [
                "bash",
                "scripts/run_tests.sh",
                "tests/plugins/test_camel_guard_plugin.py",
                "tests/plugins/test_camel_guard_information_flow.py",
            ],
            [
                "bash",
                "scripts/run-in-hermes-env",
                "python",
                str(package / "scripts/camel_guard_offline_benchmark.py"),
                "--hermes-root",
                str(root),
            ],
        ]
        for command in commands:
            subprocess.run(command, cwd=root, env=env, check=True, timeout=1800)
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
