#!/usr/bin/env python3
"""Start the profile-local LiteLLM proxy without evaluating .env as shell code."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from dotenv import dotenv_values


def profile_home() -> Path:
    configured = os.environ.get("HERMES_HOME")
    if configured:
        return Path(configured).expanduser()
    result = subprocess.run(
        ["hermes", "config", "path"], check=True, capture_output=True, text=True
    )
    return Path(result.stdout.strip()).expanduser().parent


def main() -> None:
    home = profile_home()
    root = home / "integrations" / "litellm"
    executable = root / ".venv" / "bin" / "litellm"
    config = root / "config.yaml"
    if not executable.is_file() or not config.is_file():
        raise SystemExit("LiteLLM executable or config is missing; run install and copy the template")

    values = dotenv_values(home / ".env")
    required = {
        name: values.get(name)
        for name in (
            "LITELLM_MASTER_KEY",
            "LITELLM_UPSTREAM_API_KEY",
            "LITELLM_UPSTREAM_BASE_URL",
        )
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"Missing LiteLLM secrets: {', '.join(missing)}")

    port = os.environ.get("LITELLM_PORT", "4000")
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise SystemExit("LITELLM_PORT must be an integer from 1 to 65535")

    environment = os.environ.copy()
    environment.update({name: str(value) for name, value in required.items()})
    arguments = [
        str(executable),
        "--config",
        str(config),
        "--host",
        "127.0.0.1",
        "--port",
        port,
    ]
    os.execve(str(executable), arguments, environment)


if __name__ == "__main__":
    main()
