#!/usr/bin/env python3
"""Start a KarinAI agent container in managed runtime mode.

This validates the runtime-manager handoff, maps KarinAI runtime settings onto
the upstream gateway/API-server environment, and then starts the existing
gateway entrypoint.
"""

from __future__ import annotations

import os
import sys

from .managed import (
    apply_managed_startup_env,
    load_managed_runtime_config,
    prepare_managed_runtime_filesystem,
    write_managed_gateway_config,
)


def _run_gateway_main() -> None:
    from gateway.run import main as gateway_main

    gateway_main()


def main() -> None:
    cfg = load_managed_runtime_config()
    prepare_managed_runtime_filesystem(cfg)
    write_managed_gateway_config(cfg)
    apply_managed_startup_env(cfg)
    os.chdir(cfg.workspace_path)
    _run_gateway_main()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - exercised through container startup
        print(f"KarinAI managed runtime failed to start: {exc}", file=sys.stderr)
        raise
