"""Credential proxy daemon entry point.

Spawned by ``hermes cred-proxy start`` as a detached subprocess.
Writes its PID to the given pid-file, starts the asyncio proxy server,
and waits for SIGTERM.

Usage::

    python -m proxy.daemon --socket /path/to/cred-proxy.sock --pid-file /path/to/cred-proxy.pid
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [cred-proxy] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes credential proxy daemon")
    parser.add_argument("--socket", required=True, help="Unix socket path")
    parser.add_argument("--pid-file", required=True, help="PID file path")
    args = parser.parse_args()

    socket_path = Path(args.socket)
    pid_path = Path(args.pid_file)

    # Write PID
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))

    # Load config-declared proxy credentials into env passthrough
    try:
        from proxy.config import get_proxy_credentials_list
        from tools.env_passthrough import register_env_passthrough

        creds = get_proxy_credentials_list()
        if creds:
            register_env_passthrough(creds)
            logger.info("registered %d proxy credential vars for passthrough: %s", len(creds), creds)
    except Exception as exc:
        logger.warning("failed to register proxy credentials for passthrough: %s", exc)

    # Create store and start server
    from proxy.store import CredentialStore
    from proxy.server import run_proxy

    store = CredentialStore()

    # Register store with secrets tool
    try:
        from tools.secrets_tool import set_store
        set_store(store)
    except Exception as exc:
        logger.warning("failed to register store with secrets tool: %s", exc)

    logger.info("starting credential proxy daemon (PID %d)", os.getpid())

    try:
        asyncio.run(run_proxy(socket_path, store))
    except KeyboardInterrupt:
        pass
    finally:
        pid_path.unlink(missing_ok=True)
        if socket_path.exists():
            socket_path.unlink(missing_ok=True)
        logger.info("credential proxy daemon exited")


if __name__ == "__main__":
    main()
