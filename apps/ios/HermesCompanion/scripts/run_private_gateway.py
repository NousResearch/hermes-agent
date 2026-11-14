#!/usr/bin/env python3
"""Run the shared, authenticated Hermes HTTP/WS backend on loopback only."""
import argparse
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-source", type=Path, required=True)
    parser.add_argument("--hermes-home", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9119)
    args = parser.parse_args()
    source = args.hermes_source.resolve()
    home = args.hermes_home.resolve()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    os.environ["HERMES_HOME"] = str(home)
    sys.path.insert(0, str(source))
    from hermes_cli.config import load_config_readonly

    dashboard = load_config_readonly().get("dashboard", {})
    origin = urlsplit(dashboard.get("public_url", ""))
    auth = dashboard.get("basic_auth", {})
    if not (
        origin.scheme == "https" and (origin.hostname or "").endswith(".ts.net")
        and not origin.username and not origin.password
        and origin.path in ("", "/") and not origin.query and not origin.fragment
        and all(auth.get(key) for key in ("username", "password_hash", "secret"))
    ):
        raise SystemExit("Refusing startup: private HTTPS origin and password authentication required.")
    env = os.environ.copy()
    for key in ("HERMES_DESKTOP", "HERMES_WEB_DIST", "HERMES_DESKTOP_DEV_SERVER"):
        env.pop(key, None)
    os.chdir(source)
    os.execve(sys.executable, [sys.executable, "-m", "hermes_cli.main", "-p", "default",
                              "serve", "--host", "127.0.0.1", "--port", str(args.port)], env)


if __name__ == "__main__":
    main()
