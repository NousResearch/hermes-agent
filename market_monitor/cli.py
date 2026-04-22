from __future__ import annotations

import argparse
from pathlib import Path
import threading

from market_monitor.api import start_api_server
from market_monitor.dashboard import build_dashboard_bundle, start_dashboard_server
from market_monitor.db import Database, initialize_database


LOOPBACK_HOSTS = {"127.0.0.1", "localhost"}


def build_dashboard(*, db_path: Path | str, out_dir: Path | str) -> dict[str, Path]:
    db_path = Path(db_path)
    out_dir = Path(out_dir)
    initialize_database(db_path)
    db = Database(db_path)
    return build_dashboard_bundle(db=db, out_dir=out_dir)


def start_local_servers(
    *,
    db_path: Path | str,
    out_dir: Path | str,
    api_host: str = "127.0.0.1",
    api_port: int = 8780,
    dashboard_host: str = "127.0.0.1",
    dashboard_port: int = 8765,
    allow_remote: bool = False,
) -> dict[str, object]:
    _ensure_safe_host(api_host, allow_remote=allow_remote)
    _ensure_safe_host(dashboard_host, allow_remote=allow_remote)
    db_path = Path(db_path)
    out_dir = Path(out_dir)
    initialize_database(db_path)
    db = Database(db_path)
    bundle = build_dashboard_bundle(db=db, out_dir=out_dir)
    api_server = start_api_server(db=db, host=api_host, port=api_port)
    dashboard_server = start_dashboard_server(bundle_dir=out_dir, host=dashboard_host, port=dashboard_port)
    return {"bundle": bundle, "api_server": api_server, "dashboard_server": dashboard_server}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="China EV market monitor local tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-dashboard", help="Build the static dashboard bundle")
    build_parser.add_argument("--db-path", required=True)
    build_parser.add_argument("--out-dir", required=True)

    serve_parser = subparsers.add_parser("serve-local", help="Serve both local API and dashboard")
    serve_parser.add_argument("--db-path", required=True)
    serve_parser.add_argument("--out-dir", required=True)
    serve_parser.add_argument("--api-host", default="127.0.0.1")
    serve_parser.add_argument("--api-port", type=int, default=8780)
    serve_parser.add_argument("--dashboard-host", default="127.0.0.1")
    serve_parser.add_argument("--dashboard-port", type=int, default=8765)
    serve_parser.add_argument("--allow-remote", action="store_true", help="Allow binding to non-loopback hosts")

    args = parser.parse_args(argv)
    if args.command == "build-dashboard":
        build_dashboard(db_path=args.db_path, out_dir=args.out_dir)
        return 0

    servers = start_local_servers(
        db_path=args.db_path,
        out_dir=args.out_dir,
        api_host=args.api_host,
        api_port=args.api_port,
        dashboard_host=args.dashboard_host,
        dashboard_port=args.dashboard_port,
        allow_remote=args.allow_remote,
    )
    api_server = servers["api_server"]
    dashboard_server = servers["dashboard_server"]
    print(f"API: http://{args.api_host}:{api_server.server_port}")
    print(f"Dashboard: http://{args.dashboard_host}:{dashboard_server.server_port}/index.html?api=http://{args.api_host}:{api_server.server_port}")
    dashboard_thread = threading.Thread(target=dashboard_server.serve_forever, daemon=True)
    dashboard_thread.start()
    try:
        api_server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        api_server.shutdown()
        dashboard_server.shutdown()
        api_server.server_close()
        dashboard_server.server_close()
        dashboard_thread.join(timeout=2)
    return 0


def _ensure_safe_host(host: str, *, allow_remote: bool) -> None:
    if allow_remote or host in LOOPBACK_HOSTS:
        return
    raise ValueError(f"Refusing to bind local monitor to non-loopback host: {host}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
