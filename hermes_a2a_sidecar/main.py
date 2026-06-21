from __future__ import annotations

import argparse
import sys

from .config import load_sidecar_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-a2a",
        description="Run the Hermes Agent2Agent (A2A) sidecar.",
    )
    parser.add_argument("--host", help="Bind host (default: a2a.host or 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Bind port (default: a2a.port or 8765)")
    parser.add_argument("--public-url", help="Public base URL advertised in the Agent Card")
    parser.add_argument("--rpc-path", help="A2A JSON-RPC path (default: /a2a)")
    parser.add_argument(
        "--allow-insecure-local",
        action="store_true",
        help="Development only: allow unauthenticated localhost JSON-RPC calls.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = {
        "host": args.host,
        "port": args.port,
        "public_url": args.public_url,
        "rpc_path": args.rpc_path,
    }
    if args.allow_insecure_local:
        overrides["allow_insecure_local"] = True
    config = load_sidecar_config(overrides)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exercised in packaging environments
        raise SystemExit(
            "uvicorn is required to run hermes-a2a. Install with `hermes-agent[a2a]` or `uv pip install a2a-sdk[http-server] uvicorn`."
        ) from exc
    from .app import create_app

    uvicorn.run(create_app(config), host=config.host, port=config.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
