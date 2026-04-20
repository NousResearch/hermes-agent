"""``hermes office`` subcommand entry point.

Boots the FastAPI server bound to a loopback port and (by default) opens the
browser. Designed to be safe to call multiple times: if the port is taken it
auto-picks a free one.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional


DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"


def _free_port(host: str, preferred: int) -> int:
    """Return the preferred port if free, otherwise an OS-assigned one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _ensure_built() -> Optional[Path]:
    """Locate the built frontend; return its directory or None."""
    here = Path(__file__).resolve().parent
    dist = here / "frontend" / "dist"
    if (dist / "index.html").exists():
        return dist
    return None


def _hint_build() -> None:
    print(
        "[hermes office] frontend not built — REST API will run but the UI "
        "will show a setup card.\n"
        "                run: cd hermes_office/frontend && npm install && npm run build",
        file=sys.stderr,
    )


def add_subparser(subparsers: "argparse._SubParsersAction") -> argparse.ArgumentParser:
    """Register the ``hermes office`` parser. Called from hermes_cli/main.py."""
    p = subparsers.add_parser(
        "office",
        help="Launch the Hermes Digital Office web UI",
        description=(
            "Starts a local FastAPI server (default 127.0.0.1:8765) that serves "
            "a game-like office where you can hire, edit, and watch digital "
            "employees in real time."
        ),
    )
    p.add_argument("--host", default=DEFAULT_HOST, help="Bind host (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Preferred port (default 8765)")
    p.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser")
    p.add_argument(
        "--runtime",
        choices=("simulated", "hermes"),
        default="simulated",
        help="Default runtime for new employees (simulated=safe demo, hermes=live LLM)",
    )
    p.add_argument(
        "--log-level",
        default="info",
        choices=("debug", "info", "warning", "error"),
    )
    p.set_defaults(func=cmd_office)
    return p


def cmd_office(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # type: ignore
    except ImportError:
        print(
            "[hermes office] missing dependency 'uvicorn'. Install with:\n"
            "    pip install -e .[office]",
            file=sys.stderr,
        )
        return 2

    try:
        from hermes_office.server import build_app
    except Exception as exc:
        print(f"[hermes office] cannot import server: {exc!r}", file=sys.stderr)
        return 2

    if _ensure_built() is None:
        _hint_build()

    port = _free_port(args.host, args.port)
    url = f"http://{args.host}:{port}"

    app = build_app(runtime_default=args.runtime)

    if not args.no_browser:
        try:
            webbrowser.open_new_tab(url)
        except Exception:
            pass

    print(f"[hermes office] serving {url}  (runtime={args.runtime})")
    print("[hermes office] press Ctrl-C to stop.")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=port,
            log_level=args.log_level,
            access_log=False,
        )
    except KeyboardInterrupt:
        print("\n[hermes office] stopped.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point for ``python -m hermes_office``.

    Accepts either the explicit subcommand form (``python -m hermes_office
    office --port 8765``) for parity with ``hermes office``, or the bare flag
    form (``python -m hermes_office --port 8765``) for ergonomics.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or (raw[0].startswith("-") and raw[0] not in ("-h", "--help")):
        raw = ["office", *raw]

    parser = argparse.ArgumentParser(prog="hermes-office")
    sub = parser.add_subparsers(dest="command")
    add_subparser(sub)
    args = parser.parse_args(raw)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args) or 0


if __name__ == "__main__":  # pragma: no cover - manual entry
    raise SystemExit(main())
