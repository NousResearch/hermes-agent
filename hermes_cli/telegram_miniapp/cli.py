"""Manual foreground runner for the Telegram Mini App sidecar.

No launchd/autostart registration lives here; this is intentionally an operator
command helper. Public HTTPS smoke mode is opt-in per foreground run only.
"""

from __future__ import annotations

import argparse
import socket
import sys

from .config import settings_from_config
from .server import create_app


def _ensure_port_available(host: str, port: int) -> None:
    """Fail fast with an actionable message if the bind port is already taken.

    Done as an explicit pre-bind probe rather than relying on ``uvicorn.run`` to
    surface the error: some uvicorn versions log and ``sys.exit(1)`` on a bind
    failure before it ever reaches our handler. Uses the same SO_REUSEADDR
    semantics uvicorn does, so an actively-listened port is reported as busy.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
    except OSError as exc:
        raise SystemExit(
            f"Telegram Mini App sidecar could not bind {host}:{port} "
            f"({exc.strerror or exc}). Stop the process using that port or set "
            f"telegram_miniapp.port to a free port in config.yaml."
        ) from exc
    finally:
        probe.close()


def _detect_static_dir() -> str | None:
    """Locate the built SPA (apps/telegram-miniapp/dist) relative to the repo so
    the sidecar can serve it on the same origin as the API. None if not built."""
    from pathlib import Path

    dist = Path(__file__).resolve().parents[2] / "apps" / "telegram-miniapp" / "dist"
    return str(dist) if (dist / "index.html").is_file() else None


def run_foreground(*, https_smoke: bool = False, public_base_url: str | None = None, enable_actions: bool = False) -> None:
    settings = settings_from_config()
    if settings.static_dir is None:
        settings.static_dir = _detect_static_dir()
    if settings.static_dir:
        print(f"[miniapp] Serving built SPA from {settings.static_dir}", file=sys.stderr)
    else:
        print(
            "[miniapp] No built SPA found (apps/telegram-miniapp/dist). Serving API "
            "only; run `npm run build` in apps/telegram-miniapp to serve the UI too.",
            file=sys.stderr,
        )
    if https_smoke:
        if not public_base_url:
            raise SystemExit("--https-smoke requires --public-base-url https://<smoke-host>")
        settings.host = "127.0.0.1"
        settings.public_smoke = True
        settings.public_base_url = public_base_url
        settings.cors_allowed_origins = {public_base_url.rstrip("/")}
    elif settings.host != "127.0.0.1":
        raise SystemExit("Telegram Mini App sidecar is loopback-only unless --https-smoke is used")
    if enable_actions:
        # Owner-confirmed Phase 1 action gate. Only turns on with a per-run flag
        # AND a configured owner allowlist; public smoke and actions are
        # mutually exclusive for now.
        if settings.public_smoke:
            raise SystemExit("--enable-actions cannot be combined with --https-smoke")
        if not settings.action_owners:
            raise SystemExit("--enable-actions requires telegram_miniapp.action_owners in config.yaml")
        settings.enable_actions = True
        print(
            "[miniapp] Action gate ENABLED for this run: owners can submit signed "
            "approve/reject decisions. NOTE the gateway-side resolver is not yet wired "
            "(nothing reads telegram_miniapp.bridge_enabled), so decisions are validated "
            "and recorded but NOT applied to live approvals. See "
            "docs/telegram-miniapp-runbook.md.",
            file=sys.stderr,
        )
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Telegram Mini App sidecar requires uvicorn/FastAPI dashboard extras") from exc
    _ensure_port_available(settings.host, settings.port)
    try:
        uvicorn.run(create_app(settings=settings), host=settings.host, port=settings.port)
    except OSError as exc:
        # Fallback for the small TOCTOU window between the probe above and the
        # real bind — same actionable message rather than a raw traceback.
        raise SystemExit(
            f"Telegram Mini App sidecar could not bind {settings.host}:{settings.port} "
            f"({exc.strerror or exc}). Stop the process using that port or set "
            f"telegram_miniapp.port to a free port in config.yaml."
        ) from exc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Hermes Telegram Mini App sidecar in foreground")
    parser.add_argument("serve", nargs="?", default="serve", choices=("serve",))
    parser.add_argument("--https-smoke", action="store_true", help="Enable short-lived HTTPS smoke mode for this process only")
    parser.add_argument("--public-base-url", help="Origin-only HTTPS smoke URL, e.g. https://example.tunnel")
    parser.add_argument("--enable-actions", action="store_true", help="Enable the owner-confirmed Phase 1 approve/reject action gate for this process only")
    args = parser.parse_args(argv)
    run_foreground(https_smoke=args.https_smoke, public_base_url=args.public_base_url, enable_actions=args.enable_actions)


if __name__ == "__main__":
    main()
