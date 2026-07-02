"""Manual foreground runner for the Telegram Mini App sidecar.

No launchd/autostart registration lives here; this is intentionally an operator
command helper. Public HTTPS smoke mode is opt-in per foreground run only.
"""

from __future__ import annotations

import argparse
import sys

from .config import settings_from_config
from .server import create_app


def run_foreground(*, https_smoke: bool = False, public_base_url: str | None = None, enable_actions: bool = False) -> None:
    settings = settings_from_config()
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
            "[miniapp] Action gate ENABLED for this run. The gateway must be running the "
            "Mini App bridge (config telegram_miniapp.bridge_enabled + a gateway restart) to "
            "export approvals and process decisions; otherwise /api/approvals falls back to "
            "preview and decisions stay unprocessed.",
            file=sys.stderr,
        )
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Telegram Mini App sidecar requires uvicorn/FastAPI dashboard extras") from exc
    uvicorn.run(create_app(settings=settings), host=settings.host, port=settings.port)


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
