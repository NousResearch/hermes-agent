"""Manual foreground runner for the Telegram Mini App sidecar.

No launchd/autostart registration lives here; this is intentionally a local
operator command helper for M2 development only.
"""

from __future__ import annotations

from .config import settings_from_config
from .server import create_app


def run_foreground() -> None:
    settings = settings_from_config()
    if settings.host != "127.0.0.1":
        raise SystemExit("Telegram Mini App M2 sidecar is loopback-only; host must be 127.0.0.1")
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Telegram Mini App sidecar requires uvicorn/FastAPI dashboard extras") from exc
    uvicorn.run(create_app(settings=settings), host=settings.host, port=settings.port)
