"""Foreground runner for the profile-scoped Telegram Mini App."""

from __future__ import annotations

import os
import json
from pathlib import Path


_DEDICATED_ENV_KEYS = {
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_MINI_APP_OWNER_IDS",
    "TELEGRAM_MINI_APP_PUBLIC_URL",
}
_RUNTIME_ENV_KEYS = {
    "HOME",
    "HERMES_HOME",
    "LANG",
    "LOGNAME",
    "PATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TMPDIR",
    "TZ",
    "USER",
    "VIRTUAL_ENV",
}


def _sanitize_environment() -> Path:
    """Drop inherited credentials before any Hermes or ASGI module is imported."""
    preserved = {
        key: value
        for key, value in os.environ.items()
        if key in _RUNTIME_ENV_KEYS or key.startswith("LC_")
    }
    home_raw = preserved.get("HERMES_HOME", "").strip()
    if not home_raw:
        raise RuntimeError("HERMES_HOME is required by the Mini App runner.")
    os.environ.clear()
    os.environ.update(preserved)
    return Path(home_raw).expanduser().resolve()


def _load_dedicated_env(path: Path) -> dict[str, str]:
    """Load only the two Mini App credentials, without sourcing a shell file."""
    if not path.is_file():
        raise RuntimeError(
            "Mini App credentials are missing; run `hermes gateway mini-app setup`."
        )
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RuntimeError(
                f"Malformed Mini App credential file at line {line_number}."
            )
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in _DEDICATED_ENV_KEYS:
            raise RuntimeError(f"Unexpected key in Mini App credential file: {key}")
        values[key] = value.strip()
    missing = _DEDICATED_ENV_KEYS - values.keys()
    if missing:
        raise RuntimeError(
            f"Mini App credential file is missing: {', '.join(sorted(missing))}"
        )
    for key, value in values.items():
        os.environ[key] = value
    return values


def _listen_port(state: dict) -> int:
    try:
        value = state["listen_port"]
        port = int(value)
    except (KeyError, TypeError, ValueError):
        port = 8787
    if not 1 <= port <= 65535:
        raise RuntimeError("Telegram Mini App listen_port must be between 1 and 65535.")
    return port


def serve() -> None:
    """Run the Mini App on the fixed loopback listener."""
    home = _sanitize_environment()
    root = home / "telegram-mini-app"
    _load_dedicated_env(root / "service.env")
    try:
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Mini App state is missing or invalid; run setup again."
        ) from exc
    port = _listen_port(state)

    import uvicorn

    uvicorn.run(
        "plugins.platforms.telegram.mini_app.app:app",
        host="127.0.0.1",
        port=port,
        access_log=False,
        limit_concurrency=32,
        timeout_keep_alive=5,
        proxy_headers=False,
        forwarded_allow_ips="",
    )


if __name__ == "__main__":
    serve()
