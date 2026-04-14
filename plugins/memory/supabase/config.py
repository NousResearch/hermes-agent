from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "::1"}

CONFIG_SUBDIR = "supabase_memory"
CONFIG_FILENAME = "config.json"


def _config_path(hermes_home: str | Path | None) -> Path | None:
    if not hermes_home:
        return None
    return Path(hermes_home).expanduser() / CONFIG_SUBDIR / CONFIG_FILENAME


def _normalize_value(value: str | None) -> str:
    return (value or "").strip()


def is_valid_supabase_url(value: str | None) -> bool:
    candidate = _normalize_value(value)
    if not candidate:
        return False

    parsed = urlparse(candidate)
    if not parsed.netloc:
        return False
    if parsed.scheme == "https":
        return True
    if parsed.scheme == "http":
        hostname = (parsed.hostname or "").strip().lower()
        return hostname in _LOCALHOST_HOSTS
    return False


def load_config(
    hermes_home: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    values: dict[str, str] = {}

    config_path = _config_path(hermes_home)
    if config_path and config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}

        if isinstance(loaded, dict):
            values.update(
                {
                    str(key): str(value)
                    for key, value in loaded.items()
                    if isinstance(key, str) and isinstance(value, str)
                }
            )

    source = os.environ if environ is None else environ
    env_url = _normalize_value(source.get("SUPABASE_URL"))
    env_key = _normalize_value(source.get("SUPABASE_SECRET_KEY"))

    if env_url:
        values["supabase_url"] = env_url
    if env_key:
        values["supabase_secret_key"] = env_key

    return values


def get_config_schema() -> list[dict[str, Any]]:
    return [
        {
            "key": "supabase_url",
            "description": "Supabase project URL for the PostgREST API.",
            "required": True,
            "env_var": "SUPABASE_URL",
        },
        {
            "key": "supabase_secret_key",
            "description": "Supabase service-role key used by the memory provider.",
            "required": True,
            "secret": True,
            "env_var": "SUPABASE_SECRET_KEY",
        },
    ]


def save_config(values: dict[str, Any], hermes_home: str | Path) -> None:
    config_path = _config_path(hermes_home)
    if config_path is None:
        return

    safe_values = {
        str(key): str(value)
        for key, value in values.items()
        if isinstance(key, str) and isinstance(value, (str, int, float, bool))
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(safe_values, indent=2, sort_keys=True), encoding="utf-8")
