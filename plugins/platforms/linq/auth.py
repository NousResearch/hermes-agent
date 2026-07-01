"""
Linq credential storage.

The Linq integration needs only two things: an API bearer token and the
account's "from" phone number (E.164).  Both can be supplied via environment
variables (``LINQ_API_TOKEN`` / ``LINQ_FROM_PHONE``) or persisted to
``~/.hermes/auth.json`` under ``credential_pool.linq`` by ``hermes linq
setup`` — the same file and shape Photon uses for its credentials, so the
Hermes profile machinery (``hermes_constants.get_hermes_home``) keeps working.

Precedence everywhere is: **env var → auth.json**.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

ENV_TOKEN = "LINQ_API_TOKEN"
ENV_FROM_PHONE = "LINQ_FROM_PHONE"


def _auth_json_path() -> Path:
    """Resolve ``~/.hermes/auth.json`` honouring the active Hermes profile."""
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return Path(get_hermes_home()) / "auth.json"
    except Exception:
        return Path(os.path.expanduser("~/.hermes")) / "auth.json"


def _load_auth() -> Dict[str, Any]:
    path = _auth_json_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("linq: could not read %s: %s", path, exc)
        return {}


def _save_auth(data: Dict[str, Any]) -> None:
    path = _auth_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def _stored_record() -> Dict[str, Any]:
    pool = _load_auth().get("credential_pool", {}).get("linq") or []
    if isinstance(pool, list) and pool and isinstance(pool[0], dict):
        return pool[0]
    return {}


def load_token() -> Optional[str]:
    """Return the Linq API token (env wins, then auth.json)."""
    env = os.getenv(ENV_TOKEN)
    if env and env.strip():
        return env.strip()
    token = _stored_record().get("api_token") or _stored_record().get("access_token")
    return str(token) if token else None


def load_from_phone() -> Optional[str]:
    """Return the configured Linq "from" phone number (env wins, then auth.json)."""
    env = os.getenv(ENV_FROM_PHONE)
    if env and env.strip():
        return env.strip()
    phone = _stored_record().get("from_phone")
    return str(phone) if phone else None


def load_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Return ``(api_token, from_phone)``."""
    return load_token(), load_from_phone()


def store_credentials(api_token: str, from_phone: Optional[str] = None) -> None:
    """Persist the Linq token (+ optional from-phone) under ``credential_pool.linq``."""
    auth = _load_auth()
    record: Dict[str, Any] = {
        "api_token": api_token,
        "issued_at": int(time.time()),
    }
    if from_phone:
        record["from_phone"] = from_phone
    auth.setdefault("credential_pool", {})["linq"] = [record]
    _save_auth(auth)


def print_credential_summary(emit: Callable[[str], None]) -> None:
    """Render a credential-state table via *emit* (typically ``print``).

    Mirrors Photon's pattern of confining credential-derived strings to a
    single sink so the CLI module stays free of token taint.
    """
    token, phone = load_credentials()
    token_src = "env" if os.getenv(ENV_TOKEN) else ("auth.json" if token else None)
    phone_src = "env" if os.getenv(ENV_FROM_PHONE) else ("auth.json" if phone else None)
    emit("Linq iMessage credentials:")
    emit(f"  api token           : {'✓ set (' + token_src + ')' if token else '✗ missing — run `hermes linq setup`'}")
    emit(f"  from phone          : {phone + ' (' + phone_src + ')' if phone else '✗ not set (optional)'}")
    emit(f"  auth file           : {_auth_json_path()}")
