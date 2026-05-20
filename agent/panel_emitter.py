"""panel ingest emitter — thin HMAC-signing client for /api/units/ingest.

env gates:
  PANEL_EMIT_ENABLED=1      — required, otherwise emit() silently returns disabled
  PANEL_PROFILE=hermes:base — default profile (slug derived: lowercase + ':' → '-')
  PANEL_INGEST_URL          — override ingest endpoint

secrets:
  ~/.secrets/panel-emit-<slug>.txt   ingest secret
  ~/.secrets/panel-emit-<slug>.key   site key (pk_xxx)

fail-open: any exception → {"ok": False, "reason": "..."} logged to stderr,
never raised. emitter MUST NOT break the host agent's primary task.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_INGEST_URL = "https://panel.goku.codes/api/units/ingest"
DEFAULT_PROFILE = "hermes:base"
DEFAULT_TIMEOUT = 5.0
DEDUP_WINDOW_SECONDS = 30.0
SECRETS_DIR = Path.home() / ".secrets"

_dedup_lock = threading.Lock()
_dedup_cache: dict[str, float] = {}


def _slug(profile: str) -> str:
    return profile.lower().replace(":", "-")


def _resolve_profile(profile: str | None) -> str:
    return profile or os.environ.get("PANEL_PROFILE") or DEFAULT_PROFILE


def _load_credentials(profile: str) -> tuple[str, str]:
    slug = _slug(profile)
    key_path = SECRETS_DIR / f"panel-emit-{slug}.key"
    secret_path = SECRETS_DIR / f"panel-emit-{slug}.txt"
    site_key = key_path.read_text().strip()
    secret = secret_path.read_text().strip()
    if not site_key or not secret:
        raise RuntimeError(f"empty credentials for profile {profile}")
    return site_key, secret


def _sign(secret: str, body_bytes: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()


def _filter_dedup(units: list[dict]) -> list[dict]:
    """Drop units whose external_ref was emitted within DEDUP_WINDOW_SECONDS."""
    now = time.monotonic()
    kept: list[dict] = []
    with _dedup_lock:
        # gc expired
        expired = [k for k, t in _dedup_cache.items() if now - t > DEDUP_WINDOW_SECONDS]
        for k in expired:
            _dedup_cache.pop(k, None)
        for u in units:
            ref = u.get("external_ref")
            if ref and ref in _dedup_cache:
                continue
            kept.append(u)
            if ref:
                _dedup_cache[ref] = now
    return kept


def _clear_dedup_cache() -> None:
    """test hook."""
    with _dedup_lock:
        _dedup_cache.clear()


def emit(units: list[dict], profile: str | None = None) -> dict:
    """Sign and POST a batch of units to panel ingest.

    Returns parsed JSON on success, or {'ok': False, 'reason': str} on
    any failure / when the env gate is off. Never raises.
    """
    if os.environ.get("PANEL_EMIT_ENABLED") != "1":
        return {"ok": False, "reason": "disabled"}

    if not units:
        return {"ok": False, "reason": "empty"}

    try:
        units = _filter_dedup(list(units))
        if not units:
            return {"ok": False, "reason": "dedup"}

        prof = _resolve_profile(profile)
        site_key, secret = _load_credentials(prof)

        # tag source_agent on every unit (idempotent — don't clobber if caller set it)
        for u in units:
            u.setdefault("source_agent", prof)

        body_bytes = json.dumps(units, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        sig = _sign(secret, body_bytes)

        url = os.environ.get("PANEL_INGEST_URL", DEFAULT_INGEST_URL)
        req = urllib.request.Request(
            url,
            data=body_bytes,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Panel-Site-Key": site_key,
                "X-Panel-Ingest-Sig": sig,
            },
        )
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"ok": False, "reason": "non_json_response", "body": raw[:500]}
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        print(f"[panel_emitter] HTTPError {e.code}: {body}", file=sys.stderr)
        return {"ok": False, "reason": f"http_{e.code}", "body": body}
    except Exception as e:  # noqa: BLE001 — fail open by contract
        print(f"[panel_emitter] {type(e).__name__}: {e}", file=sys.stderr)
        return {"ok": False, "reason": str(e)}
