"""Provider error-window tracking and Home Assistant problem sensors.

Records recent provider request outcomes and publishes one independent
Home Assistant binary_sensor per provider.  Sensors use device_class=problem
and turn on when more than half of the last eight observed requests for that
provider ended in an error.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

WINDOW_SIZE = 8
ERROR_THRESHOLD = WINDOW_SIZE // 2
STATE_RELATIVE_PATH = Path("state/provider_problem_sensors.json")

_thread_lock = threading.Lock()


@contextmanager
def _locked_file(path: Path) -> Iterator[None]:
    """Best-effort interprocess lock for the provider sensor state file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "a+") as lock_file:
        locked = False
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            locked = True
        except Exception:
            # Platforms without fcntl still get thread-level serialization.
            locked = False
        try:
            yield
        finally:
            if locked:
                try:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass


def _state_path() -> Path:
    return get_hermes_home() / STATE_RELATIVE_PATH


def _load_state(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.debug("provider problem sensor state read failed: %s", exc)
        return {}


def _slug_provider(provider: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]+", "_", (provider or "unknown").lower()).strip("_")
    if not cleaned:
        cleaned = "unknown"
    if len(cleaned) > 48:
        digest = hashlib.sha1(provider.encode("utf-8", "ignore")).hexdigest()[:8]
        cleaned = f"{cleaned[:39].rstrip('_')}_{digest}"
    return cleaned


def _entity_id(provider: str) -> str:
    return f"binary_sensor.hermes_provider_{_slug_provider(provider)}_problem"


def _summarize_window(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors = sum(1 for item in results if item.get("result") == "error")
    total = len(results)
    problem = total >= WINDOW_SIZE and errors > ERROR_THRESHOLD
    return {
        "total": total,
        "errors": errors,
        "error_rate": (errors / total) if total else 0.0,
        "problem": problem,
    }


def _publish_homeassistant_sensor(provider: str, results: List[Dict[str, Any]]) -> None:
    """Publish the provider's problem state to Home Assistant, if configured."""

    hass_url = os.getenv("HASS_URL", "").rstrip("/")
    hass_token = os.getenv("HASS_TOKEN", "").strip()
    if not hass_url or not hass_token:
        return

    summary = _summarize_window(results)
    entity_id = _entity_id(provider)
    payload = {
        "state": "on" if summary["problem"] else "off",
        "attributes": {
            "friendly_name": f"Hermes Provider {provider} Problem",
            "device_class": "problem",
            "provider": provider,
            "window_size": WINDOW_SIZE,
            "observed_requests": summary["total"],
            "error_count": summary["errors"],
            "error_rate": round(summary["error_rate"], 3),
            "threshold": f">{ERROR_THRESHOLD}/{WINDOW_SIZE} errors",
            "recent_results": [item.get("result") for item in results],
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    req = urllib.request.Request(
        f"{hass_url}/api/states/{entity_id}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {hass_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status >= 300:
                logger.debug("HA provider sensor update returned HTTP %s for %s", resp.status, entity_id)
    except urllib.error.URLError as exc:
        logger.debug("HA provider sensor update failed for %s: %s", entity_id, exc)
    except Exception as exc:
        logger.debug("HA provider sensor update failed for %s: %s", entity_id, exc)


def record_provider_request_result(
    *,
    provider: Optional[str],
    success: bool,
    model: Optional[str] = None,
    api_mode: Optional[str] = None,
    base_url: Optional[str] = None,
    error: Optional[str] = None,
    status_code: Optional[int] = None,
) -> Dict[str, Any]:
    """Record one provider request outcome and update its HA problem sensor.

    The rolling window and sensor are keyed only by provider so each provider
    gets an independent problem sensor.  The sensor reports a problem only when
    more than 50% of the last 8 requests for that provider are errors.
    """

    provider_name = (provider or "unknown").strip() or "unknown"
    now = time.time()
    entry = {
        "ts": now,
        "result": "success" if success else "error",
        "model": model or "",
        "api_mode": api_mode or "",
        "base_url": base_url or "",
    }
    if error and not success:
        entry["error"] = str(error)[:500]
    if status_code is not None and not success:
        entry["status_code"] = status_code

    state_path = _state_path()
    with _thread_lock:
        with _locked_file(state_path):
            state = _load_state(state_path)
            providers = state.setdefault("providers", {})
            pdata = providers.setdefault(provider_name, {})
            results = pdata.setdefault("results", [])
            if not isinstance(results, list):
                results = []
            results.append(entry)
            results = results[-WINDOW_SIZE:]
            pdata["results"] = results
            pdata["entity_id"] = _entity_id(provider_name)
            pdata["updated_at"] = now
            pdata.update(_summarize_window(results))
            state["version"] = 1
            state["updated_at"] = now
            atomic_json_write(state_path, state, indent=2)

    _publish_homeassistant_sensor(provider_name, results)
    return {
        "provider": provider_name,
        "entity_id": _entity_id(provider_name),
        **_summarize_window(results),
    }
