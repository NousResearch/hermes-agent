"""Provider-scoped usage-limit holds: one refusal parks every new run on that provider.

When a turn dies on a plan usage limit whose reset time the provider (or the credential
pool) actually reported, every other unattended run aimed at the same provider — the next
cron tick, the next kanban card, the next queued message — would hit the same wall and fail
the same way. This module remembers the wall per provider/account so those callers can wait
for the reset instead of each burning a request and posting its own failure.

Rules the callers rely on:

* A hold is only recorded from a REPORTED reset (``agent.quota_resume`` sources). An unknown
  reset time records nothing: absence never reads as "reset now" or as "held forever".
* Every hold is bounded by ``MAX_HOLD_SECONDS``; a far-future reset is clamped, not trusted.
* Holds live in ``<HERMES_HOME>/usage_holds.json`` so a gateway restart keeps them, and they
  are per profile because the file is under the profile's home.
* A run that reaches the provider successfully releases the hold early.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from hermes_constants import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_FILE_NAME = "usage_holds.json"
# A weekly window is the longest plan limit in the wild; anything past it is a bad stamp.
MAX_HOLD_SECONDS = 7 * 24 * 3600


def _endpoint_identity(base_url: Any) -> str:
    """Scheme and host are case-insensitive; the path is not (``/TeamA/v1`` is another account)."""
    raw = str(base_url).strip().rstrip("/")
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    if not parts.scheme or not parts.netloc:
        return raw
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def hold_key(provider: Any, base_url: Any = "") -> str:
    """Provider identity a hold is scoped to. Custom endpoints are distinct accounts per URL."""
    name = str(provider or "").strip().lower()
    if not name:
        return ""
    if name == "custom" and base_url:
        return f"custom:{_endpoint_identity(base_url)}"
    return name


def _path(home: Optional[Path] = None) -> Path:
    return (home or get_hermes_home()) / _FILE_NAME


@contextmanager
def _locked(home: Optional[Path] = None):
    """Cross-process lock around a whole read-change-write of the hold file: a CLI turn and the
    gateway each rewriting the file from their own read would otherwise drop each other's hold."""
    from tools.skill_usage import skill_file_lock

    with skill_file_lock(_path(home).with_suffix(".json.lock")):
        yield


def _load(home: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(_path(home).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _live(data: dict[str, dict[str, Any]], now: float) -> dict[str, dict[str, Any]]:
    return {
        k: v for k, v in data.items()
        if isinstance(v, dict) and isinstance(v.get("until"), (int, float)) and v["until"] > now
    }


def record_hold(
    provider: Any, *, until: Any, base_url: Any = "", account: str = "", source: str = "",
    now: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Park *provider* until *until* (epoch seconds). Returns the stored hold, or None when the
    reset is unknown/past — an unknown reset never creates a hold."""
    key = hold_key(provider, base_url)
    now = time.time() if now is None else now
    if not key or not isinstance(until, (int, float)) or until <= now:
        return None
    until = min(float(until), now + MAX_HOLD_SECONDS)
    with _locked():
        data = _live(_load(), now)
        prior = data.get(key) or {}
        hold = {
            "provider": str(provider), "until": until, "account": account, "source": source,
            "since": prior.get("since", now),
            # One notice per hold: a later refusal inside the same window keeps the flag.
            "notified": bool(prior.get("notified")) and prior.get("until", 0) >= until - 60,
        }
        data[key] = hold
        atomic_json_write(_path(), data)
    logger.warning("Usage limit on %s: holding new work until %s (source=%s)",
                   key, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(until)), source)
    return hold


def active_hold(
    provider: Any, base_url: Any = "", *, now: Optional[float] = None, home: Optional[Path] = None,
) -> Optional[dict]:
    """The live hold for *provider*, or None (expired holds are inert). ``home`` reads another
    profile's holds (the kanban dispatcher deciding for an assignee profile)."""
    key = hold_key(provider, base_url)
    if not key:
        return None
    return _live(_load(home), time.time() if now is None else now).get(key)


def any_active_hold(*, home: Optional[Path] = None) -> bool:
    """Cheap pre-check so callers skip per-run route resolution when nothing is held."""
    return bool(_live(_load(home), time.time()))


def claim_notice(provider: Any, base_url: Any = "", *, home: Optional[Path] = None) -> bool:
    """True exactly once per hold: the caller that gets True posts the one notice."""
    key = hold_key(provider, base_url)
    with _locked(home):
        data = _live(_load(home), time.time())
        hold = data.get(key)
        if not hold or hold.get("notified"):
            return False
        hold["notified"] = True
        atomic_json_write(_path(home), data)
    return True


def release_hold(provider: Any, base_url: Any = "") -> None:
    """A run on *provider* succeeded: the wall is gone, stop holding work for it."""
    key = hold_key(provider, base_url)
    with _locked():
        data = _load()
        if key not in data:
            return
        data.pop(key, None)
        atomic_json_write(_path(), data)
    logger.info("Usage limit on %s released after a successful run", key)


def release_after_turn(agent: Any, *, completed: bool, failed: bool) -> None:
    """The provider answered a whole turn: any usage-limit hold on it is over.

    Bookkeeping only: a hold file that cannot be written (disk full, read-only home) is logged
    and never replaces the turn's own result.
    """
    if not (completed and not failed and getattr(agent, "_turn_received_provider_response", False)):
        return
    try:
        release_hold(getattr(agent, "provider", ""), getattr(agent, "base_url", ""))
    except OSError:
        logger.warning("Could not release the usage-limit hold after a successful turn", exc_info=True)


def record_from_plan(plan: Any, *, provider: Any, base_url: Any = "", account: str = "") -> None:
    """Record a hold from an ``agent.quota_resume`` plan dict when it carries a reported reset.

    Ineligible-but-dated plans (reset beyond the resume cap) still hold, clamped by
    ``MAX_HOLD_SECONDS``: the provider told us the wall stays up that long either way.
    """
    if not isinstance(plan, dict) or plan.get("reason") != "rate_limit":
        return
    # A window that caps one model family says nothing about the provider's other models.
    if plan.get("scope") == "model":
        return
    try:
        record_hold(provider, until=plan.get("resume_at"), base_url=base_url, account=account,
                    source=str(plan.get("source") or ""))
    except OSError:
        # Never let hold bookkeeping replace the provider failure the caller is reporting.
        logger.warning("Could not record the usage-limit hold for %s", provider, exc_info=True)


def format_until(hold: dict[str, Any]) -> str:
    return time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(float(hold["until"])))


def configured_route(cfg: dict[str, Any]) -> tuple[str, str, bool]:
    """``(provider, base_url, has_fallback)`` a profile's config.yaml routes new work to.

    Only what config states: a route nothing names is never held, and a configured fallback chain
    means the normal fallback runs instead of a hold.
    """
    from hermes_cli.fallback_config import get_fallback_chain

    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    return (str(model_cfg.get("provider") or ""), str(model_cfg.get("base_url") or ""),
            bool(get_fallback_chain(cfg)))
