#!/usr/bin/env python3
"""
Rate Limit Cooldown Tracker

Generic rate limit cooldown cache for any provider.
Records rate limit events and tracks cooldown periods so subsequent
requests can skip the primary provider instead of repeatedly hitting
the limit before falling back.

File format (~/.hermes/rate_limits/{provider}.json):
{
    "model_slug": {
        "error_type": "rate_limit",
        "timestamp": "2026-04-18T18:30:00Z",
        "reset_after": 60,  # seconds from timestamp
        "http_status": 429
    }
}

This file lives in ~/.hermes/ and survives hermes-agent git resets.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_COOLDOWN_SECONDS = 60  # Fallback when API doesn't provide reset time
RATE_LIMIT_DIR_NAME = "rate_limits"


def _get_rate_limit_dir(hermes_home: Optional[str] = None) -> Path:
    """Get the rate limits cache directory."""
    if hermes_home:
        base = Path(hermes_home)
    else:
        # Try HERMES_HOME env, then ~/.hermes/
        base = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    return base / RATE_LIMIT_DIR_NAME


def _get_provider_file(provider: str, hermes_home: Optional[str] = None) -> Path:
    """Get the rate limit cache file for a specific provider."""
    provider_slug = provider.lower().replace("-", "_").replace("/", "_")
    return _get_rate_limit_dir(hermes_home) / f"{provider_slug}.json"


def _ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def _load_cache(file_path: Path) -> Dict[str, Any]:
    """Load cache from file, return empty dict if not exists or invalid."""
    if not file_path.exists():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to load rate limit cache %s: %s", file_path, e)
        return {}


def _save_cache(file_path: Path, data: Dict[str, Any]) -> None:
    """Save cache to file atomically."""
    _ensure_dir(file_path.parent)
    try:
        # Atomic write via temp file
        temp_path = file_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_path.replace(file_path)
    except IOError as e:
        logger.warning("Failed to save rate limit cache %s: %s", file_path, e)


def record_rate_limit(
    provider: str,
    model: str,
    error_type: str = "rate_limit",
    reset_after: Optional[int] = None,
    http_status: Optional[int] = None,
    hermes_home: Optional[str] = None,
) -> None:
    """Record a rate limit event for a provider/model.

    Args:
        provider: Provider name (e.g., "zai", "openrouter", "openai")
        model: Model slug (e.g., "glm-5-turbo", "claude-sonnet-4")
        error_type: Error type string for classification
        reset_after: Cooldown duration in seconds (from API retry-after or default)
        http_status: HTTP status code (e.g., 429)
        hermes_home: Override Hermes home directory
    """
    cooldown_seconds = reset_after or DEFAULT_COOLDOWN_SECONDS

    file_path = _get_provider_file(provider, hermes_home)
    cache = _load_cache(file_path)

    # Normalize model key (strip provider prefix if present)
    model_key = model.split("/")[-1] if "/" in model else model

    cache[model_key] = {
        "error_type": error_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reset_after": cooldown_seconds,
        "http_status": http_status,
    }

    _save_cache(file_path, cache)
    logger.info(
        "Rate limit recorded for %s/%s: cooldown %ds",
        provider, model_key, cooldown_seconds
    )


def is_in_cooldown(
    provider: str,
    model: str,
    hermes_home: Optional[str] = None,
) -> bool:
    """Check if a provider/model is currently in rate limit cooldown.

    Returns True if:
    - A rate limit event was recorded for this provider/model
    - The cooldown period hasn't expired yet

    Args:
        provider: Provider name
        model: Model slug
        hermes_home: Override Hermes home directory
    """
    file_path = _get_provider_file(provider, hermes_home)
    cache = _load_cache(file_path)

    model_key = model.split("/")[-1] if "/" in model else model

    if model_key not in cache:
        return False

    entry = cache[model_key]
    try:
        recorded_time = datetime.fromisoformat(entry["timestamp"])
        cooldown_seconds = entry.get("reset_after", DEFAULT_COOLDOWN_SECONDS)
    except (KeyError, ValueError):
        return False

    elapsed = (datetime.now(timezone.utc) - recorded_time).total_seconds()
    if elapsed < cooldown_seconds:
        remaining = int(cooldown_seconds - elapsed)
        logger.debug(
            "Provider %s/%s in cooldown: %ds remaining",
            provider, model_key, remaining
        )
        return True

    # Cooldown expired, clean up entry
    del cache[model_key]
    _save_cache(file_path, cache)
    logger.info("Cooldown expired for %s/%s, cleared cache", provider, model_key)
    return False


def get_cooldown_remaining(
    provider: str,
    model: str,
    hermes_home: Optional[str] = None,
) -> int:
    """Get remaining cooldown seconds for a provider/model.

    Returns 0 if not in cooldown.
    """
    file_path = _get_provider_file(provider, hermes_home)
    cache = _load_cache(file_path)

    model_key = model.split("/")[-1] if "/" in model else model

    if model_key not in cache:
        return 0

    entry = cache[model_key]
    try:
        recorded_time = datetime.fromisoformat(entry["timestamp"])
        cooldown_seconds = entry.get("reset_after", DEFAULT_COOLDOWN_SECONDS)
    except (KeyError, ValueError):
        return 0

    elapsed = (datetime.now(timezone.utc) - recorded_time).total_seconds()
    remaining = max(0, int(cooldown_seconds - elapsed))
    return remaining


def clear_cooldown(
    provider: str,
    model: str,
    hermes_home: Optional[str] = None,
) -> bool:
    """Manually clear cooldown for a provider/model.

    Returns True if entry existed and was cleared.
    """
    file_path = _get_provider_file(provider, hermes_home)
    cache = _load_cache(file_path)

    model_key = model.split("/")[-1] if "/" in model else model

    if model_key in cache:
        del cache[model_key]
        _save_cache(file_path, cache)
        logger.info("Manually cleared cooldown for %s/%s", provider, model_key)
        return True
    return False


def clear_all_cooldowns(hermes_home: Optional[str] = None) -> int:
    """Clear all rate limit cooldown caches.

    Returns number of files cleared.
    """
    rate_limit_dir = _get_rate_limit_dir(hermes_home)
    if not rate_limit_dir.exists():
        return 0

    count = 0
    for f in rate_limit_dir.glob("*.json"):
        try:
            f.unlink()
            count += 1
        except IOError:
            pass

    logger.info("Cleared %d rate limit cache files", count)
    return count