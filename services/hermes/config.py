"""Configuration for Hermes services.

Reads perf-coach connection settings from environment variables.
Credentials are never hardcoded; they must be set via env or config.
"""
from __future__ import annotations

import os
from typing import Optional


def get_perf_coach_url() -> Optional[str]:
    """Return the perf-coach base URL from PERF_COACH_URL, or None if unset."""
    return os.environ.get("PERF_COACH_URL") or None


def get_perf_coach_token() -> Optional[str]:
    """Return the perf-coach bearer token from PERF_COACH_BEARER_TOKEN, or None if unset."""
    return os.environ.get("PERF_COACH_BEARER_TOKEN") or None


def get_commander_api_url() -> Optional[str]:
    """Return the Commander dashboard API base URL from COMMANDER_API_URL, or None if unset."""
    return os.environ.get("COMMANDER_API_URL") or None
