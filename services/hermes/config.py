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


def get_approve_projects() -> list[str]:
    """Return the list of Commander-tracked ``owner/repo`` projects eligible
    for journal dev-todo [Approve] routing.

    Reads JOURNAL_APPROVE_PROJECTS as a comma-separated list, stripping
    whitespace and dropping empty entries. Falls back to a single-item list
    wrapping JOURNAL_APPROVE_PROJECT when JOURNAL_APPROVE_PROJECTS is unset.
    Returns an empty list when neither env var is set.
    """
    raw = os.environ.get("JOURNAL_APPROVE_PROJECTS")
    if raw:
        return [p.strip() for p in raw.split(",") if p.strip()]

    single = os.environ.get("JOURNAL_APPROVE_PROJECT", "").strip()
    if single:
        return [single]

    return []


def get_perf_coach_user() -> Optional[str]:
    """Return the perf-coach username from PERF_COACH_USER, or None if unset.

    When set, this is appended as the ``user`` query parameter on
    feel-entry requests. When unset, the worker falls back to its
    single-active-user resolution.
    """
    return os.environ.get("PERF_COACH_USER") or None
