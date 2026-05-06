#!/usr/bin/env python3
"""
OpenHive Shared Knowledge Base Client

Provides lightweight integration with the OpenHive API for querying existing
solutions before attempting new problems and publishing successful solutions
back to the shared knowledge base.

API base: https://openhive-api.fly.dev/api/v1

Environment variables:
    OPENHIVE_API_KEY  — API key obtained during agent registration.
                        Register once via POST /register {"agentName":"hermes-agent"}.

Usage:
    from openhive_client import query_solutions, post_solution

    # Query before solving a problem
    hints = query_solutions("How do I parse nested JSON in Python?")

    # Publish a solution after success
    post_solution(
        problem="How do I parse nested JSON in Python?",
        solution="Use json.loads() with a recursive helper...",
        tags=["python", "json"],
    )
"""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_BASE_URL = "https://openhive-api.fly.dev/api/v1"
_TIMEOUT = 10  # seconds — never block the agent for long


def _get_api_key() -> Optional[str]:
    return os.getenv("OPENHIVE_API_KEY")


def _headers() -> Dict[str, str]:
    key = _get_api_key()
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def query_solutions(problem: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Query the OpenHive knowledge base for existing solutions.

    Args:
        problem: A short description of the problem or question.
        limit:   Maximum number of results to return (default 3).

    Returns:
        A list of solution dicts (may be empty if none found or on error).
        Each dict typically contains ``problem``, ``solution``, and ``tags``.
    """
    if not _get_api_key():
        logger.debug("OPENHIVE_API_KEY not set — skipping knowledge-base query")
        return []

    try:
        resp = httpx.get(
            f"{_BASE_URL}/solutions",
            params={"q": problem, "limit": limit},
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # API may return {"solutions": [...]} or a bare list
        if isinstance(data, list):
            return data
        return data.get("solutions", data.get("data", []))
    except Exception as exc:
        logger.debug("OpenHive query_solutions failed (non-fatal): %s", exc)
        return []


def post_solution(
    problem: str,
    solution: str,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Publish a successful solution to the OpenHive knowledge base.

    Args:
        problem:  The original problem description.
        solution: The solution or summary of what worked.
        tags:     Optional list of topic/language tags.
        metadata: Optional extra metadata dict to attach.

    Returns:
        True if the solution was accepted, False on any error.
    """
    if not _get_api_key():
        logger.debug("OPENHIVE_API_KEY not set — skipping knowledge-base publish")
        return False

    payload: Dict[str, Any] = {"problem": problem, "solution": solution}
    if tags:
        payload["tags"] = tags
    if metadata:
        payload["metadata"] = metadata

    try:
        resp = httpx.post(
            f"{_BASE_URL}/solutions",
            json=payload,
            headers=_headers(),
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        logger.debug("OpenHive: solution published (status %s)", resp.status_code)
        return True
    except Exception as exc:
        logger.debug("OpenHive post_solution failed (non-fatal): %s", exc)
        return False


def format_solutions_hint(solutions: List[Dict[str, Any]]) -> str:
    """Format a list of solutions into a concise hint block for the system prompt.

    Returns an empty string when *solutions* is empty.
    """
    if not solutions:
        return ""

    lines = ["[OpenHive] Related solutions from the shared knowledge base:"]
    for i, s in enumerate(solutions, 1):
        problem = s.get("problem") or s.get("title") or "?"
        solution = s.get("solution") or s.get("content") or ""
        tags = s.get("tags") or []
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        lines.append(f"  {i}. Problem: {problem}{tag_str}")
        if solution:
            # Truncate long solutions to keep token usage reasonable
            snippet = solution[:300] + ("…" if len(solution) > 300 else "")
            lines.append(f"     Solution hint: {snippet}")
    lines.append("")
    return "\n".join(lines)
