"""Turn-level failure evidence shared by cron execution and receipt recovery."""
import logging
from typing import Optional

CRON_FAILURE_MARKER = "[CRON_FAILURE]"


def _cron_failure_marker_error(text: str) -> Optional[str]:
    """Only the exact, standalone first line declares failure; keep the rest as evidence."""
    lines = (text or "").splitlines()
    if not lines or lines[0].rstrip() != CRON_FAILURE_MARKER:
        return None
    evidence = "\n".join(lines[1:]).strip()
    return evidence or "Cron agent reported failure."


def refused_turn_error(result: dict, final_response: str) -> Optional[str]:
    """Only positive refusals of every attempt declare a failed turn."""
    if _cron_failure_marker_error(final_response) is not None:
        return None
    attempted, refused = result.get("tool_calls_attempted"), result.get("tool_calls_refused")
    if attempted is None or attempted <= 0 or refused != attempted:
        return None
    logging.getLogger(__name__).info("Cron turn refused tools: %s", result.get("tool_calls_refused_names") or [])
    return ("Every tool call in this run was refused before it ran "
            "(by a plugin, guardrail, scope, middleware or approval policy); no work was done.")
