"""Opt-in report-length policy, separate from model failure classification."""
import logging

logger = logging.getLogger(__name__)


def normalize_min_response_chars(value: object) -> int:
    """None/zero disables the floor; never coerce malformed persisted settings."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("min_response_chars must be a non-negative integer (0 disables it).")
    return value


def apply_response_floor(response: str, job: dict) -> str:
    """Route a short report through cron's existing empty-response soft failure."""
    from gateway.response_filters import is_autonomous_silence_response

    minimum = normalize_min_response_chars(job.get("min_response_chars"))
    text = response.strip()
    if minimum and text and len(text) < minimum and not is_autonomous_silence_response(text):
        logger.warning(
            "Job '%s': response below min_response_chars (%d < %d); suppressing delivery",
            job.get("id"), len(text), minimum,
        )
        return ""
    return response
