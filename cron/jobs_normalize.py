"""Text and id-list normalizers for ``create_job`` / ``update_job`` fields (moved from cron/jobs.py unchanged)."""
from typing import Any, Optional


def _normalize_job_optional_text(
    value: Any, *, strip_trailing_slash: bool = False
) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return (value.strip().rstrip("/") if strip_trailing_slash else value.strip()) or None


def _normalize_base_url(value: Any) -> Optional[str]:
    return _normalize_job_optional_text(value, strip_trailing_slash=True)


def _normalize_str_list(items: Any) -> Optional[list[str]]:
    """Non-blank stripped items of *items*, or None when nothing remains."""
    return [str(j).strip() for j in items if str(j).strip()] or None


def _normalize_context_from(value: Any) -> Optional[list[str]]:
    """Accept a job id or a list of ids; anything else is None."""
    if isinstance(value, str):
        value = [value]
    return _normalize_str_list(value) if isinstance(value, list) else None


def _normalize_failure_deliver(value: Any) -> Optional[str]:
    """failure_deliver shares deliver's value grammar; flatten str/list like the tool layer's
    _normalize_deliver_param for direct create_job callers. Semantic validation happens at
    resolution time via the shared deliver path."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(p).strip() for p in value if str(p).strip()) or None
    return _normalize_job_optional_text(value)


def _normalize_output_schema(value: Any) -> Optional[dict[str, Any]]:
    """A JSON Schema object (or its JSON text) -> the schema; ``None``/``""`` -> unset. An invalid
    schema raises ValueError before anything is stored, with delegation's own error text."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    from tools.delegation_output_schema import coerce_output_schema
    schema, error = coerce_output_schema(value)
    if error:
        raise ValueError(error)
    return schema
