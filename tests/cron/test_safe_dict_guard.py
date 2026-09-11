"""Test cron scheduler's _safe_dict guard (v0.17→v0.21 jobs.json migration, #107967)."""
import pytest


def test_safe_dict_coerces_pydantic_to_dict():
    """_safe_dict calls .model_dump() on Pydantic models and returns dict for dicts."""
    from cron.scheduler import _safe_dict

    class _MockModel:
        def model_dump(self):
            return {"key": "value"}

    # Pydantic-like object with .model_dump()
    result = _safe_dict(_MockModel())
    assert result == {"key": "value"}

    # Plain dict passes through
    plain = {"already": "dict"}
    assert _safe_dict(plain) is plain

    # None
    assert _safe_dict(None) is None


def test_resolve_job_reasoning_config_guards_pydantic_object():
    """_resolve_job_reasoning_config returns a plain dict even if resolve_reasoning_config
    somehow returned a Pydantic model (v0.17 migration scenario)."""
    from cron.scheduler import _resolve_job_reasoning_config

    class _MockReasoningModel:
        def model_dump(self):
            return {"enabled": True, "effort": "medium"}

    # Monkey-patch resolve_reasoning_config to return a Pydantic-like object
    from unittest.mock import patch
    with patch("hermes_constants.resolve_reasoning_config", return_value=_MockReasoningModel()):
        job = {"id": "test_job"}
        cfg = {}
        result = _resolve_job_reasoning_config(job, cfg, "claude-sonnet-4")

    # Result must be a plain dict, not the mock model
    assert isinstance(result, dict)
    assert result == {"enabled": True, "effort": "medium"}
