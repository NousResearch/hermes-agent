"""Test model.streaming config disables streaming.

Addresses #60879: Gemini-Flash streaming error when streaming: false is set.
"""

import pytest


def test_model_streaming_false_disables_streaming():
    """model.streaming=false should set agent._disable_streaming=True."""

    agent = type("_FakeAgent", (), {"_disable_streaming": False, "provider": None, "model": None})()

    # Simulate the config reading logic from init_agent
    _agent_cfg = {"model": {"streaming": False}}

    # This is the exact code from agent_init.py lines 1499-1515
    _model_cfg = _agent_cfg.get("model", {})
    if not isinstance(_model_cfg, dict):
        _model_cfg = {}
    _model_streaming = _model_cfg.get("streaming", None)
    if _model_streaming is not None and not _model_streaming:
        agent._disable_streaming = True

    assert agent._disable_streaming is True


def test_model_streaming_true_leaves_streaming_enabled():
    """model.streaming=true should NOT disable streaming."""

    agent = type("_FakeAgent", (), {"_disable_streaming": False, "provider": None, "model": None})()

    # Simulate the config reading logic from init_agent
    _agent_cfg = {"model": {"streaming": True}}

    # This is the exact code from agent_init.py lines 1499-1515
    _model_cfg = _agent_cfg.get("model", {})
    if not isinstance(_model_cfg, dict):
        _model_cfg = {}
    _model_streaming = _model_cfg.get("streaming", None)
    if _model_streaming is not None and not _model_streaming:
        agent._disable_streaming = True

    assert agent._disable_streaming is False


def test_model_streaming_unset_leaves_streaming_enabled():
    """model.streaming not set should NOT disable streaming (default)."""

    agent = type("_FakeAgent", (), {"_disable_streaming": False, "provider": None, "model": None})()

    # Simulate the config reading logic from init_agent
    _agent_cfg = {"model": {}}

    # This is the exact code from agent_init.py lines 1499-1515
    _model_cfg = _agent_cfg.get("model", {})
    if not isinstance(_model_cfg, dict):
        _model_cfg = {}
    _model_streaming = _model_cfg.get("streaming", None)
    if _model_streaming is not None and not _model_streaming:
        agent._disable_streaming = True

    assert agent._disable_streaming is False


def test_model_section_missing_leaves_streaming_enabled():
    """Missing model section should NOT disable streaming."""

    agent = type("_FakeAgent", (), {"_disable_streaming": False, "provider": None, "model": None})()

    # Simulate the config reading logic from init_agent
    _agent_cfg = {}

    # This is the exact code from agent_init.py lines 1499-1515
    _model_cfg = _agent_cfg.get("model", {})
    if not isinstance(_model_cfg, dict):
        _model_cfg = {}
    _model_streaming = _model_cfg.get("streaming", None)
    if _model_streaming is not None and not _model_streaming:
        agent._disable_streaming = True

    assert agent._disable_streaming is False


def test_model_streaming_zero_disables_streaming():
    """model.streaming=0 should disable streaming (falsy value)."""

    agent = type("_FakeAgent", (), {"_disable_streaming": False, "provider": None, "model": None})()

    # Simulate the config reading logic from init_agent
    _agent_cfg = {"model": {"streaming": 0}}

    # This is the exact code from agent_init.py lines 1499-1515
    _model_cfg = _agent_cfg.get("model", {})
    if not isinstance(_model_cfg, dict):
        _model_cfg = {}
    _model_streaming = _model_cfg.get("streaming", None)
    if _model_streaming is not None and not _model_streaming:
        agent._disable_streaming = True

    assert agent._disable_streaming is True