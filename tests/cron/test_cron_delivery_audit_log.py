"""Tests for the cron delivery audit log line.

Every cron delivery must record the policy decision in a single
grep-able INFO line so the operator can later tell which jobs were
ungoverned (fast path) vs approval-gated (action contract).
"""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

_HAS_TELEGRAM = pytest.importorskip("telegram", reason="python-telegram-bot not installed") is not None


def _run_async_immediately(coro):
    return asyncio.run(coro)


def _make_config():
    from gateway.config import Platform
    pconfig = SimpleNamespace(enabled=True, token="***", extra={})
    return SimpleNamespace(
        platforms={Platform.TELEGRAM: pconfig},
        get_home_channel=lambda _p: None,
    ), pconfig


def _capture_logs(caplog):
    caplog.set_level(logging.INFO, logger="cron.scheduler")


def test_audit_log_marks_fast_path_when_ungoverned(caplog):
    from cron.scheduler import _deliver_result
    _capture_logs(caplog)
    config, _pconfig = _make_config()
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})), \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        job = {"id": "j-fast", "deliver": "origin",
               "origin": {"platform": "telegram", "chat_id": "123"}}
        _deliver_result(job, "hi")
    msgs = [r.getMessage() for r in caplog.records if "delivery_policy" in r.getMessage()]
    assert any("delivery_policy=fast_path_ungoverned" in m for m in msgs)


def test_audit_log_marks_approval_gated_when_approval_id_set(caplog):
    from cron.scheduler import _deliver_result
    _capture_logs(caplog)
    config, _pconfig = _make_config()
    tool_result = json.dumps({"success": True})
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})), \
         patch("tools.send_message_tool.send_message_tool", return_value=tool_result), \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        job = {"id": "j-gated", "deliver": "origin",
               "delivery_approval_id": "approval-20260624-test-ok",
               "origin": {"platform": "telegram", "chat_id": "123"}}
        _deliver_result(job, "hi")
    msgs = [r.getMessage() for r in caplog.records if "delivery_policy" in r.getMessage()]
    assert any("delivery_policy=approval_gated" in m and "approval-20260624-test-ok" in m for m in msgs)


def test_audit_log_marks_missing_approval_when_env_var_on(caplog):
    from cron.scheduler import _deliver_result
    _capture_logs(caplog)
    config, _pconfig = _make_config()
    with patch.dict("os.environ", {"COMPANY_OS_REQUIRE_APPROVAL": "1"}, clear=False), \
         patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})), \
         patch("tools.send_message_tool.send_message_tool", return_value=json.dumps({"error": "BLOCKED"})), \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        job = {"id": "j-missing", "deliver": "origin",
               "origin": {"platform": "telegram", "chat_id": "123"}}
        _deliver_result(job, "hi")
    msgs = [r.getMessage() for r in caplog.records if "delivery_policy" in r.getMessage()]
    assert any("delivery_policy=approval_required_but_missing" in m for m in msgs)
