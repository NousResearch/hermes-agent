"""Tests for cron delivery routing through the Company OS action contract.

Companion to dec-20260624-223651: when a cron job opts into the action
contract (via ``delivery_approval_id`` or ``COMPANY_OS_REQUIRE_APPROVAL``),
``_deliver_result`` must route through ``tools.send_message_tool`` so the
guard fires. When neither is set, the existing fast path (live adapter
then standalone) is preserved.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


_HAS_TELEGRAM = pytest.importorskip("telegram", reason="python-telegram-bot not installed") is not None


def _run_async_immediately(coro):
    return asyncio.run(coro)


def _make_config(platform_name="telegram"):
    from gateway.config import Platform

    pconfig = MagicMock()
    pconfig.enabled = True
    platform = Platform(platform_name.upper())
    return SimpleNamespace(
        platforms={platform: pconfig},
        get_home_channel=lambda _p: None,
    ), pconfig


def test_delivery_approval_id_routes_through_send_message_tool(monkeypatch):
    from cron.scheduler import _deliver_result

    config, _pconfig = _make_config()

    tool_result = json.dumps({"success": True, "message_id": 99})
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch(
             "tools.send_message_tool.send_message_tool",
             return_value=tool_result,
         ) as smt_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        job = {
            "id": "job-with-approval",
            "name": "approval-job",
            "deliver": "origin",
            "delivery_approval_id": "approval-20260624-test-ok",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }
        result = _deliver_result(job, "hello world")

    assert result is None
    smt_mock.assert_called_once()
    call = smt_mock.call_args[0][0]
    assert call["action"] == "send"
    assert call["target"] == "telegram:123"
    assert call["approval_id"] == "approval-20260624-test-ok"
    assert "Cronjob Response" in call["message"]
    assert "hello world" in call["message"]
    # Fast path must NOT have been touched when the guard is in effect.
    send_mock.assert_not_called()


def test_delivery_approval_blocked_surfaces_error(monkeypatch):
    from cron.scheduler import _deliver_result

    config, _pconfig = _make_config()

    blocked = json.dumps({"error": "BLOCKED: approval_pending"})
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch(
             "tools.send_message_tool.send_message_tool",
             return_value=blocked,
         ) as smt_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        job = {
            "id": "job-blocked",
            "deliver": "origin",
            "delivery_approval_id": "approval-20260624-test-pending",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }
        result = _deliver_result(job, "hello")

    assert result is not None
    assert "BLOCKED" in result
    assert "approval_pending" in result
    send_mock.assert_not_called()
    smt_mock.assert_called_once()


def test_env_var_routes_cron_through_send_message_tool(monkeypatch):
    """When ``COMPANY_OS_REQUIRE_APPROVAL`` is set, every cron delivery
    must traverse ``send_message_tool`` so the guard (which lives inside
    the tool) has a chance to block. We mock the tool here because this
    test is about the cron-side routing decision, not the guard itself.
    """
    from cron.scheduler import _deliver_result

    monkeypatch.setenv("COMPANY_OS_REQUIRE_APPROVAL", "1")
    config, _pconfig = _make_config()

    tool_result = json.dumps({"success": True})
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch(
             "tools.send_message_tool.send_message_tool",
             return_value=tool_result,
         ) as smt_mock:
        job = {
            "id": "job-no-approval-set",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }
        result = _deliver_result(job, "hello")

    assert result is None  # mocked tool returned success
    smt_mock.assert_called_once()
    call = smt_mock.call_args[0][0]
    assert call["action"] == "send"
    assert call["target"] == "telegram:123"
    # No per-job approval_id, so the cron side forwards none either; the
    # action guard inside send_message_tool is the one that will reject it.
    assert "approval_id" not in call or call.get("approval_id") in (None, "")
    # Fast path must NOT have been touched when env var is on.
    send_mock.assert_not_called()


def test_env_var_with_blocked_tool_response_surfaces_error(monkeypatch):
    """End-to-end: env var on + no per-job approval_id + tool returns
    blocked payload → cron surfaces the error. Uses the real guard path
    by removing the tool mock and letting cron actually call into the
    tool with ``COMPANY_OS_REQUIRE_APPROVAL`` set in os.environ.
    """
    from cron.scheduler import _deliver_result

    monkeypatch.setenv("COMPANY_OS_REQUIRE_APPROVAL", "1")
    config, _pconfig = _make_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        job = {
            "id": "job-real-guard",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }
        result = _deliver_result(job, "hello")

    assert result is not None
    assert "BLOCKED" in result
    assert "approval_id was not provided" in result
    send_mock.assert_not_called()


def test_no_opt_in_preserves_existing_fast_path(monkeypatch):
    """Without delivery_approval_id and without the env var, the existing
    live-adapter / standalone path must remain the active code path."""
    from cron.scheduler import _deliver_result

    monkeypatch.delenv("COMPANY_OS_REQUIRE_APPROVAL", raising=False)
    config, pconfig = _make_config()

    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("model_tools._run_async", side_effect=_run_async_immediately), \
         patch("tools.send_message_tool._send_to_platform", new=AsyncMock(return_value={"success": True})) as send_mock, \
         patch(
             "tools.send_message_tool.send_message_tool",
         ) as smt_mock, \
         patch("gateway.mirror.mirror_to_session", return_value=True):
        job = {
            "id": "job-no-guard",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "123"},
        }
        result = _deliver_result(job, "hello")

    assert result is None
    send_mock.assert_called_once()
    smt_mock.assert_not_called()
