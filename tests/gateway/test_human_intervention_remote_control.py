"""Tests for the gateway /intervention (/iv) mobile commands.

These commands control a REMOTE CLI human-intervention prompt by code via the
``hermes_cli.human_intervention_remote_control`` store. They use a dedicated
namespace to avoid colliding with the gateway's own /approve, /deny, /status
commands (which act on in-process approvals / session state).

The handler must:
  * dispatch deny/extend/status to ``set_remote_decision`` /
    ``get_pending_intervention``;
  * dispatch ``approve`` (one-tap + typed-confirm) to ``set_remote_decision``,
    while the store refuses critical/disabled tiers with a clear message;
  * be gated by ``remote_control.enabled`` and ``allowed_targets`` config.
"""

import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _make_runner():
    """Build a minimal GatewayRunner without running __init__.

    ``_handle_intervention_command`` only reads ``event.get_command_args()``
    and ``event.source.platform`` — no other runner state is touched.
    """
    from gateway.run import GatewayRunner

    return object.__new__(GatewayRunner)


def _make_event(args: str, platform_value: str = "telegram"):
    return SimpleNamespace(
        get_command_args=lambda: args,
        source=SimpleNamespace(
            platform=SimpleNamespace(value=platform_value),
            chat_id="c",
            user_id="u",
        ),
    )


def _enabled_config(allowed_targets=None):
    return {
        "notifications": {
            "human_intervention": {
                "remote_control": {
                    "enabled": True,
                    "allowed_targets": allowed_targets or [],
                }
            }
        }
    }


@pytest.mark.asyncio
async def test_iv_deny_calls_set_remote_decision_deny():
    runner = _make_runner()
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), \
         patch(
             "hermes_cli.human_intervention_remote_control.set_remote_decision",
             return_value=(True, "ok", None),
         ) as mock_set:
        result = await runner._handle_intervention_command(_make_event("deny 7392"))

    assert mock_set.call_count == 1
    args, kwargs = mock_set.call_args
    assert args[0] == "7392"
    assert args[1] == "deny"
    assert "source" in kwargs
    assert "已拒绝" in result
    assert "7392" in result


@pytest.mark.asyncio
async def test_iv_extend_with_minutes():
    runner = _make_runner()
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), \
         patch(
             "hermes_cli.human_intervention_remote_control.set_remote_decision",
             return_value=(True, "ok", None),
         ) as mock_set:
        result = await runner._handle_intervention_command(_make_event("extend 7392 15"))

    assert mock_set.call_count == 1
    args, kwargs = mock_set.call_args
    assert args[0] == "7392"
    assert args[1] == "extend"
    assert kwargs.get("minutes") == 15
    assert "已延长" in result
    assert "15" in result


@pytest.mark.asyncio
async def test_iv_extend_defaults_to_15():
    runner = _make_runner()
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), \
         patch(
             "hermes_cli.human_intervention_remote_control.set_remote_decision",
             return_value=(True, "ok", None),
         ) as mock_set:
        await runner._handle_intervention_command(_make_event("extend 7392"))

    args, kwargs = mock_set.call_args
    assert kwargs.get("minutes") == 15


@pytest.mark.asyncio
async def test_iv_extend_rejects_non_positive_minutes_before_store():
    runner = _make_runner()
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), \
         patch(
             "hermes_cli.human_intervention_remote_control.set_remote_decision",
         ) as mock_set:
        result = await runner._handle_intervention_command(_make_event("extend 7392 -5"))

    assert mock_set.call_count == 0
    assert "分钟" in result
    assert "正整数" in result


@pytest.mark.asyncio
async def test_iv_extend_reports_max_wait_clamp():
    runner = _make_runner()
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), \
         patch(
             "hermes_cli.human_intervention_remote_control.set_remote_decision",
             return_value=(True, "clamped", None),
         ):
        result = await runner._handle_intervention_command(_make_event("extend 7392 120"))

    assert "已延长" in result
    assert "最大等待上限" in result


@pytest.mark.asyncio
async def test_iv_extend_real_store_clamps_to_max_wait(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib
    import hermes_cli.human_intervention_remote_control as hic

    hic = importlib.reload(hic)
    runner = _make_runner()
    rec = hic.create_pending_intervention(
        kind="approval",
        title="t",
        preview="p",
        session_key="s",
        timeout_seconds=60,
        max_total_wait_minutes=2,
        code="7392",
    )

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()):
        result = await runner._handle_intervention_command(_make_event("extend 7392 120"))

    stored = hic.get_pending_intervention("7392")
    assert "已延长" in result
    assert "最大等待上限" in result
    assert stored is not None
    assert stored.deadline_ts == pytest.approx(rec.max_deadline_ts)
    assert stored.deadline_ts <= rec.created_ts + 2 * 60


@pytest.mark.asyncio
async def test_iv_status_reports_state():
    runner = _make_runner()
    rec = SimpleNamespace(
        code="7392",
        kind="approval",
        state="pending",
        deadline_ts=time.time() + 45,
    )
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), \
         patch(
             "hermes_cli.human_intervention_remote_control.get_pending_intervention",
             return_value=rec,
         ):
        result = await runner._handle_intervention_command(_make_event("status 7392"))

    assert "7392" in result
    assert "pending" in result


@pytest.mark.asyncio
async def test_iv_approve_one_tap_ok():
    runner = _make_runner()
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), \
         patch(
             "hermes_cli.human_intervention_remote_control.set_remote_decision",
             return_value=(True, "ok", object()),
         ) as mock_set:
        result = await runner._handle_intervention_command(_make_event("approve 7392"))

    assert mock_set.call_count == 1
    args, kwargs = mock_set.call_args
    assert args[0] == "7392"
    assert args[1] == "approve"
    assert kwargs.get("token") is None
    assert "已批准" in result
    assert "7392" in result


@pytest.mark.asyncio
async def test_iv_approve_typed_confirm_ok():
    runner = _make_runner()
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), \
         patch(
             "hermes_cli.human_intervention_remote_control.set_remote_decision",
             return_value=(True, "ok", object()),
         ) as mock_set:
        result = await runner._handle_intervention_command(
            _make_event("approve 7392 4815")
        )

    assert mock_set.call_count == 1
    args, kwargs = mock_set.call_args
    assert args[0] == "7392"
    assert args[1] == "approve"
    assert kwargs.get("token") == "4815"
    assert "已批准" in result


@pytest.mark.asyncio
async def test_iv_approve_bad_token():
    runner = _make_runner()
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), \
         patch(
             "hermes_cli.human_intervention_remote_control.set_remote_decision",
             return_value=(False, "bad_token", object()),
         ):
        result = await runner._handle_intervention_command(
            _make_event("approve 7392 0000")
        )

    assert "令牌" in result


@pytest.mark.asyncio
async def test_iv_approve_not_allowed_critical():
    runner = _make_runner()
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), \
         patch(
             "hermes_cli.human_intervention_remote_control.set_remote_decision",
             return_value=(False, "approve_not_allowed", object()),
         ):
        result = await runner._handle_intervention_command(_make_event("approve 7392"))

    assert "不支持远程批准" in result


@pytest.mark.asyncio
async def test_iv_approve_auth_unchanged():
    runner = _make_runner()
    with patch(
        "hermes_cli.config.load_config",
        return_value=_enabled_config(allowed_targets=["telegram"]),
    ), patch(
        "hermes_cli.human_intervention_remote_control.set_remote_decision",
    ) as mock_set:
        result = await runner._handle_intervention_command(
            _make_event("approve 7392", platform_value="discord")
        )

    assert mock_set.call_count == 0
    assert "无权" in result


@pytest.mark.asyncio
async def test_iv_invalid_code_returns_error():
    runner = _make_runner()
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), \
         patch(
             "hermes_cli.human_intervention_remote_control.set_remote_decision",
             return_value=(False, "not_found", None),
         ):
        result = await runner._handle_intervention_command(_make_event("deny 0000"))

    assert "未找到" in result


@pytest.mark.asyncio
async def test_iv_disabled_returns_notice():
    runner = _make_runner()
    disabled = {
        "notifications": {
            "human_intervention": {"remote_control": {"enabled": False}}
        }
    }
    with patch("hermes_cli.config.load_config", return_value=disabled), \
         patch(
             "hermes_cli.human_intervention_remote_control.set_remote_decision",
         ) as mock_set:
        result = await runner._handle_intervention_command(_make_event("deny 7392"))

    assert mock_set.call_count == 0
    assert "未启用" in result


@pytest.mark.asyncio
async def test_iv_unauthorized_platform():
    runner = _make_runner()
    with patch(
        "hermes_cli.config.load_config",
        return_value=_enabled_config(allowed_targets=["telegram"]),
    ), patch(
        "hermes_cli.human_intervention_remote_control.set_remote_decision",
    ) as mock_set:
        result = await runner._handle_intervention_command(
            _make_event("deny 7392", platform_value="discord")
        )

    assert mock_set.call_count == 0
    assert "无权" in result


@pytest.mark.asyncio
async def test_iv_usage_when_missing_args():
    runner = _make_runner()
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()):
        result = await runner._handle_intervention_command(_make_event(""))

    assert "用法" in result
