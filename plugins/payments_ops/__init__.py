"""Slack companion surface for payments operations."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

from hermes_cli import payments as payments_store
from hermes_cli.config import load_env

logger = logging.getLogger(__name__)


def _env_value(name: str) -> str:
    try:
        env = load_env()
    except Exception:
        env = {}
    return str(os.environ.get(name) or env.get(name) or "").strip()


def _slack_client():
    try:
        from slack_sdk.web.async_client import AsyncWebClient
    except Exception as exc:  # pragma: no cover - dependency is optional in tests
        raise RuntimeError("slack_sdk is not installed") from exc
    token = _env_value("SLACK_BOT_TOKEN")
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN is not configured")
    return AsyncWebClient(token=token)


def _dashboard_url() -> str:
    return _env_value("HERMES_PAYMENTS_DASHBOARD_URL")


def _canvas_channel_key() -> str:
    return _env_value("PAYMENTS_SLACK_CANVAS_CHANNEL_KEY")


def _canvas_channel_name() -> str:
    return _env_value("PAYMENTS_SLACK_CANVAS_CHANNEL_NAME")


def _find_payment(payment_id: str) -> dict[str, Any]:
    for payment in payments_store.list_payment_requests()["requests"]:
        if str(payment.get("id")) == payment_id:
            return payment
    raise KeyError(payment_id)


def _mobile_blocks() -> list[dict[str, Any]]:
    return payments_store.build_slack_mobile_blocks(
        statuses=("needs_review", "ready_to_pay", "paid"),
        per_status_limit=5,
    )


def _status_text(status: str) -> str:
    return status.replace("_", " ")


def _parse_message_context(body: dict[str, Any]) -> tuple[str, str]:
    channel_id = str(
        body.get("channel", {}).get("id")
        or body.get("container", {}).get("channel_id")
        or ""
    ).strip()
    message_ts = str(
        body.get("message", {}).get("ts")
        or body.get("container", {}).get("message_ts")
        or ""
    ).strip()
    if not channel_id and isinstance(body.get("view"), dict):
        raw = str(body["view"].get("private_metadata") or "").strip()
        if raw:
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = {}
            channel_id = channel_id or str(parsed.get("channel_id") or "").strip()
            message_ts = message_ts or str(parsed.get("message_ts") or "").strip()
    return channel_id, message_ts


async def _refresh_mobile_message(client, body: dict[str, Any]) -> None:
    channel_id, message_ts = _parse_message_context(body)
    if not channel_id or not message_ts:
        return
    try:
        await client.chat_update(
            channel=channel_id,
            ts=message_ts,
            text="Payments mobile inbox",
            blocks=_mobile_blocks(),
        )
    except Exception:
        logger.debug("payments_ops: mobile inbox refresh failed", exc_info=True)


async def _refresh_canvas_surface() -> None:
    channel_key = _canvas_channel_key()
    channel_name = _canvas_channel_name()
    if not channel_key or not channel_name:
        return
    script = Path(__file__).resolve().parents[2] / "scripts" / "payments-slack-surface.py"
    cmd = [
        sys.executable,
        str(script),
        "publish-canvas",
        "--channel-key",
        channel_key,
        "--channel-name",
        channel_name,
    ]
    dashboard_url = _dashboard_url()
    if dashboard_url:
        cmd.extend(["--dashboard-url", dashboard_url])
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=30)
    except Exception:
        logger.debug("payments_ops: canvas refresh failed", exc_info=True)


async def _notify_result(client, body: dict[str, Any], text: str) -> None:
    channel_id = str(
        body.get("channel", {}).get("id")
        or body.get("container", {}).get("channel_id")
        or ""
    ).strip()
    user_id = str(body.get("user", {}).get("id") or "").strip()
    if not channel_id or not user_id:
        return
    try:
        await client.chat_postEphemeral(channel=channel_id, user=user_id, text=text)
    except Exception:
        logger.debug("payments_ops: ephemeral confirmation failed", exc_info=True)


async def _open_status_modal(ack, body, action) -> None:
    await ack()
    payment_id = str(action.get("value") or "").strip()
    if not payment_id:
        return
    payment = _find_payment(payment_id)
    channel_id, message_ts = _parse_message_context(body)
    private_metadata = json.dumps(
        {"channel_id": channel_id, "message_ts": message_ts},
        separators=(",", ":"),
    )
    client = _slack_client()
    await client.views_open(
        trigger_id=body.get("trigger_id"),
        view=payments_store.build_slack_status_modal_view(
            payment,
            private_metadata=private_metadata,
        ),
    )


async def _handle_status_change(target_status: str, ack, body, action) -> None:
    if body.get("container", {}).get("type") == "view":
        await ack({"response_action": "clear"})
    else:
        await ack()

    payment_id = str(action.get("value") or "").strip()
    if not payment_id:
        return

    client = _slack_client()
    try:
        updated = payments_store.update_payment_status(payment_id, target_status)
    except KeyError:
        await _notify_result(client, body, "That payment no longer exists in the review queue.")
        return
    except Exception as exc:
        await _notify_result(client, body, f"Payments update failed: {exc}")
        return

    await _refresh_mobile_message(client, body)
    await _refresh_canvas_surface()
    title = updated.get("vendor") or updated.get("title") or updated.get("id") or payment_id
    await _notify_result(client, body, f"{title} marked {_status_text(target_status)}.")


def _status_handler(target_status: str):
    async def _handler(ack, body, action) -> None:
        await _handle_status_change(target_status, ack, body, action)

    return _handler


def _command_summary(status: str, limit: int = 5) -> str:
    requests = payments_store.list_payment_requests()["requests"]
    items = [
        item
        for item in requests
        if item.get("operator_status") == status or item.get("status") == status
    ][:limit]
    heading = {
        "needs_review": "Payments due",
        "ready_to_pay": "Payments ready",
        "paid": "Payments paid",
    }.get(status, f"Payments {_status_text(status)}")
    lines = [f"{heading}:"]
    if not items:
        lines.append("- none")
    else:
        for item in items:
            title = item.get("vendor") or item.get("title") or item.get("id")
            amount = item.get("amount", {}).get("display") or "Amount missing"
            due = item.get("due_date") or "No due date"
            lines.append(f"- {title} · {amount} · due {due}")
    dashboard_url = _dashboard_url()
    if dashboard_url:
        lines.append("")
        lines.append(f"Dashboard: {dashboard_url}")
    return "\n".join(lines)


def register(ctx) -> None:
    ctx.register_slack_action_handler("payments_open_status_modal", _open_status_modal)
    ctx.register_slack_action_handler("payments_mark_review", _status_handler("needs_review"))
    ctx.register_slack_action_handler("payments_mark_ready", _status_handler("ready_to_pay"))
    ctx.register_slack_action_handler("payments_mark_paid", _status_handler("paid"))
    ctx.register_slack_action_handler("payments_mark_ignored", _status_handler("ignored"))
    ctx.register_command(
        "payments-due",
        lambda _args: _command_summary("needs_review"),
        description="Show the top payment requests still needing review",
    )
    ctx.register_command(
        "payments-ready",
        lambda _args: _command_summary("ready_to_pay"),
        description="Show the top payment requests ready to pay",
    )
