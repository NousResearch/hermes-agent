"""Configured runtime for the deterministic Gemini daily review job."""

from __future__ import annotations

import asyncio
import inspect
import re
import signal
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from agent.gemini_daily_review import (
    SOL_REVIEWER_MODEL,
    SOL_REVIEWER_PROVIDER,
    AlertDeliveryIndeterminateError,
    DailyReviewRunner,
    SolReviewer,
)
from agent.gemini_route_receipts import GeminiReceiptStore, resolve_profile_receipt_path
from hermes_constants import get_hermes_home


@contextmanager
def _hard_deadline(seconds: float):
    """Bound one scheduler tick below Hermes cron's outer timeout."""
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def expire(_signum: int, _frame: Any) -> None:
        raise TimeoutError("Gemini daily review exceeded its internal deadline")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _parse_slack_target(value: Any) -> str:
    target = str(value or "").strip()
    if not target.startswith("slack:"):
        raise ValueError("review.alert_target must be an exact slack:<channel-id> target")
    channel_id = target.split(":", 1)[1]
    if not channel_id or ":" in channel_id:
        raise ValueError("review.alert_target must identify one Slack channel")
    return channel_id


def _make_default_reviewer_factory(review: Mapping[str, Any]) -> Callable[[], SolReviewer]:
    from run_agent import AIAgent

    provider = str(review.get("review_provider") or SOL_REVIEWER_PROVIDER)
    model = str(review.get("review_model") or SOL_REVIEWER_MODEL)
    effort = str(review.get("review_reasoning_effort") or "medium")
    return lambda: SolReviewer(
        agent_factory=AIAgent,
        provider=provider,
        model=model,
        reasoning_effort=effort,
    )


def _make_default_slack_sender(
    channel_id: str,
    *,
    expected_workspace_id: str,
    standalone_send: Callable[..., Any] | None = None,
    client_factory: Callable[[str], Any] | None = None,
    token: str | None = None,
) -> Callable[[str], Mapping[str, Any]]:
    if not expected_workspace_id:
        raise ValueError("review.alert_workspace_id must pin one Slack workspace")

    def send(message: str) -> Mapping[str, Any]:
        from agent.secret_scope import get_secret
        from gateway.config import PlatformConfig
        from plugins.platforms.slack.adapter import (
            _apply_slack_proxy,
            _standalone_send,
            resolve_proxy_url,
        )

        selected_token = token or str(get_secret("SLACK_BOT_TOKEN", "")).split(",", 1)[0].strip()
        if not selected_token:
            raise RuntimeError("Gemini review Slack alert failed: bot token unavailable")

        if client_factory is None:
            from slack_sdk.web.async_client import AsyncWebClient

            def default_client_factory(raw_token: str) -> Any:
                client = AsyncWebClient(token=raw_token)
                _apply_slack_proxy(client, resolve_proxy_url())
                return client
            build_client: Callable[[str], Any] = default_client_factory
        else:
            build_client = client_factory

        client = build_client(selected_token)

        async def call(method_name: str, **kwargs: Any) -> Mapping[str, Any]:
            result = getattr(client, method_name)(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, Mapping):
                raise RuntimeError(f"Slack {method_name} returned an invalid response")
            return result

        async def preflight() -> str:
            auth = await call("auth_test")
            info = await call("conversations_info", channel=channel_id)
            channel = info.get("channel")
            if auth.get("ok") is not True or not auth.get("team_id"):
                raise RuntimeError("Slack auth.test did not verify a workspace")
            if str(auth.get("team_id")) != expected_workspace_id:
                raise RuntimeError("Slack token belongs to an unexpected workspace")
            if info.get("ok") is not True or not isinstance(channel, Mapping):
                raise RuntimeError("Slack conversations.info did not verify the channel")
            if channel.get("is_member") is not True:
                raise RuntimeError("Slack bot is not a member of the review channel")
            channel_team = channel.get("context_team_id") or channel.get("team_id")
            if channel_team and str(channel_team) != expected_workspace_id:
                raise RuntimeError("Slack workspace/channel identity mismatch")
            return expected_workspace_id

        async def find_history(
            team_id: str, expected_ts: str | None = None
        ) -> Mapping[str, Any] | None:
            cursor: str | None = None
            while True:
                kwargs: dict[str, Any] = {
                    "channel": channel_id,
                    "limit": 200,
                    "inclusive": True,
                }
                if expected_ts:
                    kwargs.update(oldest=expected_ts, latest=expected_ts)
                elif cursor:
                    kwargs["cursor"] = cursor
                history = await call("conversations_history", **kwargs)
                messages = history.get("messages")
                if history.get("ok") is not True or not isinstance(messages, list):
                    raise RuntimeError("Slack conversations.history did not verify delivery")
                for item in messages:
                    if not isinstance(item, Mapping):
                        continue
                    item_ts = str(item.get("ts") or "")
                    item_text = str(item.get("text") or "")
                    if expected_ts and item_ts != expected_ts:
                        continue
                    if item_text != message:
                        continue
                    if item_ts:
                        return {
                            "success": True,
                            "platform": "slack",
                            "team_id": team_id,
                            "chat_id": channel_id,
                            "message_id": item_ts,
                        }
                metadata = history.get("response_metadata")
                next_cursor = (
                    str(metadata.get("next_cursor") or "")
                    if isinstance(metadata, Mapping)
                    else ""
                )
                if expected_ts or not next_cursor:
                    return None
                cursor = next_cursor

        team_id = asyncio.run(preflight())
        existing = asyncio.run(find_history(team_id))
        if existing is not None:
            return existing

        send_impl = standalone_send or _standalone_send
        indeterminate = False
        try:
            result = asyncio.run(
                send_impl(PlatformConfig(enabled=True, token=selected_token), channel_id, message)
            )
        except Exception:
            indeterminate = True
            result = {"error": "unknown_send_result"}

        expected_ts = None
        if isinstance(result, Mapping):
            send_error = result.get("error")
            if isinstance(send_error, str) and send_error.startswith("Slack send failed:"):
                indeterminate = True
        if isinstance(result, Mapping) and result.get("success") is True:
            returned_channel = str(result.get("chat_id") or "")
            expected_ts = str(result.get("message_id") or "") or None
            if returned_channel != channel_id or expected_ts is None:
                raise RuntimeError("Gemini review Slack alert returned the wrong target")
            indeterminate = True
        try:
            verified = asyncio.run(find_history(team_id, expected_ts))
        except Exception as exc:
            if indeterminate:
                raise AlertDeliveryIndeterminateError(
                    "Slack delivery outcome could not be reconciled"
                ) from exc
            raise
        if verified is None:
            if indeterminate:
                raise AlertDeliveryIndeterminateError(
                    "Slack delivery outcome could not be reconciled"
                )
            raise RuntimeError("Slack delivery was not visible in channel history")
        return verified

    return send


def run_configured_review(
    *,
    config: Mapping[str, Any] | None = None,
    now: datetime | None = None,
    reviewer_factory: Callable[[], Any] | None = None,
    alert_sender: Callable[[str], Any] | None = None,
    active_profile: str | None = None,
) -> dict[str, Any]:
    """Run one configured review tick without printing successful output."""

    if config is None:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    delegation = config.get("delegation")
    routing = delegation.get("gemini_routing") if isinstance(delegation, Mapping) else None
    if not isinstance(routing, Mapping):
        return {"status": "disabled"}
    routing_enabled = routing.get("enabled", False)
    if type(routing_enabled) is not bool:
        raise ValueError("delegation.gemini_routing.enabled must be a boolean")
    if routing_enabled is False:
        return {"status": "disabled"}
    if active_profile is None:
        from hermes_cli.profiles import get_active_profile_name

        active_profile = get_active_profile_name() or "default"
    profiles = routing.get("profiles")
    if not isinstance(profiles, list) or not all(
        isinstance(profile, str) for profile in profiles
    ):
        raise ValueError("delegation.gemini_routing.profiles must be a list of strings")
    if active_profile not in profiles:
        return {"status": "disabled"}
    review = routing.get("review")
    if not isinstance(review, Mapping):
        return {"status": "disabled"}
    review_enabled = review.get("enabled", False)
    if type(review_enabled) is not bool:
        raise ValueError("delegation.gemini_routing.review.enabled must be a boolean")
    if review_enabled is False:
        return {"status": "disabled"}

    timezone_name = review.get("timezone", "America/Los_Angeles")
    if timezone_name != "America/Los_Angeles":
        raise ValueError("review.timezone must be America/Los_Angeles")
    sample_size = review.get("sample_size", 5)
    if type(sample_size) is not int or sample_size != 5:
        raise ValueError("review.sample_size must be exactly 5")
    clock = now or datetime.now(ZoneInfo(timezone_name))
    local_clock = clock.astimezone(ZoneInfo(timezone_name))
    not_before = str(review.get("not_before_local") or "00:15")
    try:
        not_before_hour, not_before_minute = (int(part) for part in not_before.split(":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("review.not_before_local must use HH:MM") from exc
    if (local_clock.hour, local_clock.minute) < (not_before_hour, not_before_minute):
        return {"status": "not_before"}
    local_date = local_clock.date()
    target_day = local_date.fromordinal(local_date.toordinal() - 1)
    alert_channel_id = _parse_slack_target(review.get("alert_target"))
    receipt_path = resolve_profile_receipt_path(
        get_hermes_home(),
        str(routing.get("receipt_db") or "state/gemini-routing.sqlite3"),
    )
    store = GeminiReceiptStore(receipt_path)
    existing_batch = store.get_review_batch(target_day.isoformat())
    delivery_channel_id = alert_channel_id
    delivery_workspace_id = str(review.get("alert_workspace_id") or "")
    if (
        existing_batch is not None
        and existing_batch.get("status") in {"failed", "pipeline_failed"}
        and existing_batch.get("alert_status") in {"pending", "sending"}
        and isinstance(existing_batch.get("slack_channel_id"), str)
        and existing_batch["slack_channel_id"]
    ):
        delivery_channel_id = str(existing_batch["slack_channel_id"])
        if isinstance(existing_batch.get("slack_workspace_id"), str):
            delivery_workspace_id = str(existing_batch["slack_workspace_id"])

    configured_provider = str(review.get("review_provider") or SOL_REVIEWER_PROVIDER)
    configured_model = str(review.get("review_model") or SOL_REVIEWER_MODEL)
    identity_error = (
        None
        if configured_provider == SOL_REVIEWER_PROVIDER
        and configured_model == SOL_REVIEWER_MODEL
        else "reviewer_identity_mismatch"
    )
    factory = reviewer_factory or _make_default_reviewer_factory(review)
    sender = alert_sender or _make_default_slack_sender(
        delivery_channel_id,
        expected_workspace_id=delivery_workspace_id,
    )
    runner = DailyReviewRunner(
        store=store,
        reviewer_factory=factory,
        alert_sender=sender,
        alert_channel_id=delivery_channel_id,
        alert_workspace_id=delivery_workspace_id,
        sample_size=sample_size,
        reviewer_provider=SOL_REVIEWER_PROVIDER,
        reviewer_model=SOL_REVIEWER_MODEL,
        timezone_name=timezone_name,
    )
    with _hard_deadline(150.0):
        result = runner.run(
            target_day=target_day,
            pipeline_preflight_error=identity_error,
        )

    retention = routing.get("retention")
    if not isinstance(retention, Mapping):
        retention = {}
    store.apply_retention(
        now=clock,
        raw_days=int(retention.get("raw_days", 30)),
        aggregate_days=int(retention.get("aggregate_days", 180)),
    )
    return result
