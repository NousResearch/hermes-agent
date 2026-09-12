"""Configured runtime for the deterministic Gemini daily review job."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from agent.gemini_daily_review import DailyReviewRunner, SolReviewer
from agent.gemini_route_receipts import GeminiReceiptStore
from hermes_constants import get_hermes_home


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

    provider = str(review.get("review_provider") or "openai-codex")
    model = str(review.get("review_model") or "gpt-5.6-sol")
    effort = str(review.get("review_reasoning_effort") or "medium")
    return lambda: SolReviewer(
        agent_factory=AIAgent,
        provider=provider,
        model=model,
        reasoning_effort=effort,
    )


def _make_default_slack_sender(channel_id: str) -> Callable[[str], None]:
    def send(message: str) -> None:
        from gateway.config import PlatformConfig
        from plugins.platforms.slack.adapter import _standalone_send

        result = asyncio.run(
            _standalone_send(
                PlatformConfig(enabled=True),
                channel_id,
                message,
            )
        )
        if not isinstance(result, dict) or result.get("success") is not True:
            error = result.get("error") if isinstance(result, dict) else "unknown Slack response"
            raise RuntimeError(f"Gemini review Slack alert failed: {error}")

    return send


def run_configured_review(
    *,
    config: Mapping[str, Any] | None = None,
    now: datetime | None = None,
    reviewer_factory: Callable[[], Any] | None = None,
    alert_sender: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one configured review tick without printing successful output."""

    if config is None:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    routing = config.get("gemini_routing")
    if not isinstance(routing, Mapping) or not bool(routing.get("enabled")):
        return {"status": "disabled"}
    review = routing.get("review")
    if not isinstance(review, Mapping) or not bool(review.get("enabled")):
        return {"status": "disabled"}

    timezone_name = str(review.get("timezone") or "America/Los_Angeles")
    clock = now or datetime.now(ZoneInfo(timezone_name))
    local_date = clock.astimezone(ZoneInfo(timezone_name)).date()
    target_day = local_date.fromordinal(local_date.toordinal() - 1)
    alert_channel_id = _parse_slack_target(review.get("alert_target"))
    receipt_path = Path(str(routing.get("receipt_db") or "state/gemini-routing.sqlite3"))
    if not receipt_path.is_absolute():
        receipt_path = get_hermes_home() / receipt_path
    store = GeminiReceiptStore(receipt_path)

    factory = reviewer_factory or _make_default_reviewer_factory(review)
    sender = alert_sender or _make_default_slack_sender(alert_channel_id)
    runner = DailyReviewRunner(
        store=store,
        reviewer_factory=factory,
        alert_sender=sender,
        alert_channel_id=alert_channel_id,
        sample_size=int(review.get("sample_size", 5)),
        reviewer_provider=str(review.get("review_provider") or "openai-codex"),
        reviewer_model=str(review.get("review_model") or "gpt-5.6-sol"),
        timezone_name=timezone_name,
    )
    result = runner.run(target_day=target_day)

    retention = routing.get("retention")
    if not isinstance(retention, Mapping):
        retention = {}
    store.apply_retention(
        now=clock,
        raw_days=int(retention.get("raw_days", 30)),
        aggregate_days=int(retention.get("aggregate_days", 180)),
    )
    return result
