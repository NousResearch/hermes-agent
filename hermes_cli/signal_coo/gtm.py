"""GTM worker for Torben's Signal COO operator."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .action_ledger import ActionLedger
from .briefs import ScopeBrief


def _compact(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


class GTMSlice:
    """Stage thought-leadership and social-content proposals."""

    def __init__(self, ledger: ActionLedger):
        self.ledger = ledger

    def generate_brief(
        self,
        evidence: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> ScopeBrief:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        research_items = _list(evidence.get("research_items") or evidence.get("signals"))
        drafts = _list(evidence.get("drafts"))
        performance_notes = _list(evidence.get("performance_notes"))

        if not research_items and not drafts:
            return ScopeBrief(
                scope="gtm",
                title="GTM",
                status="quiet",
                priority="low",
                text="GTM: no fresh content signal supplied. No post is staged.",
            )

        source = _dict(drafts[0] if drafts else research_items[0])
        platforms = source.get("platforms") or evidence.get("platforms") or ["LinkedIn", "X"]
        platform_text = "/".join(str(platform) for platform in platforms)
        thesis = _compact(source.get("thesis") or source.get("summary"), "the idea is promising but needs a sharper thesis")
        angle = _compact(source.get("angle"), "turn the signal into a direct founder POV")
        title = _compact(source.get("title") or source.get("topic"), "thought-leadership draft")
        image_direction = _compact(source.get("image_direction"), "clean diagram or sharp visual, not decorative filler")
        evidence_ids = [str(item) for item in source.get("evidence_ids") or []]

        action = self.ledger.add_action(
            scope="GTM",
            summary=f"Review {platform_text} post draft: {title}",
            evidence_ids=evidence_ids,
            allowed_next_actions=["revise", "approve_publish", "discard"],
            status="staged",
            risk_class="medium",
            now=now,
            executor_state={
                "mutation_type": "social_publish",
                "provider": "xai-oauth",
                "mutation_status": "draft_only",
                "publishing_blocked_until": "explicit_signal_approval",
                "platforms": list(platforms),
            },
        )

        lines = [
            "GTM: I found a content angle worth drafting.",
            "",
            f"Topic: {title}.",
            f"Thesis: {thesis}.",
            f"Angle: {angle}.",
            f"Image direction: {image_direction}.",
        ]
        if performance_notes:
            lines.append(f"Performance bias: {_compact(performance_notes[0], 'keep it direct and specific')}.")
        lines.extend(
            [
                "",
                f"[{action.handle}] Review the {platform_text} draft and image direction. Nothing is published.",
            ]
        )
        return ScopeBrief(
            scope="gtm",
            title="GTM",
            text="\n".join(lines),
            priority="normal",
            actions=[action],
            evidence_ids=evidence_ids,
        )
