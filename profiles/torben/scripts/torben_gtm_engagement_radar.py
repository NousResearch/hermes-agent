#!/usr/bin/env python3
"""Run Torben's GTM response-opportunity radar."""

from __future__ import annotations

import json
import os
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_cli.signal_coo.action_ledger import ActionLedger
from hermes_cli.signal_coo.gtm_engagement_radar import (
    DEFAULT_MAX_OPPORTUNITIES,
    DEFAULT_MAX_TOPICS,
    run_gtm_engagement_radar,
    write_gtm_engagement_artifacts,
)
from hermes_cli.signal_coo.gtm_radar_adapter import DEFAULT_MAGNUS_RADAR_PATH, load_magnus_gtm_radar


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except ValueError:
        return default


def main() -> int:
    home = get_hermes_home()
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    preview = _truthy(os.getenv("TORBEN_GTM_ENGAGEMENT_PREVIEW"))
    radar_path = Path(os.getenv("TORBEN_GTM_ENGAGEMENT_RADAR_PATH") or DEFAULT_MAGNUS_RADAR_PATH)

    try:
        radar = load_magnus_gtm_radar(radar_path)
        payload = run_gtm_engagement_radar(
            radar,
            ledger=ActionLedger(state_dir / "torben-action-ledger.json"),
            state_path=state_dir / "torben-gtm-engagement-radar-state.json",
            max_topics=_env_int("TORBEN_GTM_ENGAGEMENT_MAX_TOPICS", DEFAULT_MAX_TOPICS),
            max_opportunities=_env_int("TORBEN_GTM_ENGAGEMENT_MAX_OPPORTUNITIES", DEFAULT_MAX_OPPORTUNITIES),
            mark_delivered=not preview,
            stage_actions=not preview,
        )
    except Exception as exc:  # noqa: BLE001
        payload = {
            "task": "torben_gtm_engagement_radar",
            "wakeAgent": True,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc)[:300],
            },
            "public_actions_taken": 0,
            "external_mutations": 0,
            "text": (
                "Torben / GTM Response Radar\n\n"
                "Grok/X response-opportunity scan failed before it could produce a useful brief.\n"
                f"Reason: {type(exc).__name__}: {str(exc)[:180]}\n"
                "Nothing has been posted, replied to publicly, scheduled, or sent.\n"
            ),
        }

    write_gtm_engagement_artifacts(
        payload,
        json_path=state_dir / "torben-gtm-engagement-radar-latest.json",
        text_path=state_dir / "torben-gtm-engagement-radar-latest.txt",
    )
    if payload.get("wakeAgent") and payload.get("text"):
        print(str(payload["text"]), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
