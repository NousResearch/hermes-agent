from __future__ import annotations

import json
import os
from pathlib import Path

from hermes_constants import get_hermes_home
from hermes_cli.signal_coo.gmail_realtime import DEFAULT_SUBSCRIPTION_NAME, process_pubsub_pull, write_json

DEFAULT_TORBEN_HOME = Path("/Users/ericfreeman/.hermes/profiles/torben")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _torben_home() -> Path:
    if os.getenv("HERMES_HOME"):
        return get_hermes_home()
    return DEFAULT_TORBEN_HOME


def main() -> int:
    home = _torben_home()
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    config_path = home / "config" / "google_accounts.yaml"
    relationship_context_path = home / "config" / "relationship_context.yaml"
    state_path = state_dir / "torben-gmail-watch-state.json"
    output_path = state_dir / "torben-gmail-pubsub-pull-latest.json"

    payload = process_pubsub_pull(
        config_path=config_path,
        relationship_context_path=relationship_context_path,
        state_path=state_path,
        subscription_name=os.getenv("TORBEN_GMAIL_PUBSUB_SUBSCRIPTION", DEFAULT_SUBSCRIPTION_NAME),
        limit=int(os.getenv("TORBEN_GMAIL_PUBSUB_PULL_LIMIT", "10")),
        max_history_pages=int(os.getenv("TORBEN_GMAIL_HISTORY_MAX_PAGES", "10")),
        history_rate_limit_retries=int(os.getenv("TORBEN_GMAIL_HISTORY_RATE_LIMIT_RETRIES", "0")),
        history_rate_limit_backoff_seconds=float(os.getenv("TORBEN_GMAIL_HISTORY_RATE_LIMIT_BACKOFF_SECONDS", "1.0")),
        history_rate_limit_max_sleep_seconds=float(os.getenv("TORBEN_GMAIL_HISTORY_RATE_LIMIT_MAX_SLEEP_SECONDS", "30.0")),
        history_rate_limit_jitter_seconds=float(os.getenv("TORBEN_GMAIL_HISTORY_RATE_LIMIT_JITTER_SECONDS", "0.5")),
        history_rate_limit_cooldown_seconds=int(os.getenv("TORBEN_GMAIL_HISTORY_RATE_LIMIT_COOLDOWN_SECONDS", "3600")),
        max_messages_per_account=int(os.getenv("TORBEN_GMAIL_PUBSUB_MAX_MESSAGES", "40")),
        max_body_fetches_per_account=int(os.getenv("TORBEN_GMAIL_PUBSUB_MAX_BODY_FETCHES", "20")),
        max_realtime_age_seconds=int(os.getenv("TORBEN_GMAIL_PUBSUB_MAX_MESSAGE_AGE_SECONDS", "7200")),
        fetch_workers=int(os.getenv("TORBEN_GMAIL_PUBSUB_WORKERS", "6")),
        preview=_truthy(os.getenv("TORBEN_GMAIL_PUBSUB_PREVIEW")),
    )
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    from torben_job_contract import run_job

    raise SystemExit(run_job("torben-gmail-pubsub-pull", main))
