#!/usr/bin/env python3
"""Morning brief delivery script for Discord.

No-agent cron job: reads config, composes a brief from contract files,
and posts it to the configured Discord channel.

Config keys (config.yaml):
  discord.morning_brief_channel_id   — target channel (required)
  discord.morning_brief_contracts    — list of file paths to include (optional)

Required env var (.env):
  DISCORD_BOT_TOKEN   — Discord bot token with Send Messages permission

Dry-run (no Discord post, prints to stdout):
  MORNING_BRIEF_DRY_RUN=1

Exit codes:
  0 — delivered (or dry-run printed) successfully
  1 — missing required configuration (channel ID or bot token)
  2 — failed to read one or more contract files
  3 — Discord API delivery failure
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so hermes_cli.config is importable
# whether the script is invoked directly or via cron's subprocess runner.
_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hermes_cli.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

DRY_RUN = os.getenv("MORNING_BRIEF_DRY_RUN", "").strip() in ("1", "true", "yes")


def _load_config() -> dict:
    try:
        return load_config() or {}
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)


def _get_channel_id(cfg: dict) -> str:
    channel_id = str((cfg.get("discord") or {}).get("morning_brief_channel_id") or "").strip()
    if not channel_id:
        logger.error(
            "Missing required config key: discord.morning_brief_channel_id. "
            "Add it to ~/.hermes/config.yaml, e.g.:\n"
            "  discord:\n"
            "    morning_brief_channel_id: '1234567890'"
        )
        sys.exit(1)
    return channel_id


def _get_contract_paths(cfg: dict) -> list[str]:
    raw = (cfg.get("discord") or {}).get("morning_brief_contracts") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(p).strip() for p in raw if str(p).strip()]


def _load_contracts(paths: list[str]) -> str:
    if not paths:
        return ""
    parts: list[str] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            logger.error("Contract file not found: %s", path)
            sys.exit(2)
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                parts.append(text)
        except Exception as exc:
            logger.error("Failed to read contract file %s: %s", path, exc)
            sys.exit(2)
    return "\n\n---\n\n".join(parts)


def _compose_brief(contract_content: str) -> str:
    if contract_content:
        return f"Good morning! Here is your morning brief:\n\n{contract_content}"
    return "Good morning! Your morning brief is ready. Have a great day! 🌅"


def _post_to_discord(channel_id: str, message: str) -> None:
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        logger.error(
            "Missing required env var: DISCORD_BOT_TOKEN. "
            "Set it in ~/.hermes/.env"
        )
        sys.exit(1)

    try:
        import httpx
    except ImportError:
        logger.error(
            "httpx is required for Discord delivery. "
            "Install it with: pip install 'httpx>=0.28.1,<1'"
        )
        sys.exit(3)

    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }
    payload = {"content": message}

    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=30)
    except httpx.TimeoutException as exc:
        logger.error("Discord delivery timed out: %s", exc)
        sys.exit(3)
    except httpx.NetworkError as exc:
        logger.error("Discord delivery network error: %s", exc)
        sys.exit(3)

    if response.status_code == 401:
        logger.error(
            "Discord authentication failed (401). Check DISCORD_BOT_TOKEN "
            "in ~/.hermes/.env — the token may be invalid or revoked."
        )
        sys.exit(3)

    if response.status_code == 404:
        logger.error(
            "Discord channel not found (404). Verify that channel ID %r is "
            "correct and that the bot has access to it.",
            channel_id,
        )
        sys.exit(3)

    if not response.is_success:
        logger.error(
            "Discord delivery failed: HTTP %d — %s",
            response.status_code,
            response.text[:200],
        )
        sys.exit(3)

    logger.info("Morning brief delivered to Discord channel %s", channel_id)


def main() -> None:
    cfg = _load_config()
    channel_id = _get_channel_id(cfg)
    contract_paths = _get_contract_paths(cfg)
    contract_content = _load_contracts(contract_paths)
    brief = _compose_brief(contract_content)

    if DRY_RUN:
        print(f"[DRY RUN] Would post to Discord channel {channel_id}:")
        print()
        print(brief)
        return

    _post_to_discord(channel_id, brief)
    # Print delivered content to stdout — the scheduler saves this as job output.
    print(brief)


if __name__ == "__main__":
    main()
