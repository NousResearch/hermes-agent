#!/usr/bin/env python3
"""Morning brief delivery script for Discord.

No-agent cron job: reads config, composes the four-section morning brief
via scripts.morning_brief_composer, and posts it to the configured Discord
channel.

Config keys (config.yaml):
  discord.morning_brief_channel_id   — target channel (required)
  discord.morning_brief_contracts    — optional list of exactly three
                                        contract file paths (journal,
                                        perfcoach, commander — matched by
                                        filename). When absent, or when it
                                        does not contain exactly three
                                        paths that map unambiguously, the
                                        composer's own defaults (and env
                                        vars) are used instead.

Required env var (.env):
  DISCORD_BOT_TOKEN   — Discord bot token with Send Messages permission

Dry-run (no Discord post, prints to stdout):
  MORNING_BRIEF_DRY_RUN=1

Exit codes:
  0 — delivered (or dry-run printed) successfully
  1 — missing required configuration (channel ID or bot token)
  2 — reserved. The composer degrades gracefully on missing/stale
      contracts (each section renders "unavailable" instead), so this
      script no longer exits non-zero for merely-missing contract files.
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

# Imported after logging is configured: the composer module also calls
# logging.basicConfig at import time, and basicConfig is a no-op once a
# handler is already installed — importing it after our own call keeps this
# script's INFO-level logging intact.
from scripts.morning_brief_composer import (
    DEFAULT_COMMANDER_PATH,
    DEFAULT_JOURNAL_PATH,
    DEFAULT_PERFCOACH_PATH,
    compose_brief_from_paths,
)

DRY_RUN = os.getenv("MORNING_BRIEF_DRY_RUN", "").strip() in ("1", "true", "yes")

EXIT_OK = 0
EXIT_CONFIG_MISSING = 1
EXIT_CONTRACTS_UNREADABLE = 2  # reserved — see module docstring
EXIT_DISCORD_FAILURE = 3

DISCORD_MAX_CHARS = 1900  # Discord's hard cap is 2000; leave headroom.


def _load_config() -> dict:
    try:
        return load_config() or {}
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(EXIT_CONFIG_MISSING)


def _get_channel_id(cfg: dict) -> str:
    channel_id = str((cfg.get("discord") or {}).get("morning_brief_channel_id") or "").strip()
    if not channel_id:
        logger.error(
            "Missing required config key: discord.morning_brief_channel_id. "
            "Add it to ~/.hermes/config.yaml, e.g.:\n"
            "  discord:\n"
            "    morning_brief_channel_id: '1234567890'"
        )
        sys.exit(EXIT_CONFIG_MISSING)
    return channel_id


def _get_contract_paths(cfg: dict) -> list[str]:
    raw = (cfg.get("discord") or {}).get("morning_brief_contracts") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(p).strip() for p in raw if str(p).strip()]


def _resolve_contract_paths(cfg: dict) -> tuple[str, str, str]:
    """Resolve the (journal, perfcoach, commander) contract paths.

    discord.morning_brief_contracts, when present, must list exactly three
    paths that map unambiguously to journal/perfcoach/commander by
    filename (e.g. "journal_brief.latest.json"). Otherwise the composer's
    own defaults are used, which already honor JOURNAL_BRIEF_PATH,
    PERFCOACH_BRIEF_PATH, and COMMANDER_REPORT_PATH.
    """
    journal_path = os.environ.get("JOURNAL_BRIEF_PATH", DEFAULT_JOURNAL_PATH)
    perfcoach_path = os.environ.get("PERFCOACH_BRIEF_PATH", DEFAULT_PERFCOACH_PATH)
    commander_path = os.environ.get("COMMANDER_REPORT_PATH", DEFAULT_COMMANDER_PATH)

    configured = _get_contract_paths(cfg)
    if len(configured) == 3:
        mapped: dict[str, str] = {}
        for raw_path in configured:
            name = Path(raw_path).name.lower()
            if "journal" in name:
                mapped["journal"] = raw_path
            elif "perfcoach" in name:
                mapped["perfcoach"] = raw_path
            elif "commander" in name:
                mapped["commander"] = raw_path

        if set(mapped) == {"journal", "perfcoach", "commander"}:
            journal_path = mapped["journal"]
            perfcoach_path = mapped["perfcoach"]
            commander_path = mapped["commander"]
        else:
            logger.warning(
                "discord.morning_brief_contracts has 3 entries but they "
                "could not be mapped to journal/perfcoach/commander by "
                "filename; falling back to defaults."
            )

    return journal_path, perfcoach_path, commander_path


def _build_message(brief_markdown: str) -> str:
    return f"Good morning! Here is your morning brief:\n\n{brief_markdown}"


def _split_into_chunks(text: str, max_chars: int = DISCORD_MAX_CHARS) -> list[str]:
    """Split text into chunks of at most max_chars.

    Breaks only on newline boundaries — never mid-line — except when a
    single line itself exceeds max_chars, in which case that line is
    hard-split at max_chars boundaries.
    """
    chunks: list[str] = []
    current = ""

    for line in text.split("\n"):
        while len(line) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:max_chars])
            line = line[max_chars:]

        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > max_chars:
            chunks.append(current)
            current = line
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def _check_discord_response(response, channel_id: str) -> None:
    if response.status_code == 401:
        logger.error(
            "Discord authentication failed (401). Check DISCORD_BOT_TOKEN "
            "in ~/.hermes/.env — the token may be invalid or revoked."
        )
        sys.exit(EXIT_DISCORD_FAILURE)

    if response.status_code == 404:
        logger.error(
            "Discord channel not found (404). Verify that channel ID %r is "
            "correct and that the bot has access to it.",
            channel_id,
        )
        sys.exit(EXIT_DISCORD_FAILURE)

    if not response.is_success:
        logger.error(
            "Discord delivery failed: HTTP %d — %s",
            response.status_code,
            response.text[:200],
        )
        sys.exit(EXIT_DISCORD_FAILURE)


def _post_to_discord(channel_id: str, message: str) -> None:
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        logger.error(
            "Missing required env var: DISCORD_BOT_TOKEN. "
            "Set it in ~/.hermes/.env"
        )
        sys.exit(EXIT_CONFIG_MISSING)

    try:
        import httpx
    except ImportError:
        logger.error(
            "httpx is required for Discord delivery. "
            "Install it with: pip install 'httpx>=0.28.1,<1'"
        )
        sys.exit(EXIT_DISCORD_FAILURE)

    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    headers = {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
    }

    chunks = _split_into_chunks(message)
    for index, chunk in enumerate(chunks, start=1):
        try:
            response = httpx.post(url, headers=headers, json={"content": chunk}, timeout=30)
        except httpx.TimeoutException as exc:
            logger.error("Discord delivery timed out: %s", exc)
            sys.exit(EXIT_DISCORD_FAILURE)
        except httpx.NetworkError as exc:
            logger.error("Discord delivery network error: %s", exc)
            sys.exit(EXIT_DISCORD_FAILURE)

        _check_discord_response(response, channel_id)
        logger.debug("Delivered chunk %d/%d to Discord channel %s", index, len(chunks), channel_id)

    logger.info(
        "Morning brief delivered to Discord channel %s (%d message(s))",
        channel_id,
        len(chunks),
    )


def main() -> None:
    cfg = _load_config()
    channel_id = _get_channel_id(cfg)
    journal_path, perfcoach_path, commander_path = _resolve_contract_paths(cfg)
    brief_markdown = compose_brief_from_paths(journal_path, perfcoach_path, commander_path)
    message = _build_message(brief_markdown)

    if DRY_RUN:
        print(f"[DRY RUN] Would post to Discord channel {channel_id}:")
        print()
        print(message)
        return

    _post_to_discord(channel_id, message)
    # Print delivered content to stdout — the scheduler saves this as job output.
    print(message)


if __name__ == "__main__":
    main()
