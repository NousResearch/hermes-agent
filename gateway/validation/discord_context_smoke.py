"""Live Discord smoke validation for Hermes startup context.

This module intentionally uses Discord's REST API instead of a gateway client:
it posts a real validation message, polls channel history for Hermes' response,
then requests a zero-inference `/context-dump` attachment.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests  # type: ignore[import-untyped]


DISCORD_API = "https://discord.com/api/v10"
DEFAULT_CHANNEL_ID = "1501008202630696981"


@dataclass
class SmokeReport:
    ok: bool
    channel_id: str
    bot_user_id: str
    sent_message_id: str | None = None
    response_message_id: str | None = None
    context_dump_message_id: str | None = None
    context_dump_attachment: str | None = None
    context_dump_path: str | None = None
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


class DiscordRest:
    def __init__(self, token: str, *, token_kind: str = "bot") -> None:
        prefix = "Bot " if token_kind == "bot" else ""
        self.headers = {
            "Authorization": f"{prefix}{token}",
            "Content-Type": "application/json",
        }

    def send_message(self, channel_id: str, content: str) -> dict[str, Any]:
        response = requests.post(
            f"{DISCORD_API}/channels/{channel_id}/messages",
            headers=self.headers,
            json={"content": content, "allowed_mentions": {"parse": ["users"]}},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    def channel_messages_after(
        self, channel_id: str, after_message_id: str
    ) -> list[dict[str, Any]]:
        response = requests.get(
            f"{DISCORD_API}/channels/{channel_id}/messages",
            headers=self.headers,
            params={"after": after_message_id, "limit": 50},
            timeout=15,
        )
        response.raise_for_status()
        return list(reversed(response.json()))

    def download(self, url: str, path: Path) -> None:
        response = requests.get(
            url, headers={"Authorization": self.headers["Authorization"]}, timeout=30
        )
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)


def run_smoke(
    *,
    channel_id: str,
    bot_user_id: str,
    token: str,
    token_kind: str,
    timeout_seconds: int,
    output_dir: Path,
) -> SmokeReport:
    client = DiscordRest(token, token_kind=token_kind)
    report = SmokeReport(ok=False, channel_id=channel_id, bot_user_id=bot_user_id)

    stamp = int(time.time())
    ping = f"<@{bot_user_id}> hermes context smoke {stamp}: say hi and include your route banner."
    sent = client.send_message(channel_id, ping)
    report.sent_message_id = str(sent.get("id") or "")

    response = _wait_for_bot_message(
        client,
        channel_id=channel_id,
        bot_user_id=bot_user_id,
        after_message_id=report.sent_message_id,
        timeout_seconds=timeout_seconds,
    )
    if not response:
        report.errors.append(
            "Timed out waiting for Hermes response to validation ping."
        )
        report.checks["ping_response"] = False
        return report
    report.response_message_id = str(response.get("id") or "")
    content = str(response.get("content") or "")
    report.checks["ping_response"] = True
    report.checks["route_banner_mentions_model"] = (
        "using" in content.lower() and "openrouter" in content.lower()
    )
    report.checks["context_tokens_visible"] = (
        "context" in content.lower() and "token" in content.lower()
    )
    report.checks["route_banner_has_base_context_version"] = (
        "gateway-context-v1" in content
    )
    report.checks["route_banner_has_loaded_skill_names"] = "skills:" in content.lower()

    dump_request = client.send_message(channel_id, f"/context-dump <@{bot_user_id}>")
    dump_response = _wait_for_bot_message(
        client,
        channel_id=channel_id,
        bot_user_id=bot_user_id,
        after_message_id=str(dump_request.get("id") or ""),
        timeout_seconds=timeout_seconds,
    )
    if not dump_response:
        report.errors.append("Timed out waiting for /context-dump response.")
        report.checks["context_dump_response"] = False
        return report
    report.context_dump_message_id = str(dump_response.get("id") or "")
    report.checks["context_dump_response"] = True

    attachment = _first_context_dump_attachment(dump_response)
    if attachment is None:
        report.errors.append(
            "Hermes /context-dump response did not include a .message.txt attachment."
        )
        report.checks["context_dump_attachment"] = False
        return report
    report.context_dump_attachment = str(attachment.get("filename") or "")
    report.checks["context_dump_attachment"] = True

    dump_path = output_dir / report.context_dump_attachment
    client.download(str(attachment.get("url")), dump_path)
    report.context_dump_path = str(dump_path)
    text = dump_path.read_text(encoding="utf-8", errors="replace")
    report.checks["dump_has_soul"] = "SOUL" in text or "soul" in text
    report.checks["dump_has_layers"] = "## Context Layers" in text
    report.checks["dump_has_estimated_tokens"] = "Estimated total tokens:" in text
    report.checks["dump_has_raw_api_messages"] = "## Raw API Messages" in text
    report.checks["dump_has_tool_schemas"] = "## Tool Schemas" in text
    report.checks["dump_has_base_context_version"] = "base_context_version" in text
    report.checks["dump_has_loaded_skill_names"] = "loaded_skill_names" in text
    report.checks["dump_has_source_metadata"] = '"source"' in text

    report.ok = all(report.checks.values()) and not report.errors
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run live Discord validation against Hermes."
    )
    parser.add_argument(
        "--channel-id",
        default=os.getenv("HERMES_CONTEXT_VALIDATION_CHANNEL_ID", DEFAULT_CHANNEL_ID),
    )
    parser.add_argument("--bot-user-id", default=os.getenv("HERMES_BOT_USER_ID", ""))
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.getenv(
                "HERMES_CONTEXT_VALIDATION_OUTPUT_DIR", "/tmp/hermes-context-smoke"
            )
        ),
    )
    parser.add_argument(
        "--token", default=os.getenv("HERMES_CONTEXT_VALIDATION_DISCORD_TOKEN", "")
    )
    parser.add_argument(
        "--token-kind",
        choices=("bot", "bearer"),
        default=os.getenv("HERMES_CONTEXT_VALIDATION_TOKEN_KIND", "bot"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    errors = []
    if not args.bot_user_id:
        errors.append("Missing --bot-user-id or HERMES_BOT_USER_ID.")
    if not args.token:
        errors.append("Missing --token or HERMES_CONTEXT_VALIDATION_DISCORD_TOKEN.")
    report = SmokeReport(
        ok=False, channel_id=str(args.channel_id), bot_user_id=str(args.bot_user_id)
    )
    if errors:
        report.errors.extend(errors)
        report.write(args.output_dir / "discord_context_smoke_report.json")
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return 2

    report = run_smoke(
        channel_id=str(args.channel_id),
        bot_user_id=str(args.bot_user_id),
        token=str(args.token),
        token_kind=str(args.token_kind),
        timeout_seconds=int(args.timeout_seconds),
        output_dir=args.output_dir,
    )
    report.write(args.output_dir / "discord_context_smoke_report.json")
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0 if report.ok else 1


def _wait_for_bot_message(
    client: DiscordRest,
    *,
    channel_id: str,
    bot_user_id: str,
    after_message_id: str,
    timeout_seconds: int,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        for message in client.channel_messages_after(channel_id, after_message_id):
            author = message.get("author") or {}
            if str(author.get("id") or "") == str(bot_user_id):
                return message
        time.sleep(2.0)
    return None


def _first_context_dump_attachment(message: dict[str, Any]) -> dict[str, Any] | None:
    for attachment in message.get("attachments") or []:
        filename = str(attachment.get("filename") or "")
        if filename.endswith(".message.txt"):
            return attachment
    return None


if __name__ == "__main__":
    raise SystemExit(main())
