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
    mode: str = "live"
    sent_message_id: str | None = None
    response_message_id: str | None = None
    context_dump_message_id: str | None = None
    context_dump_attachment: str | None = None
    context_dump_path: str | None = None
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

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


def run_internal_validation(*, channel_id: str, output_dir: Path) -> SmokeReport:
    """Validate Hermes context composition without Discord transport."""

    from gateway.context_dump import write_context_dump_text
    from gateway.context_layers import ContextLayer
    from gateway.route_banner import (
        BASE_CONTEXT_VERSION,
        format_context_window_token_line,
    )
    from gateway.startup_context import render_reply_chain_payload
    from gateway.tool_policy import (
        select_gateway_toolsets,
        should_use_lightweight_discord_context,
    )

    report = SmokeReport(
        ok=False,
        channel_id=channel_id,
        bot_user_id="internal",
        mode="internal",
    )
    configured_toolsets = {
        "terminal",
        "file",
        "browser",
        "skills",
        "memory",
        "session_search",
        "clarify",
        "chat",
        "channel-notes",
        "budget",
    }
    selected = select_gateway_toolsets(
        platform="discord",
        configured_toolsets=configured_toolsets,
        user_text="what is the 6th highest mountain?",
    )
    route_line = format_context_window_token_line(
        approx_tokens=8106,
        context_length=128000,
        threshold_tokens=102400,
        loaded_skill_names=[],
    )
    reply_chain = render_reply_chain_payload(
        {
            "reply_chain": [
                {
                    "timestamp": "2026-06-21T00:00:00Z",
                    "author": "root",
                    "id": "1",
                    "content": "root message",
                },
                {
                    "timestamp": "2026-06-21T00:00:01Z",
                    "author": "parent",
                    "id": "2",
                    "content": "parent message",
                },
            ]
        }
    )
    payload = {
        "schema": "hermes.context_dump.v2",
        "capture_mode": "internal_context_validation",
        "base_context_version": BASE_CONTEXT_VERSION,
        "loaded_skill_names": [],
        "session_key": "discord:internal",
        "session_id": "internal",
        "estimated_total_tokens": 8106,
        "source": {"platform": "discord", "chat_id": channel_id},
        "context_layers": [
            ContextLayer.from_text("soul", source="SOUL.md", text="SOUL").to_payload(),
            ContextLayer.from_text(
                "startup_discord_context",
                source="gateway.startup_context",
                text=reply_chain,
            ).to_payload(),
        ],
        "api_messages": [{"role": "system", "content": "SOUL"}],
        "tools": [{"type": "function", "function": {"name": "skills_list"}}],
    }
    dump_path = write_context_dump_text(output_dir, "discord:internal", payload)
    dump_text = dump_path.read_text(encoding="utf-8", errors="replace")

    report.context_dump_path = str(dump_path)
    report.checks = {
        "internal_context_validation": True,
        "lightweight_discord_context": should_use_lightweight_discord_context(
            platform="discord",
            user_text="what is the 6th highest mountain?",
        ),
        "lightweight_tools_minimal": "terminal" not in selected
        and "skills" in selected,
        "route_banner_has_base_context_version": BASE_CONTEXT_VERSION in route_line,
        "route_banner_has_loaded_skill_names": "skills:" in route_line.lower(),
        "dump_has_soul": "SOUL" in dump_text,
        "dump_has_layers": "## Context Layers" in dump_text,
        "dump_has_raw_api_messages": "## Raw API Messages" in dump_text,
        "dump_has_tool_schemas": "## Tool Schemas" in dump_text,
        "dump_has_estimated_tokens": "Estimated total tokens:" in dump_text,
        "reply_chain_root_to_leaf": reply_chain.find("root message")
        < reply_chain.find("parent message"),
    }
    report.ok = all(report.checks.values())
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
    parser.add_argument(
        "--mode",
        choices=("auto", "internal", "live"),
        default=os.getenv("HERMES_CONTEXT_VALIDATION_MODE", "auto"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = str(args.mode)
    if mode == "internal":
        report = run_internal_validation(
            channel_id=str(args.channel_id),
            output_dir=args.output_dir,
        )
        report.write(args.output_dir / "discord_context_smoke_report.json")
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return 0 if report.ok else 1

    errors = []
    if not args.bot_user_id:
        errors.append("Missing --bot-user-id or HERMES_BOT_USER_ID.")
    if not args.token:
        errors.append("Missing --token or HERMES_CONTEXT_VALIDATION_DISCORD_TOKEN.")
    report = SmokeReport(
        ok=False,
        channel_id=str(args.channel_id),
        bot_user_id=str(args.bot_user_id),
        mode=mode,
    )
    if errors:
        if mode == "auto":
            report = run_internal_validation(
                channel_id=str(args.channel_id),
                output_dir=args.output_dir,
            )
            report.mode = "auto"
            report.checks["live_discord_skipped"] = True
            report.warnings.append(
                "Missing live Discord credentials; ran internal Hermes validation only."
            )
            report.write(args.output_dir / "discord_context_smoke_report.json")
            print(json.dumps(asdict(report), indent=2, sort_keys=True))
            return 0 if report.ok else 1
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
    report.mode = mode
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
