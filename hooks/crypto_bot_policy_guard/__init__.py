from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any


DEFAULT_MODE = "observe"
BLOCKING_MODES = {"block", "test"}

SECRET_VALUE = re.compile(
    r"(?i)\b(api[_-]?key|token|password|passwd|secret|credential)"
    r"\s*[:=]\s*['\"]?([A-Za-z0-9_./+=:-]{8,})"
)
PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)
BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}")
TELEGRAM_TOKEN = re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{20,}\b")


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    reason: str


RULES = (
    Rule(
        "ruff_format",
        re.compile(r"\bruff\s+format\b"),
        "ruff format is blocked",
    ),
    Rule("git_push", re.compile(r"\bgit\s+push\b"), "git push is blocked"),
    Rule("gh_pr", re.compile(r"\bgh\s+pr\b"), "gh pr commands are blocked"),
    Rule(
        "gitea_write",
        re.compile(r"\b(curl|http)\b.*\b(POST|PUT|PATCH|DELETE)\b.*?/api/v1", re.I),
        "Gitea write API calls are blocked",
    ),
    Rule(
        "workflow_start",
        re.compile(
            r"\b(workflow_dispatch|act|gitea\s+actions|actions/runners|"
            r"runner\s+(start|register|run))\b",
            re.I,
        ),
        "workflow and runner starts are blocked",
    ),
    Rule(
        "workflow_edit",
        re.compile(r"\.gitea/workflows"),
        "workflow file edits are blocked",
    ),
    Rule(
        "service_start",
        re.compile(
            r"\b(launchctl\s+(bootstrap|kickstart|load)|systemctl\s+start|"
            r"brew\s+services\s+start|docker\s+compose\s+up|docker\s+run|"
            r"kubectl\s+apply|flux\s+reconcile|uvicorn|gunicorn|"
            r"npm\s+(run\s+)?start|qmd)\b",
            re.I,
        ),
        "service starts are blocked",
    ),
    Rule(
        "branch_rewrite",
        re.compile(
            r"\bgit\s+(merge|rebase|reset\s+--hard|commit\s+--amend|"
            r"cherry-pick|push\s+--force|branch\s+-D)\b"
        ),
        "branch rewrite or merge commands are blocked",
    ),
    Rule(
        "blocked_writer",
        re.compile(
            r"\b(apply_approved_write_plan|branch_local_writer|"
            r"execute_forge_issue_create)\.py\b"
        ),
        "blocked writer/executor script is not allowed",
    ),
    Rule(
        "secret_read",
        re.compile(
            r"\b(cat|less|more|head|tail|sed|grep|rg|open|sqlite3)\b.*"
            r"(\.env|token|credential|cookie|keychain|private[_-]?key|"
            r"oauth|\.git-credentials|runtime\.db)",
            re.I,
        ),
        "secret or credential reads are blocked",
    ),
    Rule(
        "trading_api",
        re.compile(
            r"\b(robinhood[_-]?api|robinhood|broker|exchange|live-market|"
            r"account|order|position|wallet|trading|financial)\w*.*"
            r"\b(api|curl|http|request|client|order)\b",
            re.I,
        ),
        "broker, trading, or financial API commands are blocked",
    ),
    Rule(
        "deploy",
        re.compile(
            r"\b(deploy|release|gitops|harbor|openbao|rabbitmq|"
            r"redis-server|temporal)\b",
            re.I,
        ),
        "deployment or production service commands are blocked",
    ),
)


def mode() -> str:
    value = os.environ.get("HERMES_CRYPTO_BOT_POLICY_GUARD_MODE", DEFAULT_MODE)
    return value.strip().lower() or DEFAULT_MODE


def redact_sensitive(text: str) -> str:
    text = PRIVATE_KEY_BLOCK.sub("[REDACTED_PRIVATE_KEY]", text)
    text = BEARER.sub("Bearer [REDACTED]", text)
    text = TELEGRAM_TOKEN.sub("[REDACTED_TELEGRAM_TOKEN]", text)
    return SECRET_VALUE.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)


def command_from_tool(tool_name: str, args: dict[str, Any]) -> str:
    if not isinstance(args, dict):
        return ""
    for key in ("command", "cmd", "input", "script"):
        value = args.get(key)
        if isinstance(value, str):
            return value
    if tool_name in {"read_file", "open_file"}:
        value = args.get("path") or args.get("file")
        if isinstance(value, str):
            return f"read_file {value}"
    return ""


def evaluate_command(command: str) -> dict[str, Any]:
    matches = [
        {"rule": rule.name, "reason": rule.reason}
        for rule in RULES
        if rule.pattern.search(command)
    ]
    return {
        "blocked": bool(matches),
        "matches": matches,
        "command": command,
    }


def block_message(decision: dict[str, Any]) -> str:
    reasons = ", ".join(match["reason"] for match in decision["matches"])
    return f"crypto_bot policy guard blocked command: {reasons}"


def pre_tool_call(
    tool_name: str = "",
    args: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, str] | None:
    command = command_from_tool(tool_name, args or {})
    if not command:
        return None
    decision = evaluate_command(command)
    if not decision["blocked"]:
        return None
    if mode() in BLOCKING_MODES:
        return {"action": "block", "message": block_message(decision)}
    return {"action": "observe", "message": block_message(decision)}


def transform_terminal_output(command: str, output: str, **_: Any) -> str | None:
    redacted = redact_sensitive(output)
    return redacted if redacted != output else None


def transform_tool_result(tool_name: str, result: str, **_: Any) -> str | None:
    _ = tool_name
    redacted = redact_sensitive(result)
    return redacted if redacted != result else None


def register(ctx: Any) -> None:
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("transform_terminal_output", transform_terminal_output)
    ctx.register_hook("transform_tool_result", transform_tool_result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument(
        "--mode",
        choices=sorted({DEFAULT_MODE, *BLOCKING_MODES}),
        default=DEFAULT_MODE,
    )
    parser.add_argument("--format", choices=("json",), default="json")
    args = parser.parse_args(argv)

    os.environ["HERMES_CRYPTO_BOT_POLICY_GUARD_MODE"] = args.mode
    decision = evaluate_command(args.command)
    payload = {
        "schema": "hermes.autonomy.crypto_bot_policy_guard.v1",
        "mode": args.mode,
        **decision,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if args.mode in BLOCKING_MODES and decision["blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
