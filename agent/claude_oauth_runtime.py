"""Official Claude Code OAuth runtime for the ``claude-oauth`` provider.

This path is deliberately fail-closed: it strips API billing credentials and
accepts only a live first-party Claude.ai subscription session. It never calls
the Anthropic Messages API directly and has no provider fallback.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any


class ClaudeOAuthError(RuntimeError):
    """The subscription-only Claude runtime could not complete a request."""


@dataclass(frozen=True)
class ClaudeOAuthResult:
    text: str
    usage: dict[str, Any]
    raw: dict[str, Any]


def subscription_environment() -> dict[str, str]:
    """Return a child environment that cannot select Anthropic PAYG auth."""
    env = dict(os.environ)
    forbidden = {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "CLAUDE_CODE_USE_ANTHROPIC_AWS",
        "ANTHROPIC_AWS_WORKSPACE_ID",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_TENANT_ID",
    }
    for key in forbidden:
        env.pop(key, None)
    return env


def _run(command: list[str], *, cli_path: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [cli_path, *command], input=input_text, text=True,
            capture_output=True, env=subscription_environment(), timeout=900,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ClaudeOAuthError(f"claude-oauth runtime unavailable: {exc}") from exc


def verify_subscription_auth(cli_path: str | None = None) -> dict[str, Any]:
    cli = cli_path or shutil.which("claude")
    if not cli:
        raise ClaudeOAuthError("claude-oauth requires the Claude Code CLI")
    proc = _run(["auth", "status"], cli_path=cli)
    try:
        status = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ClaudeOAuthError("claude-oauth could not read Claude auth status") from exc
    if (
        proc.returncode != 0
        or status.get("loggedIn") is not True
        or status.get("authMethod") != "claude.ai"
        or status.get("apiProvider") != "firstParty"
    ):
        detail = (proc.stderr or proc.stdout or "not logged in").strip()
        raise ClaudeOAuthError(
            "claude-oauth requires first-party Claude.ai subscription OAuth; "
            f"no paid API fallback was attempted ({detail})"
        )
    return status


def _render_messages(messages: list[dict[str, Any]]) -> str:
    parts = ["Continue this conversation. Follow the SYSTEM instructions exactly."]
    for message in messages:
        role = str(message.get("role") or "user").upper()
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        parts.append(f"\n<{role}>\n{content}\n</{role}>")
    return "\n".join(parts)


def run_claude_oauth(*, model: str, messages: list[dict[str, Any]], cli_path: str | None = None) -> ClaudeOAuthResult:
    """Run one text-only turn through official ``claude -p`` subscription auth.

    Claude built-in tools are disabled. This MVP therefore returns text only;
    Hermes tool calls are not available on this dedicated early-return path.
    """
    cli = cli_path or shutil.which("claude")
    if not cli:
        raise ClaudeOAuthError("claude-oauth requires the Claude Code CLI")
    verify_subscription_auth(cli)
    proc = _run(
        ["-p", "", "--model", model, "--tools", "", "--max-turns", "1",
         "--no-session-persistence", "--output-format", "json"],
        cli_path=cli, input_text=_render_messages(messages),
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "Claude OAuth request failed").strip()
        raise ClaudeOAuthError(f"claude-oauth failed; no fallback attempted: {detail}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ClaudeOAuthError("claude-oauth returned invalid JSON; no fallback attempted") from exc
    if payload.get("is_error") or payload.get("subtype") != "success":
        detail = payload.get("result") or payload.get("error") or "unknown Claude error"
        raise ClaudeOAuthError(f"claude-oauth failed; no fallback attempted: {detail}")
    text = payload.get("result")
    if not isinstance(text, str) or not text:
        raise ClaudeOAuthError("claude-oauth returned no response; no fallback attempted")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return ClaudeOAuthResult(text=text, usage=usage, raw=payload)


def run_claude_oauth_turn(agent, *, messages: list[dict[str, Any]], original_user_message: Any = None) -> dict[str, Any]:
    """Hermes early-return adapter for one official Claude OAuth text turn."""
    outbound = list(messages)
    system_prompt = getattr(agent, "_cached_system_prompt", None)
    if isinstance(system_prompt, str) and system_prompt:
        outbound = [{"role": "system", "content": system_prompt}, *outbound]
    result = run_claude_oauth(model=agent.model, messages=outbound)
    messages.append({"role": "assistant", "content": result.text})
    try:
        agent._flush_messages_to_session_db(messages)
    except Exception:
        pass
    try:
        agent._sync_external_memory_for_turn(
            original_user_message=original_user_message,
            final_response=result.text,
            interrupted=False,
            messages=messages,
        )
    except Exception:
        pass
    return {
        "final_response": result.text,
        "messages": messages,
        "api_calls": 1,
        "completed": True,
        "partial": False,
        "error": None,
        "agent_persisted": True,
        "usage": result.usage,
    }
