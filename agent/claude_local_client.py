"""Subprocess-based Anthropic client using the official ``claude`` CLI.

When a Claude Max OAuth account routes tools-carrying Bearer requests into the
overage lane (HTTP 400 "out of extra usage"), direct SDK calls fail regardless
of header or payload variation.  The official ``claude`` CLI issues the same
requests from a first-class Claude Code session that Anthropic infrastructure
keeps in the Max lane.

This shim intercepts the ``client.messages.create()`` call that
``_anthropic_messages_create`` makes, translates the Anthropic-format kwargs
into a prompt the claude CLI understands, spawns it as a subprocess, and
returns a SimpleNamespace shaped like an Anthropic Message so that
``AnthropicTransport.normalize_response()`` works unchanged.

Activation: set ``HERMES_ANTHROPIC_CLAUDE_LOCAL=1`` in ``~/.hermes/.env``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from types import SimpleNamespace
from typing import Any

import logging

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 600.0


# ── Public client class ─────────────────────────────────────────────────


class ClaudeLocalClient:
    """Drop-in replacement for the Anthropic SDK client (messages.create path)."""

    def __init__(self, **kwargs):
        self._timeout = float(kwargs.get("timeout") or _DEFAULT_TIMEOUT)

    @property
    def messages(self) -> "ClaudeLocalClient":
        return self

    def create(self, **kwargs) -> SimpleNamespace:
        """Run an Anthropic-format messages call via the claude CLI subprocess."""
        system_blocks = kwargs.get("system") or []
        messages = kwargs.get("messages") or []
        model = kwargs.get("model") or "claude-sonnet-4-6"

        prompt = _build_prompt(system_blocks, messages)

        cmd = [
            "claude",
            "--print",
            "--output-format", "stream-json",
            "--input-format", "stream-json",
            "--no-session-persistence",
            "--verbose",
        ]

        input_payload = json.dumps({
            "type": "user",
            "message": {"role": "user", "content": prompt},
        }) + "\n"

        logger.debug("claude_local: spawning subprocess, prompt length=%d", len(prompt))

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate(
                input=input_payload,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise RuntimeError(
                f"claude local: subprocess timed out after {self._timeout}s"
            )
        except FileNotFoundError:
            raise RuntimeError(
                "claude local: 'claude' CLI not found in PATH. "
                "Install Claude Code (npm install -g @anthropic-ai/claude-code) "
                "and try again."
            )

        result_event = _parse_result_event(stdout)

        if result_event is None or result_event.get("subtype") != "success":
            detail = (stderr or "").strip()[:400] or stdout.strip()[:400]
            raise RuntimeError(f"claude local: subprocess returned no success result. {detail}")

        result_text = result_event.get("result") or ""
        usage_raw = result_event.get("usage") or {}

        logger.debug(
            "claude_local: done, result_len=%d in_tok=%s out_tok=%s",
            len(result_text),
            usage_raw.get("input_tokens"),
            usage_raw.get("output_tokens"),
        )

        return SimpleNamespace(
            id=f"claude-local-{int(time.time())}",
            type="message",
            role="assistant",
            model=model,
            content=[SimpleNamespace(type="text", text=result_text)],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=SimpleNamespace(
                input_tokens=int(usage_raw.get("input_tokens") or 0),
                output_tokens=int(usage_raw.get("output_tokens") or 0),
                cache_creation_input_tokens=int(
                    usage_raw.get("cache_creation_input_tokens") or 0
                ),
                cache_read_input_tokens=int(
                    usage_raw.get("cache_read_input_tokens") or 0
                ),
            ),
        )

    # claude CLI doesn't expose a streaming interface through this shim
    def stream(self, **kwargs):
        raise NotImplementedError(
            "ClaudeLocalClient does not support streaming — "
            "set _disable_streaming or check isinstance before calling."
        )


# ── Prompt formatting ───────────────────────────────────────────────────


def _build_prompt(system_blocks: list, messages: list) -> str:
    """Serialize Anthropic-format system + messages into a single text prompt."""
    parts: list[str] = []

    # System context — skip the huge Hermes system prompt; include only the
    # short Claude Code identity prefix (≤500 chars) so claude knows the role.
    for block in system_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = (block.get("text") or "").strip()
        if not text:
            continue
        if len(text) <= 500:
            parts.append(f"[Context: {text}]")
        break  # only first block

    # Conversation history
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or ""
        rendered = _render_content(msg.get("content"))
        if not rendered.strip():
            continue
        label = "Human" if role == "user" else "Assistant" if role == "assistant" else role.title()
        parts.append(f"{label}: {rendered}")

    return "\n\n".join(p for p in parts if p.strip())


def _render_content(content: Any) -> str:
    """Render an Anthropic content value (str or list of blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content).strip()

    rendered: list[str] = []
    for block in content:
        if isinstance(block, str):
            rendered.append(block.strip())
            continue
        if not isinstance(block, dict):
            continue
        btype = block.get("type") or ""
        if btype == "text":
            text = (block.get("text") or "").strip()
            if text:
                rendered.append(text)
        elif btype == "tool_use":
            name = block.get("name") or "?"
            inp = block.get("input") or {}
            rendered.append(f"[Tool call: {name}({json.dumps(inp, ensure_ascii=False)[:300]})]")
        elif btype == "tool_result":
            result_content = block.get("content") or ""
            if isinstance(result_content, list):
                result_text = " ".join(
                    r.get("text", "") for r in result_content
                    if isinstance(r, dict) and r.get("type") == "text"
                )
            else:
                result_text = str(result_content)
            rendered.append(f"[Tool result: {result_text[:1000]}]")
        elif btype == "thinking":
            pass  # skip reasoning blocks
    return "\n".join(rendered).strip()


# ── Stream-json output parsing ──────────────────────────────────────────


def _parse_result_event(stdout: str) -> dict | None:
    """Find the ``result`` event in claude CLI stream-json output."""
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            return event
    return None
