"""Hermes Claude Code CLI subprocess adapter.

This package implements the subprocess transport for routing Anthropic-bound
calls through the official ``claude`` CLI binary instead of direct HTTPS to
``api.anthropic.com``. See ``docs/superpowers/specs/2026-05-16-hermes-claude-
code-cli-adapter-design.md`` for the design and rationale.

This commit (Task 1 of PR 1) ships only the exception hierarchy. The stream-
json parser, probe, and adapter land in subsequent tasks of PR 1 and in
subsequent PRs (2-6) after the probe settles the CLI contract.
"""

from agent.claude_cli.errors import (
    ClaudeCliAuthMissing,
    ClaudeCliError,
    ClaudeCliIncompatible,
    ClaudeCliUnavailable,
    ClaudeCliVersionTooOld,
    HermesDirectAnthropicEgressDetected,
    ProtocolError,
    PromptTooLarge,
)

__all__ = [
    "ClaudeCliAuthMissing",
    "ClaudeCliError",
    "ClaudeCliIncompatible",
    "ClaudeCliUnavailable",
    "ClaudeCliVersionTooOld",
    "HermesDirectAnthropicEgressDetected",
    "ProtocolError",
    "PromptTooLarge",
]
