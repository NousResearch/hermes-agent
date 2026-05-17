"""Exception hierarchy for the claude_cli package.

All exceptions inherit from ``ClaudeCliError`` so callers can catch the
package's failures with a single except clause.
"""


class ClaudeCliError(Exception):
    """Base class for all claude_cli package errors."""


class ClaudeCliUnavailable(ClaudeCliError):
    """The ``claude`` binary was not found on PATH (or the configured path)."""


class ClaudeCliVersionTooOld(ClaudeCliError):
    """The detected ``claude`` binary version is below the pinned minimum."""


class ClaudeCliAuthMissing(ClaudeCliError):
    """Required Claude Code OAuth token is missing from the environment."""


class ClaudeCliIncompatible(ClaudeCliError):
    """The installed ``claude`` binary does not behave as the adapter expects.

    Raised when the probe detects a flag/protocol/precedence mismatch that
    would make the adapter unsafe to use against this binary version.
    """


class HermesDirectAnthropicEgressDetected(ClaudeCliError):
    """Hermes parent process emitted outbound HTTPS to api.anthropic.com.

    This is a regression guard against accidental direct-HTTPS fallback to
    the existing ``anthropic_adapter.py`` while the operator believes they
    are routed through Claude Code. NOT a billing proof — actual plan
    billing must be verified externally.
    """


class ProtocolError(ClaudeCliError):
    """Stream-json output from the ``claude`` subprocess failed to parse.

    Raised for malformed JSON lines, oversized lines, exceeded event-count
    budgets, and persistent non-JSON content on stdout.
    """


class PromptTooLarge(ClaudeCliError):
    """The prompt payload exceeds the configured ``max_prompt_bytes`` limit.

    Only reachable if PR 1 probe determines stdin prompt transport is NOT
    supported and a bounded-argv fallback mode is used. Default v1 path
    uses stdin (no bound).
    """
