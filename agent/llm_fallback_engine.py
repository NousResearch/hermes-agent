"""
LLM Fallback Engine — Graceful Degradation for Hermes Agent.

When all LLM providers are unavailable (network failure, 503, auth errors, etc.),
this engine provides pattern-based canned responses that keep Hermes functional
for basic user intents.  This is the last line of defense for GL (Generative Loop).

Design principles:
  GL (Generative Loop)  — system keeps running even when LLM is down
  OS (Observable State)  — every fallback activation is logged via EventBus
  MI (Module Independence) — pure-Python, no external deps beyond stdlib
  LP (Least Privilege)   — only activates when all LLM options are exhausted

Fallback hierarchy:
  1. Primary LLM (main provider)
  2. Fallback LLM chain (from config)
  3. Rule engine (this module) ← last resort

Rule engine coverage:
  - Help / usage intents
  - Error acknowledgment
  - Diagnostic commands (doctor, status)
  - Simple greetings and acknowledgments
  - MCP/tool availability queries
  - Generic graceful degradation message

Rule engine does NOT attempt to:
  - Execute tools or code
  - Make decisions requiring reasoning
  - Access external resources
  - Perform file operations
  - Replace the LLM for complex tasks
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Rule Definition ──────────────────────────────────────────────────────────

@dataclass
class FallbackRule:
    """A single pattern → response mapping."""
    # Regex pattern (case-insensitive). Matched against the user message.
    pattern: str
    # Response template. May contain {user_message} substitution.
    response: str
    # Priority: higher numbers match first. Ties broken by insertion order.
    priority: int = 0
    # Whether to also show the diagnostic footer.
    show_diagnostics: bool = True


@dataclass
class FallbackResult:
    """Outcome of fallback engine evaluation."""
    # Whether a rule matched.
    matched: bool
    # The response text (empty if no match).
    response: str
    # Which rule matched (None if no match).
    rule: Optional[FallbackRule] = None
    # How long evaluation took (seconds).
    eval_time_ms: float = 0.0


# ─── Built-in Rules ───────────────────────────────────────────────────────────

def _build_rules() -> list[FallbackRule]:
    """Build the default rule set.  Override via constructor for testing."""

    rules: list[FallbackRule] = [
        # ── High-priority: explicit help requests ──────────────────────
        FallbackRule(
            pattern=r"^(help|\?)$",
            response=(
                "I'm currently unable to reach the AI service (all LLM providers are unavailable).\n\n"
                "Here's what you can try:\n"
                "  1. Run `hermes doctor` to diagnose configuration issues\n"
                "  2. Check your internet connection\n"
                "  3. Verify your API key: `hermes auth status`\n"
                "  4. Try a different model: `hermes model`\n"
                "  5. Restart Hermes: `exit` then relaunch\n\n"
                "Once the AI service is restored, I can help with coding, file operations, "
                "web search, and more."
            ),
            priority=90,
        ),

        FallbackRule(
            pattern=r"^(how do I|how can I|what is|what are|explain|tell me about|what does)",
            response=(
                "The AI service is currently unavailable, so I can't answer that question right now.\n\n"
                "Please try again in a moment, or run `hermes doctor` to check your configuration."
            ),
            priority=80,
            show_diagnostics=True,
        ),

        # ── Diagnostic / admin commands ───────────────────────────────
        FallbackRule(
            pattern=r"^(doctor|diagnose|check|status|debug)",
            response=(
                "LLM is currently unavailable. Run `hermes doctor` from a separate terminal "
                "to check your configuration and API connectivity.\n\n"
                "Common fixes:\n"
                "  • Set ANTHROPIC_API_KEY or ANTHROPIC_TOKEN\n"
                "  • Run `hermes auth` to configure credentials\n"
                "  • Check network connectivity to your LLM provider"
            ),
            priority=85,
            show_diagnostics=False,
        ),

        FallbackRule(
            pattern=r"^(exit|quit|bye|goodbye)$",
            response="Goodbye! When the AI service recovers, Hermes will be ready to help again.",
            priority=85,
            show_diagnostics=False,
        ),

        FallbackRule(
            pattern=r"^(retry|try again|reload|refresh)$",
            response=(
                "The LLM service is currently unavailable. Your last request will not be "
                "retried automatically.\n\n"
                "Please try again manually when connectivity is restored, or check "
                "`hermes doctor` for configuration issues."
            ),
            priority=80,
        ),

        FallbackRule(
            pattern=r"^(list tools|available tools|what tools|show tools)",
            response=(
                "LLM is unavailable, so I cannot execute tools right now.\n\n"
                "Tools that would normally be available:\n"
                "  • File operations (read, write, edit, glob)\n"
                "  • Shell command execution\n"
                "  • Web search and browsing\n"
                "  • Memory and session management\n"
                "  • Code execution and debugging\n"
                "  • Delegate to subagents\n\n"
                "These will work normally once the LLM service is restored."
            ),
            priority=80,
            show_diagnostics=False,
        ),

        FallbackRule(
            pattern=r"^((list|show) (tools|commands|skills))",
            response=(
                "LLM is unavailable — I cannot enumerate tools right now.\n\n"
                "Run `hermes doctor` to check configuration, then try again."
            ),
            priority=75,
            show_diagnostics=False,
        ),

        FallbackRule(
            pattern=r"^(use (|tool )|run |execute |call )",
            response=(
                "LLM is unavailable, so I cannot select or execute tools right now.\n\n"
                "Tool execution requires the LLM to determine which tools to use and how "
                "to call them. Once the service is restored, normal tool use will resume."
            ),
            priority=70,
            show_diagnostics=False,
        ),

        # ── Acknowledgment / low-complexity intents ────────────────────
        FallbackRule(
            pattern=r"^(yes|yeah|yep|ok|okay|sure|please|go ahead|continue)",
            response=(
                "Acknowledged. The AI service is currently unavailable, so I cannot proceed.\n\n"
                "Please try again when the service is restored, or let me know if you'd "
                "like to restart Hermes (`exit`) to retry the connection."
            ),
            priority=60,
        ),

        FallbackRule(
            pattern=r"^(no|nah|nope|cancel|never mind|nevermind|stop)$",
            response="Understood. Let me know if there's anything else.",
            priority=60,
            show_diagnostics=False,
        ),

        FallbackRule(
            pattern=r"^(thank you|thanks|thx|appreciate)$",
            response="You're welcome! I hope the AI service recovers soon.",
            priority=60,
            show_diagnostics=False,
        ),

        FallbackRule(
            pattern=r"^(hello|hi|hey|greetings)$",
            response=(
                "Hello! The AI service is currently unavailable, so I'm operating in "
                "limited mode.\n\n"
                "I can still:\n"
                "  • Acknowledge your requests\n"
                "  • Provide diagnostic guidance\n"
                "  • Direct you to manual workarounds\n\n"
                "The full experience will resume once the LLM service is restored."
            ),
            priority=60,
        ),

        FallbackRule(
            pattern=r"^(sorry|apologize|my bad|excuse)$",
            response="No need to apologize! The issue is on my end — the AI service is unavailable.",
            priority=55,
            show_diagnostics=False,
        ),

        # ── Error acknowledgment ────────────────────────────────────────
        FallbackRule(
            pattern=r"\b(error|failed|exception|crash|bug|broken|doesn't work|not working)\b",
            response=(
                "It sounds like something went wrong. Since the AI service is unavailable, "
                "I can't investigate further right now.\n\n"
                "Try running `hermes doctor` from another terminal to diagnose the issue, "
                "or check `hermes logs` for error details."
            ),
            priority=50,
        ),

        FallbackRule(
            pattern=r"(config|configuration|setup|install)",
            response=(
                "LLM is unavailable. For configuration help, run `hermes doctor` to check "
                "your setup, or see `hermes auth --help` and `hermes model --help`."
            ),
            priority=50,
        ),
    ]

    return rules


# ─── Diagnostic footer ────────────────────────────────────────────────────────

def _build_diagnostic_footer(
    primary_error: str,
    fallback_count: int,
    fallback_index: int,
) -> str:
    """Build the standard diagnostic footer appended to fallback responses."""
    parts = [
        "\n─── System Status ───",
        f"LLM: UNAVAILABLE (primary error: {primary_error})",
    ]
    if fallback_count > 0:
        parts.append(
            f"Fallback chain: {fallback_index}/{fallback_count} providers tried"
        )
    else:
        parts.append("Fallback chain: not configured")
    parts.append(
        "Run `hermes doctor` to diagnose, or `hermes logs` for error details."
    )
    return "\n".join(parts)


# ─── Main Engine ──────────────────────────────────────────────────────────────

class LLMFallbackEngine:
    """
    Pattern-based fallback engine for graceful LLM degradation.

    Activates when all LLM providers in the fallback chain have failed.
    Matches the user's message against registered patterns and returns
    a canned response appropriate to their intent.

    Events emitted (via optional EventBus):
      - llm.fallback.activated   — fallback engine was invoked
      - llm.fallback.matched     — a rule matched (includes rule pattern)
      - llm.fallback.no_match    — no rule matched
      - llm.fallback.recovered    — LLM became available again (future use)

    Usage::

        engine = LLMFallbackEngine(
            event_bus=event_bus,        # optional EventBus instance
            rules=custom_rules,          # optional; uses built-in if omitted
        )

        result = engine.handle(user_message, primary_error="timeout after 3 retries")
        if result.matched:
            print(result.response)
        else:
            print("No fallback rule matched — system error")
    """

    def __init__(
        self,
        event_bus=None,
        rules: list[FallbackRule] = None,
        include_diagnostics: bool = True,
    ):
        """
        Args:
            event_bus: Optional EventBus for emitting fallback events.
            rules: Override rule set.  Uses built-in rules if None.
            include_diagnostics: Append diagnostic footer to responses.
        """
        self._event_bus = event_bus
        self._rules = rules if rules is not None else _build_rules()
        self._include_diagnostics = include_diagnostics

        # Compile patterns once at construction time
        self._compiled: list[Tuple[re.Pattern, FallbackRule]] = []
        for rule in self._rules:
            try:
                compiled = re.compile(rule.pattern, re.IGNORECASE)
                self._compiled.append((compiled, rule))
            except re.error as exc:
                logger.warning(
                    "LLMFallbackEngine: invalid rule pattern %r: %s",
                    rule.pattern, exc,
                )

        # Sort by priority descending (highest priority first)
        self._compiled.sort(key=lambda x: x[1].priority, reverse=True)

        # Metrics
        self._activation_count = 0
        self._match_count = 0
        self._no_match_count = 0
        self._last_activation_time: Optional[float] = None
        self._last_primary_error: Optional[str] = None
        self._last_fallback_index: int = 0
        self._last_fallback_count: int = 0

        logger.info(
            "LLMFallbackEngine initialized with %d rules (diagnostics=%s)",
            len(self._compiled), include_diagnostics,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def handle(
        self,
        user_message: str,
        primary_error: str = "unknown",
        fallback_index: int = 0,
        fallback_count: int = 0,
    ) -> FallbackResult:
        """
        Evaluate the user's message and return a fallback response.

        Args:
            user_message: The raw user input string.
            primary_error: Human-readable description of the LLM failure.
            fallback_index: Which fallback provider was last tried (0 = none).
            fallback_count: Total providers in the fallback chain.

        Returns:
            FallbackResult with matched flag, response text, and metadata.
        """
        start = time.monotonic()
        self._activation_count += 1
        self._last_activation_time = start
        self._last_primary_error = primary_error
        self._last_fallback_index = fallback_index
        self._last_fallback_count = fallback_count

        # Emit activation event
        self._emit_event("llm.fallback.activated", {
            "primary_error": primary_error,
            "fallback_index": fallback_index,
            "fallback_count": fallback_count,
            "activation_count": self._activation_count,
        })

        # Try each rule in priority order
        message = (user_message or "").strip()

        for pattern, rule in self._compiled:
            match = pattern.search(message)
            if match:
                self._match_count += 1
                eval_time_ms = (time.monotonic() - start) * 1000

                response = rule.response
                if "{user_message}" in response:
                    response = response.replace("{user_message}", message[:200])

                if self._include_diagnostics and rule.show_diagnostics:
                    response += _build_diagnostic_footer(
                        primary_error, fallback_count, fallback_index,
                    )

                logger.debug(
                    "LLMFallbackEngine: rule %r matched for %r (priority=%d, %.1fms)",
                    rule.pattern, message[:50], rule.priority, eval_time_ms,
                )

                self._emit_event("llm.fallback.matched", {
                    "pattern": rule.pattern,
                    "priority": rule.priority,
                    "eval_time_ms": eval_time_ms,
                    "user_message_preview": message[:100],
                })

                return FallbackResult(
                    matched=True,
                    response=response,
                    rule=rule,
                    eval_time_ms=eval_time_ms,
                )

        # No rule matched
        self._no_match_count += 1
        eval_time_ms = (time.monotonic() - start) * 1000

        logger.debug(
            "LLMFallbackEngine: no rule matched for %r (%.1fms)",
            message[:50], eval_time_ms,
        )

        self._emit_event("llm.fallback.no_match", {
            "user_message_preview": message[:100],
            "eval_time_ms": eval_time_ms,
            "no_match_count": self._no_match_count,
        })

        # Generic graceful degradation response
        generic = (
            "I'm sorry, but the AI service is currently unavailable and I'm unable to "
            "process your request right now.\n\n"
            "What happened: All LLM providers have failed (primary error: {error}).\n\n"
            "What you can do:\n"
            "  • Check your internet connection\n"
            "  • Run `hermes doctor` to diagnose configuration issues\n"
            "  • Verify API credentials: `hermes auth status`\n"
            "  • Try again in a few minutes\n\n"
            "The system will automatically recover when the AI service is restored."
        ).format(error=primary_error[:100])

        if self._include_diagnostics:
            generic += _build_diagnostic_footer(
                primary_error, fallback_count, fallback_index,
            )

        return FallbackResult(
            matched=False,
            response=generic,
            rule=None,
            eval_time_ms=eval_time_ms,
        )

    def is_available(self) -> bool:
        """Return True if the engine is ready (always True — no external deps)."""
        return True

    def snapshot(self) -> dict:
        """
        Return an immutable snapshot of engine metrics for observability.

        Implements OS (Observable State) from the design principles.
        """
        return {
            "activation_count": self._activation_count,
            "match_count": self._match_count,
            "no_match_count": self._no_match_count,
            "rule_count": len(self._compiled),
            "last_activation_time": self._last_activation_time,
            "last_primary_error": self._last_primary_error,
            "last_fallback_index": self._last_fallback_index,
            "last_fallback_count": self._last_fallback_count,
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _emit_event(self, event_type: str, payload: dict) -> None:
        """Emit an event via EventBus (fire-and-forget)."""
        if self._event_bus is None:
            return
        try:
            from agent.hermes.analytics import Event
            event = Event(type=event_type, payload=payload)
            self._event_bus.emit(event)
        except Exception as exc:
            logger.debug("LLMFallbackEngine: EventBus emit failed: %s", exc)


# ─── Singleton ────────────────────────────────────────────────────────────────

_engine_instance: Optional[LLMFallbackEngine] = None
_instance_lock = threading.Lock()


def get_fallback_engine(event_bus=None) -> LLMFallbackEngine:
    """
    Get or create the global LLMFallbackEngine singleton.

    This ensures a single engine instance is shared across the entire agent
    lifecycle, so metrics accumulate correctly.
    """
    global _engine_instance
    with _instance_lock:  # Thread-safe singleton initialization
        if _engine_instance is None:
            _engine_instance = LLMFallbackEngine(event_bus=event_bus)
        elif event_bus is not None:
            # Only update _event_bus if still None (double-checked locking)
            if _engine_instance._event_bus is None:
                _engine_instance._event_bus = event_bus
    return _engine_instance


def reset_fallback_engine() -> None:
    """Reset the global singleton (for testing)."""
    global _engine_instance
    _engine_instance = None
