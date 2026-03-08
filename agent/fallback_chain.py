"""Provider fallback chain for resilient inference.

When the primary inference provider fails (rate-limit, auth, overload),
the fallback chain provides an ordered list of alternatives. In CLI mode,
the user gets an interactive countdown prompt to pick one. In gateway mode
(Telegram, Discord), it auto-cascades through the chain with status
messages sent to the user.

Config (in ~/.hermes/config.yaml):

    fallback:
      mode: interactive    # "auto" | "interactive"
      timeout: 30          # seconds before auto-selecting (interactive mode)
      chain:
        - provider: openrouter
          model: anthropic/claude-opus-4.6
        - provider: lmstudio
          base_url: http://localhost:1234/v1
          model: qwen3-30b-a3b
        - provider: ollama
          base_url: http://localhost:11434/v1
          model: llama3.1:70b

If no fallback config exists but OPENROUTER_API_KEY is set and the
primary provider is Anthropic, a single-entry OpenRouter chain is
auto-generated for backward compatibility.
"""

from __future__ import annotations

import logging
import os
import select
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from hermes_constants import OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)

# Status codes that trigger a fallback switch
FALLBACK_STATUS_CODES = {401, 429, 503, 529}


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class FallbackEntry:
    """A single provider slot in the fallback chain."""

    provider: str
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""  # env var name to read key from
    api_mode: str = "chat_completions"

    @property
    def display_name(self) -> str:
        """Human-readable name for the provider."""
        host = ""
        if self.base_url:
            # Show host for local endpoints
            from urllib.parse import urlparse
            parsed = urlparse(self.base_url)
            if parsed.hostname and parsed.hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
                host = f" ({parsed.hostname}:{parsed.port or 80})"
        return f"{self.provider}{host}"

    def resolve_api_key(self) -> str:
        """Resolve the API key from env var or provider defaults."""
        # Explicit env var specified in config
        if self.api_key_env:
            key = os.getenv(self.api_key_env, "").strip()
            if key:
                return key

        # Provider-specific defaults
        provider_lower = self.provider.lower()
        if provider_lower == "openrouter":
            return os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        elif provider_lower in ("lmstudio", "ollama", "llamacpp", "local"):
            # Local providers typically don't need a key
            return os.getenv("OPENAI_API_KEY", "") or "not-needed"
        elif provider_lower == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY", "") or os.getenv("ANTHROPIC_TOKEN", "")
        else:
            # Generic fallback: try OPENAI_API_KEY
            return os.getenv("OPENAI_API_KEY", "")

    def build_client_kwargs(self) -> Dict[str, Any]:
        """Build OpenAI client kwargs for this fallback entry."""
        provider_lower = self.provider.lower()
        base_url = self.base_url or OPENROUTER_BASE_URL
        api_key = self.resolve_api_key()

        kwargs: Dict[str, Any] = {
            "base_url": base_url.rstrip("/"),
            "api_key": api_key,
        }

        # OpenRouter-specific headers
        if "openrouter" in base_url.lower():
            kwargs["default_headers"] = {
                "HTTP-Referer": "https://github.com/NousResearch/hermes-agent",
                "X-OpenRouter-Title": "Hermes Agent",
                "X-OpenRouter-Categories": "productivity,cli-agent",
            }

        return kwargs


# =============================================================================
# Fallback Chain
# =============================================================================

class FallbackChain:
    """Ordered chain of fallback inference providers.

    The chain tracks which entries have been tried and exhausted.
    It supports both sequential auto-cascade and interactive selection.
    """

    def __init__(
        self,
        entries: List[FallbackEntry],
        mode: str = "auto",
        timeout: int = 30,
        enabled: bool = True,
    ):
        self.entries = entries
        self.mode = mode  # "auto", "interactive", or "off"
        self.timeout = timeout
        self.enabled = enabled  # Master toggle — False disables all fallback
        self._exhausted: Set[int] = set()

    @classmethod
    def from_config(cls, config: dict) -> "FallbackChain":
        """Build chain from config.yaml fallback section."""
        fallback_cfg = config.get("fallback", {})
        if not fallback_cfg or not isinstance(fallback_cfg, dict):
            return cls(entries=[], mode="auto")

        # Master toggle: enabled: false or mode: off disables fallback
        enabled = fallback_cfg.get("enabled", True)
        mode = fallback_cfg.get("mode", "auto")
        if mode == "off":
            enabled = False
        timeout = fallback_cfg.get("timeout", 30)

        entries: List[FallbackEntry] = []
        for entry_cfg in fallback_cfg.get("chain", []):
            if not isinstance(entry_cfg, dict):
                continue
            provider = entry_cfg.get("provider", "").strip()
            if not provider:
                continue
            entries.append(FallbackEntry(
                provider=provider,
                base_url=entry_cfg.get("base_url", ""),
                model=entry_cfg.get("model", ""),
                api_key_env=entry_cfg.get("api_key_env", ""),
                api_mode=entry_cfg.get("api_mode", "chat_completions"),
            ))

        return cls(entries=entries, mode=mode, timeout=timeout, enabled=enabled)

    @classmethod
    def build_legacy_chain(cls, primary_model: str = "") -> "FallbackChain":
        """Build a backward-compatible single-entry OpenRouter chain.

        Called when there's no explicit fallback config but OPENROUTER_API_KEY
        exists. Preserves the pre-chain behavior: Anthropic -> OpenRouter.
        """
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            return cls(entries=[], mode="auto")

        # Map the primary model to OpenRouter format
        model = primary_model
        if model and not model.startswith("anthropic/") and "claude" in model.lower():
            model = f"anthropic/{model}"

        return cls(
            entries=[FallbackEntry(
                provider="openrouter",
                base_url=OPENROUTER_BASE_URL,
                model=model,
                api_key_env="OPENROUTER_API_KEY",
            )],
            mode="auto",
            timeout=30,
        )

    def has_fallbacks(self) -> bool:
        """Check if fallback is enabled and entries exist."""
        return self.enabled and len(self.entries) > 0

    def is_exhausted(self) -> bool:
        """Check if all fallback entries have been tried."""
        return len(self._exhausted) >= len(self.entries)

    def available_entries(self) -> List[Tuple[int, FallbackEntry]]:
        """Get (index, entry) pairs for entries not yet exhausted."""
        return [
            (i, e) for i, e in enumerate(self.entries)
            if i not in self._exhausted
        ]

    def next_available(self) -> Optional[FallbackEntry]:
        """Get the next untried entry in order."""
        for i, entry in enumerate(self.entries):
            if i not in self._exhausted:
                return entry
        return None

    def mark_failed(self, entry: FallbackEntry):
        """Mark an entry as failed/exhausted."""
        try:
            idx = self.entries.index(entry)
            self._exhausted.add(idx)
        except ValueError:
            pass

    def select_by_index(self, index: int) -> Optional[FallbackEntry]:
        """Select a specific entry by its chain index."""
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None

    def reset(self):
        """Reset all exhaustion state for a fresh round."""
        self._exhausted.clear()


# =============================================================================
# Fallback selectors (Strategy pattern)
# =============================================================================

def select_interactive(
    chain: FallbackChain,
    error_msg: str,
    current_model: str = "",
    log_prefix: str = "",
) -> Optional[FallbackEntry]:
    """Interactive countdown prompt for fallback selection (CLI mode).

    Shows a numbered menu of available fallback providers with a countdown
    timer. Auto-selects the first available if the timer expires.

    Returns the selected FallbackEntry, or None to skip (keep retrying).
    """
    available = chain.available_entries()
    if not available:
        return None

    # Display error and menu
    print(f"\n{log_prefix}⚠️  Primary provider failed: {error_msg[:120]}")
    print(f"{log_prefix}🔄 Fallback options:\n")

    for display_idx, (_, entry) in enumerate(available, 1):
        model_str = f"  →  {entry.model}" if entry.model else ""
        print(f"{log_prefix}  [{display_idx}] {entry.display_name}{model_str}")

    print(f"{log_prefix}  [m] Change model before switching")
    print(f"{log_prefix}  [s] Skip — keep retrying primary")
    print()

    default_idx, default_entry = available[0]
    timeout = chain.timeout

    try:
        start = time.time()
        while True:
            remaining = max(0, timeout - int(time.time() - start))

            if remaining == 0:
                print(f"\r{log_prefix}  Auto-selecting [{1}] {default_entry.display_name}                    ")
                print()
                return default_entry

            # Progress bar
            bar_total = 20
            bar_filled = int((remaining / timeout) * bar_total)
            bar = "\u2588" * bar_filled + "\u2591" * (bar_total - bar_filled)
            sys.stdout.write(
                f"\r{log_prefix}  Auto-selecting [{1}] in {remaining}s  {bar}  "
            )
            sys.stdout.flush()

            # Poll stdin for 1 second
            if sys.stdin.isatty():
                rlist, _, _ = select.select([sys.stdin], [], [], 1.0)
                if rlist:
                    choice = sys.stdin.readline().strip().lower()
                    # Clear the countdown line
                    sys.stdout.write(f"\r{log_prefix}{'':60}\r")
                    sys.stdout.flush()

                    if choice == "s":
                        print(f"{log_prefix}  ↩ Continuing with primary provider...")
                        return None

                    elif choice == "m":
                        return _handle_model_override(
                            available, log_prefix
                        )

                    else:
                        try:
                            sel_idx = int(choice) - 1
                            if 0 <= sel_idx < len(available):
                                _, entry = available[sel_idx]
                                print(f"{log_prefix}  ✓ Selected {entry.display_name}")
                                return entry
                        except (ValueError, IndexError):
                            pass
                        # Invalid input — auto-select default
                        print(f"{log_prefix}  Invalid choice, selecting {default_entry.display_name}")
                        return default_entry
            else:
                # Non-interactive stdin — auto-cascade
                time.sleep(1)

    except (KeyboardInterrupt, EOFError):
        print(f"\n{log_prefix}  Interrupted — staying on primary provider")
        return None


def _handle_model_override(
    available: List[Tuple[int, FallbackEntry]],
    log_prefix: str,
) -> Optional[FallbackEntry]:
    """Handle the [m] model override choice in interactive mode."""
    try:
        model = input(f"{log_prefix}  Enter model name: ").strip()
        if not model:
            print(f"{log_prefix}  No model entered, using default")
            return available[0][1]

        if len(available) == 1:
            entry = available[0][1]
            entry.model = model
            print(f"{log_prefix}  ✓ Using {entry.display_name} with model {model}")
            return entry

        provider_q = input(
            f"{log_prefix}  Which provider? [1-{len(available)}, default=1]: "
        ).strip()

        sel_idx = 0
        if provider_q:
            try:
                sel_idx = int(provider_q) - 1
                if not (0 <= sel_idx < len(available)):
                    sel_idx = 0
            except ValueError:
                sel_idx = 0

        _, entry = available[sel_idx]
        entry.model = model
        print(f"{log_prefix}  ✓ Using {entry.display_name} with model {model}")
        return entry

    except (KeyboardInterrupt, EOFError):
        return None


def select_auto(
    chain: FallbackChain,
    error_msg: str,
    current_model: str = "",
    log_prefix: str = "",
    notify: Optional[Callable[[str], None]] = None,
) -> Optional[FallbackEntry]:
    """Auto-select the next fallback in the chain (gateway mode).

    Args:
        notify: Optional callback to send status messages to the user
                (e.g., Telegram message, Discord embed).
    """
    entry = chain.next_available()
    if entry is None:
        msg = "⚠️ All fallback providers exhausted. Could not complete request."
        print(f"{log_prefix}{msg}")
        if notify:
            try:
                notify(msg)
            except Exception:
                pass
        return None

    model_str = f" ({entry.model})" if entry.model else ""
    msg = f"🔄 Primary failed. Switching to {entry.display_name}{model_str}..."
    print(f"{log_prefix}{msg}")
    if notify:
        try:
            notify(msg)
        except Exception:
            pass

    return entry


# =============================================================================
# High-level API used by AIAgent
# =============================================================================

def should_trigger_fallback(status_code: Optional[int], error: Exception) -> bool:
    """Determine if an API error should trigger a fallback switch.

    Only triggers on errors that are provider-level failures, not client
    errors that would persist across providers (e.g., bad request format).
    """
    if status_code and status_code in FALLBACK_STATUS_CODES:
        return True

    # Connection errors suggest the provider is unreachable
    error_type = type(error).__name__
    if error_type in ("ConnectError", "ConnectTimeout", "ReadTimeout", "ConnectionError"):
        return True

    error_msg = str(error).lower()
    if "overloaded" in error_msg or "rate limit" in error_msg:
        return True

    return False
