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

If no explicit fallback config exists, the chain is auto-built by scanning
environment variables for known provider API keys. The primary provider is
excluded to avoid circular fallback. Priority order:

    1. OpenRouter (broadest model selection, good fallback for any primary)
    2. Anthropic (direct — for when primary is OpenRouter/other)
    3. OpenAI (direct)
    4. Google Gemini
    5. DeepSeek
    6. Together AI
    7. Groq
    8. Fireworks AI
    9. Mistral AI
    10. Local providers (LM Studio, Ollama — checked via port probe)
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
# Known provider definitions for auto-chain building
# =============================================================================

@dataclass
class _KnownProvider:
    """Describes a known inference provider for auto-detection."""
    id: str
    name: str
    env_vars: Tuple[str, ...]  # API key env vars to check (first found wins)
    base_url: str = ""
    default_model: str = ""
    api_mode: str = "chat_completions"
    # Aliases that match this provider as "primary" (to exclude from chain)
    primary_aliases: Tuple[str, ...] = ()


# Ordered by fallback priority — first match with a valid key wins
KNOWN_PROVIDERS: List[_KnownProvider] = [
    _KnownProvider(
        id="openrouter",
        name="OpenRouter",
        env_vars=("OPENROUTER_API_KEY",),
        base_url=OPENROUTER_BASE_URL,
        default_model="anthropic/claude-sonnet-4",
        primary_aliases=("openrouter",),
    ),
    _KnownProvider(
        id="anthropic",
        name="Anthropic",
        env_vars=("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN"),
        base_url="https://api.anthropic.com/v1",
        default_model="claude-sonnet-4-20250514",
        primary_aliases=("anthropic",),
    ),
    _KnownProvider(
        id="openai",
        name="OpenAI",
        env_vars=("OPENAI_API_KEY",),
        base_url="https://api.openai.com/v1",
        default_model="gpt-4.1",
        primary_aliases=("openai",),
    ),
    _KnownProvider(
        id="gemini",
        name="Google Gemini",
        env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-2.5-flash",
        primary_aliases=("gemini", "google"),
    ),
    _KnownProvider(
        id="deepseek",
        name="DeepSeek",
        env_vars=("DEEPSEEK_API_KEY",),
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        primary_aliases=("deepseek",),
    ),
    _KnownProvider(
        id="together",
        name="Together AI",
        env_vars=("TOGETHER_API_KEY",),
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        primary_aliases=("together",),
    ),
    _KnownProvider(
        id="groq",
        name="Groq",
        env_vars=("GROQ_API_KEY",),
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        primary_aliases=("groq",),
    ),
    _KnownProvider(
        id="fireworks",
        name="Fireworks AI",
        env_vars=("FIREWORKS_API_KEY",),
        base_url="https://api.fireworks.ai/inference/v1",
        default_model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        primary_aliases=("fireworks",),
    ),
    _KnownProvider(
        id="mistral",
        name="Mistral AI",
        env_vars=("MISTRAL_API_KEY",),
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-large-latest",
        primary_aliases=("mistral",),
    ),
]

# Local providers checked via port probe (no API key needed)
LOCAL_PROVIDERS = [
    _KnownProvider(
        id="lmstudio",
        name="LM Studio",
        env_vars=(),
        base_url="http://localhost:1234/v1",
        default_model="",  # auto-detected
        primary_aliases=("lmstudio", "lm-studio", "lm_studio"),
    ),
    _KnownProvider(
        id="ollama",
        name="Ollama",
        env_vars=(),
        base_url="http://localhost:11434/v1",
        default_model="",  # auto-detected
        primary_aliases=("ollama",),
    ),
]


# =============================================================================
# Helper functions
# =============================================================================

def _infer_primary_provider(
    provider: str, model: str, base_url: str
) -> str:
    """Infer the primary provider id from provider name, model, or base_url.

    Returns a lowercase provider id that matches KNOWN_PROVIDERS aliases,
    or empty string if unknown.
    """
    if provider:
        p = provider.lower().strip()
        # Direct match against known provider aliases
        for kp in KNOWN_PROVIDERS + LOCAL_PROVIDERS:
            if p in kp.primary_aliases or p == kp.id:
                return kp.id
        # Fuzzy match: provider string contains known id
        for kp in KNOWN_PROVIDERS + LOCAL_PROVIDERS:
            if kp.id in p:
                return kp.id
        return p  # Unknown provider — return as-is for exclusion

    # Infer from model name
    if model:
        m = model.lower()
        if "claude" in m or m.startswith("anthropic/"):
            return "anthropic"
        if "gpt" in m or "o1" in m or "o3" in m or "o4" in m:
            return "openai"
        if "gemini" in m:
            return "gemini"
        if "deepseek" in m:
            return "deepseek"
        if "llama" in m or "mistral" in m:
            pass  # Could be any provider, don't guess
        # Model has provider prefix (e.g. "anthropic/claude-3")
        if "/" in m:
            prefix = m.split("/")[0]
            for kp in KNOWN_PROVIDERS:
                if prefix in kp.primary_aliases:
                    return kp.id

    # Infer from base_url
    if base_url:
        url = base_url.lower()
        if "openrouter" in url:
            return "openrouter"
        if "anthropic" in url:
            return "anthropic"
        if "openai.com" in url:
            return "openai"
        if "googleapis" in url or "generativelanguage" in url:
            return "gemini"
        if "deepseek" in url:
            return "deepseek"
        if "together" in url:
            return "together"
        if "groq" in url:
            return "groq"
        if "fireworks" in url:
            return "fireworks"
        if "mistral" in url:
            return "mistral"
        if "localhost:1234" in url:
            return "lmstudio"
        if "localhost:11434" in url:
            return "ollama"

    return ""


def _probe_local_endpoint(base_url: str, timeout: float = 0.5) -> bool:
    """Quick check if a local endpoint is reachable (TCP connect only).

    Non-blocking, fast timeout. Returns True if port is open.
    """
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 80
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


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
        """Build a backward-compatible chain. Delegates to build_auto_chain."""
        return cls.build_auto_chain(primary_model=primary_model)

    @classmethod
    def build_auto_chain(
        cls,
        primary_provider: str = "",
        primary_model: str = "",
        primary_base_url: str = "",
    ) -> "FallbackChain":
        """Auto-build a fallback chain by scanning environment for API keys.

        Detects all known providers with valid API keys, excludes the primary
        provider, and returns an ordered chain. Also probes local endpoints
        (LM Studio, Ollama) if reachable.

        Args:
            primary_provider: The active provider id (e.g. "anthropic", "openrouter").
                              Used to exclude it from the fallback chain.
            primary_model: The active model name. Used to infer primary provider
                           if primary_provider is empty.
            primary_base_url: The active base URL. Used to infer primary provider.
        """
        # Infer primary provider from model/base_url if not explicitly given
        primary_id = _infer_primary_provider(
            primary_provider, primary_model, primary_base_url
        )

        entries: List[FallbackEntry] = []

        # Scan cloud providers (need API keys)
        for kp in KNOWN_PROVIDERS:
            # Skip the primary provider
            if primary_id and primary_id in kp.primary_aliases:
                continue

            # Check if any env var has a key set
            api_key = ""
            api_key_env = ""
            for env_var in kp.env_vars:
                val = os.getenv(env_var, "").strip()
                if val:
                    api_key = val
                    api_key_env = env_var
                    break

            if not api_key:
                continue

            # For OpenRouter: use the same model as primary (it's a proxy)
            model = kp.default_model
            if kp.id == "openrouter" and primary_model:
                # Map model to OpenRouter format if needed
                or_model = primary_model
                if "claude" in or_model.lower() and not or_model.startswith("anthropic/"):
                    or_model = f"anthropic/{or_model}"
                model = or_model

            entries.append(FallbackEntry(
                provider=kp.id,
                base_url=kp.base_url,
                model=model,
                api_key_env=api_key_env,
            ))

        # Probe local providers (no API key needed — just check if port is open)
        for lp in LOCAL_PROVIDERS:
            if primary_id and primary_id in lp.primary_aliases:
                continue
            if _probe_local_endpoint(lp.base_url):
                entries.append(FallbackEntry(
                    provider=lp.id,
                    base_url=lp.base_url,
                    model=lp.default_model,
                ))

        if entries:
            logger.info(
                "Auto-configured fallback chain: %s",
                ", ".join(e.provider for e in entries),
            )

        return cls(entries=entries, mode="auto", timeout=30)

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
