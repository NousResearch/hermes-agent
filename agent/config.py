"""Configuration dataclasses for AIAgent.

Decomposes the 60-parameter AIAgent constructor into four focused config
objects, each with single responsibility:

  - ProviderConfig: API routing, provider selection, credentials
  - SessionConfig:  Session identity, platform context, user/chat IDs
  - BudgetConfig:   Iteration limits, token caps, trajectory settings
  - CallbackConfig: All callback hooks for tool/stream/clarify events

SOLID principles applied:
  S — each class owns one concern
  O — new fields can be added without breaking existing consumers
  L — all configs are usable wherever their type is expected
  I — consumers depend only on the config they need
  D — AIAgent depends on abstract config interfaces, not raw params

DRY: validation, serialization, defaults live once per class.
"""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_lower(value: Optional[str]) -> str:
    """Normalize a string: strip whitespace, lowercase. Empty string if None."""
    if not isinstance(value, str) or not value.strip():
        return ""
    return value.strip().lower()


def _hostname(url: str) -> str:
    """Extract hostname from a URL (best-effort, no urllib dependency)."""
    # Strip scheme
    h = url
    if "://" in h:
        h = h.split("://", 1)[1]
    # Strip path/query/fragment
    h = h.split("/", 1)[0]
    # Strip port
    h = h.split(":", 1)[0]
    return h.lower()


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    """All provider/API routing parameters.

    Replaces: base_url, api_key, provider, api_mode, model,
              max_iterations, fallback_model, providers_allowed/ignored/order,
              provider_sort, openrouter_min_coding_score, credential_pool,
              service_tier, acp_command, acp_args, api_mode detection logic.
    """

    base_url: str = ""
    api_key: Optional[str] = None
    provider: str = ""
    api_mode: Optional[str] = None
    model: str = ""
    max_iterations: int = 90

    # Routing
    fallback_model: Optional[Dict[str, Any]] = None
    providers_allowed: Optional[List[str]] = None
    providers_ignored: Optional[List[str]] = None
    providers_order: Optional[List[str]] = None
    provider_sort: Optional[str] = None
    openrouter_min_coding_score: Optional[float] = None
    credential_pool: Any = None
    service_tier: Optional[str] = None

    # ACP
    acp_command: Optional[str] = None
    acp_args: Optional[List[str]] = None

    def __post_init__(self):
        # Normalize provider
        self.provider = _clean_lower(self.provider)
        # Auto-detect api_mode if not explicitly set
        if self.api_mode is None:
            self.api_mode = self._detect_api_mode()

    def _detect_api_mode(self) -> str:
        """Auto-detect API mode from provider name and base URL."""
        hostname = _hostname(self.base_url)

        if self.provider == "openai-codex" or self.provider == "xai":
            return "codex_responses"
        if hostname == "chatgpt.com" and "/backend-api/codex" in self.base_url.lower():
            self.provider = "openai-codex"
            return "codex_responses"
        if hostname == "api.x.ai":
            if not self.provider:
                self.provider = "xai"
            return "codex_responses"
        if self.provider == "anthropic" or hostname == "api.anthropic.com":
            if not self.provider:
                self.provider = "anthropic"
            return "anthropic_messages"
        if self.base_url.lower().rstrip("/").endswith("/anthropic"):
            return "anthropic_messages"
        if self.provider == "bedrock" or (
            hostname.startswith("bedrock-runtime.") and "amazonaws.com" in hostname
        ):
            return "bedrock_converse"
        return "chat_completions"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (excludes None callables)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ProviderConfig:
        """Deserialize from a plain dict. Ignores unknown keys."""
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def replace(self, **kwargs) -> ProviderConfig:
        """Return a new ProviderConfig with the given fields replaced."""
        return dataclasses.replace(self, **kwargs)


# ---------------------------------------------------------------------------
# SessionConfig
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    """Session identity and platform context.

    Replaces: session_id, platform, user_id, user_name, chat_id,
              chat_name, chat_type, thread_id, gateway_session_key,
              parent_session_id, skip_context_files, load_soul_identity,
              skip_memory, pass_session_id.
    """

    session_id: Optional[str] = None
    platform: Optional[str] = None
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    chat_id: Optional[str] = None
    chat_name: Optional[str] = None
    chat_type: Optional[str] = None
    thread_id: Optional[str] = None
    gateway_session_key: Optional[str] = None
    parent_session_id: Optional[str] = None

    # Flags
    skip_context_files: bool = False
    load_soul_identity: bool = False
    skip_memory: bool = False
    pass_session_id: bool = False

    @property
    def effective_session_id(self) -> str:
        """Return session_id or generate one if missing."""
        return self.session_id or str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SessionConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# BudgetConfig
# ---------------------------------------------------------------------------

@dataclass
class BudgetConfig:
    """Iteration/token budget and trajectory settings.

    Replaces: max_iterations, save_trajectories, max_tokens,
              reasoning_config, request_overrides, prefill_messages,
              checkpoints_enabled, checkpoint_max_snapshots,
              checkpoint_max_total_size_mb, checkpoint_max_file_size_mb.
    """

    max_iterations: int = 90
    save_trajectories: bool = False
    max_tokens: Optional[int] = None
    reasoning_config: Optional[Dict[str, Any]] = None
    request_overrides: Optional[Dict[str, Any]] = None
    prefill_messages: Optional[List[Dict[str, Any]]] = None

    # Checkpoints
    checkpoints_enabled: bool = False
    checkpoint_max_snapshots: int = 20
    checkpoint_max_total_size_mb: int = 500
    checkpoint_max_file_size_mb: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BudgetConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_iteration_budget(self):
        """Create an IterationBudget from this config.

        Lazy import to avoid circular dependency with run_agent.
        """
        from run_agent import IterationBudget
        return IterationBudget(self.max_iterations)


# ---------------------------------------------------------------------------
# CallbackConfig
# ---------------------------------------------------------------------------

@dataclass
class CallbackConfig:
    """All callback hooks for tool/stream/clarify events.

    Replaces: tool_progress_callback, tool_start_callback,
              tool_complete_callback, thinking_callback,
              reasoning_callback, clarify_callback, step_callback,
              stream_delta_callback, interim_assistant_callback,
              tool_gen_callback, status_callback.
    """

    tool_progress_callback: Optional[Callable] = None
    tool_start_callback: Optional[Callable] = None
    tool_complete_callback: Optional[Callable] = None
    thinking_callback: Optional[Callable] = None
    reasoning_callback: Optional[Callable] = None
    clarify_callback: Optional[Callable] = None
    step_callback: Optional[Callable] = None
    stream_delta_callback: Optional[Callable] = None
    interim_assistant_callback: Optional[Callable] = None
    tool_gen_callback: Optional[Callable] = None
    status_callback: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict, excluding non-serializable callables."""
        d = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if val is not None and not callable(val):
                d[f.name] = val
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CallbackConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
