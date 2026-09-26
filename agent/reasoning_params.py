"""Provider reasoning-parameter policy for ``AIAgent``.

When ``reasoning`` extra_body is safe to send, LM Studio / Ollama / GitHub Models capability probes,
``reasoning_content`` echo families, and strict-API tool-call sanitising.
Extracted from ``run_agent.py``; every method resolves through ``AIAgent``'s MRO unchanged.
"""
import time
from typing import Any

from agent.lazy_forward import forward as _forward, forward_static as _forward_static
from agent.message_sanitization import matches_reasoning_echo_family
from utils import base_url_host_matches

# ``model.reasoning_echo`` modes. Read once per route change, never per message.
REASONING_ECHO_ALWAYS = "always"
REASONING_ECHO_NEVER = "never"


def normalize_reasoning_echo_mode(value: Any) -> "str | None":
    """Normalize a raw ``reasoning_echo`` value (config key or fallback-entry field) to a mode.

    ``True`` / ``"always"`` → ``"always"``; ``"never"`` → ``"never"``; everything else (including
    ``False``, ``"auto"`` and ``None``) → ``None`` (auto — defer to the family rules).
    """
    if isinstance(value, str):
        mode = value.strip().lower()
        return mode if mode in (REASONING_ECHO_ALWAYS, REASONING_ECHO_NEVER) else None
    return REASONING_ECHO_ALWAYS if value is True else None


def read_reasoning_echo_mode() -> "str | None":
    """Read ``model.reasoning_echo`` as a mode: ``"always"`` (force echo-back), ``"never"`` (force the
    strict strip side), or ``None`` (auto — defer to the family rules in
    ``message_sanitization._REASONING_ECHO_RULES``).

    Accepts the booleans the key historically carried (``True`` → ``"always"``) plus the strings
    ``always`` / ``never`` / ``auto``. An absent key, ``False``, ``"auto"`` or anything unparseable is
    ``None``. ``False`` deliberately stays *auto*, not *never*: the flag has always meant "no opt-in,
    defer to detection", so a detected family must keep winning for callers that only ever set the
    boolean — use ``"never"`` to opt a route out.

    Single source of truth for both wire-truth consumers (the agent's payload policy and the
    compressor's tail walk). They must agree or compaction loops, and two independent config reads
    could drift; one reader cannot.
    """
    try:
        from hermes_cli.config import load_config_readonly

        model_cfg = load_config_readonly().get("model") or {}
        if "reasoning_echo" not in model_cfg:
            return None
        return normalize_reasoning_echo_mode(model_cfg.get("reasoning_echo"))
    except Exception:
        return None


# Static OpenRouter fallback when the live /v1/models capability cache is cold.
_OPENROUTER_REASONING_PREFIXES = (
    "deepseek/", "anthropic/", "openai/", "x-ai/", "google/gemini-2", "google/gemma-4",
    "qwen/qwen3", "tencent/hy", "xiaomi/",
)

# Probe results cache per (model, base_url). Definitive values cache permanently; an
# "unknown" value (empty list / None) caches 60s so a transient failure neither sticks
# for the session nor round-trips every turn.
_PROBE_TTL_S = 60


def _cached_probe(agent, cache_attr: str, probe, unknown, definitive):
    """``probe(model, base_url, api_key)`` once per (model, base_url); ``unknown`` is what a raising probe
    yields, ``definitive(value)`` decides permanent vs 60s TTL."""
    cache = getattr(agent, cache_attr, None)
    if cache is None:
        cache = {}
        setattr(agent, cache_attr, cache)
    key = (agent.model, agent.base_url)
    cached = cache.get(key)
    if cached is not None:
        value, ts = cached
        if definitive(value) or (time.monotonic() - ts) < _PROBE_TTL_S:
            return value
    try:
        value = probe(agent.model, agent.base_url, getattr(agent, "api_key", ""))
    except Exception:
        value = unknown
    cache[key] = (value, time.monotonic())
    return value


def unset_reasoning_default(agent) -> dict | None:
    """Reasoning config for a main-loop request whose ``agent.reasoning_effort`` is unset.

    Asks the active provider profile (``ProviderProfile.default_reasoning_config``; the custom /
    OpenAI-compatible profile answers medium) so a route's own default never silently applies —
    kimi-k3 behind a relay defaults to ``max``, 3x the reasoning tokens of medium. Resolved at
    request time, so ``/model`` and fallback activation re-evaluate it. None keeps the field off
    the wire: a non chat-completions transport, a profile without a default (those decide inside
    ``build_api_kwargs_extras``), a model the catalog / ``model_overrides`` mark
    ``supports_reasoning: false``, or a local Ollama model whose ``/api/show`` lacks ``thinking``.
    """
    if getattr(agent, "api_mode", None) != "chat_completions":
        return None
    provider = str(getattr(agent, "provider", "") or "")
    model = str(getattr(agent, "model", "") or "")
    try:
        from providers import get_provider_profile

        profile = get_provider_profile(provider)
        default = profile.default_reasoning_config(model) if profile is not None else None
    except Exception:
        return None
    if not default:
        return None
    try:
        from agent.models_dev import get_model_capabilities

        caps = get_model_capabilities(provider, model, allow_network=False)
    except Exception:
        caps = None
    if caps is not None and caps.supports_reasoning is False:
        return None
    # ``_ollama_num_ctx`` is only ever set for a server detected as Ollama (agent_init); Ollama
    # 400s ``reasoning_effort`` on a model pulled without the thinking capability.
    if getattr(agent, "_ollama_num_ctx", None) and not agent._ollama_supports_thinking_cached():
        return None
    return dict(default)


class ReasoningParamsMixin:
    """Reasoning-parameter gating and echo policy (see module docstring)."""

    def _supports_reasoning_extra_body(self) -> bool:
        """True when reasoning extra_body is safe to send: OpenRouter forwards unknown extra_body upstream and
        some routes 400 on ``reasoning``, so gate to known reasoning-capable families and direct Nous Portal."""
        url = self._base_url_lower
        if base_url_host_matches(url, "nousresearch.com") or base_url_host_matches(url, "ai-gateway.vercel.sh"):
            return True
        if base_url_host_matches(url, "models.github.ai") or base_url_host_matches(url, "githubcopilot.com"):
            try:
                from hermes_cli.models import github_model_reasoning_efforts

                return bool(github_model_reasoning_efforts(self.model))
            except Exception:
                return False
        if (self.provider or "").strip().lower() == "lmstudio":
            # "off-only" (or absent) means no real reasoning capability.
            return any(opt and opt != "off" for opt in self._lmstudio_reasoning_options_cached())
        if base_url_host_matches(url, "ollama.com"):
            # Ollama Cloud: /api/show capabilities are authoritative.
            return self._ollama_supports_thinking_cached()
        if not self._is_openrouter_url() or base_url_host_matches(url, "api.mistral.ai"):
            return False
        # Live-catalog metadata first (OpenRouter /v1/models supported_parameters) — the static prefix
        # allowlist repeatedly went stale one vendor at a time. Unknown falls back to the static list.
        try:
            from hermes_cli.models_reasoning_caps import openrouter_model_reasoning_capabilities, warm_openrouter_reasoning_caps_async
            caps = openrouter_model_reasoning_capabilities(self.model)
            if caps is None:
                warm_openrouter_reasoning_caps_async()  # cache cold — warm in the background, never block
        except Exception:
            caps = None
        if caps is not None:
            return bool(caps.get("supports_reasoning"))
        model = (self.model or "").lower()
        return any(model.startswith(prefix) for prefix in _OPENROUTER_REASONING_PREFIXES)

    def _lmstudio_reasoning_options_cached(self) -> list[str]:
        """LM Studio's published reasoning ``allowed_options`` (gate + clamp so toggle models don't 400 on ``high``)."""
        try:
            from hermes_cli.models_local import lmstudio_model_reasoning_options
        except Exception:
            return []
        return _cached_probe(self, "_lm_reasoning_opts_cache", lmstudio_model_reasoning_options, [], bool)

    def _ollama_supports_thinking_cached(self) -> bool:
        """True only if Ollama's ``/api/show`` declares the ``thinking`` capability."""
        try:
            from hermes_cli.models_local import ollama_model_supports_thinking
        except Exception:
            return False
        return bool(_cached_probe(self, "_ollama_thinking_cache", ollama_model_supports_thinking, None, lambda v: v is not None))

    def _github_models_reasoning_extra_body(self) -> dict | None:
        """Format reasoning payload for GitHub Models/OpenAI-compatible routes."""
        try:
            from hermes_cli.models import github_model_reasoning_efforts
        except Exception:
            return None

        supported = github_model_reasoning_efforts(self.model)
        if not supported:
            return None

        cfg = self.reasoning_config if isinstance(self.reasoning_config, dict) else {}
        if cfg.get("enabled") is False:
            return None
        effort = str(cfg.get("effort", "medium")).strip().lower()

        if effort not in supported:
            # Nearest-neighbour fallbacks: xhigh→high, minimal→low, else medium, else the first published level.
            nearest = {"xhigh": "high", "minimal": "low"}.get(effort)
            effort = nearest if nearest in supported else "medium" if "medium" in supported else supported[0]
        return {"effort": effort}

    _build_assistant_message = _forward("agent.chat_completion_helpers", "build_assistant_message")

    def _needs_thinking_reasoning_pad(self) -> bool:
        """True when the provider enforces ``reasoning_content`` echo-back on tool-call replays (DeepSeek, Kimi,
        MiMo thinking all 400 without it). Cached per (provider, model, base_url), invalidated by
        ``switch_model()`` / ``_try_activate_fallback()`` — called ~16× per turn.

        DeepSeek v4 thinking and Kimi / Moonshot thinking both reject replays of assistant tool-call
        messages that omit ``reasoning_content`` (refs 15250, #17400). Xiaomi MiMo thinking mode has the
        same requirement.

        ``model.reasoning_echo: never`` wins over every rule below. A gateway can serve a family model
        (detected by name) whose own backend does not enforce the echo — opencode.ai accepts a replayed
        assistant turn with the field absent — and echoing there only hands the model its own stale
        chain-of-thought back, which it then re-emits.
        """
        key = (self.provider, self.model, getattr(self, "_base_url_lower", self.base_url))
        cached = getattr(self, "_thinking_pad_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        if self._reasoning_echo_forced_strip():
            result = False
        else:
            result = (self._needs_deepseek_tool_reasoning() or self._needs_kimi_tool_reasoning()
                      or self._needs_mimo_tool_reasoning() or self._reasoning_echo_opt_in())
        self._thinking_pad_cache = (key, result)
        return result

    def _reasoning_echo_opt_in(self) -> bool:
        """``model.reasoning_echo`` opt-in for the *current* provider (covers gateways the host rules miss);
        fallback activation swaps the flag and ``restore_primary_runtime()`` restores it."""
        return (bool(getattr(self, "_reasoning_echo_flag", False))
                or getattr(self, "_reasoning_echo_mode", "") == REASONING_ECHO_ALWAYS)

    def _reasoning_echo_forced_strip(self) -> bool:
        """``model.reasoning_echo: never`` — force the strict strip side over family detection."""
        return getattr(self, "_reasoning_echo_mode", "") == REASONING_ECHO_NEVER

    def _sync_reasoning_echo_from_config(self) -> None:
        """Sync ``_reasoning_echo_mode`` / ``_reasoning_echo_flag`` to ``model.reasoning_echo`` and drop the
        per-route pad cache so the next ``_needs_thinking_reasoning_pad()`` re-derives. Called at init,
        ``switch_model()``, fallback activation and ``restore_primary_runtime()``."""
        mode = self._read_reasoning_echo_from_config() or ""
        self._reasoning_echo_mode = mode
        self._reasoning_echo_flag = mode == REASONING_ECHO_ALWAYS
        self._thinking_pad_cache = None

    @staticmethod
    def _read_reasoning_echo_from_config() -> "str | None":
        """``model.reasoning_echo`` as a mode for the current route; ``None`` (auto) on any error."""
        return read_reasoning_echo_mode()

    # Echo families are host/provider-driven, not model-name-driven: aggregators re-exporting Kimi reject the
    # echo. Rule table: ``message_sanitization._REASONING_ECHO_RULES``. Kimi deliberately passes the raw
    # provider and no model (its rule matches exact provider ids + hosts only).
    def _needs_kimi_tool_reasoning(self) -> bool:
        """True when the current provider is Kimi / Moonshot thinking mode."""
        return matches_reasoning_echo_family("kimi", self.provider, None, self.base_url)

    def _needs_deepseek_tool_reasoning(self) -> bool:
        """True when the current provider is DeepSeek thinking mode (omitting the echo is an HTTP 400).

        DeepSeek V4 thinking mode requires ``reasoning_content`` on every assistant tool-call turn; omitting
        it causes HTTP 400 when the message is replayed in a subsequent API request (#15250).
        """
        return matches_reasoning_echo_family("deepseek", (self.provider or "").lower(), self.model, self.base_url)

    def _needs_mimo_tool_reasoning(self) -> bool:
        """True when the current provider is Xiaomi MiMo thinking mode."""
        return matches_reasoning_echo_family("mimo", (self.provider or "").lower(), self.model, self.base_url)

    _copy_reasoning_content_for_api = _forward("agent.agent_runtime_helpers", "copy_reasoning_content_for_api")

    _reapply_reasoning_echo_for_provider = _forward("agent.agent_runtime_helpers", "reapply_reasoning_echo_for_provider")

    @staticmethod
    def _sanitize_tool_calls_for_strict_api(api_msg: dict, model: "str | None" = None) -> dict:
        """Strip Codex Responses fields from tool_calls for strict Chat Completions APIs (Mistral, Fireworks
        400/422 on unknown fields). ``extra_content`` (Gemini thought_signature) is kept only for Gemini-family
        models. Builds new dicts so the internal history keeps the Codex fields for a later fallback."""
        tool_calls = api_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return api_msg
        from agent.transports.chat_completions import _model_consumes_thought_signature
        strip = {"call_id", "response_item_id"} | (set() if _model_consumes_thought_signature(model) else {"extra_content"})
        api_msg["tool_calls"] = [{k: v for k, v in tc.items() if k not in strip} if isinstance(tc, dict) else tc
                                 for tc in tool_calls]
        return api_msg

    _sanitize_tool_call_arguments = _forward_static("agent.agent_runtime_helpers", "sanitize_tool_call_arguments")

    def _should_sanitize_tool_calls(self) -> bool:
        """True for every non-Codex API: Codex Responses fields are not Chat Completions schema and 400 elsewhere."""
        return self.api_mode != "codex_responses"
