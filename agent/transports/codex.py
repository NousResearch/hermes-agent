"""OpenAI Responses API (Codex) transport.

Owns format conversion/normalization on top of agent/codex_responses_adapter.py —
NOT client lifecycle, streaming, or the _run_codex_stream() call path.
"""

import hashlib
import json
import logging
import re
from typing import Any, Callable, Optional

from agent.reasoning_effort import (
    ACTUAL_RELAY_EFFORTS, XAI_GROK46_EFFORTS, XAI_LEGACY_EFFORTS, clamp_effort,
    # Same declared vocabulary + shared clamp as the main Codex transport (agent.reasoning_effort):
    # per-model — "max" is gpt-5.6-only, "minimal"/"ultra" always rejected (live-verified, #68365).
    codex_supported_efforts,
)
from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall

logger = logging.getLogger(__name__)

# Cron fires use ``cron_<job_id>_<YYYYMMDD_HHMMSS>``; the per-fire timestamp is
# stripped so repeat fires of one job share a cache scope.
# See #51395, #52295.
_CRON_SESSION_ID_RE = re.compile(r"^(cron_.+)_\d{8}_\d{6}$")


def _cache_scope_from_session_id(session_id: Optional[str]) -> str:
    """Normalize a physical session_id into a stable logical cache scope."""
    sid = str(session_id or "")
    match = _CRON_SESSION_ID_RE.match(sid)
    return match.group(1) if match else sid


def _bounded_prompt_cache_key(value: Any) -> Optional[str]:
    """Return a provider-safe (<=64 char) cache key without changing session identity."""
    key = "" if value is None else str(value).strip()
    if not key:
        return None
    return key if len(key) <= 64 else "pck_" + hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:24]


def _bound_prompt_cache_key_field(container: Any) -> None:
    """Bound (or drop, when empty) an in-place ``prompt_cache_key`` entry."""
    if isinstance(container, dict) and "prompt_cache_key" in container:
        bounded = _bounded_prompt_cache_key(container["prompt_cache_key"])
        if bounded:
            container["prompt_cache_key"] = bounded
        else:
            container.pop("prompt_cache_key", None)


def _merge_extra_headers(kwargs: dict[str, Any], **headers: str) -> None:
    """Merge ``headers`` into a str-coerced copy of ``kwargs['extra_headers']`` (SDK kwarg -> HTTP headers)."""
    existing = kwargs.get("extra_headers")
    merged = {str(k): str(v) for k, v in existing.items() if k and v is not None} if isinstance(existing, dict) else {}
    merged.update(headers)
    kwargs["extra_headers"] = merged


# Client-side ``web_search`` on xAI Responses collides with Grok's native tool
# (incomplete hang / HTTP 400); it goes on the wire under this alias.
_XAI_CLIENT_WEB_SEARCH_ALIAS = "hermes_web_search"

# OpenCode /v1/responses rejects client tools using these names (HTTP 400
# "custom function name 'X' is reserved"); xAI reserves ``tool_search`` for
# Grok's native Tool Search. Aliased as hermes_<name>.
# OpenCode's /v1/responses endpoints (Zen and Go, including custom providers pointing at opencode.ai)
# reserve certain function names server-side and reject client tools that use them with HTTP 400 ("custom
# function name 'X' is reserved"). Same treatment as the xAI web_search collision: rename on the wire
# (hermes_<name>), map back in normalize_response so Hermes dispatch is unaffected. See #85589.
_OPENCODE_RESERVED_TOOL_NAMES = ("web_search", "search_files")
_XAI_RESERVED_TOOL_NAMES = ("tool_search",)
_RESERVED_TOOL_ALIAS_PREFIX = "hermes_"

# Reverse map used ONLY when normalize_response runs on a transport that never
# built a request; real requests carry request-local ``_last_wire_aliases``.
_LEGACY_ALIAS_FALLBACK = {
    f"{_RESERVED_TOOL_ALIAS_PREFIX}{name}": name
    for name in (*_OPENCODE_RESERVED_TOOL_NAMES, *_XAI_RESERVED_TOOL_NAMES)
}
_LEGACY_ALIAS_FALLBACK[_XAI_CLIENT_WEB_SEARCH_ALIAS] = "web_search"


def _is_opencode_responses_backend(params: dict[str, Any]) -> bool:
    """True for opencode-zen/go providers, ``opencode-*`` families, or opencode.ai hosts."""
    try:
        from hermes_cli.models import opencode_provider_family

        if opencode_provider_family(params.get("provider")) is not None:
            return True
    except Exception:
        pass
    try:
        from utils import base_url_hostname

        return base_url_hostname(str(params.get("base_url") or "")).lower() == "opencode.ai"
    except Exception:
        return False


def _alias_reserved_tools(
    response_tools: list[dict[str, Any]], reserved_names: tuple[str, ...],
    name_of: Callable[[dict], Any] = lambda t: t.get("name"),
    rename: Callable[[dict, str], dict] = lambda t, alias: {**t, "name": alias},
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Alias provider-reserved function names on the wire; returns ``(tools, {alias: original_name})``.

    An alias already taken by a real tool gets a ``_2``/``_3`` suffix. ``name_of``/``rename``
    adapt the tool shape (Responses ``{name}`` by default; chat_completions passes ``function.name``).
    """
    rewritten: list[dict[str, Any]] = []
    alias_map: dict[str, str] = {}
    taken = {name_of(tool) for tool in response_tools if isinstance(tool, dict) and name_of(tool)}
    for tool in response_tools:
        name = name_of(tool) if isinstance(tool, dict) else None
        if name not in reserved_names:
            rewritten.append(tool)
            continue
        base = alias = f"{_RESERVED_TOOL_ALIAS_PREFIX}{name}"
        suffix = 2
        while alias in taken:
            alias, suffix = f"{base}_{suffix}", suffix + 1
        taken.add(alias)
        alias_map[alias] = name
        rewritten.append(rename(tool, alias))
    return rewritten, alias_map


def _xai_prefers_native_web_search() -> bool:
    """True when xAI Responses should use Grok's native ``web_search`` built-in.

    Web-search registry first, then the legacy ``_get_search_backend`` probe; fails closed to native (True).

    Delegates to the web-search registry's provider resolution (which reads ``web.search_backend`` /
    ``web.backend`` from config) and checks whether the resolved provider is xAI. On any resolution failure,
    returns True (fail-closed to native — preserves the #48108 incomplete-hang fix rather than risk
    reintroducing it).
    """
    try:
        from agent.web_search_registry import get_active_search_provider

        provider = get_active_search_provider()
        if provider is not None:
            return getattr(provider, "name", None) == "xai"

        from tools.web_tools import _get_search_backend

        return (_get_search_backend() or "").strip().lower() == "xai"
    except Exception:
        return True


def _alias_wire_tools(response_tools: Any, params: dict[str, Any], is_xai_responses: bool) -> tuple[Any, dict[str, str]]:
    """Apply provider-reserved tool-name aliasing; returns ``(tools, {alias: original})`` for THIS request.

    xAI: a client ``web_search`` collides with Grok's native search — native mode
    swaps it 1:1 for the built-in, client mode keeps Hermes dispatch under an alias.
    """
    wire_aliases: dict[str, str] = {}

    def is_client_web_search(t: Any) -> bool:
        return isinstance(t, dict) and t.get("name") == "web_search"

    if is_xai_responses and response_tools and any(is_client_web_search(t) for t in response_tools):
        if _xai_prefers_native_web_search():
            response_tools = [t for t in response_tools if not is_client_web_search(t)] + [{"type": "web_search"}]
        else:
            response_tools = [
                {**t, "name": _XAI_CLIENT_WEB_SEARCH_ALIAS} if is_client_web_search(t) else t for t in response_tools
            ]
            wire_aliases[_XAI_CLIENT_WEB_SEARCH_ALIAS] = "web_search"
    # OpenCode Responses backends reserve web_search / search_files as function names (HTTP 400 "custom
    # function name 'X' is reserved", #85589). Alias them on the wire; normalize_response maps them back.
    if response_tools and _is_opencode_responses_backend(params):
        response_tools, _oc_aliases = _alias_reserved_tools(response_tools, _OPENCODE_RESERVED_TOOL_NAMES)
        wire_aliases.update(_oc_aliases)
    # xAI server-side web search vs Hermes web providers. grok models on xAI's /v1/responses surface have a
    # *native*, server-executed web search. A client-side function literally named ``web_search`` collides
    # with that engine: declared as a plain ``function`` rather than ``{"type": "web_search"}``, the search
    # dispatches but never reconciles → incomplete turn + 3 retries. Verified live against
    # grok-composer-2.5-fast (2026-06); see #48108. Two modes, chosen by the user's web-search backend
    # config: 1. **Native** (active/configured backend is ``xai``, or resolution fails): drop the client
    # ``web_search`` function and declare xAI's built-in instead. 1:1 swap only when client ``web_search``
    # was already present — never an additive grant. 2. **Client** (Firecrawl / Tavily / Exa / … configured
    # or resolved): keep Hermes dispatch so ``web.backend`` / ``web.search_backend`` is honored, but rename
    # the wire tool to ``hermes_web_search`` so Grok cannot hijack the name. The alias is mapped back to
    # ``web_search`` in ``normalize_response``. Request-local alias provenance: every wire alias THIS
    # request emits is recorded here and stashed on the transport, so the reverse rewrite in
    # ``normalize_response`` applies only to aliases that were actually sent (never to a real tool that
    # merely shares an alias-shaped name).
    if is_xai_responses and response_tools:
        response_tools, _xai_aliases = _alias_reserved_tools(response_tools, _XAI_RESERVED_TOOL_NAMES)
        wire_aliases.update(_xai_aliases)
    return response_tools, wire_aliases


def _resolve_reasoning(model: str, params: dict[str, Any]) -> tuple[Any, bool]:
    """``(effort, enabled)`` for the request, effort clamped (never escalated) to the endpoint's vocabulary.

    A profile-declared ``()`` means "no reasoning parameters accepted" (400 on any
    reasoning field) and disables reasoning outright.
    """
    reasoning_effort, reasoning_enabled = "medium", True
    reasoning_config = params.get("reasoning_config")
    if reasoning_config and isinstance(reasoning_config, dict):
        if reasoning_config.get("enabled") is False:
            reasoning_enabled = False
        elif reasoning_config.get("effort"):
            reasoning_effort = reasoning_config["effort"]

    # Wire vocabularies are declared in agent.reasoning_effort; the shared clamp policy (nearest weaker
    # supported level, never escalate, never invert the ladder) replaces the per-backend hand maps that
    # repeatedly leaked internal levels like "ultra" to the wire (#89503 class) or clamped one rung below a
    # model's real ceiling (#87279).
    if params.get("is_xai_responses", False):
        from agent.model_metadata import is_grok_46_family

        # Grok 4.6 accepts xhigh; older Grok tops out at high.
        supported = XAI_GROK46_EFFORTS if is_grok_46_family(model) else XAI_LEGACY_EFFORTS
    elif (params.get("provider") or "").strip().lower() == "actual":
        supported = ACTUAL_RELAY_EFFORTS
    else:
        declared = _profile_declared_efforts(params.get("provider"), model, params.get("base_url"))
        if declared is not None and not declared:
            reasoning_enabled = False
        supported = declared or codex_supported_efforts(model)
    return clamp_effort(reasoning_effort, supported), reasoning_enabled


_EXTENDED_PROMPT_CACHE_MODELS = (
    "gpt-5.5-pro", "gpt-5.5", "gpt-5.4", "gpt-5.2",
    "gpt-5.1-codex-max", "gpt-5.1-codex-mini", "gpt-5.1-chat-latest", "gpt-5.1-codex", "gpt-5.1",
    "gpt-5-codex", "gpt-5", "gpt-4.1",
)
_EXTENDED_PROMPT_CACHE_MODEL_RE = re.compile(
    rf"(?:^|[./:])(?:{'|'.join(re.escape(name) for name in _EXTENDED_PROMPT_CACHE_MODELS)})"
    r"(?:-\d{4}-\d{2}-\d{2})?$"
)


def _default_prompt_cache_retention_for_request(model: str, base_url: Any) -> Optional[str]:
    """Return ``24h`` for supported hosts/models (Bedrock Mantle, Meta)."""
    from utils import base_url_hostname

    hostname = base_url_hostname(str(base_url or "")).lower()
    # Meta Model API: caching is opt-in via prompt_cache_retention (0% hits without).
    # Meta Model API (api.meta.ai) only achieves prompt-cache hits on the Responses API with
    # prompt_cache_retention; chat/completions stays cache-cold (0% vs 93-99% measured). Exact-hostname
    # match per #32243.
    # Meta Model API: prompt caching only on Responses API (0% on chat/completions vs 93-99% on /responses
    # with retention). See #32243.
    if hostname == "api.meta.ai":
        return "24h"
    parts = hostname.split(".")
    is_bedrock_mantle = len(parts) == 4 and parts[0] == "bedrock-mantle" and bool(parts[1]) and parts[2:] == ["api", "aws"]
    if not is_bedrock_mantle:
        return None
    normalized = str(model or "").strip().lower().replace("_", "-")
    return "24h" if _EXTENDED_PROMPT_CACHE_MODEL_RE.search(normalized) else None


def _content_cache_key(instructions: str, tools: Optional[list[dict[str, Any]]], scope_id: str = "") -> Optional[str]:
    """``pck_<sha256[:24]>`` of (scope_id, instructions, name-sorted tools), or None if nothing static.

    Routing hint only; ``scope_id`` keeps unrelated sessions off one bucket.

    ``scope_id`` (pass ``_cache_scope_from_session_id(session_id)``) keeps unrelated sessions — independent
    conversations, main vs. child/subagent, sibling children — from concentrating onto the same bucket
    merely because their static prefix matches (see #78941), while still letting recurring cron fires of one
    job share a stable key across their timestamped session_ids (the original #51395/#52295 fix this built
    on). Sorting tools by name keeps the hash insertion-order independent.
    """
    if not instructions and not tools:
        return None
    tools_part = ""
    if tools:
        sorted_tools = sorted(
            (t for t in tools if isinstance(t, dict)), key=lambda t: str(t.get("name") or t.get("type") or ""),
        )
        tools_part = json.dumps(sorted_tools, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    # \x00 separators so a boundary can't be forged by content containing the same bytes.
    content = f"{scope_id}\x00{instructions or ''}\x00{tools_part}"
    return "pck_" + hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:24]


def _profile_declared_efforts(provider: Any, model: Optional[str], base_url: Any = None) -> Optional[tuple]:
    """Provider-profile-declared reasoning-effort vocabulary, or None (fail-open).

    Resolves by provider name, then by endpoint host. Lazy import: provider
    plugins import this transport during registry discovery.
    """
    try:
        from providers import get_provider_profile

        name = str(provider or "").strip().lower()
        profile = get_provider_profile(name) if name else None
        declared = profile.supported_reasoning_efforts(model) if profile is not None else None
        if declared is None and base_url:
            from agent.model_metadata import _infer_provider_from_url

            inferred = _infer_provider_from_url(str(base_url))
            if inferred and inferred != name:
                inferred_profile = get_provider_profile(inferred)
                if inferred_profile is not None:
                    declared = inferred_profile.supported_reasoning_efforts(model)
    except Exception as exc:
        logger.debug("profile-declared efforts lookup failed: %s", exc)
        return None
    return None if declared is None else tuple(declared)


def _is_azure_foundry_responses(params: dict[str, Any]) -> bool:
    """True for Microsoft Foundry's Responses API (provider id, else host match — not substring)."""
    from utils import base_url_host_matches

    if str(params.get("provider") or "").strip().lower() == "azure-foundry":
        return True
    return base_url_host_matches(str(params.get("base_url") or ""), "services.ai.azure.com")


def _is_post_tool_replay(messages: Optional[list[dict[str, Any]]]) -> bool:
    """True when ``messages`` end on a tool-result run issued by the preceding assistant turn.

    Azure Foundry rejects only this post-tool shape when encrypted reasoning is
    replayed, so only the *trailing* messages are checked (a whole-history scan
    would make suppression sticky). Call ids resolve like ``_chat_messages_to_responses_input``.
    """
    from agent.codex_responses_adapter import _canonical_call_id_from_fc, _split_responses_tool_id

    def _pair_ids(raw: Any, explicit: Any = None) -> set:
        embedded_call_id, item_id = _split_responses_tool_id(raw)
        ids = {embedded_call_id} if embedded_call_id else set()
        if isinstance(explicit, str) and explicit.strip():
            ids.add(explicit.strip())
        if not ids and isinstance(raw, str) and raw.strip():
            ids.add(raw.strip())
        canonical = _canonical_call_id_from_fc(item_id)
        if canonical:
            ids.add(canonical)
        return ids

    trailing = set()
    for msg in reversed(messages or ()):
        role = msg.get("role") if isinstance(msg, dict) else None
        if role == "system":
            continue
        if role == "tool":
            ids = _pair_ids(msg.get("tool_call_id"))
            if not ids:
                return False
            trailing |= ids
            continue
        # First non-tool message must be the assistant turn that issued the run.
        if role != "assistant":
            return False
        return any(
            trailing & _pair_ids(call.get("id"), call.get("call_id"))
            for call in msg.get("tool_calls") or []
            if isinstance(call, dict)
        )
    return False


def _native_compaction_active(context_management: Any) -> bool:
    """True only when the caller's eligibility gate produced a non-empty payload.

    Every native-compaction wire effect hangs off this predicate, so a persisted
    checkpoint cannot keep reshaping requests after the gate closes.
    """
    return isinstance(context_management, list) and bool(context_management)


def _coerce_timeout(timeout: Any) -> Optional[float]:
    """Finite positive number -> float; anything else (None, bool, str, inf) -> None."""
    if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and 0 < float(timeout) < float("inf"):
        return float(timeout)
    return None


def _reasoning_fields(
    model: str, params: dict[str, Any], *, effort: Any, enabled: bool, replay_encrypted_reasoning: bool,
    is_xai_responses: bool, is_github_responses: bool,
) -> dict[str, Any]:
    """``reasoning`` / ``include`` request fields for the endpoint family.

    xAI 400s on ``reasoning.effort`` outside its allowlist; GitHub Models takes a
    verbatim ``github_reasoning_extra`` and never ``include``.
    """
    include = ["reasoning.encrypted_content"] if replay_encrypted_reasoning else []
    fields: dict[str, Any] = {}
    if enabled and is_xai_responses:
        from agent.model_metadata import grok_supports_reasoning_effort

        fields["include"] = include
        if grok_supports_reasoning_effort(model):
            fields["reasoning"] = {"effort": effort}
    elif enabled:
        if is_github_responses:
            if params.get("github_reasoning_extra") is not None:
                fields["reasoning"] = params["github_reasoning_extra"]
        else:
            fields["reasoning"] = {"effort": effort, "summary": "auto"}
            fields["include"] = include
    elif not is_github_responses and not is_xai_responses:
        fields["include"] = []
    return fields


class ResponsesApiTransport(ProviderTransport):
    """Transport for api_mode='codex_responses'."""

    # Codex response.status -> OpenAI finish_reason (caller checks incomplete_details).
    _STOP_REASON_MAP = {"completed": "stop", "incomplete": "length", "failed": "stop", "cancelled": "stop"}

    # Issuer kind of the most recent build_kwargs/convert_messages call (normalize_response fallback).
    _last_issuer_kind: Optional[str] = None
    # ``{wire_alias: original}`` of the most recent build_kwargs. None = no request built (legacy map).
    _last_wire_aliases: Optional[dict[str, str]] = None

    @property
    def api_mode(self) -> str:
        return "codex_responses"

    def _resolve_issuer_kind(self, params: dict[str, Any]) -> str:
        """Classify the current Responses endpoint from transport params (stashed for normalize_response)."""
        from agent.codex_responses_adapter import _classify_responses_issuer

        self._last_issuer_kind = _classify_responses_issuer(
            is_xai_responses=params.get("is_xai_responses") is True,
            is_github_responses=params.get("is_github_responses") is True,
            is_codex_backend=params.get("is_codex_backend") is True,
            base_url=params.get("base_url"),
        )
        return self._last_issuer_kind

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs) -> Any:
        """Convert OpenAI chat messages to Responses API input items."""
        from agent.codex_responses_adapter import _chat_messages_to_responses_input

        return _chat_messages_to_responses_input(
            messages, is_xai_responses=kwargs.get("is_xai_responses") is True,
            is_github_responses=kwargs.get("is_github_responses") is True,
            replay_encrypted_reasoning=bool(kwargs.get("replay_encrypted_reasoning", True)),
            current_issuer_kind=self._resolve_issuer_kind(kwargs),
            native_compaction_eligible=_native_compaction_active(kwargs.get("context_management")),
        )

    def convert_tools(self, tools: Optional[list[dict[str, Any]]]) -> Any:
        """Convert OpenAI tool schemas to Responses API function definitions."""
        from agent.codex_responses_adapter import _responses_tools

        return _responses_tools(tools)

    def build_kwargs(
        self, model: str, messages: list[dict[str, Any]], tools: Optional[list[dict[str, Any]]] = None, **params,
    ) -> dict[str, Any]:
        """Build Responses API kwargs (calls convert_messages/convert_tools internally).

        params: instructions, reasoning_config ({effort, enabled}), session_id (transcript id;
        Codex header; cache-scope fallback), cache_scope_id (rotation-stable scope for the
        cache key / xAI conv header), max_tokens, timeout, request_overrides, provider, base_url,
        is_github_responses, is_codex_backend, is_xai_responses, github_reasoning_extra,
        context_management, replay_encrypted_reasoning.

        params: instructions: str — system prompt (extracted from messages[0] if not given)
        reasoning_config: dict | None — {effort, enabled} session_id: str | None — transcript/session id;
        drives the Codex ``session_id`` header, and is the cache-scope fallback when no ``cache_scope_id``
        is given cache_scope_id: str | None — rotation-stable logical scope id (compression-lineage root;
        see agent/prompt_cache_scope.py). Preferred over session_id when deriving the prompt_cache_key
        content hash and the xAI x-grok-conv-id header; the Codex x-client-request-id header mirrors the
        resulting body key. Keeps the cache warm across context-compression session rotation (#79017)
        max_tokens: int | None — max_output_tokens timeout: float | None — per-request timeout forwarded to
        the SDK request_overrides: dict | None — extra kwargs merged in provider: str | None — provider name
        for backend-specific logic base_url: str | None — endpoint URL base_url_hostname: str | None —
        hostname for backend detection is_github_responses: bool — Copilot/GitHub models backend
        is_codex_backend: bool — chatgpt.com/backend-api/codex is_xai_responses: bool — xAI/Grok backend
        github_reasoning_extra: dict | None — Copilot reasoning params
        """
        from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

        instructions = params.get("instructions", "")
        payload_messages = messages
        if not instructions and messages and messages[0].get("role") == "system":
            instructions = str(messages[0].get("content") or "").strip()
            payload_messages = messages[1:]
        instructions = instructions or DEFAULT_AGENT_IDENTITY

        is_github_responses = params.get("is_github_responses") is True
        is_codex_backend = params.get("is_codex_backend") is True
        is_xai_responses = params.get("is_xai_responses") is True
        # Foundry 400s on encrypted-reasoning replay only in the post-tool follow-up turn.
        replay_encrypted_reasoning = bool(params.get("replay_encrypted_reasoning", True)) and not (
            _is_azure_foundry_responses(params) and _is_post_tool_replay(payload_messages)
        )
        # One predicate decides whether context_management goes out AND whether the converter may replay a checkpoint.
        context_management = params.get("context_management")
        native_compaction_active = _native_compaction_active(context_management)

        reasoning_effort, reasoning_enabled = _resolve_reasoning(model, params)
        response_tools, self._last_wire_aliases = _alias_wire_tools(self.convert_tools(tools), params, is_xai_responses)

        # Lazy: provider plugins import this transport during model_metadata init.
        from agent.model_metadata import strip_codex_context_variant_suffix as _strip_ctx_variant
        kwargs = {
            # ``-900k`` picker variants are Hermes-side aliases; the backend knows only the base slug.
            "model": _strip_ctx_variant(model),
            "instructions": instructions,
            "input": self.convert_messages(
                payload_messages, is_xai_responses=is_xai_responses, is_github_responses=is_github_responses,
                replay_encrypted_reasoning=replay_encrypted_reasoning, base_url=params.get("base_url"),
                is_codex_backend=is_codex_backend, context_management=context_management,
            ),
            "store": False,
        }
        # ``tools`` MUST be omitted when empty: the openai SDK iterates it without a None guard.
        if response_tools:
            kwargs["tools"] = response_tools
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = True
        if native_compaction_active:
            kwargs["context_management"] = context_management

        session_id = params.get("session_id")
        # Content-addressed (instructions + tools) within a logical scope that survives
        # compression rotation; session_id itself stays untouched for transcript isolation.
        _cache_scope = _cache_scope_from_session_id(params.get("cache_scope_id") or session_id)
        cache_key = _content_cache_key(instructions, response_tools, _cache_scope) or _cache_scope
        # xAI takes prompt_cache_key in extra_body (below); GitHub Models opts out entirely.
        if not is_github_responses and not is_xai_responses and cache_key:
            kwargs["prompt_cache_key"] = cache_key

        cache_retention = _default_prompt_cache_retention_for_request(model, params.get("base_url"))
        if cache_retention:
            kwargs.setdefault("prompt_cache_retention", cache_retention)

        kwargs.update(_reasoning_fields(
            model, params, effort=reasoning_effort, enabled=reasoning_enabled,
            replay_encrypted_reasoning=replay_encrypted_reasoning,
            is_xai_responses=is_xai_responses, is_github_responses=is_github_responses,
        ))
        if params.get("request_overrides"):
            kwargs.update(params["request_overrides"])

        _bound_prompt_cache_key_field(kwargs)

        # Older xAI models reject ``service_tier`` (HTTP 400); only Grok 4.6 accepts Priority Processing.
        # Grok 4.6 accepts Priority Processing, but continue stripping stale or unsupported tier values on
        # every other xAI path. See #28490 and #84799.
        if is_xai_responses:
            from agent.model_metadata import is_grok_46_family

            if not (is_grok_46_family(model) and kwargs.get("service_tier") == "priority"):
                kwargs.pop("service_tier", None)

        # Forward per-request timeout to the SDK (providers.<id>.request_timeout_seconds).
        timeout = _coerce_timeout(kwargs.get("timeout", params.get("timeout")))
        if timeout is not None:
            kwargs["timeout"] = timeout
        else:
            kwargs.pop("timeout", None)

        if is_codex_backend:
            # SDK kwarg -> HTTP headers. ``session_id`` = raw physical id (transcript
            # identity); ``x-client-request-id`` mirrors the body cache key so both agree.
            headers = {
                "session_id": str(session_id) if session_id else None,
                "x-client-request-id": kwargs.get("prompt_cache_key") or _bounded_prompt_cache_key(_cache_scope),
            }
            headers = {k: v for k, v in headers.items() if v}
            if headers:
                _merge_extra_headers(kwargs, **headers)
        elif params.get("max_tokens") is not None:
            kwargs["max_output_tokens"] = params["max_tokens"]

        if is_xai_responses and session_id:
            # Scoped like the body key so cron fires don't each pin a different xAI backend server.
            _merge_extra_headers(kwargs, **{"x-grok-conv-id": _cache_scope})
            # xAI reads prompt_cache_key from the body; extra_body survives SDK builds whose
            # Responses.stream() dropped the typed kwarg. An explicit request_overrides value wins.
            # Scoped like the body cache key below — otherwise cron's per-fire timestamp in session_id
            # (cron_<id>_<ts>) pins every fire of the same job to a different xAI backend server (#78941).
            # xAI Responses cache-routing — body-level field per
            # https://docs.x.ai/developers/advanced-api-usage/prompt-caching/maximizing-cache-hits. A
            # caller's request_overrides={"prompt_cache_key": ...} lands on the top-level kwarg set above —
            # read it back here so an explicit override actually governs the field xAI reads, instead of
            # being silently outrun by the auto-derived cache_key (#78941).
            existing_extra_body = kwargs.get("extra_body")
            kwargs["extra_body"] = dict(existing_extra_body) if isinstance(existing_extra_body, dict) else {}
            kwargs["extra_body"].setdefault("prompt_cache_key", kwargs.get("prompt_cache_key", cache_key))

        _bound_prompt_cache_key_field(kwargs.get("extra_body"))
        return kwargs

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize Codex Responses API response to NormalizedResponse."""
        from agent.codex_responses_adapter import _normalize_codex_response

        msg, finish_reason = _normalize_codex_response(
            response, issuer_kind=kwargs.get("issuer_kind") or self._last_issuer_kind
        )

        tool_calls = None
        if msg and msg.tool_calls:
            tool_calls = []
            alias_map = self._last_wire_aliases
            for tc in msg.tool_calls:
                provider_data = {
                    key: getattr(tc, key) for key in ("call_id", "response_item_id") if getattr(tc, key, None)
                }
                has_fn = hasattr(tc, "function")
                name = tc.function.name if has_fn else getattr(tc, "name", "")
                # Undo only aliases THIS request emitted; the legacy map is for normalize-only call sites.
                if alias_map is None:
                    name = _LEGACY_ALIAS_FALLBACK.get(name, name)
                elif name in alias_map:
                    name = alias_map[name]
                tool_calls.append(ToolCall(
                    id=tc.id if hasattr(tc, "id") else (name or None), name=name,
                    arguments=tc.function.arguments if has_fn else getattr(tc, "arguments", "{}"),
                    provider_data=provider_data or None,
                ))

        provider_data = {
            key: getattr(msg, key, None)
            for key in ("codex_reasoning_items", "codex_message_items", "reasoning_details")
            if msg and getattr(msg, key, None)
        }
        return NormalizedResponse(
            content=msg.content if msg else None, tool_calls=tool_calls, finish_reason=finish_reason or "stop",
            reasoning=getattr(msg, "reasoning", None) if msg else None,
            usage=None,  # Codex usage is extracted separately in normalize_usage()
            provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        """True if response.output is a non-empty list, or a terminal content_filter refusal.

        An incomplete/content_filter response with no output must reach normalization,
        not a retry. Does NOT check output_text fallback — the caller handles that.
        """
        if response is None:
            return False
        output = getattr(response, "output", None)
        if isinstance(output, list) and output:
            return True
        status = str(getattr(response, "status", "") or "").strip().lower()
        details = getattr(response, "incomplete_details", None)
        raw_reason = details.get("reason") if isinstance(details, dict) else getattr(details, "reason", "")
        return status == "incomplete" and str(raw_reason or "").strip().lower() == "content_filter"

    def preflight_kwargs(
        self, api_kwargs: Any, *, allow_stream: bool = False, is_github_responses: bool = False,
        sanitize_harmony_tokens: bool = False,
    ) -> dict:
        """Validate and sanitize Codex API kwargs before the call.

        ``sanitize_harmony_tokens`` is for the ChatGPT Codex backend only (rejects literal Harmony tokens).
        """
        from agent.codex_responses_adapter import _preflight_codex_api_kwargs

        normalized = _preflight_codex_api_kwargs(
            api_kwargs, allow_stream=allow_stream, is_github_responses=is_github_responses,
            sanitize_harmony_tokens=sanitize_harmony_tokens,
        )
        _bound_prompt_cache_key_field(normalized)
        _bound_prompt_cache_key_field(normalized.get("extra_body"))
        return normalized


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("codex_responses", ResponsesApiTransport)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Dict  # noqa: F401,E402
from typing import List  # noqa: F401,E402
from typing import Tuple  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
