"""OpenAI Responses API (Codex) transport.

Delegates to the existing adapter functions in agent/codex_responses_adapter.py.
This transport owns format conversion and normalization — NOT client lifecycle,
streaming, or the _run_codex_stream() call path.
"""

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Cron fires build session_id as ``cron_<job_id>_<YYYYMMDD_HHMMSS>`` (see
# cron/scheduler.py). The trailing timestamp is per-fire noise; stripped so
# repeat fires of the same job share a cache scope (see #51395/#52295).
_CRON_SESSION_ID_RE = re.compile(r"^(cron_.+)_\d{8}_\d{6}$")


def _cache_scope_from_session_id(session_id: Optional[str]) -> str:
    """Normalize a physical session_id into a stable logical cache scope.

    Every non-cron session_id already identifies one conversation/agent
    instance (main run, a specific child/subagent, a sibling child, ...),
    so it is used unchanged. Only cron's per-fire timestamp needs stripping.
    """
    sid = str(session_id or "")
    match = _CRON_SESSION_ID_RE.match(sid)
    return match.group(1) if match else sid

from agent.reasoning_effort import (
    ACTUAL_RELAY_EFFORTS,
    XAI_GROK46_EFFORTS,
    XAI_LEGACY_EFFORTS,
    clamp_effort,
    codex_supported_efforts,
)
from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall


def _bounded_prompt_cache_key(value: Any) -> Optional[str]:
    """Return a provider-safe cache key without changing session identity."""
    if value is None:
        return None
    key = str(value).strip()
    if not key:
        return None
    if len(key) <= 64:
        return key
    # Match _content_cache_key's compact, collision-resistant routing-key shape.
    digest = hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"pck_{digest}"


# Wire-name used when Hermes keeps client-side web_search on xAI Responses.
# A function literally named ``web_search`` collides with Grok's native
# server-side tool (incomplete hang or HTTP 400 duplicate names); this alias
# avoids that while still dispatching through Hermes's configured provider
# (Firecrawl / Tavily / …). Mapped back to ``web_search`` in normalize_response.
_XAI_CLIENT_WEB_SEARCH_ALIAS = "hermes_web_search"

# OpenCode's /v1/responses endpoints (Zen and Go, including custom providers
# pointing at opencode.ai) reserve certain function names server-side and
# reject client tools that use them with HTTP 400 ("custom function name
# 'X' is reserved"). Reported for grok-4.5 on Go with `search_files` and
# `web_search` (#85589). Same treatment as the xAI web_search collision:
# rename on the wire (hermes_<name>), map back in normalize_response so
# Hermes dispatch is unaffected.
_OPENCODE_RESERVED_TOOL_NAMES = ("web_search", "search_files")

# xAI reserves ``tool_search`` server-side for Grok's own native Tool Search
# and rejects *any* client function declared with that name:
#   HTTP 400 {"code":"invalid-argument","error":"The function name
#   tool_search is reserved for the tool_search tool"}
# Hermes's progressive-disclosure bridge registers exactly that literal
# (``tools.tool_search.TOOL_SEARCH_NAME``), and assembly is not provider
# gated, so with ``tools.tool_search.enabled: auto`` a grok turn dies the
# moment the catalog crosses the threshold — mid-session, and only for
# sessions large enough to activate the bridge. Same treatment as the two
# collisions above. ``tool_describe`` / ``tool_call`` are not reserved.
# Refs #95003.
_XAI_RESERVED_TOOL_NAMES = ("tool_search",)

_RESERVED_TOOL_ALIAS_PREFIX = "hermes_"
_RESERVED_ALIAS_TO_NAME = {
    f"{_RESERVED_TOOL_ALIAS_PREFIX}{name}": name
    for name in (*_OPENCODE_RESERVED_TOOL_NAMES, *_XAI_RESERVED_TOOL_NAMES)
}

# Legacy reverse map used ONLY when normalize_response runs on a transport
# instance that never built a request (normalize-only call sites / tests).
# Production requests carry request-local provenance instead — see
# ``_last_wire_aliases`` — so a real user/plugin/MCP tool that happens to be
# named ``hermes_tool_search`` is never silently rewritten to ``tool_search``
# unless THIS request actually emitted that alias (#95003 review contract).
_LEGACY_ALIAS_FALLBACK = {
    **_RESERVED_ALIAS_TO_NAME,
    "hermes_web_search": "web_search",
}


def _is_opencode_responses_backend(params: Dict[str, Any]) -> bool:
    """True when this Responses request targets an OpenCode endpoint.

    Matches the built-in opencode-zen/go providers, custom ``opencode-go-*`` /
    ``opencode-zen-*`` family providers, and any base_url hosted on
    opencode.ai (covers custom providers with arbitrary names pointing at
    the OpenCode gateway).
    """
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
    response_tools: List[Dict[str, Any]],
    reserved_names: Tuple[str, ...],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Alias provider-reserved client function names on the wire.

    Single owner for every reserved-name collision on this transport.
    Returns ``(rewritten_tools, alias_map)`` where ``alias_map`` maps each
    wire alias emitted by THIS request back to the original tool name.
    The caller stashes the map for ``normalize_response`` so the reverse
    rewrite only ever applies to aliases this request actually sent —
    a legitimate user/plugin/MCP tool already named ``hermes_<x>`` is
    neither shadowed (the alias picks a ``_2``/``_3`` suffix instead of
    duplicating a wire name) nor mis-dispatched on the response path.
    """
    rewritten: List[Dict[str, Any]] = []
    alias_map: Dict[str, str] = {}
    taken = {
        tool.get("name")
        for tool in response_tools
        if isinstance(tool, dict) and tool.get("name")
    }
    for tool in response_tools:
        if isinstance(tool, dict) and tool.get("name") in reserved_names:
            base = f"{_RESERVED_TOOL_ALIAS_PREFIX}{tool['name']}"
            alias = base
            suffix = 2
            while alias in taken:
                alias = f"{base}_{suffix}"
                suffix += 1
            taken.add(alias)
            alias_map[alias] = tool["name"]
            aliased = dict(tool)
            aliased["name"] = alias
            rewritten.append(aliased)
        else:
            rewritten.append(tool)
    return rewritten, alias_map


def _xai_prefers_native_web_search() -> bool:
    """True when xAI Responses should use Grok's native ``web_search`` built-in.

    Delegates to the web-search registry's provider resolution (which reads
    ``web.search_backend`` / ``web.backend`` from config) and checks whether
    the resolved provider is xAI. Falls back to the legacy ``_get_search_backend``
    probe when the registry has no providers loaded. On any resolution failure,
    returns True (fail-closed to native — preserves the #48108 incomplete-hang
    fix rather than risk reintroducing it).
    """
    try:
        from agent.web_search_registry import get_active_search_provider

        provider = get_active_search_provider()
        if provider is not None:
            return getattr(provider, "name", None) == "xai"

        from tools.web_tools import _get_search_backend

        return (_get_search_backend() or "").strip().lower() == "xai"
    except Exception:
        # Fail closed to native — same behavior as pre-fix main.
        return True


def _rename_client_web_search_for_xai(response_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rename client ``web_search`` → alias so xAI won't hijack it server-side."""
    rewritten: List[Dict[str, Any]] = []
    for tool in response_tools:
        if isinstance(tool, dict) and tool.get("name") == "web_search":
            aliased = dict(tool)
            aliased["name"] = _XAI_CLIENT_WEB_SEARCH_ALIAS
            rewritten.append(aliased)
        else:
            rewritten.append(tool)
    return rewritten


_EXTENDED_PROMPT_CACHE_MODELS = (
    "gpt-5.5-pro",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.2",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
    "gpt-5.1-chat-latest",
    "gpt-5.1-codex",
    "gpt-5.1",
    "gpt-5-codex",
    "gpt-5",
    "gpt-4.1",
)
_EXTENDED_PROMPT_CACHE_MODEL_RE = re.compile(
    rf"(?:^|[./:])(?:{'|'.join(re.escape(name) for name in _EXTENDED_PROMPT_CACHE_MODELS)})"
    r"(?:-\d{4}-\d{2}-\d{2})?$"
)


def _default_prompt_cache_retention_for_request(
    model: str,
    base_url: Any,
) -> Optional[str]:
    """Return ``24h`` for supported hosts/models (Bedrock Mantle, Meta)."""
    from utils import base_url_hostname

    hostname = base_url_hostname(str(base_url or "")).lower()
    # Meta Model API: prompt caching is opt-in via prompt_cache_retention.
    # Measured 0% hits on /chat/completions vs 93-99% on /responses with 24h.
    if hostname == "api.meta.ai":
        return "24h"

    hostname_parts = hostname.split(".")
    is_bedrock_mantle = (
        len(hostname_parts) == 4
        and hostname_parts[0] == "bedrock-mantle"
        and bool(hostname_parts[1])
        and hostname_parts[2:] == ["api", "aws"]
    )
    if not is_bedrock_mantle:
        return None

    normalized = str(model or "").strip().lower().replace("_", "-")
    if _EXTENDED_PROMPT_CACHE_MODEL_RE.search(normalized):
        return "24h"
    return None


def _content_cache_key(
    instructions: str,
    tools: Optional[List[Dict[str, Any]]],
    scope_id: str = "",
) -> Optional[str]:
    """Content-address the prompt cache key within a logical cache scope.

    Returns ``pck_<sha256[:24]>`` of (scope_id + instructions + sorted tool
    schemas), or None when there is nothing static to key on. The cache key
    is a routing hint only — never a correctness boundary — so two requests
    sharing a scope, system prompt, and tool set intentionally resolve to the
    same warm prefix bucket.

    ``scope_id`` (pass ``_cache_scope_from_session_id(session_id)``) keeps
    unrelated sessions — independent conversations, main vs. child/subagent,
    sibling children — from concentrating onto the same bucket merely because
    their static prefix matches (see #78941), while still letting recurring
    cron fires of one job share a stable key across their timestamped
    session_ids (the original #51395/#52295 fix this built on). Sorting tools
    by name keeps the hash insertion-order independent.
    """
    if not instructions and not tools:
        return None
    tools_part = ""
    if tools:
        sorted_tools = sorted(
            (t for t in tools if isinstance(t, dict)),
            key=lambda t: str(t.get("name") or t.get("type") or ""),
        )
        tools_part = json.dumps(
            sorted_tools, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
    # \x00 separators so a scope/instructions/tools boundary can't be forged
    # by content that happens to contain the same bytes.
    content = f"{scope_id}\x00{instructions or ''}\x00{tools_part}"
    digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"pck_{digest}"


def _profile_declared_efforts(
    provider: Any, model: Optional[str], base_url: Any = None
) -> Optional[tuple]:
    """Provider-profile-declared reasoning-effort vocabulary, or None.

    Thin, fail-open wrapper around
    ``ProviderProfile.supported_reasoning_efforts`` (see providers/base.py
    for the tri-state contract). Lazy import: provider plugins import this
    transport during registry discovery, so a module-level import of
    ``providers`` would cycle.

    Resolution is by provider name first, then by the endpoint's host: a
    named custom provider pointed at a known provider's endpoint (e.g. a
    ``providers.my-proxy`` entry with base_url ``https://api.router.com/v1``,
    which the host mandate routes onto this transport) must get that
    provider's declared vocabulary too — the host, not the config-entry
    name, is what validates the request.
    """
    try:
        from providers import get_provider_profile

        name = str(provider or "").strip().lower()
        profile = get_provider_profile(name) if name else None
        declared = (
            profile.supported_reasoning_efforts(model)
            if profile is not None
            else None
        )
        if declared is None and base_url:
            from agent.model_metadata import _infer_provider_from_url

            inferred = _infer_provider_from_url(str(base_url))
            if inferred and inferred != name:
                inferred_profile = get_provider_profile(inferred)
                if inferred_profile is not None:
                    declared = inferred_profile.supported_reasoning_efforts(model)
    except Exception as exc:
        # Fail-open by design: a broken profile hook must never block the
        # request — the transport falls back to its default vocabulary.
        logger.debug("profile-declared efforts lookup failed: %s", exc)
        return None
    if declared is None:
        return None
    return tuple(declared)


def _is_azure_foundry_responses(params: Dict[str, Any]) -> bool:
    """Return True for Microsoft Foundry's OpenAI-compatible Responses API.

    Matched on the registered provider id first, then on the endpoint host.
    Host matching goes through ``base_url_host_matches`` rather than a
    substring test, so a path or query segment carrying the Foundry domain
    (``https://proxy.example.com/.services.ai.azure.com/v1``) is not
    misclassified as Foundry.
    """
    from utils import base_url_host_matches

    provider = str(params.get("provider") or "").strip().lower()
    if provider == "azure-foundry":
        return True

    return base_url_host_matches(
        str(params.get("base_url") or ""), "services.ai.azure.com"
    )


def _is_post_tool_replay(messages: Optional[List[Dict[str, Any]]]) -> bool:
    """Return True when ``messages`` end on a tool result awaiting a follow-up.

    Azure Foundry only rejects the *post-tool follow-up* payload — the shape
    where a prior assistant ``function_call`` and its ``function_call_output``
    are replayed alongside an encrypted ``reasoning`` item (HTTP 400
    invalid_payload). Detecting that shape here keeps reasoning suppression
    scoped to the failing turn, so ordinary (non-tool) Foundry multi-turn
    continuity is left unchanged.

    The test is on the *trailing* messages, not on the history as a whole.
    Scanning the whole history for any tool call plus any tool result makes
    the predicate sticky: one tool call early in a conversation would then
    suppress reasoning on every later turn, including plain user follow-ups
    that Foundry accepts. The rejected payload is specifically the turn whose
    last item is a tool result, so that is what this matches: the final
    non-system message is a ``tool`` result, and the assistant message that
    issued its ``tool_call_id`` is present.

    Tool-call identity is resolved the same way
    ``_chat_messages_to_responses_input`` resolves it, because the pairing
    that matters is the one that reaches the wire. A stored tool call can
    carry the function call id in ``call_id``, in ``id``, or in a composite
    ``"call_x|fc_y"`` id, and a bare ``fc_``-prefixed ``id`` is a response
    item id that the converter turns into ``call_<rest>``. Matching only
    ``id`` would miss the ``id=fc_… / call_id=call_…`` shape that resumed
    legacy sessions and host-fed histories still use, and let the rejected
    payload through.
    """
    from agent.codex_responses_adapter import (
        _canonical_call_id_from_fc,
        _split_responses_tool_id,
    )

    def _pair_ids(raw: Any, explicit: Any = None) -> set:
        """Every call id a stored tool id could pair on, converter-order."""
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
        if not isinstance(msg, dict):
            return False
        role = msg.get("role")
        if role == "system":
            continue
        if role == "tool":
            ids = _pair_ids(msg.get("tool_call_id"))
            if not ids:
                return False
            trailing |= ids
            continue
        # First message before the trailing run of tool results. It must be
        # the assistant turn that issued them for this to be the follow-up
        # payload; a non-empty ``trailing`` is what proves the run existed.
        if role != "assistant":
            return False
        return any(
            trailing & _pair_ids(call.get("id"), call.get("call_id"))
            for call in msg.get("tool_calls") or []
            if isinstance(call, dict)
        )

    return False


def _native_compaction_active(context_management: Any) -> bool:
    """Is THIS request natively compacted?

    True only when the caller's eligibility gate
    (``native_compaction.native_compaction_context_management``) produced a
    non-empty payload. Every native-compaction side effect on the wire —
    sending ``context_management``, replaying a ``type: "compaction"``
    checkpoint, restructuring the input around it — hangs off this one
    predicate, so a checkpoint that outlives the gate (model swapped out of
    the gpt-5.6 family, compression disabled, rejection kill switch, resumed
    session) cannot keep reshaping requests on its own.
    """
    return isinstance(context_management, list) and bool(context_management)


class ResponsesApiTransport(ProviderTransport):
    """Transport for api_mode='codex_responses'.

    Wraps the functions extracted into codex_responses_adapter.py (PR 1).
    """

    # Issuer kind of the most recent build_kwargs / convert_messages call.
    # Used as a fallback when normalize_response is invoked without an
    # explicit ``issuer_kind`` kwarg, so reasoning items captured from a
    # response are stamped with the endpoint that minted them. Plain class
    # attribute default; mutated on the instance, not the class.
    _last_issuer_kind: Optional[str] = None

    # Wire-alias provenance of the most recent build_kwargs call:
    # ``{alias_sent_on_wire: original_tool_name}``. ``None`` means "no
    # request built on this instance" (normalize-only call sites), in which
    # case normalize_response falls back to the static legacy map. An empty
    # dict means the last request emitted no aliases, so no reverse rewrite
    # is permitted (#95003 provenance contract).
    _last_wire_aliases: Optional[Dict[str, str]] = None

    @property
    def api_mode(self) -> str:
        return "codex_responses"

    def _resolve_issuer_kind(self, params: Dict[str, Any]) -> str:
        """Classify the current Responses endpoint from transport params."""
        from agent.codex_responses_adapter import _classify_responses_issuer
        return _classify_responses_issuer(
            is_xai_responses=params.get("is_xai_responses") is True,
            is_github_responses=params.get("is_github_responses") is True,
            is_codex_backend=params.get("is_codex_backend") is True,
            base_url=params.get("base_url"),
        )

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Convert OpenAI chat messages to Responses API input items."""
        from agent.codex_responses_adapter import _chat_messages_to_responses_input
        issuer = self._resolve_issuer_kind(kwargs)
        self._last_issuer_kind = issuer
        return _chat_messages_to_responses_input(
            messages,
            is_xai_responses=kwargs.get("is_xai_responses") is True,
            is_github_responses=kwargs.get("is_github_responses") is True,
            replay_encrypted_reasoning=bool(
                kwargs.get("replay_encrypted_reasoning", True)
            ),
            current_issuer_kind=issuer,
            native_compaction_eligible=_native_compaction_active(
                kwargs.get("context_management")
            ),
            reasoning_replay_keep_last=kwargs.get("reasoning_replay_keep_last"),
        )

    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        """Convert OpenAI tool schemas to Responses API function definitions."""
        from agent.codex_responses_adapter import _responses_tools
        return _responses_tools(tools)

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Build Responses API kwargs.

        Calls convert_messages and convert_tools internally.

        params:
            instructions: str — system prompt (extracted from messages[0] if not given)
            reasoning_config: dict | None — {effort, enabled}
            session_id: str | None — transcript/session id; drives the Codex
                ``session_id`` header, and is the cache-scope fallback when no
                ``cache_scope_id`` is given
            cache_scope_id: str | None — rotation-stable logical scope id
                (compression-lineage root; see agent/prompt_cache_scope.py).
                Preferred over session_id when deriving the prompt_cache_key
                content hash and the xAI x-grok-conv-id header; the Codex
                x-client-request-id header mirrors the resulting body key.
                Keeps the cache warm across context-compression session
                rotation (#79017)
            max_tokens: int | None — max_output_tokens
            timeout: float | None — per-request timeout forwarded to the SDK
            request_overrides: dict | None — extra kwargs merged in
            provider: str | None — provider name for backend-specific logic
            base_url: str | None — endpoint URL
            base_url_hostname: str | None — hostname for backend detection
            is_github_responses: bool — Copilot/GitHub models backend
            is_codex_backend: bool — chatgpt.com/backend-api/codex
            is_xai_responses: bool — xAI/Grok backend
            github_reasoning_extra: dict | None — Copilot reasoning params
        """
        from agent.codex_responses_adapter import (
            _chat_messages_to_responses_input,
            _responses_tools,
        )

        from run_agent import DEFAULT_AGENT_IDENTITY

        instructions = params.get("instructions", "")
        payload_messages = messages
        if not instructions:
            if messages and messages[0].get("role") == "system":
                instructions = str(messages[0].get("content") or "").strip()
                payload_messages = messages[1:]
        if not instructions:
            instructions = DEFAULT_AGENT_IDENTITY

        is_github_responses = params.get("is_github_responses") is True
        is_codex_backend = params.get("is_codex_backend") is True
        is_xai_responses = params.get("is_xai_responses") is True
        replay_encrypted_reasoning = bool(
            params.get("replay_encrypted_reasoning", True)
        )
        if replay_encrypted_reasoning and _is_azure_foundry_responses(params):
            # Microsoft Foundry accepts the initial Responses function-call
            # request and ordinary (non-tool) multi-turn continuity, but
            # rejects the post-tool follow-up payload that carries prior
            # encrypted reasoning items alongside function_call /
            # function_call_output, with HTTP 400 invalid_payload. Scope the
            # suppression to that follow-up turn: keep function_call /
            # function_call_output continuity intact and drop only the
            # encrypted reasoning replay for this endpoint.
            if _is_post_tool_replay(payload_messages):
                replay_encrypted_reasoning = False
        # Native server-side compaction (gpt-5.6 on direct OpenAI/Codex routes
        # only). The caller resolves eligibility via
        # agent.native_compaction.native_compaction_context_management();
        # None means the field is never added to the request.
        context_management = params.get("context_management")
        # Single source of truth for "this request is natively compacted":
        # the same value decides whether the field goes out AND whether the
        # converter may replay/prune around a compaction checkpoint. Keeping
        # them derived from one expression is what stops a persisted
        # checkpoint from restructuring the wire after the gate closes.
        native_compaction_active = _native_compaction_active(context_management)

        # Resolve the issuing endpoint for this call. Stashed on the
        # transport so normalize_response can stamp it onto reasoning
        # items captured from the response, and passed to the input
        # converter so foreign-issuer reasoning blocks in history are
        # dropped before the API rejects them.
        issuer_kind = self._resolve_issuer_kind(params)
        self._last_issuer_kind = issuer_kind

        # Resolve reasoning effort
        reasoning_effort = "medium"
        reasoning_enabled = True
        reasoning_config = params.get("reasoning_config")
        if reasoning_config and isinstance(reasoning_config, dict):
            if reasoning_config.get("enabled") is False:
                reasoning_enabled = False
            elif reasoning_config.get("effort"):
                reasoning_effort = reasoning_config["effort"]

        # Wire vocabularies are declared in agent.reasoning_effort; the shared
        # clamp policy (nearest weaker supported level, never escalate,
        # never invert the ladder) replaces the per-backend hand maps that
        # repeatedly leaked internal levels like "ultra" to the wire
        # (#89503 class) or clamped one rung below a model's real ceiling
        # (#87279).
        if params.get("is_xai_responses", False):
            from agent.model_metadata import is_grok_46_family

            # Grok 4.6 accepts xhigh as a wire value; older Grok tops out
            # at high.
            _supported = (
                XAI_GROK46_EFFORTS if is_grok_46_family(model)
                else XAI_LEGACY_EFFORTS
            )
        elif (params.get("provider") or "").strip().lower() == "actual":
            # Actual Computer relays to SGLang/vLLM backends:
            # none/low/medium/high/max.
            _supported = ACTUAL_RELAY_EFFORTS
        else:
            # Profile-declared vocabulary first: gateways that validate
            # reasoning.effort per model (Ramp Router reads its live catalog)
            # declare it via ProviderProfile.supported_reasoning_efforts.
            # ``()`` is the definitive "this model takes no reasoning
            # parameters" verdict — such backends 400 on any reasoning field
            # rather than ignoring it, so suppress reasoning entirely.
            _supported = None
            _declared = _profile_declared_efforts(
                params.get("provider"), model, params.get("base_url")
            )
            if _declared is not None:
                if not _declared:
                    reasoning_enabled = False
                else:
                    _supported = _declared
            if _supported is None:
                # OpenAI/Codex Responses backend — per-model vocabulary
                # (live-verified: "max" is gpt-5.6-only, "minimal" always
                # rejected). #68365 premise confirmed.
                _supported = codex_supported_efforts(model)
        reasoning_effort = clamp_effort(reasoning_effort, _supported)

        response_tools = _responses_tools(tools)

        # xAI server-side web search vs Hermes web providers.
        #
        # grok models on xAI's /v1/responses surface have a *native*,
        # server-executed web search.  A client-side function literally named
        # ``web_search`` collides with that engine: declared as a plain
        # ``function`` rather than ``{"type": "web_search"}``, the search
        # dispatches but never reconciles → incomplete turn + 3 retries.
        # Verified live against grok-composer-2.5-fast (2026-06); see #48108.
        #
        # Two modes, chosen by the user's web-search backend config:
        #
        # 1. **Native** (active/configured backend is ``xai``, or resolution
        #    fails): drop the client ``web_search`` function and declare
        #    xAI's built-in instead. 1:1 swap only when client ``web_search``
        #    was already present — never an additive grant.
        # 2. **Client** (Firecrawl / Tavily / Exa / … configured or resolved):
        #    keep Hermes dispatch so ``web.backend`` / ``web.search_backend``
        #    is honored, but rename the wire tool to
        #    ``hermes_web_search`` so Grok cannot hijack the name. The alias
        #    is mapped back to ``web_search`` in ``normalize_response``.
        # Request-local alias provenance: every wire alias THIS request
        # emits is recorded here and stashed on the transport, so the
        # reverse rewrite in ``normalize_response`` applies only to aliases
        # that were actually sent (never to a real tool that merely shares
        # an alias-shaped name).
        wire_aliases: Dict[str, str] = {}
        if is_xai_responses and response_tools:
            has_client_web_search = any(
                isinstance(t, dict) and t.get("name") == "web_search"
                for t in response_tools
            )
            if has_client_web_search:
                if _xai_prefers_native_web_search():
                    filtered = [
                        t for t in response_tools
                        if not (isinstance(t, dict) and t.get("name") == "web_search")
                    ]
                    filtered.append({"type": "web_search"})
                    response_tools = filtered
                else:
                    response_tools = _rename_client_web_search_for_xai(response_tools)
                    wire_aliases[_XAI_CLIENT_WEB_SEARCH_ALIAS] = "web_search"

        # OpenCode Responses backends reserve web_search / search_files as
        # function names (HTTP 400 "custom function name 'X' is reserved",
        # #85589). Alias them on the wire; normalize_response maps them back.
        if response_tools and _is_opencode_responses_backend(params):
            response_tools, _oc_aliases = _alias_reserved_tools(
                response_tools, _OPENCODE_RESERVED_TOOL_NAMES
            )
            wire_aliases.update(_oc_aliases)

        # xAI reserves ``tool_search`` for its native server-side tool and
        # rejects the client declaration outright (#95003). Alias it on the
        # wire; normalize_response maps it back before dispatch.
        if is_xai_responses and response_tools:
            response_tools, _xai_aliases = _alias_reserved_tools(
                response_tools, _XAI_RESERVED_TOOL_NAMES
            )
            wire_aliases.update(_xai_aliases)

        # Stash for normalize_response (same request/response pairing model
        # as ``_last_issuer_kind``). An empty dict is meaningful: it means
        # this request emitted NO aliases, so no reverse rewrite may run.
        self._last_wire_aliases = wire_aliases

        # ``tools`` MUST be omitted entirely when there are no functions to
        # expose: the openai SDK's ``responses.stream()`` / ``responses.parse()``
        # eagerly call ``_make_tools(tools)`` which does ``for tool in tools``
        # without a None guard, so passing ``tools=None`` raises
        # ``TypeError: 'NoneType' object is not iterable`` before any HTTP
        # request is issued (openai==2.24.0).  Reported for the
        # ``openai-codex`` / ``gpt-5.5`` combo on chatgpt.com/backend-api/codex
        # (#32892) when the agent runs without external tools registered.
        # Function-level import: agent.model_metadata is imported lazily
        # because provider plugins import this transport during
        # model_metadata's own module init (circular otherwise).
        from agent.model_metadata import (
            strip_codex_context_variant_suffix as _strip_ctx_variant,
        )
        kwargs = {
            # ``-900k`` large-context picker variants are Hermes-side aliases
            # (gpt-5.6-sol-900k etc.) — the Codex/OpenAI backend only knows
            # the base slug, so strip the suffix before it hits the wire.
            "model": _strip_ctx_variant(model),
            "instructions": instructions,
            "input": _chat_messages_to_responses_input(
                payload_messages,
                is_xai_responses=is_xai_responses,
                is_github_responses=is_github_responses,
                replay_encrypted_reasoning=replay_encrypted_reasoning,
                current_issuer_kind=issuer_kind,
                native_compaction_eligible=native_compaction_active,
                reasoning_replay_keep_last=params.get("reasoning_replay_keep_last"),
            ),
            "store": False,
        }
        if response_tools:
            kwargs["tools"] = response_tools
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = True
        if native_compaction_active:
            kwargs["context_management"] = context_management

        session_id = params.get("session_id")
        # prompt_cache_key is content-addressed from the static prefix
        # (instructions + tools) scoped by session, NOT the raw session_id —
        # recurring cron jobs carry a per-fire timestamp in session_id
        # (cron_<id>_<ts>) that made every run cache-cold, so the scope strips
        # that suffix (see _cache_scope_from_session_id). session_id is left
        # untouched for transcript isolation (the Codex ``session_id`` header
        # below). Falls back to session_id when there is no static content to
        # hash.
        #
        # cache_scope_id, when provided, is the rotation-stable logical scope
        # (compression-lineage root — agent/prompt_cache_scope.py): legacy
        # ``compression.in_place: false`` compaction rotates session_id
        # mid-conversation, and scoping by the physical id went cache-cold at
        # every rotation boundary (#79017).
        _cache_scope = _cache_scope_from_session_id(
            params.get("cache_scope_id") or session_id
        )
        cache_key = _content_cache_key(
            instructions, response_tools, _cache_scope
        ) or _cache_scope
        # xAI Responses takes prompt_cache_key in extra_body (set further
        # down); GitHub Models opts out of cache-key routing entirely.
        if not is_github_responses and not is_xai_responses and cache_key:
            kwargs["prompt_cache_key"] = cache_key

        cache_retention = _default_prompt_cache_retention_for_request(
            model,
            params.get("base_url"),
        )
        if cache_retention:
            kwargs.setdefault("prompt_cache_retention", cache_retention)

        if reasoning_enabled and is_xai_responses:
            from agent.model_metadata import grok_supports_reasoning_effort

            # Ask xAI to echo back encrypted reasoning items so we can
            # replay them on subsequent turns for cross-turn coherence.
            # See agent/codex_responses_adapter._chat_messages_to_responses_input
            # for the May 2026 reversal of the earlier suppression gate.
            kwargs["include"] = (
                ["reasoning.encrypted_content"] if replay_encrypted_reasoning else []
            )
            # xAI rejects `reasoning.effort` on grok-4 / grok-4-fast / grok-3
            # / grok-code-fast / grok-4.20-0309-* with HTTP 400 even though
            # those models reason natively. Only send the effort dial when
            # the target model is on the allowlist; otherwise send no
            # `reasoning` key at all and let the model reason on its own.
            if grok_supports_reasoning_effort(model):
                kwargs["reasoning"] = {"effort": reasoning_effort}
        elif reasoning_enabled:
            if is_github_responses:
                github_reasoning = params.get("github_reasoning_extra")
                if github_reasoning is not None:
                    kwargs["reasoning"] = github_reasoning
            else:
                kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
                kwargs["include"] = (
                    ["reasoning.encrypted_content"] if replay_encrypted_reasoning else []
                )
        elif not is_github_responses and not is_xai_responses:
            kwargs["include"] = []

        request_overrides = params.get("request_overrides")
        if request_overrides:
            kwargs.update(request_overrides)

        if "prompt_cache_key" in kwargs:
            bounded_cache_key = _bounded_prompt_cache_key(kwargs["prompt_cache_key"])
            if bounded_cache_key:
                kwargs["prompt_cache_key"] = bounded_cache_key
            else:
                kwargs.pop("prompt_cache_key", None)

        # Older xAI Responses models reject ``service_tier`` (HTTP 400
        # "Argument not supported: service_tier"). Grok 4.6 accepts Priority
        # Processing, but continue stripping stale or unsupported tier values
        # on every other xAI path. See #28490 and #84799.
        if is_xai_responses:
            from agent.model_metadata import is_grok_46_family

            if not (
                is_grok_46_family(model)
                and kwargs.get("service_tier") == "priority"
            ):
                kwargs.pop("service_tier", None)

        # Forward per-request timeout to the SDK so OpenAI/Anthropic clients
        # honor it.  Without this, ``providers.<id>.request_timeout_seconds``
        # is silently dropped on the main agent Codex path while the
        # chat_completions path and auxiliary Codex adapter both forward it.
        timeout = kwargs.get("timeout", params.get("timeout"))
        if (
            isinstance(timeout, (int, float))
            and not isinstance(timeout, bool)
            and 0 < float(timeout) < float("inf")
        ):
            kwargs["timeout"] = float(timeout)
        else:
            kwargs.pop("timeout", None)

        if is_codex_backend:
            # The Codex backend rejects body-level ``extra_headers`` with
            # HTTP 400, but the OpenAI SDK's ``extra_headers`` kwarg maps
            # to actual HTTP request headers (not body fields).  ``session_id``
            # carries the raw physical session id — transcript/identity, per
            # the #57012 contract — while ``x-client-request-id`` mirrors the
            # body's effective ``prompt_cache_key`` so header and body always
            # agree on the same routing bucket instead of diverging (#78941).
            final_cache_key = kwargs.get("prompt_cache_key") or _bounded_prompt_cache_key(_cache_scope)
            if session_id or final_cache_key:
                existing_extra_headers = kwargs.get("extra_headers")
                merged_extra_headers: Dict[str, str] = {}
                if isinstance(existing_extra_headers, dict):
                    merged_extra_headers.update(
                        {
                            str(key): str(value)
                            for key, value in existing_extra_headers.items()
                            if key and value is not None
                        }
                    )
                if session_id:
                    merged_extra_headers["session_id"] = str(session_id)
                if final_cache_key:
                    merged_extra_headers["x-client-request-id"] = final_cache_key
                kwargs["extra_headers"] = merged_extra_headers

        max_tokens = params.get("max_tokens")
        if max_tokens is not None and not is_codex_backend:
            kwargs["max_output_tokens"] = max_tokens

        if is_xai_responses and session_id:
            existing_extra_headers = kwargs.get("extra_headers")
            merged_extra_headers: Dict[str, str] = {}
            if isinstance(existing_extra_headers, dict):
                merged_extra_headers.update(
                    {
                        str(key): str(value)
                        for key, value in existing_extra_headers.items()
                        if key and value is not None
                    }
                )
            # Scoped like the body cache key below — otherwise cron's
            # per-fire timestamp in session_id (cron_<id>_<ts>) pins every
            # fire of the same job to a different xAI backend server (#78941).
            merged_extra_headers["x-grok-conv-id"] = _cache_scope
            kwargs["extra_headers"] = merged_extra_headers

            # xAI Responses cache-routing — body-level field per
            # https://docs.x.ai/developers/advanced-api-usage/prompt-caching/maximizing-cache-hits.
            # Sent via extra_body (not the typed kwarg) so it survives openai
            # SDK builds whose Responses.stream() signature has dropped the field.
            # A caller's request_overrides={"prompt_cache_key": ...} lands on
            # the top-level kwarg set above — read it back here so an explicit
            # override actually governs the field xAI reads, instead of being
            # silently outrun by the auto-derived cache_key (#78941).
            existing_extra_body = kwargs.get("extra_body")
            merged_extra_body: Dict[str, Any] = {}
            if isinstance(existing_extra_body, dict):
                merged_extra_body.update(existing_extra_body)
            merged_extra_body.setdefault(
                "prompt_cache_key", kwargs.get("prompt_cache_key", cache_key)
            )
            kwargs["extra_body"] = merged_extra_body

        extra_body = kwargs.get("extra_body")
        if isinstance(extra_body, dict) and "prompt_cache_key" in extra_body:
            bounded_cache_key = _bounded_prompt_cache_key(extra_body["prompt_cache_key"])
            if bounded_cache_key:
                extra_body["prompt_cache_key"] = bounded_cache_key
            else:
                extra_body.pop("prompt_cache_key", None)

        return kwargs

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize Codex Responses API response to NormalizedResponse."""
        from agent.codex_responses_adapter import (
            _normalize_codex_response,
        )

        # Issuer for this response = explicit kwarg if the caller knows it,
        # otherwise the stash from the matching build_kwargs/convert_messages
        # call. Either way it gets stamped onto reasoning items so future
        # turns can detect a model swap and drop foreign-issuer blobs.
        issuer_kind = kwargs.get("issuer_kind") or self._last_issuer_kind
        # _normalize_codex_response returns (SimpleNamespace, finish_reason_str)
        msg, finish_reason = _normalize_codex_response(response, issuer_kind=issuer_kind)

        tool_calls = None
        if msg and msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                provider_data = {}
                if hasattr(tc, "call_id") and tc.call_id:
                    provider_data["call_id"] = tc.call_id
                if hasattr(tc, "response_item_id") and tc.response_item_id:
                    provider_data["response_item_id"] = tc.response_item_id
                name = tc.function.name if hasattr(tc, "function") else getattr(tc, "name", "")
                # Undo THIS request's wire aliases before Hermes dispatch.
                # Request-local provenance: only aliases the paired
                # build_kwargs call actually emitted are rewritten, so a
                # legitimate tool that happens to be named
                # ``hermes_tool_search`` etc. is dispatched as itself when
                # no alias was sent. The static legacy map is used only for
                # normalize-only call sites that never built a request on
                # this transport instance.
                alias_map = self._last_wire_aliases
                if alias_map is None:
                    if name == _XAI_CLIENT_WEB_SEARCH_ALIAS:
                        name = "web_search"
                    elif name in _LEGACY_ALIAS_FALLBACK:
                        name = _LEGACY_ALIAS_FALLBACK[name]
                elif name in alias_map:
                    name = alias_map[name]
                tool_calls.append(ToolCall(
                    id=tc.id if hasattr(tc, "id") else (name or None),
                    name=name,
                    arguments=tc.function.arguments if hasattr(tc, "function") else getattr(tc, "arguments", "{}"),
                    provider_data=provider_data or None,
                ))

        # Extract reasoning items for provider_data
        provider_data = {}
        if msg and hasattr(msg, "codex_reasoning_items") and msg.codex_reasoning_items:
            provider_data["codex_reasoning_items"] = msg.codex_reasoning_items
        if msg and hasattr(msg, "codex_message_items") and msg.codex_message_items:
            provider_data["codex_message_items"] = msg.codex_message_items
        if msg and hasattr(msg, "reasoning_details") and msg.reasoning_details:
            provider_data["reasoning_details"] = msg.reasoning_details

        return NormalizedResponse(
            content=msg.content if msg else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            reasoning=msg.reasoning if msg and hasattr(msg, "reasoning") else None,
            usage=None,  # Codex usage is extracted separately in normalize_usage()
            provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        """Check Codex Responses API response has valid output structure.

        Returns True only if response.output is a non-empty list. Also treats
        terminal content-filter incomplete responses as valid: the Responses API
        may return status=incomplete with incomplete_details.reason='content_filter'
        and no output items. That is a provider refusal signal, not a malformed
        response, and must reach normalization so the agent loop can use the
        content-policy / fallback path instead of invalid-response retries.

        Does NOT check output_text fallback — the caller handles that with
        diagnostic logging for stream backfill recovery.
        """
        if response is None:
            return False
        output = getattr(response, "output", None)
        if not isinstance(output, list) or not output:
            status = str(getattr(response, "status", "") or "").strip().lower()
            incomplete_details = getattr(response, "incomplete_details", None)
            if isinstance(incomplete_details, dict):
                reason = str(incomplete_details.get("reason") or "").strip().lower()
            else:
                reason = str(getattr(incomplete_details, "reason", "") or "").strip().lower()
            return status == "incomplete" and reason == "content_filter"
        return True

    def preflight_kwargs(
        self,
        api_kwargs: Any,
        *,
        allow_stream: bool = False,
        is_github_responses: bool = False,
        sanitize_harmony_tokens: bool = False,
    ) -> dict:
        """Validate and sanitize Codex API kwargs before the call.

        Normalizes input items, strips unsupported fields, validates structure.
        ``sanitize_harmony_tokens`` is enabled only for the ChatGPT Codex
        backend, which rejects literal reserved Harmony wire tokens in text.
        """
        from agent.codex_responses_adapter import _preflight_codex_api_kwargs

        normalized = _preflight_codex_api_kwargs(
            api_kwargs,
            allow_stream=allow_stream,
            is_github_responses=is_github_responses,
            sanitize_harmony_tokens=sanitize_harmony_tokens,
        )
        if "prompt_cache_key" in normalized:
            bounded = _bounded_prompt_cache_key(normalized["prompt_cache_key"])
            if bounded:
                normalized["prompt_cache_key"] = bounded
            else:
                normalized.pop("prompt_cache_key", None)
        extra_body = normalized.get("extra_body")
        if isinstance(extra_body, dict) and "prompt_cache_key" in extra_body:
            bounded = _bounded_prompt_cache_key(extra_body["prompt_cache_key"])
            if bounded:
                extra_body["prompt_cache_key"] = bounded
            else:
                extra_body.pop("prompt_cache_key", None)
        return normalized

    def map_finish_reason(self, raw_reason: str) -> str:
        """Map Codex response.status to OpenAI finish_reason.

        Codex uses response.status ('completed', 'incomplete') +
        response.incomplete_details.reason for granular mapping.
        This method handles the simple status string; the caller
        should check incomplete_details separately for 'max_output_tokens'.
        """
        _MAP = {
            "completed": "stop",
            "incomplete": "length",
            "failed": "stop",
            "cancelled": "stop",
        }
        return _MAP.get(raw_reason, "stop")


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("codex_responses", ResponsesApiTransport)
