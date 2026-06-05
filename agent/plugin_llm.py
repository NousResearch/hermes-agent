"""
Plugin LLM facade — host-owned LLM access for trusted plugins.
==============================================================

Plugins built on Hermes Agent often need to make their own LLM calls
out-of-band — a hook that rewrites a tool error before the user sees
it, a gateway adapter that translates inbound text, a slash command
that summarises a paste, a scheduled job that scores yesterday's
activity into a single line on a status board.

Today the only stable plugin surfaces extend an existing Hermes
subsystem: ``register_tool``, ``register_platform``,
``register_memory_provider``, etc. None of those help when the
plugin's job is to make its own model call. This module is the
supported lane for that case.

The plugin gets ``ctx.llm`` exposed on its
:class:`~hermes_cli.plugins.PluginContext`:

* ``complete(messages, ...)`` — chat completion against the user's
  active model + auth.
* ``complete_structured(instructions=..., input=[...], json_schema=...)``
  — bounded structured inference with optional image inputs, JSON
  schema validation, and parsed JSON output.
* async siblings ``acomplete()`` / ``acomplete_structured()`` for
  plugins running on asyncio loops (gateway adapters, hooks).

Provider/model/agent_id/profile are explicit keyword arguments — no
embedded slugs, no shorthands. This mirrors Hermes' main config
shape (``model.provider`` + ``model.model``) so plugin authors who
already understand the host config don't have to learn anything new.

The host owns provider routing, auth resolution, timeouts, and
fallback. The plugin never sees raw OAuth tokens or API keys. All
override knobs (``provider=``, ``model=``, ``agent_id=``,
``profile=``) are gated behind explicit per-plugin trust flags in
``config.yaml``::

    plugins:
      entries:
        my-plugin:
          llm:
            allow_provider_override: true
            allow_model_override: true
            allowed_providers: [openrouter, anthropic]   # optional
            allowed_models:    [openai/gpt-4o-mini]       # optional
            allow_agent_id_override: false
            allow_profile_override: false

Untrusted plugins still get the default surface — they just can't
steer provider, model, agent, or auth-profile selection. The trust
gate is fail-closed: a missing config block means "no overrides,"
not "anything goes."

Backed by :func:`agent.auxiliary_client.call_llm`, which already
handles every provider, fallback chain, and per-task override Hermes
supports.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PluginLlmTextInput:
    """Text block in a structured input list."""

    text: str
    type: str = "text"


@dataclass
class PluginLlmImageInput:
    """Image block in a structured input list.

    Either ``data`` (raw bytes) or ``url`` (http(s) or data: URL) must be
    provided. ``mime_type`` defaults to ``image/png`` when ``data`` is
    used and is required for non-PNG bytes to render correctly across
    providers.
    """

    data: Optional[bytes] = None
    url: Optional[str] = None
    mime_type: str = "image/png"
    file_name: str = ""
    type: str = "image"


PluginLlmInput = Union[PluginLlmTextInput, PluginLlmImageInput, Dict[str, Any]]
"""A single structured input block.

Plugins may pass either the dataclasses above or plain dicts with the
same shape — dicts are normalized internally. Dict shape::

    {"type": "text", "text": "..."}
    {"type": "image", "data": <bytes>, "mime_type": "image/png", "file_name": "receipt.png"}
    {"type": "image", "url": "https://..."}
"""


@dataclass
class PluginLlmUsage:
    """Token + cost usage for a completion. All fields optional — providers
    differ on what they return. ``cost_usd`` is the host's best estimate."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: Optional[float] = None


@dataclass
class PluginLlmCompleteResult:
    """Result of :meth:`PluginLlm.complete`."""

    text: str
    provider: str
    model: str
    agent_id: str
    usage: PluginLlmUsage = field(default_factory=PluginLlmUsage)
    audit: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginLlmStructuredResult:
    """Result of :meth:`PluginLlm.complete_structured`.

    ``parsed`` is set only when ``json_mode=True`` or ``json_schema`` is
    provided AND the response was valid JSON. ``content_type`` is
    ``"json"`` in that case, ``"text"`` otherwise (e.g. the model
    refused or the response wasn't requested as JSON)."""

    text: str
    provider: str
    model: str
    agent_id: str
    usage: PluginLlmUsage = field(default_factory=PluginLlmUsage)
    parsed: Optional[Any] = None
    content_type: str = "text"
    audit: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trust gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TrustPolicy:
    """Resolved trust gate for one plugin's LLM access."""

    plugin_id: str
    allow_provider_override: bool = False
    allowed_providers: Optional[frozenset] = None  # None = no allowlist
    allow_any_provider: bool = False  # True when allowed_providers == ["*"]
    allow_model_override: bool = False
    allowed_models: Optional[frozenset] = None  # None = no allowlist
    allow_any_model: bool = False  # True when allowed_models == ["*"]
    allow_agent_id_override: bool = False
    allow_profile_override: bool = False


def _normalize_ref(raw: str) -> str:
    """Lower-case + strip whitespace. Used for allowlist matching."""
    return (raw or "").strip().lower()


def _coerce_allowlist(raw: Any) -> tuple[Optional[frozenset], bool]:
    """Coerce a YAML list into ``(frozenset_or_None, allow_any)``.

    ``["*"]`` (or any list containing ``"*"``) → ``(frozenset(), True)``.
    Any other list → ``(frozenset({...}), False)``.
    Missing / non-list → ``(None, False)`` meaning "no allowlist."
    """
    if not isinstance(raw, list):
        return None, False
    normalized = [_normalize_ref(item) for item in raw if isinstance(item, str)]
    allow_any = "*" in normalized
    cleaned = {item for item in normalized if item and item != "*"}
    if allow_any and not cleaned:
        return frozenset(), True
    if cleaned:
        return frozenset(cleaned), allow_any
    return frozenset(), allow_any


def _resolve_trust_policy(plugin_id: str) -> _TrustPolicy:
    """Read ``plugins.entries.<plugin_id>.llm`` from config.yaml.

    Missing config → fully restrictive policy (default deny on every
    override). The policy is resolved per-call rather than cached so
    config edits take effect without restarting the agent.
    """
    if not plugin_id:
        return _TrustPolicy(plugin_id="")

    try:
        from hermes_cli.config import load_config
        config = load_config() or {}
    except Exception:  # pragma: no cover — config IO failure
        return _TrustPolicy(plugin_id=plugin_id)

    plugins_cfg = config.get("plugins")
    if not isinstance(plugins_cfg, dict):
        return _TrustPolicy(plugin_id=plugin_id)
    entries = plugins_cfg.get("entries")
    if not isinstance(entries, dict):
        return _TrustPolicy(plugin_id=plugin_id)
    entry = entries.get(plugin_id)
    if not isinstance(entry, dict):
        return _TrustPolicy(plugin_id=plugin_id)
    llm_cfg = entry.get("llm")
    if not isinstance(llm_cfg, dict):
        return _TrustPolicy(plugin_id=plugin_id)

    allowed_models, allow_any_model = _coerce_allowlist(llm_cfg.get("allowed_models"))
    allowed_providers, allow_any_provider = _coerce_allowlist(
        llm_cfg.get("allowed_providers")
    )

    return _TrustPolicy(
        plugin_id=plugin_id,
        allow_provider_override=bool(llm_cfg.get("allow_provider_override", False)),
        allowed_providers=allowed_providers,
        allow_any_provider=allow_any_provider,
        allow_model_override=bool(llm_cfg.get("allow_model_override", False)),
        allowed_models=allowed_models,
        allow_any_model=allow_any_model,
        allow_agent_id_override=bool(llm_cfg.get("allow_agent_id_override", False)),
        allow_profile_override=bool(llm_cfg.get("allow_profile_override", False)),
    )


class PluginLlmTrustError(PermissionError):
    """Raised when a plugin attempts an LLM override without trust."""


def _check_overrides(
    policy: _TrustPolicy,
    *,
    requested_provider: Optional[str],
    requested_model: Optional[str],
    requested_agent_id: Optional[str],
    requested_profile: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Apply the trust gate. Returns the validated overrides as
    ``(provider, model, agent_id, profile)`` or raises
    :class:`PluginLlmTrustError`.

    Each override (``provider``, ``model``, ``agent_id``, ``profile``)
    is independently gated. ``provider`` and ``model`` each have an
    optional allowlist via ``allowed_providers`` / ``allowed_models``.
    """
    final_provider: Optional[str] = None
    final_model: Optional[str] = None
    final_profile: Optional[str] = None

    if requested_provider:
        if not policy.allow_provider_override:
            raise PluginLlmTrustError(
                f"Plugin {policy.plugin_id!r} cannot override the provider "
                f"(set plugins.entries.{policy.plugin_id}.llm.allow_provider_override "
                f"to true to allow)."
            )
        normalized = _normalize_ref(requested_provider)
        if (
            not policy.allow_any_provider
            and policy.allowed_providers is not None
            and normalized not in policy.allowed_providers
        ):
            raise PluginLlmTrustError(
                f"Plugin {policy.plugin_id!r} provider override "
                f"{requested_provider!r} is not in plugins.entries."
                f"{policy.plugin_id}.llm.allowed_providers."
            )
        final_provider = requested_provider.strip()

    if requested_model:
        if not policy.allow_model_override:
            raise PluginLlmTrustError(
                f"Plugin {policy.plugin_id!r} cannot override the model "
                f"(set plugins.entries.{policy.plugin_id}.llm.allow_model_override "
                f"to true to allow)."
            )
        normalized = _normalize_ref(requested_model)
        if (
            not policy.allow_any_model
            and policy.allowed_models is not None
            and normalized not in policy.allowed_models
        ):
            raise PluginLlmTrustError(
                f"Plugin {policy.plugin_id!r} model override "
                f"{requested_model!r} is not in plugins.entries."
                f"{policy.plugin_id}.llm.allowed_models."
            )
        final_model = requested_model.strip()

    if requested_agent_id and not policy.allow_agent_id_override:
        raise PluginLlmTrustError(
            f"Plugin {policy.plugin_id!r} cannot run completions against a "
            f"non-default agent id (set plugins.entries.{policy.plugin_id}."
            f"llm.allow_agent_id_override to true to allow)."
        )

    if requested_profile:
        if not policy.allow_profile_override:
            raise PluginLlmTrustError(
                f"Plugin {policy.plugin_id!r} cannot override the auth profile "
                f"(set plugins.entries.{policy.plugin_id}.llm.allow_profile_override "
                f"to true to allow)."
            )
        final_profile = requested_profile.strip()

    return final_provider, final_model, requested_agent_id, final_profile


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------


def _normalize_input_block(block: PluginLlmInput) -> Dict[str, Any]:
    """Coerce a structured input block to a plain dict the message
    builder understands. Unknown shapes raise ``ValueError``."""
    if isinstance(block, PluginLlmTextInput):
        return {"type": "text", "text": block.text}
    if isinstance(block, PluginLlmImageInput):
        d: Dict[str, Any] = {
            "type": "image",
            "mime_type": block.mime_type,
            "file_name": block.file_name,
        }
        if block.data is not None:
            d["data"] = block.data
        if block.url:
            d["url"] = block.url
        return d
    if isinstance(block, dict):
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError("text input block requires 'text' string")
            return {"type": "text", "text": text}
        if kind == "image":
            if "data" not in block and not block.get("url"):
                raise ValueError("image input block requires 'data' bytes or 'url'")
            return {
                "type": "image",
                "data": block.get("data"),
                "url": block.get("url"),
                "mime_type": block.get("mime_type") or "image/png",
                "file_name": block.get("file_name") or "",
            }
        raise ValueError(f"Unknown input block type: {kind!r}")
    raise ValueError(f"Unsupported input block: {type(block).__name__}")


def _build_structured_messages(
    *,
    instructions: str,
    inputs: Sequence[PluginLlmInput],
    json_mode: bool,
    json_schema: Optional[Any],
    schema_name: Optional[str],
    system_prompt: Optional[str],
) -> List[Dict[str, Any]]:
    """Build the OpenAI-style messages list for a structured call.

    The instructions become the first text part of the user message,
    followed by an optional ``Schema name: <name>`` hint and an optional
    JSON-only directive when JSON output is requested. Image inputs are
    encoded as ``image_url`` parts.
    """
    messages: List[Dict[str, Any]] = []
    sys_parts: List[str] = []
    if system_prompt:
        sys_parts.append(system_prompt.strip())
    if json_mode or json_schema is not None:
        sys_parts.append(
            "Respond with a single JSON object that matches the requested shape. "
            "Do not include prose or markdown fences."
        )
    if sys_parts:
        messages.append({"role": "system", "content": "\n\n".join(sys_parts)})

    user_parts: List[Dict[str, Any]] = []
    header = instructions.strip()
    if schema_name:
        header = f"{header}\n\nSchema name: {schema_name}"
    if json_schema is not None:
        try:
            schema_text = json.dumps(json_schema, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            schema_text = str(json_schema)
        header = f"{header}\n\nJSON schema:\n{schema_text}"
    user_parts.append({"type": "text", "text": header})

    for block in inputs:
        norm = _normalize_input_block(block)
        if norm["type"] == "text":
            user_parts.append({"type": "text", "text": norm["text"]})
        elif norm["type"] == "image":
            if norm.get("url"):
                user_parts.append({
                    "type": "image_url",
                    "image_url": {"url": norm["url"]},
                })
            else:
                data = norm.get("data") or b""
                if not isinstance(data, (bytes, bytearray)):
                    raise ValueError("image input 'data' must be bytes")
                b64 = base64.b64encode(data).decode("ascii")
                mime = norm.get("mime_type") or "image/png"
                user_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })

    messages.append({"role": "user", "content": user_parts})
    return messages


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    """Pull the first fenced code block out of ``text`` if any. Returns
    ``text`` unchanged when no fence is present."""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_structured_text(
    *, text: str, json_mode: bool, json_schema: Optional[Any]
) -> tuple[Optional[Any], str]:
    """Return ``(parsed, content_type)``. ``content_type`` is ``"json"``
    when parsing succeeded and (when a schema was given) validation
    passed; ``"text"`` otherwise."""
    if not (json_mode or json_schema is not None):
        return None, "text"
    if not text:
        return None, "text"

    try:
        parsed = json.loads(_strip_code_fences(text))
    except (json.JSONDecodeError, ValueError):
        return None, "text"

    if json_schema is not None:
        try:
            import jsonschema  # type: ignore[import-untyped]
            jsonschema.validate(parsed, json_schema)
        except ImportError:
            # jsonschema is optional; skip strict validation when absent.
            logger.debug("jsonschema unavailable; skipping schema validation")
        except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            raise ValueError(
                f"Plugin LLM structured output did not match schema: {exc.message}"
            ) from exc

    return parsed, "json"


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------


def _extract_usage(response: Any) -> PluginLlmUsage:
    """Pull token usage out of an OpenAI-shaped response object.

    Tolerant of provider differences — Anthropic via the auxiliary
    adapter exposes ``usage.prompt_tokens`` / ``usage.completion_tokens``;
    direct OpenAI also exposes ``cache_read_input_tokens``."""
    usage = PluginLlmUsage()
    raw = getattr(response, "usage", None)
    if raw is None:
        return usage

    def _g(name: str) -> int:
        v = getattr(raw, name, None)
        if v is None and isinstance(raw, dict):
            v = raw.get(name)
        try:
            return int(v) if v is not None else 0
        except (TypeError, ValueError):
            return 0

    usage.input_tokens = _g("prompt_tokens") or _g("input_tokens")
    usage.output_tokens = _g("completion_tokens") or _g("output_tokens")
    usage.total_tokens = _g("total_tokens") or (usage.input_tokens + usage.output_tokens)
    usage.cache_read_tokens = _g("cache_read_input_tokens") or _g("cache_read_tokens")
    usage.cache_write_tokens = _g("cache_creation_input_tokens") or _g("cache_write_tokens")
    return usage


def _extract_text(response: Any) -> str:
    """Pull the assistant text out of an OpenAI-shaped response object."""
    try:
        msg = response.choices[0].message
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        parts.append(part["text"])
                else:
                    txt = getattr(part, "text", None)
                    if isinstance(txt, str):
                        parts.append(txt)
            return "".join(parts)
    except (AttributeError, IndexError, TypeError):
        pass
    return ""


def _resolve_attribution(
    *,
    provider_override: Optional[str],
    model_override: Optional[str],
    response: Any,
) -> tuple[str, str]:
    """Decide what to record as ``result.provider`` / ``result.model``.

    Precedence:

    1. Explicit overrides win — if the plugin asked for ``provider="x"``
       or ``model="y"``, that's what we record (it's what the call
       actually targeted).
    2. Otherwise we ask the host for the current main provider/model
       via :func:`_read_main_provider` / :func:`_read_main_model`, since
       those are what ``call_llm`` resolves to when ``provider=None``
       and ``model=None`` are passed through. They reflect runtime
       overrides set by ``set_runtime_main()``.
    3. ``response.model`` (if present) overrides the recorded model
       string. Providers post-resolution often return a slightly
       different model id than the request (e.g. ``gpt-4o`` →
       ``gpt-4o-2024-08-06``); the plugin's audit log should reflect
       what actually ran.
    4. If everything above is empty, fall back to ``"auto"`` /
       ``"default"`` so the result object has non-empty strings.
    """
    if provider_override:
        provider = provider_override
    else:
        try:
            from agent.auxiliary_client import _read_main_provider
            provider = (_read_main_provider() or "").strip() or "auto"
        except Exception:  # pragma: no cover — defensive
            provider = "auto"

    response_model = getattr(response, "model", None)
    if isinstance(response_model, str) and response_model.strip():
        model = response_model.strip()
    elif model_override:
        model = model_override
    else:
        try:
            from agent.auxiliary_client import _read_main_model
            model = (_read_main_model() or "").strip() or "default"
        except Exception:  # pragma: no cover — defensive
            model = "default"

    return provider, model


# ---------------------------------------------------------------------------
# PluginLlm facade
# ---------------------------------------------------------------------------


class PluginLlm:
    """Host-owned LLM access for one trusted plugin.

    Instances are constructed by :class:`hermes_cli.plugins.PluginContext`
    and exposed as ``ctx.llm``. Plugins should not instantiate this
    directly — the constructor binds plugin identity for trust-gate
    enforcement.
    """

    def __init__(
        self,
        *,
        plugin_id: str,
        policy_loader: Optional[Callable[[str], _TrustPolicy]] = None,
        sync_caller: Optional[Callable[..., Any]] = None,
        async_caller: Optional[Callable[..., Awaitable[Any]]] = None,
    ) -> None:
        self._plugin_id = plugin_id
        self._policy_loader = policy_loader or _resolve_trust_policy
        self._sync_caller = sync_caller
        self._async_caller = async_caller

    # -- public sync API ----------------------------------------------------

    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        agent_id: Optional[str] = None,
        profile: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> PluginLlmCompleteResult:
        """Run a host-owned chat completion against the user's active model.

        ``messages`` is the standard OpenAI shape. ``provider``,
        ``model``, ``agent_id``, and ``profile`` follow the same
        explicit shape as the host's main config (``model.provider``
        + ``model.model``). Each is independently gated by
        ``plugins.entries.<id>.llm.allow_*_override`` (see module
        docstring).
        """
        policy = self._policy_loader(self._plugin_id)
        eff_provider, eff_model, eff_agent, eff_profile = _check_overrides(
            policy,
            requested_provider=provider,
            requested_model=model,
            requested_agent_id=agent_id,
            requested_profile=profile,
        )
        real_provider, real_model, response = self._invoke_sync(
            messages=messages,
            provider_override=eff_provider,
            model_override=eff_model,
            profile_override=eff_profile,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        text = _extract_text(response)
        usage = _extract_usage(response)
        result = PluginLlmCompleteResult(
            text=text,
            provider=real_provider,
            model=real_model,
            agent_id=eff_agent or "default",
            usage=usage,
            audit={
                "plugin_id": self._plugin_id,
                "purpose": purpose or "",
                "profile": eff_profile or "",
            },
        )
        logger.info(
            "plugin_llm.complete plugin=%s provider=%s model=%s purpose=%s "
            "tokens=%d",
            self._plugin_id, real_provider, real_model, purpose or "",
            usage.total_tokens,
        )
        return result

    def complete_structured(
        self,
        *,
        instructions: str,
        input: Sequence[PluginLlmInput],
        json_schema: Optional[Any] = None,
        json_mode: bool = False,
        schema_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        agent_id: Optional[str] = None,
        profile: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> PluginLlmStructuredResult:
        """Run a bounded host-owned structured completion.

        ``input`` accepts text and image blocks (see
        :class:`PluginLlmTextInput` / :class:`PluginLlmImageInput`). When
        ``json_mode=True`` or ``json_schema`` is provided, the response
        is parsed and (if a schema is given) validated; the parsed value
        is returned in :attr:`PluginLlmStructuredResult.parsed`.

        Validation requires the optional ``jsonschema`` package. When it
        isn't installed, JSON mode still works but schema enforcement is
        skipped with a debug log.
        """
        if not instructions or not instructions.strip():
            raise ValueError("complete_structured requires non-empty instructions")
        if not input:
            raise ValueError("complete_structured requires at least one input block")

        policy = self._policy_loader(self._plugin_id)
        eff_provider, eff_model, eff_agent, eff_profile = _check_overrides(
            policy,
            requested_provider=provider,
            requested_model=model,
            requested_agent_id=agent_id,
            requested_profile=profile,
        )

        messages = _build_structured_messages(
            instructions=instructions,
            inputs=list(input),
            json_mode=json_mode,
            json_schema=json_schema,
            schema_name=schema_name,
            system_prompt=system_prompt,
        )
        extra_body = self._json_response_format(json_mode=json_mode, json_schema=json_schema)

        real_provider, real_model, response = self._invoke_sync(
            messages=messages,
            provider_override=eff_provider,
            model_override=eff_model,
            profile_override=eff_profile,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body=extra_body,
        )
        text = _extract_text(response)
        usage = _extract_usage(response)
        parsed, content_type = _parse_structured_text(
            text=text, json_mode=json_mode, json_schema=json_schema
        )
        result = PluginLlmStructuredResult(
            text=text,
            provider=real_provider,
            model=real_model,
            agent_id=eff_agent or "default",
            usage=usage,
            parsed=parsed,
            content_type=content_type,
            audit={
                "plugin_id": self._plugin_id,
                "purpose": purpose or "",
                "profile": eff_profile or "",
                "schema_name": schema_name or "",
            },
        )
        logger.info(
            "plugin_llm.complete_structured plugin=%s provider=%s model=%s "
            "purpose=%s content_type=%s tokens=%d",
            self._plugin_id, real_provider, real_model, purpose or "",
            content_type, usage.total_tokens,
        )
        return result

    # -- public async API ---------------------------------------------------

    async def acomplete(
        self,
        messages: List[Dict[str, Any]],
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        agent_id: Optional[str] = None,
        profile: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> PluginLlmCompleteResult:
        """Async sibling of :meth:`complete`."""
        policy = self._policy_loader(self._plugin_id)
        eff_provider, eff_model, eff_agent, eff_profile = _check_overrides(
            policy,
            requested_provider=provider,
            requested_model=model,
            requested_agent_id=agent_id,
            requested_profile=profile,
        )
        real_provider, real_model, response = await self._invoke_async(
            messages=messages,
            provider_override=eff_provider,
            model_override=eff_model,
            profile_override=eff_profile,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        text = _extract_text(response)
        usage = _extract_usage(response)
        return PluginLlmCompleteResult(
            text=text,
            provider=real_provider,
            model=real_model,
            agent_id=eff_agent or "default",
            usage=usage,
            audit={
                "plugin_id": self._plugin_id,
                "purpose": purpose or "",
                "profile": eff_profile or "",
            },
        )

    async def acomplete_structured(
        self,
        *,
        instructions: str,
        input: Sequence[PluginLlmInput],
        json_schema: Optional[Any] = None,
        json_mode: bool = False,
        schema_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        agent_id: Optional[str] = None,
        profile: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> PluginLlmStructuredResult:
        """Async sibling of :meth:`complete_structured`."""
        if not instructions or not instructions.strip():
            raise ValueError("acomplete_structured requires non-empty instructions")
        if not input:
            raise ValueError("acomplete_structured requires at least one input block")

        policy = self._policy_loader(self._plugin_id)
        eff_provider, eff_model, eff_agent, eff_profile = _check_overrides(
            policy,
            requested_provider=provider,
            requested_model=model,
            requested_agent_id=agent_id,
            requested_profile=profile,
        )
        messages = _build_structured_messages(
            instructions=instructions,
            inputs=list(input),
            json_mode=json_mode,
            json_schema=json_schema,
            schema_name=schema_name,
            system_prompt=system_prompt,
        )
        extra_body = self._json_response_format(json_mode=json_mode, json_schema=json_schema)
        real_provider, real_model, response = await self._invoke_async(
            messages=messages,
            provider_override=eff_provider,
            model_override=eff_model,
            profile_override=eff_profile,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body=extra_body,
        )
        text = _extract_text(response)
        usage = _extract_usage(response)
        parsed, content_type = _parse_structured_text(
            text=text, json_mode=json_mode, json_schema=json_schema
        )
        return PluginLlmStructuredResult(
            text=text,
            provider=real_provider,
            model=real_model,
            agent_id=eff_agent or "default",
            usage=usage,
            parsed=parsed,
            content_type=content_type,
            audit={
                "plugin_id": self._plugin_id,
                "purpose": purpose or "",
                "profile": eff_profile or "",
                "schema_name": schema_name or "",
            },
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _json_response_format(
        *, json_mode: bool, json_schema: Optional[Any]
    ) -> Optional[Dict[str, Any]]:
        """Build the ``extra_body.response_format`` payload for the
        provider request. Falls back to ``json_object`` when no schema
        is given so providers that ignore json_schema still get a hint."""
        if json_schema is not None:
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "plugin_structured_output",
                        "schema": json_schema,
                        "strict": False,
                    },
                }
            }
        if json_mode:
            return {"response_format": {"type": "json_object"}}
        return None

    def _invoke_sync(
        self,
        *,
        messages: List[Dict[str, Any]],
        provider_override: Optional[str],
        model_override: Optional[str],
        profile_override: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        timeout: Optional[float],
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str, Any]:
        """Invoke the host's ``call_llm``. Lazy-imports
        ``agent.auxiliary_client`` to avoid circular deps at plugin
        discovery time."""
        if self._sync_caller is not None:
            return self._sync_caller(
                messages=messages,
                provider_override=provider_override,
                model_override=model_override,
                profile_override=profile_override,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_body=extra_body,
            )
        from agent.auxiliary_client import call_llm
        merged_extra = dict(extra_body or {})
        if profile_override:
            merged_extra.setdefault("metadata", {})["auth_profile"] = profile_override
        response = call_llm(
            task=None,
            provider=provider_override,
            model=model_override,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body=merged_extra or None,
        )
        provider, model = _resolve_attribution(
            provider_override=provider_override,
            model_override=model_override,
            response=response,
        )
        return provider, model, response

    async def _invoke_async(
        self,
        *,
        messages: List[Dict[str, Any]],
        provider_override: Optional[str],
        model_override: Optional[str],
        profile_override: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        timeout: Optional[float],
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str, Any]:
        if self._async_caller is not None:
            return await self._async_caller(
                messages=messages,
                provider_override=provider_override,
                model_override=model_override,
                profile_override=profile_override,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_body=extra_body,
            )
        from agent.auxiliary_client import async_call_llm
        merged_extra = dict(extra_body or {})
        if profile_override:
            merged_extra.setdefault("metadata", {})["auth_profile"] = profile_override
        response = await async_call_llm(
            task=None,
            provider=provider_override,
            model=model_override,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body=merged_extra or None,
        )
        provider, model = _resolve_attribution(
            provider_override=provider_override,
            model_override=model_override,
            response=response,
        )
        return provider, model, response


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def make_plugin_llm_for_test(
    *,
    plugin_id: str,
    policy: _TrustPolicy,
    sync_caller: Optional[Callable[..., Any]] = None,
    async_caller: Optional[Callable[..., Awaitable[Any]]] = None,
) -> PluginLlm:
    """Construct a :class:`PluginLlm` with an injected policy and caller.

    Used by unit tests that don't want to round-trip through config.yaml
    or hit a real provider. Not part of the public plugin API.
    """
    return PluginLlm(
        plugin_id=plugin_id,
        policy_loader=lambda _pid: policy,
        sync_caller=sync_caller,
        async_caller=async_caller,
    )


__all__ = [
    "PluginLlm",
    "PluginLlmTextInput",
    "PluginLlmImageInput",
    "PluginLlmInput",
    "PluginLlmUsage",
    "PluginLlmCompleteResult",
    "PluginLlmStructuredResult",
    "PluginLlmTrustError",
    "make_plugin_llm_for_test",
]
