"""Plugin LLM facade — host-owned LLM access for trusted plugins (``ctx.llm``).

``complete`` / ``complete_structured`` (text + image inputs, JSON schema validation)
and their async siblings. Provider/model/agent_id/profile are explicit keyword
arguments mirroring the host config shape; the host owns routing, auth, timeouts
and fallback, so the plugin never sees raw tokens or keys. Every override knob is
gated by the per-plugin ``plugins.entries.<id>.llm.allow_*_override`` trust flags
(fail-closed: a missing block means "no overrides"). Backed by
:func:`agent.auxiliary_client.call_llm`.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


@dataclass
class PluginLlmTextInput:
    """Text block in a structured input list."""

    text: str
    type: str = "text"


@dataclass
class PluginLlmImageInput:
    """Image block: ``data`` (raw bytes) or ``url`` (http(s)/data: URL).
    ``mime_type`` is required for non-PNG bytes to render across providers."""

    data: Optional[bytes] = None
    url: Optional[str] = None
    mime_type: str = "image/png"
    file_name: str = ""
    type: str = "image"


PluginLlmInput = Union[PluginLlmTextInput, PluginLlmImageInput, Dict[str, Any]]
"""One structured input block: a dataclass above or a plain dict of the same shape."""


@dataclass
class PluginLlmUsage:
    """Token + cost usage; every field optional. ``cost_usd`` is the host's best estimate."""

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

    ``parsed`` is set only when JSON output was requested AND the response was
    valid JSON; ``content_type`` is then ``"json"``, otherwise ``"text"``."""

    text: str
    provider: str
    model: str
    agent_id: str
    usage: PluginLlmUsage = field(default_factory=PluginLlmUsage)
    parsed: Optional[Any] = None
    content_type: str = "text"
    audit: Dict[str, Any] = field(default_factory=dict)


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
    # Lets ``complete(task=...)`` borrow the host's *built-in* aux task slots.
    # Slots the plugin registered itself are always allowed. Fail-closed.
    allow_task_override: bool = False


# The ``allow_*_override`` config keys; each is a same-named ``_TrustPolicy`` field.
_OVERRIDE_FLAGS = ("allow_provider_override", "allow_model_override", "allow_agent_id_override",
                   "allow_profile_override", "allow_task_override")


def _coerce_allowlist(raw: Any) -> tuple[Optional[frozenset], bool]:
    """YAML list → ``(frozenset_or_None, allow_any)``. A ``"*"`` entry sets
    ``allow_any``; missing / non-list → ``(None, False)`` = no allowlist."""
    if not isinstance(raw, list):
        return None, False
    normalized = [item.strip().lower() for item in raw if isinstance(item, str)]
    return frozenset(item for item in normalized if item and item != "*"), "*" in normalized


def _resolve_trust_policy(plugin_id: str) -> _TrustPolicy:
    """Read ``plugins.entries.<plugin_id>.llm`` from config.yaml (missing → fully
    restrictive). Resolved per call so config edits apply without a restart."""
    if not plugin_id:
        return _TrustPolicy(plugin_id="")
    try:
        from hermes_cli.config import load_config_readonly
        llm_cfg: Any = (load_config_readonly() or {}).get("plugins")
    except Exception:  # pragma: no cover — config IO failure
        llm_cfg = None
    for key in ("entries", plugin_id, "llm"):
        llm_cfg = llm_cfg.get(key) if isinstance(llm_cfg, dict) else None
    if not isinstance(llm_cfg, dict):
        return _TrustPolicy(plugin_id=plugin_id)
    allowed_models, allow_any_model = _coerce_allowlist(llm_cfg.get("allowed_models"))
    allowed_providers, allow_any_provider = _coerce_allowlist(llm_cfg.get("allowed_providers"))
    return _TrustPolicy(
        plugin_id=plugin_id, allowed_providers=allowed_providers, allow_any_provider=allow_any_provider,
        allowed_models=allowed_models, allow_any_model=allow_any_model,
        **{name: bool(llm_cfg.get(name, False)) for name in _OVERRIDE_FLAGS},
    )


class PluginLlmTrustError(PermissionError):
    """Raised when a plugin attempts an LLM override without trust."""


def _denied(plugin_id: str, what: str, flag: str) -> PluginLlmTrustError:
    return PluginLlmTrustError(
        f"Plugin {plugin_id!r} cannot {what} (set plugins.entries.{plugin_id}.llm.{flag} to true to allow).")


def _gate_ref_override(policy: _TrustPolicy, kind: str, requested: str) -> str:
    """Gate a ``provider`` / ``model`` override: trust flag, then optional allowlist."""
    if not getattr(policy, f"allow_{kind}_override"):
        raise _denied(policy.plugin_id, f"override the {kind}", f"allow_{kind}_override")
    allowed = getattr(policy, f"allowed_{kind}s")
    if not getattr(policy, f"allow_any_{kind}") and allowed is not None and requested.strip().lower() not in allowed:
        raise PluginLlmTrustError(f"Plugin {policy.plugin_id!r} {kind} override {requested!r} is not in "
                                  f"plugins.entries.{policy.plugin_id}.llm.allowed_{kind}s.")
    return requested.strip()


# Overrides gated by a bare trust flag (no allowlist): ``kind`` -> denial wording.
_FLAG_ONLY_OVERRIDES = {
    "agent_id": "run completions against a non-default agent id", "profile": "override the auth profile",
}


def _check_overrides(
    policy: _TrustPolicy, *, requested_provider: Optional[str], requested_model: Optional[str],
    requested_agent_id: Optional[str], requested_profile: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Gate each override independently, in the order provider, model, agent_id,
    profile. Returns ``(provider, model, agent_id, profile)`` (agent_id unstripped)."""
    final_provider = _gate_ref_override(policy, "provider", requested_provider) if requested_provider else None
    final_model = _gate_ref_override(policy, "model", requested_model) if requested_model else None
    for kind, requested in (("agent_id", requested_agent_id), ("profile", requested_profile)):
        if requested and not getattr(policy, f"allow_{kind}_override"):
            raise _denied(policy.plugin_id, _FLAG_ONLY_OVERRIDES[kind], f"allow_{kind}_override")
    return final_provider, final_model, requested_agent_id, requested_profile.strip() if requested_profile else None


def _resolve_task_ownership(plugin_id: str) -> tuple[frozenset, frozenset]:
    """``(owned_keys, builtin_keys)`` for the task trust gate.

    Imports are lazy (circular import at plugin discovery); an unreadable registry
    yields empty sets, failing the gate closed. Ownership matches on the canonical id
    ``ctx.llm`` is bound to (``manifest.key or manifest.name``), which is what
    ``register_auxiliary_task`` stores as the entry's ``plugin``."""
    owned: set = set()
    builtin: set = set()
    try:
        from hermes_cli.plugins import get_plugin_auxiliary_tasks
        owned = {e.get("key") for e in get_plugin_auxiliary_tasks()
                 if e.get("plugin") == plugin_id and isinstance(e.get("key"), str) and e.get("key")}
    except Exception:  # pragma: no cover — registry unavailable
        pass
    try:
        from hermes_cli.main_provider_setup import _AUX_TASKS
        builtin = {k for k, _name, _desc in _AUX_TASKS}
    except Exception:  # pragma: no cover — main import failure
        pass
    return frozenset(owned), frozenset(builtin)


def _check_task(policy: _TrustPolicy, *, plugin_id: str, requested_task: Optional[str]) -> Optional[str]:
    """Validate a plugin's requested auxiliary ``task`` key.

    unset / ``""`` / ``"auto"`` → ``None`` (main-model path); a key the plugin
    registered itself → allowed; a built-in key → only with ``allow_task_override``;
    anything else raises + logs. Never silently downgraded to ``auto``: that would
    mask the misconfiguration and could route to a main model the user steered
    elsewhere on purpose.

    A foreign/unknown key raises :class:`PluginLlmTrustError` and logs a warning naming the offending plugin
    and key. See #64174, #64182.
    """
    task = (requested_task or "").strip()
    if not task or task.lower() == "auto":
        return None
    owned, builtin = _resolve_task_ownership(plugin_id)
    if task in owned or (task in builtin and policy.allow_task_override):
        return task
    if task in builtin:
        logger.warning("plugin_llm task routing denied: plugin %r requested built-in "
                       "auxiliary task %r without plugins.entries.%s.llm.allow_task_override", plugin_id, task, plugin_id)
        raise _denied(plugin_id, f"route through the built-in auxiliary task {task!r}", "allow_task_override")
    logger.warning("plugin_llm task routing denied: plugin %r requested auxiliary task %r it did not register",
                   plugin_id, task)
    raise PluginLlmTrustError(
        f"Plugin {plugin_id!r} cannot route through auxiliary task {task!r} — a "
        f"plugin may only pass a task key it registered itself via "
        f"ctx.register_auxiliary_task() (or a built-in key when plugins.entries."
        f"{plugin_id}.llm.allow_task_override is true)."
    )


def _normalize_input_block(block: PluginLlmInput) -> Dict[str, Any]:
    """Coerce a structured input block to a plain dict. Unknown shapes raise ``ValueError``."""
    if isinstance(block, PluginLlmTextInput):
        return {"type": "text", "text": block.text}
    if isinstance(block, PluginLlmImageInput):
        d: Dict[str, Any] = {"type": "image", "mime_type": block.mime_type, "file_name": block.file_name}
        if block.data is not None:
            d["data"] = block.data
        if block.url:
            d["url"] = block.url
        return d
    if not isinstance(block, dict):
        raise ValueError(f"Unsupported input block: {type(block).__name__}")
    kind = block.get("type")
    if kind == "text":
        if not isinstance(block.get("text"), str):
            raise ValueError("text input block requires 'text' string")
        return {"type": "text", "text": block["text"]}
    if kind == "image":
        if "data" not in block and not block.get("url"):
            raise ValueError("image input block requires 'data' bytes or 'url'")
        return {"type": "image", "data": block.get("data"), "url": block.get("url"),
                "mime_type": block.get("mime_type") or "image/png", "file_name": block.get("file_name") or ""}
    raise ValueError(f"Unknown input block type: {kind!r}")


def _image_part(norm: Dict[str, Any]) -> Dict[str, Any]:
    """Normalized image block → OpenAI ``image_url`` part (data: URL for bytes)."""
    url = norm.get("url")
    if not url:
        data = norm.get("data") or b""
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("image input 'data' must be bytes")
        url = f"data:{norm.get('mime_type') or 'image/png'};base64,{base64.b64encode(data).decode('ascii')}"
    return {"type": "image_url", "image_url": {"url": url}}


def _build_structured_messages(
    *, instructions: str, inputs: Sequence[PluginLlmInput], json_mode: bool,
    json_schema: Optional[Any], schema_name: Optional[str], system_prompt: Optional[str],
) -> List[Dict[str, Any]]:
    """OpenAI-style messages for a structured call: optional system message (prompt +
    JSON-only directive), then a user message whose first text part is the
    instructions (+ schema name / JSON schema) followed by the input blocks."""
    messages: List[Dict[str, Any]] = []
    sys_parts: List[str] = [system_prompt.strip()] if system_prompt else []
    if json_mode or json_schema is not None:
        sys_parts.append("Respond with a single JSON object that matches the requested shape. "
                         "Do not include prose or markdown fences.")
    if sys_parts:
        messages.append({"role": "system", "content": "\n\n".join(sys_parts)})
    header = instructions.strip()
    if schema_name:
        header = f"{header}\n\nSchema name: {schema_name}"
    if json_schema is not None:
        try:
            schema_text = json.dumps(json_schema, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            schema_text = str(json_schema)
        header = f"{header}\n\nJSON schema:\n{schema_text}"
    user_parts: List[Dict[str, Any]] = [{"type": "text", "text": header}]
    for block in inputs:
        norm = _normalize_input_block(block)  # always "text" or "image"
        user_parts.append({"type": "text", "text": norm["text"]} if norm["type"] == "text" else _image_part(norm))
    messages.append({"role": "user", "content": user_parts})
    return messages


_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    """The first fenced code block's body, or the stripped text when unfenced."""
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _parse_structured_text(*, text: str, json_mode: bool, json_schema: Optional[Any]) -> tuple[Optional[Any], str]:
    """``(parsed, content_type)``: ``"json"`` when parsing (and schema validation, if
    given) succeeded, ``"text"`` otherwise. Schema violations raise ``ValueError``;
    a missing ``jsonschema`` package skips validation with a debug log."""
    if not (json_mode or json_schema is not None) or not text:
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
            logger.debug("jsonschema unavailable; skipping schema validation")
        except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            raise ValueError(f"Plugin LLM structured output did not match schema: {exc.message}") from exc
    return parsed, "json"


def _extract_usage(response: Any) -> PluginLlmUsage:
    """Token usage from an OpenAI-shaped response, tolerating provider naming
    (``prompt_tokens``/``completion_tokens`` vs ``input_tokens``/``output_tokens``,
    ``cache_read_input_tokens`` vs ``cache_read_tokens``)."""
    raw = getattr(response, "usage", None)
    if raw is None:
        return PluginLlmUsage()

    def _g(*names: str) -> int:
        for name in names:
            v = getattr(raw, name, None)
            if v is None and isinstance(raw, dict):
                v = raw.get(name)
            try:
                if v is not None and int(v):
                    return int(v)
            except (TypeError, ValueError):
                pass
        return 0
    inp, out = _g("prompt_tokens", "input_tokens"), _g("completion_tokens", "output_tokens")
    return PluginLlmUsage(
        input_tokens=inp, output_tokens=out, total_tokens=_g("total_tokens") or (inp + out),
        cache_read_tokens=_g("cache_read_input_tokens", "cache_read_tokens"),
        cache_write_tokens=_g("cache_creation_input_tokens", "cache_write_tokens"),
    )


def _extract_text(response: Any) -> str:
    """Assistant text of an OpenAI-shaped response (string or text-part list content)."""
    try:
        content = getattr(response.choices[0].message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = ((part.get("text") if part.get("type") == "text" else None)
                     if isinstance(part, dict) else getattr(part, "text", None) for part in content)
            return "".join(t for t in texts if isinstance(t, str))
    except (AttributeError, IndexError, TypeError):
        pass
    return ""


def _main_config_value(reader: str, default: str) -> str:
    """Read the current main provider/model via ``agent.auxiliary_client``."""
    try:
        import agent.auxiliary_client as ac
        return (getattr(ac, reader)() or "").strip() or default
    except Exception:  # pragma: no cover — defensive
        return default


def _resolve_attribution(*, provider_override: Optional[str], model_override: Optional[str], response: Any,
                         route_info: Optional[Dict[str, str]] = None) -> tuple[str, str]:
    """``(provider, model)`` to record on the result.

    Provider: route selected by ``auxiliary_client`` > explicit override > current
    main provider > ``"auto"``. Model: ``response.model`` (the canonical id that
    actually ran) > route > override > current main model > ``"default"``."""
    route_info = route_info or {}
    provider = route_info.get("provider") or provider_override or _main_config_value("_read_main_provider", "auto")
    response_model = getattr(response, "model", None)
    if isinstance(response_model, str) and response_model.strip():
        return provider, response_model.strip()
    return provider, route_info.get("model") or model_override or _main_config_value("_read_main_model", "default")


def _json_response_format(*, json_mode: bool, json_schema: Optional[Any]) -> Optional[Dict[str, Any]]:
    """``extra_body.response_format``; falls back to ``json_object`` without a
    schema so schema-blind providers still get a hint."""
    if json_schema is not None:
        schema = {"name": "plugin_structured_output", "schema": json_schema, "strict": False}
        return {"response_format": {"type": "json_schema", "json_schema": schema}}
    if json_mode:
        return {"response_format": {"type": "json_object"}}
    return None


def _structured_spec(
    name: str, instructions: str, input: Sequence[PluginLlmInput], system_prompt: Optional[str],
    json_mode: bool, json_schema: Optional[Any], schema_name: Optional[str],
) -> Dict[str, Any]:
    """Argument check for the structured methods (runs before the trust gate)."""
    if not instructions or not instructions.strip():
        raise ValueError(f"{name} requires non-empty instructions")
    if not input:
        raise ValueError(f"{name} requires at least one input block")
    return dict(instructions=instructions, inputs=list(input), system_prompt=system_prompt,
                json_mode=json_mode, json_schema=json_schema, schema_name=schema_name)


class PluginLlm:
    """Host-owned LLM access for one trusted plugin.

    Constructed by :class:`hermes_cli.plugins.PluginContext` and exposed as
    ``ctx.llm``; the constructor binds plugin identity for trust enforcement, so
    plugins should not instantiate it directly. Every public method is ``_gate``
    (trust checks → call kwargs) → ``_invoke_*`` (host ``call_llm`` or injected
    caller) → ``_finish`` (result + audit log)."""

    def __init__(
        self, *, plugin_id: str, policy_loader: Optional[Callable[[str], _TrustPolicy]] = None,
        sync_caller: Optional[Callable[..., Any]] = None,
        async_caller: Optional[Callable[..., Awaitable[Any]]] = None,
    ) -> None:
        self._plugin_id = plugin_id
        self._policy_loader = policy_loader or _resolve_trust_policy
        self._sync_caller = sync_caller
        self._async_caller = async_caller

    def complete(
        self, messages: List[Dict[str, Any]], *, provider: Optional[str] = None, model: Optional[str] = None,
        temperature: Optional[float] = None, max_tokens: Optional[int] = None,
        timeout: Optional[float] = None, agent_id: Optional[str] = None, profile: Optional[str] = None,
        purpose: Optional[str] = None, task: Optional[str] = None,
    ) -> PluginLlmCompleteResult:
        """Run a host-owned chat completion against the user's active model.

        ``provider``/``model``/``agent_id``/``profile`` are each gated by
        ``plugins.entries.<id>.llm.allow_*_override``. ``task`` routes through a
        plugin-registered auxiliary slot (see :func:`_check_task`)."""
        agent, kw = self._gate(provider, model, agent_id, profile, task, messages, temperature, max_tokens, timeout)
        return self._finish("complete", agent, kw, self._invoke_sync(kw), purpose)

    def complete_structured(
        self, *, instructions: str, input: Sequence[PluginLlmInput], json_schema: Optional[Any] = None,
        json_mode: bool = False, schema_name: Optional[str] = None, system_prompt: Optional[str] = None,
        provider: Optional[str] = None, model: Optional[str] = None, temperature: Optional[float] = None,
        max_tokens: Optional[int] = None, timeout: Optional[float] = None, agent_id: Optional[str] = None,
        profile: Optional[str] = None, purpose: Optional[str] = None, task: Optional[str] = None,
    ) -> PluginLlmStructuredResult:
        """Run a bounded host-owned structured completion.

        ``input`` accepts text and image blocks. With ``json_mode=True`` or a
        ``json_schema`` the response is parsed (and validated when the optional
        ``jsonschema`` package is installed) into ``result.parsed``."""
        spec = _structured_spec("complete_structured", instructions, input, system_prompt, json_mode, json_schema, schema_name)
        agent, kw = self._gate(provider, model, agent_id, profile, task, None, temperature, max_tokens, timeout, spec)
        return self._finish("complete_structured", agent, kw, self._invoke_sync(kw), purpose, spec)

    async def acomplete(
        self, messages: List[Dict[str, Any]], *, provider: Optional[str] = None, model: Optional[str] = None,
        temperature: Optional[float] = None, max_tokens: Optional[int] = None,
        timeout: Optional[float] = None, agent_id: Optional[str] = None, profile: Optional[str] = None,
        purpose: Optional[str] = None, task: Optional[str] = None,
    ) -> PluginLlmCompleteResult:
        """Async sibling of :meth:`complete`."""
        agent, kw = self._gate(provider, model, agent_id, profile, task, messages, temperature, max_tokens, timeout)
        return self._finish("acomplete", agent, kw, await self._invoke_async(kw), purpose)

    async def acomplete_structured(
        self, *, instructions: str, input: Sequence[PluginLlmInput], json_schema: Optional[Any] = None,
        json_mode: bool = False, schema_name: Optional[str] = None, system_prompt: Optional[str] = None,
        provider: Optional[str] = None, model: Optional[str] = None, temperature: Optional[float] = None,
        max_tokens: Optional[int] = None, timeout: Optional[float] = None, agent_id: Optional[str] = None,
        profile: Optional[str] = None, purpose: Optional[str] = None, task: Optional[str] = None,
    ) -> PluginLlmStructuredResult:
        """Async sibling of :meth:`complete_structured`."""
        spec = _structured_spec("acomplete_structured", instructions, input, system_prompt, json_mode, json_schema, schema_name)
        agent, kw = self._gate(provider, model, agent_id, profile, task, None, temperature, max_tokens, timeout, spec)
        return self._finish("acomplete_structured", agent, kw, await self._invoke_async(kw), purpose, spec)

    def _gate(
        self, provider: Optional[str], model: Optional[str], agent_id: Optional[str], profile: Optional[str],
        task: Optional[str], messages: Optional[List[Dict[str, Any]]], temperature: Optional[float],
        max_tokens: Optional[int], timeout: Optional[float], spec: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[str], Dict[str, Any]]:
        """Trust gate (task first, then overrides), then — for a structured ``spec`` —
        build messages/response_format (input-shape errors surface only after trust
        passes). Returns the effective agent id and the call kwargs, in the documented
        order: messages, provider_override, model_override, profile_override,
        temperature, max_tokens, timeout, extra_body, task."""
        policy = self._policy_loader(self._plugin_id)
        eff_task = _check_task(policy, plugin_id=self._plugin_id, requested_task=task)
        eff_provider, eff_model, eff_agent, eff_profile = _check_overrides(
            policy, requested_provider=provider, requested_model=model, requested_agent_id=agent_id,
            requested_profile=profile)
        extra_body = None
        if spec is not None:
            messages = _build_structured_messages(**spec)
            extra_body = _json_response_format(json_mode=spec["json_mode"], json_schema=spec["json_schema"])
        return eff_agent, dict(messages=messages, provider_override=eff_provider, model_override=eff_model,
                               profile_override=eff_profile, temperature=temperature, max_tokens=max_tokens,
                               timeout=timeout, extra_body=extra_body, task=eff_task)

    def _finish(
        self, name: str, agent_id: Optional[str], kw: Dict[str, Any], invoked: tuple[str, str, Any],
        purpose: Optional[str], spec: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Build the result object + audit dict and emit the INFO audit line."""
        real_provider, real_model, response = invoked
        text = _extract_text(response)
        usage = _extract_usage(response)
        eff_task = kw["task"] or ""
        audit: Dict[str, Any] = {"plugin_id": self._plugin_id, "purpose": purpose or "", "profile": kw["profile_override"] or ""}
        fields: Dict[str, Any] = dict(text=text, provider=real_provider, model=real_model, agent_id=agent_id or "default", usage=usage)
        fmt = f"plugin_llm.{name} plugin=%s provider=%s model=%s task=%s purpose=%s "
        log_args = [self._plugin_id, real_provider, real_model, eff_task, purpose or ""]
        cls: Any = PluginLlmCompleteResult
        if spec is not None:
            parsed, content_type = _parse_structured_text(text=text, json_mode=spec["json_mode"], json_schema=spec["json_schema"])
            audit["schema_name"] = spec["schema_name"] or ""
            fields.update(parsed=parsed, content_type=content_type)
            fmt += "content_type=%s "
            log_args.append(content_type)
            cls = PluginLlmStructuredResult
        audit["task"] = eff_task
        logger.info(fmt + "tokens=%d", *log_args, usage.total_tokens)
        return cls(**fields, audit=audit)

    @staticmethod
    def _host_kwargs(kw: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[Dict[str, str]]]:
        """Call kwargs → ``call_llm`` kwargs. The auth profile rides in
        ``extra_body.metadata.auth_profile``; ``route_info`` is only requested when
        routing through a task slot."""
        merged_extra = dict(kw["extra_body"] or {})
        if kw["profile_override"]:
            merged_extra.setdefault("metadata", {})["auth_profile"] = kw["profile_override"]
        route_info: Optional[Dict[str, str]] = {} if kw["task"] else None
        return dict(task=kw["task"], provider=kw["provider_override"], model=kw["model_override"],
                    messages=kw["messages"], temperature=kw["temperature"], max_tokens=kw["max_tokens"],
                    timeout=kw["timeout"], extra_body=merged_extra or None, route_info=route_info), route_info

    @staticmethod
    def _attributed(kw: Dict[str, Any], response: Any, route_info: Optional[Dict[str, str]]) -> tuple[str, str, Any]:
        return (*_resolve_attribution(provider_override=kw["provider_override"], model_override=kw["model_override"],
                                      response=response, route_info=route_info), response)

    def _invoke_sync(self, kw: Dict[str, Any]) -> tuple[str, str, Any]:
        """Host ``call_llm`` (lazy import: circular deps at plugin discovery) →
        ``(provider, model, response)``. An injected ``sync_caller`` replaces the
        whole path and receives the call kwargs."""
        if self._sync_caller is not None:
            return self._sync_caller(**kw)
        from agent.auxiliary_client import call_llm
        call_kw, route_info = self._host_kwargs(kw)
        return self._attributed(kw, call_llm(**call_kw), route_info)

    async def _invoke_async(self, kw: Dict[str, Any]) -> tuple[str, str, Any]:
        """Async sibling of :meth:`_invoke_sync` (``async_call_llm`` / ``async_caller``)."""
        if self._async_caller is not None:
            return await self._async_caller(**kw)
        from agent.auxiliary_client import async_call_llm
        call_kw, route_info = self._host_kwargs(kw)
        return self._attributed(kw, await async_call_llm(**call_kw), route_info)


def make_plugin_llm_for_test(*, plugin_id: str, policy: _TrustPolicy, sync_caller: Optional[Callable[..., Any]] = None,
                             async_caller: Optional[Callable[..., Awaitable[Any]]] = None) -> PluginLlm:
    """:class:`PluginLlm` with an injected policy and caller (no config.yaml, no
    provider). Not part of the public plugin API."""
    return PluginLlm(plugin_id=plugin_id, policy_loader=lambda _pid: policy, sync_caller=sync_caller, async_caller=async_caller)


__all__ = [
    "PluginLlm", "PluginLlmTextInput", "PluginLlmImageInput", "PluginLlmInput", "PluginLlmUsage",
    "PluginLlmCompleteResult", "PluginLlmStructuredResult", "PluginLlmTrustError", "make_plugin_llm_for_test",
]
