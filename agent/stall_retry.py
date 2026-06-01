"""
Agentic stall-retry (dflash Q4 premature-EOS workaround).

dflash (Qwen3.6-27B Q4_K_M, lucebox spec-decode) sometimes emits EOS right
after a short action preamble ("Let me check X:") on agentic decision turns,
ending the turn with NO tool_call -> the agent loop treats it as a final
answer and stops mid-task. Higher-precision weights (the stock Q6 lane on the
same host) continue to a real tool call on the identical prompt.

This module detects that stall signature on a no-tool-call turn and retries
the turn against a higher-quality model lane, with a small recovery nudge that
asks the model to emit the tool call it just promised. If the retry produces
tool_calls, the loop adopts that response and continues; otherwise the caller
should fail the turn as partial rather than persist the planning-only text as
a final assistant message.

Entirely opt-in: does nothing unless ``HERMES_STALL_RETRY_MODEL`` is set
(e.g. ``qwen3.6-27b-256k``). Default-off => zero change to existing behavior.

Env:
  HERMES_STALL_RETRY_MODEL  retry lane/model name (required to enable)
  HERMES_STALL_RETRY_PROVIDER  optional provider override for the retry lane
  HERMES_STALL_RETRY_BASE_URL  optional OpenAI-compatible retry endpoint
  HERMES_STALL_RETRY_MAX_PER_TURN  max unrecovered retries per user turn
                                (default 5). Successful retry-lane tool calls
                                are progress and do not consume this budget.
  HERMES_STALL_RETRY_PROMOTE_AFTER  successful rescues before routing the rest
                                of the turn through the retry lane (default 2,
                                0 disables)
  HERMES_STALL_RETRY_MAX_CHARS  max content length to still count as a stall
                                (default 400; longer open action preambles
                                ending in ":" get a bounded exception)
  HERMES_STALL_RETRY_NUDGE  true/false; add a retry-only continuation nudge
                            (default true)
  HERMES_STALL_RETRY_TELEMETRY  true/false; append local NDJSON telemetry
                                (default true)
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# Action-preamble signature: the turn announced an action but produced no tool
# call. These English phrases match the observed dflash stall corpus; broader
# language-agnostic fallbacks below still catch trailing-colon and incomplete
# final fragments without pretending this regex is multilingual.
_ACTION_RE = re.compile(
    r"(let me\b|let's\b|i'?ll\b|i will\b|i'?m going to\b|i am going to\b|"
    r"now i\b|first,?\s+i\b|next,?\s+i\b|i need to\b|i should\b|"
    r"going to (check|look|run|start|examine|search|read|list|create|write|edit|use))",
    re.IGNORECASE,
)
# Genuine completion signature: the model declared it is done / nothing to do.
# These English phrases must NOT be retried (they are correct no-tool-call
# turns); other languages still rely on the neutral structural checks below.
_COMPLETION_RE = re.compile(
    r"(\bdone\b|\bcomplete(d)?\b|nothing to (do|save|change|report|fix)|"
    r"no changes?\b|no action\b|already (complete|done|finished)|\bfinished\b|"
    r"all set\b|no further\b|nothing left\b|here('?s| is| are)\b|"
    r"in summary\b|to summarize\b|the answer is\b)",
    re.IGNORECASE,
)
_NATURAL_END_CHARS = '.!?:)"\']}。！？：）】」』》^'
_MIN_INCOMPLETE_FINAL_CHARS = 80
_ACTION_TAIL_CHARS = 500
_INCOMPLETE_TAIL_RE = re.compile(
    r"\b("
    r"and|or|but|so|because|while|with|without|for|to|of|in|on|at|by|from|"
    r"the|a|an|this|that|these|those|some|any|another|more|other|which|who|"
    r"where|when|if|then|also"
    r")\s*$",
    re.IGNORECASE,
)
_STALL_RETRY_NUDGE = (
    "Your previous assistant response ended after describing the next action, "
    "but it did not include the required tool call. Continue the same task now "
    "by making the tool call immediately. Do not summarize or apologize; call "
    "the tool that performs the action you just announced."
)
EMPTY_AFTER_TOOL_RETRY_NUDGE = (
    "Your previous assistant response after the tool results was empty. "
    "Continue the same task using the tool results above. If the next step "
    "requires another tool, call it immediately; otherwise provide the next "
    "concise response. Do not summarize or apologize."
)


def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def has_recent_tool_result(messages: Sequence[Any], *, lookback: int = 24) -> bool:
    """Return whether the current turn has a recent tool result.

    Stop at the current turn's user message so a tool-heavy previous turn does
    not make an unrelated empty first response look like post-tool fallout.
    """

    checked = 0
    for msg in reversed(messages):
        if checked >= lookback:
            return False
        if not isinstance(msg, Mapping):
            continue
        role = msg.get("role")
        if role == "tool":
            return True
        if role == "user":
            return False
        checked += 1
    return False


def _stall_retry_config(agent: Any | None = None) -> Mapping[str, Any]:
    loaded_cfg: Mapping[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        loaded = load_config()
    except Exception:
        loaded = {}
    cfg = loaded.get("stall_retry") if isinstance(loaded, Mapping) else None
    if isinstance(cfg, Mapping):
        loaded_cfg = cfg

    agent_cfg = getattr(agent, "_stall_retry_config", None)
    if not isinstance(agent_cfg, Mapping):
        return loaded_cfg
    if not agent_cfg:
        return loaded_cfg

    merged = dict(loaded_cfg)
    for key, value in agent_cfg.items():
        if value is None or value == "":
            continue
        merged[str(key)] = value
    return merged


def get_stall_retry_model(agent: Any | None = None) -> str:
    """Return the configured retry model, with env taking precedence."""
    env_model = os.environ.get("HERMES_STALL_RETRY_MODEL", "").strip()
    if env_model:
        return env_model
    cfg_model = _stall_retry_config(agent).get("model")
    return str(cfg_model or "").strip()


def get_stall_retry_provider(agent: Any | None = None) -> str:
    """Return an optional provider override for the retry lane."""
    env_provider = os.environ.get("HERMES_STALL_RETRY_PROVIDER", "").strip()
    if env_provider:
        return env_provider
    cfg_provider = _stall_retry_config(agent).get("provider")
    return str(cfg_provider or "").strip()


def get_stall_retry_base_url(agent: Any | None = None) -> str:
    """Return an optional base URL override for the retry lane."""
    env_base_url = os.environ.get("HERMES_STALL_RETRY_BASE_URL", "").strip()
    if env_base_url:
        return env_base_url
    cfg_base_url = _stall_retry_config(agent).get("base_url")
    return str(cfg_base_url or "").strip()


def get_stall_retry_max_chars(agent: Any | None = None) -> int:
    env_value = os.environ.get("HERMES_STALL_RETRY_MAX_CHARS")
    if env_value is not None:
        return _as_positive_int(env_value, 400)
    return _as_positive_int(_stall_retry_config(agent).get("max_chars"), 400)


def get_stall_retry_max_per_turn(agent: Any | None = None) -> int:
    env_value = os.environ.get("HERMES_STALL_RETRY_MAX_PER_TURN")
    if env_value is not None:
        try:
            return max(0, int(env_value))
        except ValueError:
            return 5
    cfg_value = _stall_retry_config(agent).get("max_per_turn")
    try:
        return max(0, int(cfg_value))
    except (TypeError, ValueError):
        return 5


def get_stall_retry_promote_after(agent: Any | None = None) -> int:
    """Return successful retry rescues needed before turn-scoped promotion."""
    env_value = os.environ.get("HERMES_STALL_RETRY_PROMOTE_AFTER")
    if env_value is not None:
        try:
            return max(0, int(env_value))
        except ValueError:
            return 2
    cfg_value = _stall_retry_config(agent).get("promote_after")
    try:
        return max(0, int(cfg_value))
    except (TypeError, ValueError):
        return 2


def get_stall_retry_nudge_enabled(agent: Any | None = None) -> bool:
    env_value = os.environ.get("HERMES_STALL_RETRY_NUDGE")
    if env_value is not None:
        return _as_bool(env_value, True)
    return _as_bool(_stall_retry_config(agent).get("nudge"), True)


def get_stall_retry_telemetry_enabled(agent: Any | None = None) -> bool:
    env_value = os.environ.get("HERMES_STALL_RETRY_TELEMETRY")
    if env_value is not None:
        return _as_bool(env_value, True)
    return _as_bool(_stall_retry_config(agent).get("telemetry"), True)


def _has_natural_response_ending(content: str) -> bool:
    stripped = (content or "").rstrip()
    if not stripped:
        return False
    if stripped.endswith("```"):
        return True
    last = stripped[-1]
    if last in _NATURAL_END_CHARS:
        return True
    return ord(last) >= 0x1F300


def looks_like_incomplete_final_fragment(
    content: str,
    finish_reason: str,
    has_tool_calls: bool,
    max_chars: int,
) -> bool:
    """True when a short no-tool final looks cut off mid-sentence.

    dflash can occasionally stop with ordinary visible prose after a tool
    result, e.g. ``"... and some"``. That is not an action preamble, but it is
    still unsafe to persist as the final answer in a tool loop.
    """
    if has_tool_calls or finish_reason not in ("stop", "length"):
        return False
    c = (content or "").strip()
    c = re.sub(r"^<think>.*?</think>\s*", "", c, flags=re.IGNORECASE | re.DOTALL).strip()
    if not c or len(c) > max_chars:
        return False
    if _COMPLETION_RE.search(c) or _has_natural_response_ending(c):
        return False
    if len(c) >= _MIN_INCOMPLETE_FINAL_CHARS:
        return True
    return len(c) >= 40 and bool(_INCOMPLETE_TAIL_RE.search(c))


def _action_after_completion(content: str) -> bool:
    """True when a response says some step is complete, then promises work.

    The dflash phone failure hit this exact shape:
    "Onboarding complete. Now let me read the STATUS.md ..."

    The earlier completion word is not a final answer when a later clause is
    still announcing the next tool step.
    """
    action_matches = list(_ACTION_RE.finditer(content or ""))
    if not action_matches:
        return False
    completion_matches = list(_COMPLETION_RE.finditer(content or ""))
    if not completion_matches:
        return False
    return action_matches[-1].start() > completion_matches[-1].end()


def _ends_with_action_promise(content: str) -> bool:
    """True when the visible tail promises immediate work but stops there.

    The generic stall heuristic is intentionally length-capped because long
    prose is often a real answer. Explicit tail promises are different: a long
    diagnostic can still end with "Let me check that:" and no tool call, which
    is the exact dflash premature-stop shape this module exists to recover.
    """
    tail = (content or "").strip()[-_ACTION_TAIL_CHARS:]
    if not tail:
        return False
    if not _ACTION_RE.search(tail):
        return False
    return tail.rstrip().endswith(":")


def _safe_preview(value: Any, max_chars: int = 240) -> str:
    text = value if isinstance(value, str) else str(value or "")
    text = re.sub(r"^<think>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _stall_retry_log_path(agent: Any | None = None) -> Path:
    cfg_path = _stall_retry_config(agent).get("telemetry_path")
    if cfg_path:
        return Path(str(cfg_path)).expanduser()
    try:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home())
    except Exception:
        home = Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()
    return home / "logs" / "stall-retry.ndjson"


def record_stall_retry_event(agent: Any, event: str, **fields: Any) -> None:
    """Record local, bounded stall-retry telemetry."""
    entry: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event": str(event),
        "session_id": str(getattr(agent, "session_id", "") or ""),
        "model": str(getattr(agent, "model", "") or ""),
        "provider": str(getattr(agent, "provider", "") or ""),
    }
    content = fields.pop("content", None)
    if content is not None:
        text = content if isinstance(content, str) else str(content)
        entry["content_chars"] = len(text)
        entry["content_preview"] = _safe_preview(text)
    entry.update({str(k): _jsonable(v) for k, v in fields.items()})

    events = getattr(agent, "_stall_retry_events", None)
    if not isinstance(events, list):
        events = []
        try:
            setattr(agent, "_stall_retry_events", events)
        except Exception:
            pass
    events.append(entry)

    if not get_stall_retry_telemetry_enabled(agent):
        return
    try:
        path = _stall_retry_log_path(agent)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        return


def stall_retry_summary(agent: Any) -> dict[str, Any] | None:
    events = getattr(agent, "_stall_retry_events", None)
    if not isinstance(events, list) or not events:
        return None
    counts = {
        "detected": 0,
        "attempted": 0,
        "recovered": 0,
        "failed": 0,
        "limit_exhausted": 0,
        "exceptions": 0,
    }
    for item in events:
        kind = item.get("event") if isinstance(item, Mapping) else None
        if kind == "detected":
            counts["detected"] += 1
        elif kind == "attempt":
            counts["attempted"] += 1
        elif kind == "recovered":
            counts["recovered"] += 1
        elif kind in {"failed_no_tool_call", "api_none", "skipped_same_model"}:
            counts["failed"] += 1
        elif kind == "limit_exhausted":
            counts["limit_exhausted"] += 1
        elif kind == "exception":
            counts["exceptions"] += 1
    summary: dict[str, Any] = dict(counts)
    summary["events"] = len(events)
    if get_stall_retry_telemetry_enabled(agent):
        summary["log_path"] = str(_stall_retry_log_path(agent))
    return summary


def _retry_messages_with_nudge(
    agent: Any,
    api_messages: list[dict[str, Any]],
    stalled_content: str,
    retry_nudge: str | None = None,
) -> list[dict[str, Any]]:
    if not get_stall_retry_nudge_enabled(agent):
        return api_messages
    retry_messages = [msg.copy() if isinstance(msg, dict) else msg for msg in api_messages]
    visible = (stalled_content or "").strip()
    if visible:
        retry_messages.append({"role": "assistant", "content": visible})
    retry_messages.append({"role": "user", "content": retry_nudge or _STALL_RETRY_NUDGE})
    return retry_messages


def _load_env_value(name: str) -> str:
    env_name = str(name or "").strip()
    if not env_name:
        return ""
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    try:
        from hermes_cli.env_loader import load_hermes_dotenv
        from hermes_constants import get_hermes_home

        load_hermes_dotenv(hermes_home=get_hermes_home())
    except Exception:
        pass
    return os.environ.get(env_name, "").strip()


def _base_url_keys(value: Any) -> set[str]:
    raw = str(value or "").strip().lower().rstrip("/")
    if not raw:
        return set()
    keys = {raw}
    if raw.endswith("/v1"):
        keys.add(raw[:-3].rstrip("/"))
    else:
        keys.add(f"{raw}/v1")
    return keys


def _custom_provider_entry(agent: Any | None, provider: str, base_url: str) -> Mapping[str, Any]:
    try:
        from hermes_cli.config import get_compatible_custom_providers, load_config

        entries = get_compatible_custom_providers(load_config())
    except Exception:
        entries = getattr(agent, "_custom_providers", []) if agent is not None else []
    if not isinstance(entries, list):
        return {}

    provider_name = str(provider or "").strip().lower()
    target_keys = _base_url_keys(base_url or getattr(agent, "base_url", ""))
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        entry_name = str(entry.get("name") or "").strip().lower()
        if provider_name and entry_name == provider_name:
            return entry
        entry_keys = _base_url_keys(entry.get("base_url"))
        if target_keys and entry_keys and target_keys.intersection(entry_keys):
            return entry
    return {}


def _configured_retry_api_key(agent: Any, provider: str, base_url: str) -> str:
    cfg = _stall_retry_config(agent)
    explicit = str(cfg.get("api_key") or "").strip()
    if explicit:
        return explicit
    key_env = str(cfg.get("key_env") or cfg.get("api_key_env") or "").strip()
    if key_env:
        return _load_env_value(key_env)

    entry = _custom_provider_entry(agent, provider, base_url)
    explicit = str(entry.get("api_key") or "").strip()
    if explicit:
        return explicit
    key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
    return _load_env_value(key_env) if key_env else ""


def _retry_api_call(agent: Any, api_kwargs: dict[str, Any], retry_model: str) -> Any:
    """Execute the retry request, optionally through a configured provider.

    The original implementation reused the active client and only changed the
    model name. That is fine when both models are served anonymously by the
    same endpoint, but it breaks when the retry lane is a named custom provider
    whose auth lives in ``key_env``. Resolve that provider explicitly when the
    retry config names one or supplies endpoint/auth details.
    """
    retry_provider = get_stall_retry_provider(agent)
    retry_base_url = get_stall_retry_base_url(agent)
    retry_api_key = _configured_retry_api_key(agent, retry_provider, retry_base_url)
    if not retry_base_url and retry_api_key:
        retry_base_url = str(getattr(agent, "base_url", "") or "").strip()

    if retry_provider or retry_base_url or retry_api_key:
        from agent.auxiliary_client import resolve_provider_client

        provider = retry_provider or str(getattr(agent, "provider", "") or "custom")
        client, resolved_model = resolve_provider_client(
            provider,
            model=retry_model,
            raw_codex=True,
            explicit_base_url=retry_base_url or None,
            explicit_api_key=retry_api_key or None,
        )
        if client is None:
            raise RuntimeError(f"Could not resolve stall retry provider {provider!r}")
        retry_kwargs = dict(api_kwargs)
        retry_kwargs["model"] = resolved_model or retry_model
        return client.chat.completions.create(**retry_kwargs)

    return agent._interruptible_api_call(api_kwargs)


def activate_stall_retry_runtime(
    agent: Any,
    retry_model: str,
    *,
    promote_after: int,
    successful_retries: int,
) -> bool:
    """Route the rest of this turn through the retry lane.

    A repeated pattern of primary-model stalls followed by retry-lane tool-call
    recovery means the primary is no longer reliable for this agentic turn. This
    helper promotes the already-configured retry lane into the active runtime
    without updating ``_primary_runtime``; Hermes' normal start-of-turn restore
    then switches back to the user's selected primary on the next user turn.
    """

    retry_model = str(retry_model or "").strip()
    if not retry_model:
        return False
    if getattr(agent, "_stall_retry_runtime_promoted", False):
        return False

    original_model = str(getattr(agent, "model", "") or "")
    original_provider = str(getattr(agent, "provider", "") or "")
    original_base_url = str(getattr(agent, "base_url", "") or "")
    retry_provider = get_stall_retry_provider(agent)
    retry_base_url = get_stall_retry_base_url(agent)
    retry_api_key = _configured_retry_api_key(agent, retry_provider, retry_base_url)
    if not retry_base_url and retry_api_key:
        retry_base_url = original_base_url

    provider = retry_provider or original_provider
    base_url = retry_base_url or original_base_url
    api_key = retry_api_key or getattr(agent, "api_key", "")
    client = None
    resolved_model = retry_model
    client_kwargs: dict[str, Any] = {}

    try:
        if retry_provider or retry_base_url or retry_api_key:
            from agent.auxiliary_client import resolve_provider_client

            client, provider_model = resolve_provider_client(
                provider or "custom",
                model=retry_model,
                raw_codex=True,
                explicit_base_url=retry_base_url or None,
                explicit_api_key=retry_api_key or None,
            )
            if client is None:
                record_stall_retry_event(
                    agent,
                    "promotion_skipped",
                    retry_model=retry_model,
                    original_model=original_model,
                    reason="provider_unavailable",
                    promote_after=promote_after,
                    successful_retries=successful_retries,
                )
                return False
            resolved_model = provider_model or retry_model
            base_url = str(getattr(client, "base_url", base_url) or base_url)
            api_key = getattr(client, "api_key", api_key)
            headers = (
                getattr(client, "_custom_headers", None)
                or getattr(client, "default_headers", None)
            )
            client_kwargs = {
                "api_key": api_key,
                "base_url": base_url,
                **({"default_headers": dict(headers)} if headers else {}),
            }
        else:
            if retry_model == original_model:
                record_stall_retry_event(
                    agent,
                    "promotion_skipped",
                    retry_model=retry_model,
                    original_model=original_model,
                    reason="same_model",
                    promote_after=promote_after,
                    successful_retries=successful_retries,
                )
                return False
            client_kwargs = dict(getattr(agent, "_client_kwargs", {}) or {})
            if not client_kwargs:
                client_kwargs = {
                    "api_key": api_key,
                    "base_url": base_url,
                }

        try:
            from hermes_cli.providers import determine_api_mode

            api_mode = determine_api_mode(provider, base_url)
        except Exception:
            api_mode = getattr(agent, "api_mode", "") or "chat_completions"
        if api_mode in {"anthropic_messages", "bedrock_converse", "codex_responses"}:
            record_stall_retry_event(
                agent,
                "promotion_skipped",
                retry_model=retry_model,
                original_model=original_model,
                reason=f"unsupported_api_mode:{api_mode}",
                promote_after=promote_after,
                successful_retries=successful_retries,
            )
            return False

        try:
            from hermes_cli.timeouts import get_provider_request_timeout

            timeout = get_provider_request_timeout(provider, resolved_model)
            if timeout is not None:
                client_kwargs["timeout"] = timeout
        except Exception:
            pass

        agent._config_context_length = None
        agent.model = resolved_model
        agent.provider = provider
        agent.base_url = base_url
        agent.api_key = api_key
        agent.api_mode = api_mode or "chat_completions"
        agent._client_kwargs = client_kwargs
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()
        if client is not None:
            agent.client = client
        elif hasattr(agent, "_create_openai_client"):
            agent.client = agent._create_openai_client(
                dict(client_kwargs),
                reason="stall_retry_runtime_promotion",
                shared=True,
            )

        try:
            agent._use_prompt_caching, agent._use_native_cache_layout = (
                agent._anthropic_prompt_cache_policy(
                    provider=provider,
                    base_url=base_url,
                    api_mode=agent.api_mode,
                    model=resolved_model,
                )
            )
        except Exception:
            pass

        if hasattr(agent, "_ensure_lmstudio_runtime_loaded"):
            agent._ensure_lmstudio_runtime_loaded()

        context_length = None
        if getattr(agent, "context_compressor", None):
            try:
                from agent.model_metadata import get_model_context_length

                ctx_api_key = api_key if isinstance(api_key, str) else ""
                context_length = get_model_context_length(
                    resolved_model,
                    base_url=base_url,
                    api_key=ctx_api_key,
                    provider=provider,
                    config_context_length=getattr(agent, "_config_context_length", None),
                    custom_providers=getattr(agent, "_custom_providers", None),
                )
                agent.context_compressor.update_model(
                    model=resolved_model,
                    context_length=context_length,
                    base_url=base_url,
                    api_key=api_key,
                    provider=provider,
                    api_mode=agent.api_mode,
                )
            except Exception:
                pass

        agent._fallback_activated = True
        agent._stall_retry_runtime_promoted = True
        agent._stall_retry_promoted_from = original_model

        record_stall_retry_event(
            agent,
            "runtime_promoted",
            retry_model=resolved_model,
            original_model=original_model,
            original_provider=original_provider,
            promote_after=promote_after,
            successful_retries=successful_retries,
            context_length=context_length,
        )
        try:
            agent._emit_status(
                "↻ dflash stalled repeatedly; using "
                f"{resolved_model} for the rest of this turn. "
                "Primary model will be restored next turn."
            )
        except Exception:
            try:
                agent._vprint(
                    f"{getattr(agent, 'log_prefix', '')}↻ dflash stalled "
                    f"repeatedly; using {resolved_model} for the rest of this turn.",
                    force=True,
                )
            except Exception:
                pass
        return True
    except Exception as exc:
        record_stall_retry_event(
            agent,
            "promotion_exception",
            retry_model=retry_model,
            original_model=original_model,
            error_type=type(exc).__name__,
            error=str(exc)[:300],
            promote_after=promote_after,
            successful_retries=successful_retries,
        )
        return False


def looks_like_stall(content: str, finish_reason: str, has_tool_calls: bool,
                     max_chars: int) -> bool:
    """True when a no-tool-call turn looks like a premature agentic stall
    (announced an action, didn't call a tool) rather than a real final answer."""
    if has_tool_calls:
        return False
    if finish_reason not in ("stop", "length"):
        return False
    c = (content or "").strip()
    # Strip a leading <think>...</think> block if present; judge the visible tail.
    c = re.sub(r"^<think>.*?</think>\s*", "", c, flags=re.IGNORECASE | re.DOTALL).strip()
    if not c:
        # Truly empty responses have their own recovery path in the
        # conversation loop. Do not let stall retry preempt that machinery.
        return False
    if _action_after_completion(c):
        return True
    if _COMPLETION_RE.search(c):
        return False  # model said it's done => respect it
    if _ends_with_action_promise(c):
        return True
    if len(c) > max_chars:
        return False  # long => almost certainly a real answer
    if _ACTION_RE.search(c):
        return True   # announced an action, no tool call => stall
    # Short prose that doesn't declare completion and isn't an obvious answer:
    # a trailing colon strongly implies "about to do something".
    if c.endswith(":"):
        return True
    # dflash can also stop after a tool result with ordinary-looking prose that
    # is simply cut off mid-sentence (for example after a CLI interrupt resumes
    # the turn). In an agentic tool loop, a short no-tool stop that declares no
    # completion and lacks a natural ending is safer to retry than to persist as
    # a final assistant message.
    if looks_like_incomplete_final_fragment(c, finish_reason, False, max_chars):
        return True
    return False


def retry_on_stall(
    agent,
    api_messages,
    finish_reason,
    stalled_content: str = "",
    retry_index: int | None = None,
    *,
    accept_content: bool = False,
    retry_nudge: str | None = None,
):
    """If the just-finished no-tool-call turn looks like a stall and a retry
    lane is configured, re-issue the turn against that lane (same provider /
    client / endpoint — only the model name changes). A retry-only nudge is
    appended by default so the fallback model is told to continue with a tool
    call instead of repeating the action preamble.

    Returns the normalized assistant_message from the retry IF it produced tool
    calls (caller should adopt it + its finish_reason='tool_calls'), else None.
    Never raises into the caller — any failure returns None so the caller can
    fail closed without storing the stalled assistant message.
    """
    retry_model = get_stall_retry_model(agent)
    if not retry_model:
        return None

    try:
        retry_messages = _retry_messages_with_nudge(
            agent,
            api_messages,
            stalled_content,
            retry_nudge=retry_nudge,
        )
        # Build kwargs exactly as the normal turn would, then override only the
        # model name. Safe when the retry lane is served by the SAME provider/
        # endpoint as agent.model (e.g. taro serves both dflash and the Q6 lane),
        # so no client rebuild is needed.
        api_kwargs = agent._build_api_kwargs(retry_messages)
        orig_model = api_kwargs.get("model")
        if retry_model == orig_model:
            record_stall_retry_event(
                agent,
                "skipped_same_model",
                retry_model=retry_model,
                finish_reason=finish_reason,
                retry_index=retry_index,
            )
            return None  # nothing to gain retrying the same model
        api_kwargs = dict(api_kwargs)
        api_kwargs["model"] = retry_model
        # Force non-streaming for the retry (simpler, we only inspect the result).
        api_kwargs.pop("stream", None)
        api_kwargs["stream"] = False

        try:
            agent._vprint(
                f"{getattr(agent, 'log_prefix', '')}↻ stall detected "
                f"(no tool call) — retrying turn on '{retry_model}'",
                force=True,
            )
        except Exception:
            pass

        record_stall_retry_event(
            agent,
            "attempt",
            retry_model=retry_model,
            original_model=orig_model,
            finish_reason=finish_reason,
            retry_index=retry_index,
            nudge=get_stall_retry_nudge_enabled(agent),
            content=stalled_content,
        )
        response = _retry_api_call(agent, api_kwargs, retry_model)
        if response is None:
            record_stall_retry_event(
                agent,
                "api_none",
                retry_model=retry_model,
                retry_index=retry_index,
            )
            return None
        transport = agent._get_transport()
        normalize_kwargs = {}
        if getattr(agent, "api_mode", None) == "anthropic_messages":
            normalize_kwargs["strip_tool_prefix"] = getattr(agent, "_is_anthropic_oauth", False)
        normalized = transport.normalize_response(response, **normalize_kwargs)
        tool_calls = getattr(normalized, "tool_calls", None)
        content = getattr(normalized, "content", "") or ""
        if tool_calls or (accept_content and content.strip()):
            record_stall_retry_event(
                agent,
                "recovered",
                retry_model=retry_model,
                retry_index=retry_index,
                tool_call_count=len(tool_calls or []),
                content=content,
            )
            return normalized
        record_stall_retry_event(
            agent,
            "failed_no_tool_call",
            retry_model=retry_model,
            retry_index=retry_index,
            content=getattr(normalized, "content", "") or "",
        )
        return None
    except Exception as exc:
        record_stall_retry_event(
            agent,
            "exception",
            retry_model=retry_model,
            retry_index=retry_index,
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        # Any error => silently fall back to the original response.
        return None
