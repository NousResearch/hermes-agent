"""Langfuse LLM observability plugin for Hermes (SDK v3).

Maps Hermes lifecycle hooks to Langfuse observations:

  Trace      = Session (on_session_start -> on_session_end / on_session_finalize)
  Generation = LLM API call (pre_api_request -> post_api_request)
  Span       = Tool call (pre_tool_call -> post_tool_call)

Uses Langfuse v3 SDK: client.start_observation() with manual .update()/.end().

Config
------
Two switches must both be on for the plugin to activate:

1. ``plugins.enabled`` must include ``langfuse-tracing`` (standard plugin gate).
2. ``langfuse.enabled: true`` in ``config.yaml`` (plugin-specific opt-in).

Environment variables:

- ``LANGFUSE_PUBLIC_KEY`` -- project public key
- ``LANGFUSE_SECRET_KEY`` -- project secret key
- ``LANGFUSE_HOST`` or ``LANGFUSE_BASE_URL`` -- base URL (default: cloud)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# -- Lazy imports (langfuse may not be installed) --
_langfuse_module: Any = None


def _get_langfuse():
    global _langfuse_module
    if _langfuse_module is not None:
        return _langfuse_module
    try:
        import langfuse  # type: ignore

        _langfuse_module = langfuse
        return _langfuse_module
    except ImportError:
        _langfuse_module = False
        return None


# ======================================================================
# Configuration helpers
# ======================================================================


def _is_langfuse_enabled() -> bool:
    """Check ``langfuse.enabled`` in config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        return bool(cfg.get("langfuse", {}).get("enabled", False))
    except Exception:
        return False


def _get_env(key: str) -> Optional[str]:
    """Read an env var (os.environ first, then ~/.hermes/.env)."""
    val = os.getenv(key)
    if val is not None:
        return val
    try:
        from hermes_cli.config import get_env_value

        return get_env_value(key)
    except Exception:
        return None


def _get_langfuse_config() -> Optional[Dict[str, Any]]:
    """Collect Langfuse connection config from environment and config.yaml."""
    public_key = _get_env("LANGFUSE_PUBLIC_KEY")
    secret_key = _get_env("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        return None

    host = (
        _get_env("LANGFUSE_HOST")
        or _get_env("LANGFUSE_BASE_URL")
        or "https://cloud.langfuse.com"
    )

    return {
        "public_key": public_key,
        "secret_key": secret_key,
        "host": host,
    }


# ======================================================================
# Global state
# ======================================================================

_CLIENT: Any = None
_CLIENT_READY = False
_CLIENT_LOCK = threading.Lock()

# session_id -> ActiveObjects
_STATE_LOCK = threading.RLock()
_SESSIONS: Dict[str, _ActiveObjects] = {}


class _ActiveObjects:
    """In-flight Langfuse observations for a single session."""

    __slots__ = ("trace", "user_message", "assistant_response", "user_id", "company_id", "active_generations", "active_spans")

    def __init__(self) -> None:
        self.trace: Any = None  # LangfuseSpan (root)
        self.user_message: str = ""  # latest user message (from pre_llm_call)
        self.assistant_response: str = ""  # latest assistant response (from post_llm_call)
        self.user_id: str = ""  # decoded from x-user JWT (pre_llm_call)
        self.company_id: str = ""  # decoded from x-user JWT (pre_llm_call)
        self.active_generations: Dict[str, Any] = {}
        self.active_spans: Dict[str, Any] = {}


def _init_client() -> Any:
    """Lazily initialise the Langfuse client (thread-safe, idempotent)."""
    global _CLIENT, _CLIENT_READY

    if _CLIENT_READY:
        return _CLIENT

    with _CLIENT_LOCK:
        if _CLIENT_READY:
            return _CLIENT

        langfuse = _get_langfuse()
        if langfuse is False:
            logger.debug("langfuse-tracing: langfuse package not installed")
            _CLIENT_READY = True
            return None

        lf_cfg = _get_langfuse_config()
        if lf_cfg is None:
            logger.debug(
                "langfuse-tracing: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set"
            )
            _CLIENT_READY = True
            return None

        try:
            _CLIENT = langfuse.Langfuse(
                public_key=lf_cfg["public_key"],
                secret_key=lf_cfg["secret_key"],
                host=lf_cfg["host"],
            )
            logger.info("Langfuse 客户端初始化完成: %s", lf_cfg["host"])
        except Exception as exc:
            logger.debug("langfuse-tracing: failed to initialise client: %s", exc)
            _CLIENT = None

        _CLIENT_READY = True
        return _CLIENT


# -- State helpers -----------------------------------------------------


def _get_session(session_id: str) -> _ActiveObjects:
    with _STATE_LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = _ActiveObjects()
        return _SESSIONS[session_id]


def _create_trace(
    client: Any, session_id: str, model: str, platform: str, *, lazy: bool = False
) -> Any:
    """Create a root observation (trace) for *session_id*.

    Returns the LangfuseSpan, or None on failure.
    """
    try:
        trace = client.start_observation(
            name="hermes-session",
            as_type="span",
            input={"model": model, "platform": platform},
            metadata={
                "session_id": session_id,
                "model": model,
                "platform": platform,
                "component": "hermes-agent",
            },
        )
        trace.update_trace(
            user_id=session_id[:64],
            session_id=session_id[:64],
        )
        return trace
    except Exception as exc:
        logger.warning("Langfuse trace 创建失败: %s", exc)
        return None


# ======================================================================
# Hook callbacks
# ======================================================================


def _on_session_start(**kwargs: Any) -> None:
    """Create the Langfuse trace for this session.

    Hermes kwargs: session_id, model, platform, x_user_token
    """
    client = _init_client()
    if client is None:
        return

    session_id = kwargs.get("session_id", "") or ""
    if not session_id:
        return

    model = kwargs.get("model", "")
    platform = kwargs.get("platform", "")

    trace = _create_trace(client, session_id, model, platform)
    if trace is not None:
        active = _get_session(session_id)
        active.trace = trace

        # 解析 x-user JWT，提取用户和企业信息
        x_user_token = kwargs.get("x_user_token", "") or ""
        if x_user_token:
            jwt_payload = _decode_jwt_payload(x_user_token)
            user_id = str(jwt_payload.get("id", "")) or None
            company_id = str(jwt_payload.get("companyId", "")) or None
            if user_id or company_id:
                logger.info("解析 x-user JWT 成功: 用户=%s, 企业=%s",
                             user_id or "-", company_id or "-")
            if user_id:
                active.user_id = user_id
            if company_id:
                active.company_id = company_id
            if user_id or company_id:
                tags = [platform] if platform else []
                meta: Dict[str, Any] = {"platform": platform}
                if user_id:
                    meta["user_id"] = user_id
                if company_id:
                    meta["company_id"] = company_id
                    tags.append(f"company:{company_id}")
                try:
                    active.trace.update_trace(
                        user_id=user_id or session_id[:64],
                        session_id=session_id[:64],
                        tags=tags,
                        metadata=meta,
                    )
                except Exception as exc:
                    logger.warning("update_trace 失败: %s", exc)


def _on_session_end(**kwargs: Any) -> None:
    """Update trace output at turn-end.  Flush when the session truly ends.

    Hermes kwargs: session_id, completed, interrupted, model, platform
    """
    session_id = kwargs.get("session_id", "") or ""
    active = _SESSIONS.get(session_id) if session_id else None
    if active is None or active.trace is None:
        return

    try:
        active.trace.update(
            output={
                "completed": kwargs.get("completed", True),
                "interrupted": kwargs.get("interrupted", False),
                "model": kwargs.get("model", ""),
                "assistant_response": _truncate(active.assistant_response, 10000),
                "user_message": _truncate(active.user_message, 5000),
            }
        )
    except Exception as exc:
        logger.warning("on_session_end 更新失败: %s", exc)

    if kwargs.get("completed") or kwargs.get("interrupted"):
        _flush_and_cleanup(session_id)


def _on_session_finalize(**kwargs: Any) -> None:
    """Flush when the Gateway destroys a session (timeout, /reset, /new).

    Hermes kwargs: session_id, platform
    """
    session_id = kwargs.get("session_id", "") or ""
    if session_id and _SESSIONS.get(session_id):
        try:
            active = _SESSIONS[session_id]
            if active.trace is not None:
                active.trace.update(
                    output={
                        "status": "finalized",
                        "platform": kwargs.get("platform", ""),
                    }
                )
        except Exception as exc:
            logger.warning("on_session_finalize 更新失败: %s", exc)

    _flush_and_cleanup(session_id)


def _decode_jwt_payload(token: str) -> Dict[str, Any]:
    """Decode a JWT payload without signature verification.

    Returns an empty dict on any parse failure.
    """
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1]
        # base64url: replace URL-safe chars, add padding
        payload = payload.replace("-", "+").replace("_", "/")
        payload += "=" * ((4 - len(payload) % 4) % 4)
        decoded = __import__("base64").b64decode(payload)
        return __import__("json").loads(decoded)
    except Exception as exc:
        logger.warning("JWT 解析失败: %s", exc)
        return {}


def _on_pre_llm_call(**kwargs: Any) -> None:
    """Store user message and update trace attributes before each turn.

    Hermes kwargs: session_id, user_message, conversation_history,
                    is_first_turn, model, platform, sender_id, x_user_token
    """
    session_id = kwargs.get("session_id", "") or ""
    user_message = kwargs.get("user_message", "") or ""
    if not session_id or not user_message:
        return

    active = _ensure_trace(
        session_id, kwargs.get("model", ""), kwargs.get("platform", "")
    )
    if active is None:
        return

    active.user_message = user_message

    try:
        active.trace.update(input={"user_message": _truncate(user_message, 10000)})

        # 解析 x-user JWT，提取用户和企业信息
        x_user_token = kwargs.get("x_user_token", "") or ""
        if x_user_token:
            jwt_payload = _decode_jwt_payload(x_user_token)
            user_id = str(jwt_payload.get("id", "")) or None
            company_id = str(jwt_payload.get("companyId", "")) or None

            # 存入状态供其他 hook 读取
            active.user_id = user_id or ""
            active.company_id = company_id or ""

            if user_id or company_id:
                tags = [kwargs.get("platform", "")]
                meta: Dict[str, Any] = {"platform": kwargs.get("platform", "")}
                if user_id:
                    meta["user_id"] = user_id
                if company_id:
                    meta["company_id"] = company_id
                    tags.append(f"company:{company_id}")

                active.trace.update_trace(
                    user_id=user_id or session_id[:64],
                    session_id=session_id[:64],
                    tags=tags,
                    metadata=meta,
                )
                logger.info("解析 x-user JWT 成功: 用户=%s, 企业=%s",
                             user_id or "-", company_id or "-")
            else:
                logger.warning("JWT 解析成功但缺少 id/companyId 字段: %s",
                               list(jwt_payload.keys()))
    except Exception as exc:
        logger.warning("pre_llm_call 异常: %s", exc)


def _on_post_llm_call(**kwargs: Any) -> None:
    """Update trace output with the assistant response after each turn.

    Hermes kwargs: session_id, user_message, assistant_response,
                    conversation_history, model, platform
    """
    session_id = kwargs.get("session_id", "") or ""
    assistant_response = kwargs.get("assistant_response", "") or ""
    if not session_id:
        return

    active = _SESSIONS.get(session_id)
    if active is None or active.trace is None:
        return

    active.assistant_response = assistant_response

    try:
        active.trace.update(
            output={"assistant_response": _truncate(assistant_response, 10000)}
        )
    except Exception as exc:
        logger.warning("post_llm_call 异常: %s", exc)


def _ensure_trace(
    session_id: str, model: str, platform: str
) -> Optional[_ActiveObjects]:
    """Return ActiveObjects for *session_id*, creating the trace lazily if needed.

    Some platforms (API server) may never fire ``on_session_start``.
    """
    active = _get_session(session_id)
    if active.trace is not None:
        return active

    client = _init_client()
    if client is None:
        return None

    trace = _create_trace(client, session_id, model, platform, lazy=True)
    if trace is None:
        return None

    active.trace = trace
    return active


def _on_pre_api_request(**kwargs: Any) -> None:
    """Create a Generation observation for this API call.

    Hermes kwargs:
      task_id, session_id, platform, model, provider, base_url, api_mode,
      api_call_count, message_count, tool_count, approx_input_tokens,
      request_char_count, max_tokens
    """
    session_id = kwargs.get("session_id", "") or ""
    if not session_id:
        return

    model = kwargs.get("model", "")
    platform = kwargs.get("platform", "")

    active = _ensure_trace(session_id, model, platform)
    if active is None:
        return

    task_id = kwargs.get("task_id", "")
    api_call_count = kwargs.get("api_call_count", 0)
    gen_key = f"{task_id}:{api_call_count}"

    try:
        gen = active.trace.start_observation(
            name=f"llm:{model}",
            as_type="generation",
            model=model,
            input={
                "message_count": kwargs.get("message_count", 0),
                "tool_count": kwargs.get("tool_count", 0),
                "approx_input_tokens": kwargs.get("approx_input_tokens", 0),
                "user_message": _truncate(active.user_message, 5000),
            },
            model_parameters={
                "max_tokens": kwargs.get("max_tokens", 0),
                "provider": kwargs.get("provider", ""),
            },
            metadata={
                "api_mode": kwargs.get("api_mode", ""),
                "platform": kwargs.get("platform", ""),
                "task_id": task_id,
                "api_call_count": api_call_count,
            },
        )
        active.active_generations[gen_key] = gen
    except Exception as exc:
        logger.warning("pre_api_request 异常: %s", exc)


def _on_post_api_request(**kwargs: Any) -> None:
    """End the Generation with usage, latency, and output info.

    Hermes kwargs (same as pre_api_request +):
      api_duration, finish_reason, response_model,
      usage (dict), assistant_content_chars, assistant_tool_call_count,
      reasoning_content (str)
    """
    session_id = kwargs.get("session_id", "") or ""
    active = _SESSIONS.get(session_id)
    if active is None:
        return

    task_id = kwargs.get("task_id", "")
    api_call_count = kwargs.get("api_call_count", 0)
    gen_key = f"{task_id}:{api_call_count}"
    gen = active.active_generations.pop(gen_key, None)
    if gen is None:
        return

    usage: Dict[str, Any] = kwargs.get("usage") or {}
    reasoning = kwargs.get("reasoning_content", "") or ""

    try:
        gen.update(
            output={
                "finish_reason": kwargs.get("finish_reason", ""),
                "content_chars": kwargs.get("assistant_content_chars", 0),
                "tool_call_count": kwargs.get("assistant_tool_call_count", 0),
                "reasoning": _truncate(reasoning, 10000),
            },
            usage_details={
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
                "total": usage.get("total_tokens", 0),
            },
            metadata={
                "duration_s": kwargs.get("api_duration", 0),
                "response_model": kwargs.get("response_model", ""),
                "cache_read_tokens": usage.get("cache_read_tokens", 0),
                "cache_write_tokens": usage.get("cache_write_tokens", 0),
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
            },
        )
        gen.end()
        logger.info("LLM 调用完成: model=%s tokens(in=%s out=%s total=%s) 耗时=%.1fs",
                     kwargs.get("model", ""),
                     usage.get("input_tokens", 0),
                     usage.get("output_tokens", 0),
                     usage.get("total_tokens", 0),
                     kwargs.get("api_duration", 0))
    except Exception as exc:
        logger.warning("post_api_request 异常: %s", exc)


def _on_pre_tool_call(**kwargs: Any) -> None:
    """Create a Span observation for this tool call.

    Hermes kwargs: tool_name, args, task_id, session_id, tool_call_id
    """
    session_id = kwargs.get("session_id", "") or ""
    if not session_id:
        return

    active = _ensure_trace(session_id, "", "")
    if active is None:
        return

    tool_call_id = kwargs.get("tool_call_id", "") or ""
    if not tool_call_id:
        return

    try:
        span = active.trace.start_observation(
            name=kwargs.get("tool_name", "tool"),
            as_type="span",
            input=kwargs.get("args", {}),
            metadata={
                "tool_call_id": tool_call_id,
                "task_id": kwargs.get("task_id", ""),
            },
        )
        active.active_spans[tool_call_id] = span
    except Exception as exc:
        logger.warning("pre_tool_call 异常: %s", exc)


def _on_post_tool_call(**kwargs: Any) -> None:
    """End the tool Span with result and duration.

    Hermes kwargs (same as pre_tool_call +): result, duration_ms
    """
    session_id = kwargs.get("session_id", "") or ""
    active = _SESSIONS.get(session_id)
    if active is None:
        return

    tool_call_id = kwargs.get("tool_call_id", "") or ""
    span = active.active_spans.pop(tool_call_id, None)
    if span is None:
        return

    result = kwargs.get("result", "") or ""
    duration_ms = kwargs.get("duration_ms", 0)

    try:
        span.update(
            output={"result": _truncate(result, 5000)},
            metadata={
                "duration_ms": duration_ms,
                "result_length": len(result),
            },
        )
        span.end()
    except Exception as exc:
        logger.warning("post_tool_call 异常: %s", exc)


# -- Helpers ------------------------------------------------------------


def _truncate(s: str, max_len: int) -> str:
    if not s or len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def _flush_and_cleanup(session_id: str) -> None:
    """End dangling observations, flush, and remove session state."""
    active = _SESSIONS.get(session_id)
    if active is None:
        return

    dangling_gen = 0
    dangling_span = 0

    # End any dangling generations (those not ended by post_api_request)
    for _key, gen in list(active.active_generations.items()):
        try:
            gen.end()
            dangling_gen += 1
        except Exception:
            pass
    active.active_generations.clear()

    # End any dangling spans
    for _key, span in list(active.active_spans.items()):
        try:
            span.end()
            dangling_span += 1
        except Exception:
            pass
    active.active_spans.clear()

    # End the root trace
    if active.trace is not None:
        try:
            active.trace.end()
        except Exception:
            pass

    # 发送到 Langfuse 后端
    client = _init_client()
    if client:
        try:
            client.flush()
        except Exception as exc:
            logger.warning("Langfuse flush 失败: %s", exc)

    # Remove session state
    with _STATE_LOCK:
        _SESSIONS.pop(session_id, None)


# ======================================================================
# Plugin entry point
# ======================================================================


def register(ctx: Any) -> None:
    """Register Langfuse tracing hooks with the Hermes plugin system."""

    # 1. Plugin-level opt-in switch
    if not _is_langfuse_enabled():
        logger.debug(
            "langfuse-tracing: disabled (langfuse.enabled not set in config.yaml)"
        )
        return

    # 2. Ensure the langfuse package is available
    if _get_langfuse() is False:
        logger.warning(
            "langfuse-tracing: 'langfuse' package not installed. "
            "Install it with: pip install langfuse"
        )
        return

    # 3. Credentials must be present
    if not _get_env("LANGFUSE_PUBLIC_KEY") or not _get_env("LANGFUSE_SECRET_KEY"):
        logger.warning(
            "langfuse-tracing: LANGFUSE_PUBLIC_KEY and/or LANGFUSE_SECRET_KEY "
            "not configured. Plugin will not be active.\n"
            "  Set them in ~/.hermes/.env or as environment variables.\n"
            "  Get your keys at https://cloud.langfuse.com"
        )
        return

    # 4. Register hooks
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("on_session_finalize", _on_session_finalize)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("pre_api_request", _on_pre_api_request)
    ctx.register_hook("post_api_request", _on_post_api_request)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)

    host = (
        _get_env("LANGFUSE_HOST")
        or _get_env("LANGFUSE_BASE_URL")
        or "https://cloud.langfuse.com"
    )
    logger.info("Langfuse 插件注册完成: %s", host)

if __name__ == "__main__":
    jwt_token = "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjEyMyIsImNvbXBhbnlJZCI6IjEyMzEyMyJ9.xQfvluj44irrX3xM641CGbzw0TOcxeLIHoDKIhebCtSjzsRc8oCdTq_e3EMpbT_UtJuC_O91de-xgVQXkYvX7g"
    dict = _decode_jwt_payload(jwt_token)
    print(dict)