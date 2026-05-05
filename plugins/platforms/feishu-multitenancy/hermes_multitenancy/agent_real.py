"""Phase 3 — real (non-stub) agent runner.

Replaces ``runtime._default_run_agent`` (Phase-1 echo stub) with a thin
OpenAI-compatible LLM call that reads its config + credentials from the
profile_home directory. Designed to plug into ``ProfileRuntime`` via:

    ProfileRuntime(profile_home, run_agent_fn=real_run_agent)

Resolution order for an API key, given a model spec like ``"zai/glm-5.1"``:
  1. Environment variable ``<PROVIDER>_API_KEY`` (e.g. ``GLM_API_KEY``,
     ``ZAI_API_KEY``, ``OPENROUTER_API_KEY``) — sourced from the profile's
     ``.env`` file via ``python-dotenv``.
  2. ``auth.json`` ``credential_pool[provider]`` — first entry whose
     ``last_status`` is not ``"exhausted"``.

Fallback strategy: try the primary ``model.default``; on any error, walk the
``fallback`` list. Returns the first non-empty content string.

Spike scope: deliberate one-shot LLM call, no SessionStore, no tool-loop, no
streaming. Phase 4 will graduate to a real AIAgent loop (~1700 LOC per the
architect estimate). For end-to-end demo today, this thin runner is enough.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Map a model provider prefix to the env-var name that holds its API key.
# Keep this list short and explicit — adding a provider is one line.
_PROVIDER_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "zai": ("GLM_API_KEY", "ZAI_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "moonshot": ("MOONSHOT_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
}


# Each provider's API base URL when the model spec has no explicit override.
# Values mirror what ``hermes_cli/config.py`` infers for the same providers.
_PROVIDER_BASE_URLS: dict[str, str] = {
    "zai": "https://api.z.ai/api/coding/paas/v4",
    "openrouter": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    "deepseek": "https://api.deepseek.com",
}


async def stream_run_agent(  # type: ignore[override]
    event: Any,
    profile_home: Path,
    *,
    messages: Optional[list[dict]] = None,
):
    """Yields ``(kind, text)`` tuples — Phase 4 routes through the real AIAgent.

    With the AIAgent path enabled, the LLM gets the full toolset and can
    actually call PR-added UAT tools. The subprocess bridge emits NDJSON
    events so the parent can forward tool/reasoning/text deltas into the
    Feishu streaming card while the synchronous AIAgent loop is still running.

    Falls back to the legacy streaming path on AIAgent failure (preserves
    the visible-typing UX even when tools cannot fire). ``messages`` is the
    router-provided conversation history; the subprocess receives prior turns
    as ``conversation_history`` and the current event text as the active user
    message.
    """
    try:
        content_parts: list[str] = []
        final_text = ""
        stream = (
            _stream_aiagent_subprocess(event, profile_home, messages=messages)
            if messages is not None
            else _stream_aiagent_subprocess(event, profile_home)
        )
        async for kind, payload in stream:
            if kind == "done":
                final_text = str(payload or "")
                continue
            if kind == "content":
                text = str(payload or "")
                if text:
                    content_parts.append(text)
                    yield "content", text
                continue
            yield kind, payload
        if final_text and not "".join(content_parts).strip():
            yield "content", final_text
        if final_text or content_parts:
            return
    except Exception as exc:
        logger.warning(
            "[multitenancy] streaming AIAgent path failed (%s); falling back to legacy stream",
            exc, exc_info=True,
        )

    async for kind, text in _stream_loop(event, profile_home, messages=messages):
        yield kind, text


async def _stream_loop(
    event: Any,
    profile_home: Path,
    *,
    messages: Optional[list[dict]] = None,
):
    """Streaming counterpart to ``real_run_agent`` — yields content chunks.

    Used by the multitenancy router to stream LLM tokens into a Feishu
    ``edit_message`` loop, restoring the typewriter UX that hermes' main
    flow provides natively. Falls through provider candidates the same way
    as ``real_run_agent`` — first one whose first chunk is non-empty wins.

    Yields
    ------
    str
        Each non-empty content chunk from the live model.

    Raises
    ------
    RuntimeError
        If every candidate model+credential combination fails or yields
        nothing. Caller should fall back to ``real_run_agent`` for a final
        non-streamed attempt before giving up.
    """
    import yaml
    from openai import AsyncOpenAI
    from dotenv import dotenv_values

    config = _load_yaml(profile_home / "config.yaml")
    auth = _load_json(profile_home / "auth.json")
    env_overrides = (
        dotenv_values(profile_home / ".env") if (profile_home / ".env").exists() else {}
    )

    primary = config.get("model", {}).get("default")
    fallback_models = config.get("fallback") or []
    candidates: list[str] = [primary] if primary else []
    candidates.extend(fallback_models)

    soul_text = _load_soul(profile_home)
    user_text = getattr(event, "text", "") or ""

    # Caller can override the message list (used for multi-turn history).
    # Default: system prompt + single user message.
    if messages is None:
        effective_messages: list[dict] = [
            {"role": "system", "content": soul_text},
            {"role": "user", "content": user_text},
        ]
    else:
        # Caller supplies the conversation. We still inject SOUL as system
        # to guarantee the profile's persona stays in force.
        effective_messages = [
            {"role": "system", "content": soul_text},
            *messages,
        ]

    last_error: Optional[BaseException] = None

    for model_spec in candidates:
        if not model_spec:
            continue
        try:
            provider, model_name = _split_model_spec(model_spec)
        except ValueError:
            continue
        api_key = _resolve_api_key(provider, env_overrides, auth)
        if not api_key:
            continue
        base_url = _resolve_base_url(provider, model_spec == primary, config)

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        try:
            stream = await client.chat.completions.create(
                model=model_name,
                messages=effective_messages,
                max_tokens=512,
                stream=True,
            )
            got_content = False
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if not delta:
                    continue
                # Reasoning models (e.g. GLM 5.1) stream reasoning_content
                # BEFORE content; surfacing it gives the user real-time feedback
                # instead of a 5-15s placeholder freeze.
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield "thinking", reasoning
                if delta.content:
                    got_content = True
                    yield "content", delta.content
            if got_content:
                return
            logger.info("stream_run_agent: %s yielded no content, falling back", model_spec)
        except Exception as exc:
            last_error = exc
            logger.info("stream_run_agent: %s failed (%s), falling back", model_spec, exc)

    if last_error is not None:
        raise RuntimeError(f"streaming failed; last error: {last_error}") from last_error
    raise RuntimeError("streaming exhausted (no usable provider returned content)")


async def real_run_agent(
    event: Any,
    profile_home: Path,
    *,
    messages: Optional[list[dict]] = None,
) -> str:
    """Run the inbound event through hermes' real AIAgent (with tool-loop).

    Phase 4 — replaces the spike one-shot ``chat.completions.create`` call
    with a full ``AIAgent.run_conversation()`` loop, so PR-added UAT tools
    (e.g. feishu_calendar_list_events) actually fire. Sets
    ``sender_open_id_scope`` so per-user UAT files are loaded correctly.

    Falls back to the legacy thin LLM call (kept below as
    ``_legacy_real_run_agent``) on any AIAgent failure so the spike-style
    fallback path still answers the user — without tools, but at least with
    a coherent reply.
    """
    try:
        if messages is not None:
            return await _run_aiagent_subprocess(event, profile_home, messages=messages)
        return await _run_aiagent_subprocess(event, profile_home)
    except Exception as exc:
        logger.warning(
            "[multitenancy] AIAgent path failed (%s); falling back to legacy spike",
            exc, exc_info=True,
        )
    # Legacy / fallback path — no tool-loop, but still answers.
    return await _legacy_real_run_agent(event, profile_home, messages=messages)


async def _legacy_real_run_agent(
    event: Any,
    profile_home: Path,
    *,
    messages: Optional[list[dict]] = None,
) -> str:
    """Original spike implementation — kept as a fallback for the AIAgent path."""
    from openai import AsyncOpenAI
    from dotenv import dotenv_values

    config = _load_yaml(profile_home / "config.yaml")
    auth = _load_json(profile_home / "auth.json")
    env_overrides = dotenv_values(profile_home / ".env") if (profile_home / ".env").exists() else {}

    primary = config.get("model", {}).get("default")
    fallback_models = config.get("fallback") or []
    candidates: list[str] = [primary] if primary else []
    candidates.extend(fallback_models)

    soul_text = _load_soul(profile_home)
    user_text = getattr(event, "text", "") or ""

    if messages is None:
        effective_messages: list[dict] = [
            {"role": "system", "content": soul_text},
            {"role": "user", "content": user_text},
        ]
    else:
        effective_messages = [
            {"role": "system", "content": soul_text},
            *messages,
        ]

    last_error: Optional[BaseException] = None

    for model_spec in candidates:
        if not model_spec:
            continue
        try:
            provider, model_name = _split_model_spec(model_spec)
        except ValueError as exc:
            logger.debug("real_run_agent: bad model spec %r: %s", model_spec, exc)
            continue

        api_key = _resolve_api_key(provider, env_overrides, auth)
        if not api_key:
            logger.debug("real_run_agent: no API key for provider %s", provider)
            continue

        base_url = _resolve_base_url(provider, model_spec == primary, config)

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        try:
            rsp = await client.chat.completions.create(
                model=model_name,
                messages=effective_messages,
                max_tokens=512,
            )
            text = (rsp.choices[0].message.content or "").strip()
            if text:
                logger.debug(
                    "real_run_agent: %s ok (prompt=%d completion=%d)",
                    model_spec,
                    rsp.usage.prompt_tokens if rsp.usage else -1,
                    rsp.usage.completion_tokens if rsp.usage else -1,
                )
                return text
            # Empty content (often signals quota exhausted) — try next.
            logger.info("real_run_agent: %s returned empty, falling back", model_spec)
        except Exception as exc:
            last_error = exc
            logger.info("real_run_agent: %s failed (%s), falling back", model_spec, exc)

    if last_error is not None:
        raise RuntimeError(f"all providers failed; last error: {last_error}") from last_error
    raise RuntimeError("all providers exhausted (no usable key or non-empty response)")


# -- helpers ---------------------------------------------------------------


def _split_model_spec(spec: str) -> tuple[str, str]:
    """Split ``provider/model_name`` into its parts."""
    if "/" not in spec:
        raise ValueError(f"model spec missing provider prefix: {spec!r}")
    provider, name = spec.split("/", 1)
    return provider.strip().lower(), name.strip()


def _resolve_api_key(
    provider: str,
    env_overrides: dict[str, Any],
    auth: dict[str, Any],
) -> Optional[str]:
    """Find an API key for *provider* — env vars first, auth.json second."""
    for env_name in _PROVIDER_ENV_KEYS.get(provider, ()):
        key = env_overrides.get(env_name) or os.environ.get(env_name)
        if key:
            return key
    pool = auth.get("credential_pool", {}).get(provider)
    if isinstance(pool, list):
        for cred in pool:
            if not isinstance(cred, dict):
                continue
            if cred.get("last_status") == "exhausted":
                continue
            token = cred.get("access_token")
            if token:
                return token
    return None


def _resolve_base_url(provider: str, is_primary: bool, config: dict[str, Any]) -> Optional[str]:
    """Resolve the API base URL for *provider*. Primary model honors config.model.base_url."""
    if is_primary:
        explicit = config.get("model", {}).get("base_url")
        if explicit:
            return explicit
    return _PROVIDER_BASE_URLS.get(provider)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_soul(profile_home: Path) -> str:
    """Read SOUL.md as the system prompt; fall back to a generic prompt."""
    soul = profile_home / "SOUL.md"
    if soul.exists():
        text = soul.read_text().strip()
        if text:
            return text
    return "You are a helpful assistant."


def _resolve_shared_hermes_home(profile_home: Path) -> Path:
    """Return the default Hermes root that stores cross-profile shared auth."""
    explicit = os.getenv("HERMES_SHARED_HOME")
    if explicit:
        return Path(explicit).expanduser()
    profile_home = Path(profile_home).expanduser()
    if profile_home.parent.name == "profiles":
        return profile_home.parent.parent
    return profile_home


def _configure_feishu_uat_home(feishu_oapi_module: Any, profile_home: Path) -> Path:
    """Point Feishu UAT lookups at the shared Hermes root, not the profile dir."""
    shared_home = _resolve_shared_hermes_home(profile_home)
    feishu_oapi_module.FEISHU_UAT_PATH = shared_home / "feishu_uat.json"
    feishu_oapi_module.FEISHU_UAT_DIR = shared_home / "feishu_uat"
    return shared_home


def _log_aiagent_tool_progress(
    event_type: str,
    tool_name: str,
    preview: Any = None,
    args: Any = None,
    **kwargs: Any,
) -> None:
    """Persist AIAgent tool progress for gateway stress-test observability."""
    if event_type == "tool.started":
        logger.info("[multitenancy] tool.started %s preview=%s", tool_name, preview or "")
    elif event_type == "tool.completed":
        logger.info(
            "[multitenancy] tool.completed %s duration=%.2fs error=%s",
            tool_name,
            float(kwargs.get("duration") or 0.0),
            bool(kwargs.get("is_error")),
        )


# ---------------------------------------------------------------------------
# Isolated AIAgent subprocess bridge
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Return a JSON-safe representation for dataclass/enum-ish event fields."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    return str(value)


def _jsonable_deep(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable_deep(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_deep(v) for v in value]
    return _jsonable(value)


def _get_nested_value(obj: Any, path: tuple[str, ...]) -> Any:
    cur = obj
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
        if cur is None:
            return None
    return cur


def _find_ou_value(obj: Any) -> str:
    """Best-effort recursive search for a Feishu open_id in raw event data."""
    if isinstance(obj, str):
        return obj if obj.startswith("ou_") else ""
    if isinstance(obj, dict):
        for key in ("open_id", "openId"):
            value = obj.get(key)
            if isinstance(value, str) and value.startswith("ou_"):
                return value
        for value in obj.values():
            found = _find_ou_value(value)
            if found:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _find_ou_value(value)
            if found:
                return found
    return ""


def _resolve_subprocess_sender_open_id(event: Any) -> str:
    """Resolve sender ou_* for the child process after Feishu batching."""
    try:
        from tools.feishu_oapi_client import current_sender_open_id
        current = current_sender_open_id.get()
        if current and str(current).startswith("ou_"):
            return str(current)
    except Exception:
        pass

    source = getattr(event, "source", None)
    for candidate in (
        getattr(event, "sender_open_id", None),
        getattr(source, "open_id", None) if source is not None else None,
        getattr(source, "user_id", None) if source is not None else None,
        getattr(source, "user_id_alt", None) if source is not None else None,
    ):
        if candidate and str(candidate).startswith("ou_"):
            return str(candidate)

    raw = getattr(event, "raw_message", None)
    for path in (
        ("event", "sender", "sender_id", "open_id"),
        ("event", "message", "sender", "sender_id", "open_id"),
        ("sender", "sender_id", "open_id"),
        ("message", "sender", "sender_id", "open_id"),
        ("sender_id", "open_id"),
    ):
        value = _get_nested_value(raw, path)
        if value and str(value).startswith("ou_"):
            return str(value)
    return _find_ou_value(raw)


def _jsonable_messages(messages: Optional[list[dict]]) -> list[dict] | None:
    if not messages:
        return None
    payload_messages: list[dict] = []
    for message in messages:
        if isinstance(message, dict):
            jsonable = _jsonable_deep(message)
            if isinstance(jsonable, dict):
                payload_messages.append(jsonable)
    return payload_messages


def _event_to_subprocess_payload(
    event: Any,
    profile_home: Path,
    *,
    messages: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """Serialize the small MessageEvent surface needed by the child runner."""
    source = getattr(event, "source", None)
    source_payload: dict[str, Any] = {}
    if source is not None:
        for key in (
            "platform",
            "chat_id",
            "chat_name",
            "chat_type",
            "user_id",
            "user_name",
            "thread_id",
            "chat_topic",
            "user_id_alt",
            "chat_id_alt",
            "is_bot",
            "guild_id",
            "parent_chat_id",
            "message_id",
        ):
            if hasattr(source, key):
                source_payload[key] = _jsonable(getattr(source, key))

    message_id = (
        getattr(event, "message_id", None)
        or source_payload.get("message_id")
        or ""
    )
    payload = {
        "event": {
            "text": getattr(event, "text", "") or "",
            "message_id": _jsonable(message_id),
            "sender_open_id": _resolve_subprocess_sender_open_id(event),
            "source": source_payload,
        },
        "profile_home": str(profile_home),
    }
    payload_messages = _jsonable_messages(messages)
    if payload_messages is not None:
        payload["messages"] = payload_messages
    return payload


async def _run_aiagent_subprocess(
    event: Any,
    profile_home: Path,
    *,
    messages: Optional[list[dict]] = None,
) -> str:
    """Run the sync AIAgent body in a fresh Python process.

    The gateway stays fully async while the child process owns the synchronous
    AIAgent/tool loop. This avoids the gateway event-loop deadlock observed
    when ``_run_with_aiagent`` runs through ``asyncio.to_thread``.
    """
    import asyncio

    payload = json.dumps(
        _event_to_subprocess_payload(event, profile_home, messages=messages),
        ensure_ascii=False,
    ).encode("utf-8")
    timeout_s = float(os.getenv("HERMES_AIAGENT_SUBPROCESS_TIMEOUT", "300"))
    env = os.environ.copy()
    env["HERMES_SHARED_HOME"] = str(_resolve_shared_hermes_home(profile_home))
    env["HERMES_HOME"] = str(profile_home)
    child_script = Path(__file__).with_name("aiagent_subprocess.py")

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(child_script),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(payload), timeout_s)
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"AIAgent subprocess timed out after {timeout_s:g}s") from exc
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise

    stderr_text = stderr.decode("utf-8", errors="replace").strip()
    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    if stderr_text:
        logger.debug("[multitenancy] AIAgent subprocess stderr: %s", stderr_text[-4000:])

    try:
        data = json.loads(stdout_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "AIAgent subprocess returned invalid JSON "
            f"(exit={proc.returncode}, stdout={stdout_text[-1000:]!r}, stderr={stderr_text[-1000:]!r})"
        ) from exc

    if proc.returncode != 0:
        raise RuntimeError(
            f"AIAgent subprocess exited {proc.returncode}: "
            f"{data.get('error') or stderr_text or stdout_text}"
        )
    if data.get("error"):
        raise RuntimeError(f"AIAgent subprocess failed: {data['error']}")
    return str(data.get("result") or "")


async def _stream_aiagent_subprocess(
    event: Any,
    profile_home: Path,
    *,
    messages: Optional[list[dict]] = None,
):
    """Run AIAgent in a child process and yield its NDJSON progress events."""
    import asyncio

    payload = json.dumps(
        _event_to_subprocess_payload(event, profile_home, messages=messages),
        ensure_ascii=False,
    ).encode("utf-8")
    timeout_s = float(os.getenv("HERMES_AIAGENT_SUBPROCESS_TIMEOUT", "300"))
    env = os.environ.copy()
    env["HERMES_SHARED_HOME"] = str(_resolve_shared_hermes_home(profile_home))
    env["HERMES_HOME"] = str(profile_home)
    env["HERMES_AIAGENT_EVENT_STREAM"] = "1"
    child_script = Path(__file__).with_name("aiagent_subprocess.py")

    started_at = time.monotonic()
    logger.info(
        "[multitenancy] AIAgent subprocess spawning profile_home=%s timeout=%.1fs",
        profile_home,
        timeout_s,
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(child_script),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    logger.info(
        "[multitenancy] AIAgent subprocess spawned pid=%s elapsed=%.3fs",
        proc.pid,
        time.monotonic() - started_at,
    )
    stderr_task = asyncio.create_task(proc.stderr.read())
    saw_done = False
    first_event_logged = False
    try:
        assert proc.stdin is not None
        proc.stdin.write(payload)
        await proc.stdin.drain()
        proc.stdin.close()
        try:
            await proc.stdin.wait_closed()
        except Exception:
            pass

        assert proc.stdout is not None
        while True:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout_s)
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                logger.debug("[multitenancy] ignoring non-json child stream line: %r", text[-500:])
                continue
            event_name = data.get("event")
            if not first_event_logged:
                first_event_logged = True
                logger.info(
                    "[multitenancy] AIAgent subprocess first event kind=%s elapsed=%.3fs",
                    event_name,
                    time.monotonic() - started_at,
                )
            if event_name == "done":
                saw_done = True
                if data.get("error"):
                    raise RuntimeError(f"AIAgent subprocess failed: {data['error']}")
                logger.info(
                    "[multitenancy] AIAgent subprocess done elapsed=%.3fs result_len=%s",
                    time.monotonic() - started_at,
                    len(str(data.get("result") or "")),
                )
                yield "done", str(data.get("result") or "")
                continue
            if event_name == "content":
                yield "content", str(data.get("text") or "")
            elif event_name == "thinking":
                yield "thinking", str(data.get("text") or "")
            elif event_name in {"tool_started", "tool_completed"}:
                payload_data = {k: v for k, v in data.items() if k != "event"}
                yield str(event_name), payload_data
            else:
                logger.debug("[multitenancy] ignoring unknown child stream event: %s", event_name)

        returncode = await asyncio.wait_for(proc.wait(), timeout=5)
        stderr_text = (await stderr_task).decode("utf-8", errors="replace").strip()
        if stderr_text:
            logger.debug("[multitenancy] AIAgent subprocess stderr: %s", stderr_text[-4000:])
        logger.info(
            "[multitenancy] AIAgent subprocess exited returncode=%s elapsed=%.3fs",
            returncode,
            time.monotonic() - started_at,
        )
        if returncode != 0:
            raise RuntimeError(f"AIAgent subprocess exited {returncode}: {stderr_text[-1000:]}")
        if not saw_done:
            raise RuntimeError("AIAgent subprocess stream ended without done event")
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
    except Exception:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        raise
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        if not stderr_task.done():
            stderr_task.cancel()


# ---------------------------------------------------------------------------
# Phase 4 — real AIAgent runner with tool-loop (replaces the spike one-shot)
# ---------------------------------------------------------------------------


def _resolve_sender_open_id(event: Any) -> str:
    """Pick the real Feishu open_id (ou_*) from the event source for UAT lookup.

    Prefer source.user_id (typical Feishu open_id); fall back to user_id_alt
    when the former is a union_id (on_*). Returns "" if nothing usable is found.
    """
    source = getattr(event, "source", None)
    event_sender = getattr(event, "sender_open_id", None)
    if event_sender and str(event_sender).startswith("ou_"):
        return str(event_sender)
    if source is None:
        return ""
    for candidate in (
        getattr(source, "open_id", None),
        getattr(source, "user_id", None),
        getattr(source, "user_id_alt", None),
    ):
        if candidate and str(candidate).startswith("ou_"):
            return str(candidate)
    # Fallback: any non-empty user_id, even if not ou_-prefixed
    return str(getattr(source, "user_id", "") or "")


def _session_part(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip() or default
    safe = "".join(ch if (ch.isalnum() or ch in "._:-") else "_" for ch in text)
    return safe[:160] or default


def _resolve_platform_value(source: Any, default: str = "feishu") -> str:
    if source is None:
        return default
    platform = getattr(source, "platform", None)
    return str(getattr(platform, "value", None) or platform or default)


def _resolve_aiagent_session_id(
    event: Any,
    profile_home: Path,
    sender_open_id: str = "",
) -> str:
    """Build a stable, per-profile/per-user AIAgent session id.

    Feishu ``message_id`` changes every turn, so it must only be a last-ditch
    fallback. Keeping profile and sender in the key prevents cross-tenant or
    cross-user history bleed when multiple Feishu users hit the same bot.
    """
    source = getattr(event, "source", None)
    platform = _resolve_platform_value(source)
    chat_type = getattr(source, "chat_type", "") if source else ""
    chat_id = (
        getattr(source, "chat_id", None)
        or getattr(source, "parent_chat_id", None)
        or getattr(source, "chat_id_alt", None)
        if source
        else ""
    )
    thread_id = (
        getattr(source, "thread_id", None)
        or getattr(source, "chat_topic", None)
        if source
        else ""
    )
    user_id = (
        sender_open_id
        or (getattr(source, "user_id", None) if source else "")
        or (getattr(source, "user_id_alt", None) if source else "")
    )
    message_id = (
        getattr(event, "message_id", None)
        or (getattr(source, "message_id", None) if source else "")
    )

    parts = [
        "agent",
        "profile",
        profile_home.name,
        "platform",
        platform,
        "chat_type",
        chat_type or "unknown",
    ]
    if chat_id:
        parts.extend(["chat", chat_id])
    if thread_id:
        parts.extend(["thread", thread_id])
    if user_id:
        parts.extend(["user", user_id])
    elif message_id:
        parts.extend(["message", message_id])
    else:
        parts.append("fallback")

    session_id = ":".join(_session_part(part) for part in parts)
    if len(session_id) <= 220:
        return session_id
    digest = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:12]
    return f"{session_id[:200]}:{digest}"


def _conversation_history_for_aiagent(
    messages: Optional[list[dict]],
    user_text: str,
) -> Optional[list[dict]]:
    if not messages:
        return None
    history = [dict(message) for message in messages if isinstance(message, dict)]
    if (
        history
        and history[-1].get("role") == "user"
        and str(history[-1].get("content") or "") == user_text
    ):
        history = history[:-1]
    return history or None


def _run_with_aiagent(
    event: Any,
    profile_home: Path,
    *,
    messages: Optional[list[dict]] = None,
    event_sink=None,
) -> str:
    """Synchronous body — runs hermes' real AIAgent against the profile config.

    Constructs an AIAgent with the profile's enabled toolsets + LLM
    credentials, sets the per-user open_id contextvar so UAT tools load the
    right token file, and runs a full tool-loop conversation.

    Designed to run inside ``aiagent_subprocess.py`` so the parent gateway
    keeps its async event loop isolated from the synchronous AIAgent/tool loop.
    """
    # 1) Anchor HERMES_HOME so any module that reads it sees the profile.
    os.environ["HERMES_HOME"] = str(profile_home)

    # 2) Read profile LLM config + credentials (mirrors the spike loader).
    config = _load_yaml(profile_home / "config.yaml")
    auth = _load_json(profile_home / "auth.json")
    from dotenv import dotenv_values
    env_overrides = dict(
        dotenv_values(profile_home / ".env")
        if (profile_home / ".env").exists()
        else {}
    )

    primary = (config.get("model") or {}).get("default")
    if not primary:
        raise RuntimeError("profile config missing model.default")
    fallback_models = config.get("fallback") or []

    provider, model_only = _split_model_spec(primary)
    api_key = _resolve_api_key(provider, env_overrides, auth)
    if not api_key:
        raise RuntimeError(f"no API key for primary provider {provider!r}")

    base_url = _resolve_base_url(provider, True, config)

    # 3) Lazy-import hermes core (only when this code path is hit).
    from run_agent import AIAgent
    from tools import feishu_oapi_client as feishu_oapi
    sender_open_id_scope = feishu_oapi.sender_open_id_scope
    current_sender_open_id = feishu_oapi.current_sender_open_id
    shared_hermes_home = _configure_feishu_uat_home(feishu_oapi, profile_home)
    if shared_hermes_home != profile_home:
        logger.info(
            "[multitenancy] using shared Feishu UAT dir: %s",
            shared_hermes_home / "feishu_uat",
        )
    try:
        from hermes_cli.tools_config import _get_platform_tools
    except Exception:
        _get_platform_tools = None  # graceful: fall back to None toolsets

    # 4) Resolve toolsets enabled for this platform on this profile.
    platform_key = "feishu"
    enabled_toolsets = _resolve_enabled_toolsets(
        config,
        platform_key,
        platform_tools_resolver=_get_platform_tools,
    )

    # 5) Sender's real Feishu open_id (ou_*) for per-user UAT routing.
    # The feishu adapter already set this contextvar in
    # _process_inbound_message before dispatching us — prefer that value
    # because it comes straight from sender_id.open_id (the SDK gives the
    # ou_* form). Only fall back to event.source on weird code paths
    # (e.g., synthetic events constructed without going through the adapter).
    sender_open_id = (current_sender_open_id.get() or "") or _resolve_sender_open_id(event)

    # 6) Pull source / session metadata for AIAgent kwargs.
    source = getattr(event, "source", None)
    user_text = getattr(event, "text", "") or ""
    session_id = _resolve_aiagent_session_id(event, profile_home, sender_open_id)
    conversation_history = _conversation_history_for_aiagent(messages, user_text)

    runtime_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        runtime_kwargs["base_url"] = base_url

    fallback_model = fallback_models[0] if fallback_models else None

    # 7) Wrap the agent run in sender_open_id_scope so per-user UAT tools
    #    pick up the right token from ~/.hermes/feishu_uat/<open_id>.json.
    logger.info(
        "[multitenancy] running AIAgent for sender=%s profile=%s toolsets=%s",
        sender_open_id, profile_home.name,
        enabled_toolsets if enabled_toolsets is not None else "<default>",
    )
    logger.info(
        "[multitenancy] Feishu UAT lookup sender=%s dir=%s",
        sender_open_id,
        shared_hermes_home / "feishu_uat",
    )
    with sender_open_id_scope(sender_open_id or None):
        try:
            from gateway.session_context import clear_session_vars, set_session_vars
        except Exception:
            clear_session_vars = None
            set_session_vars = None

        platform_value = str(
            getattr(getattr(source, "platform", ""), "value", None)
            or getattr(source, "platform", "")
            or platform_key
        )
        session_tokens = None
        if set_session_vars is not None:
            session_tokens = set_session_vars(
                platform=platform_value,
                chat_id=str(getattr(source, "chat_id", "") or "") if source else "",
                chat_name=str(getattr(source, "chat_name", "") or "") if source else "",
                thread_id=str(getattr(source, "thread_id", "") or "") if source else "",
                user_id=str(getattr(source, "user_id", "") or "") if source else "",
                user_name=str(getattr(source, "user_name", "") or "") if source else "",
                session_key=str(session_id),
            )
        def _emit(event_name: str, **payload: Any) -> None:
            if event_sink is None:
                return
            try:
                event_sink(event_name, **payload)
            except Exception:
                logger.debug("[multitenancy] event_sink failed", exc_info=True)

        def _tool_progress_event_callback(
            event_type: str,
            tool_name: str,
            preview: Any = None,
            args: Any = None,
            **kwargs: Any,
        ) -> None:
            _log_aiagent_tool_progress(event_type, tool_name, preview, args, **kwargs)
            if event_type == "tool.started":
                _emit(
                    "tool_started",
                    name=str(tool_name or ""),
                    preview=str(preview or "") if preview is not None else None,
                )
            elif event_type == "tool.completed":
                _emit(
                    "tool_completed",
                    name=str(tool_name or ""),
                    duration=float(kwargs.get("duration") or 0.0),
                    is_error=bool(kwargs.get("is_error")),
                )
            elif event_type in {"reasoning.available", "_thinking"}:
                text = str(preview or tool_name or "")
                if text:
                    _emit("thinking", text=text)

        def _stream_delta_event_callback(text: Any) -> None:
            if text is None:
                return
            text = str(text)
            if text:
                _emit("content", text=text)

        def _reasoning_event_callback(text: Any) -> None:
            if text is None:
                return
            text = str(text)
            if text:
                _emit("thinking", text=text)

        def _tool_gen_event_callback(tool_name: str) -> None:
            if tool_name:
                _emit("tool_started", name=str(tool_name), preview="generating arguments")

        agent_kwargs: dict[str, Any] = {
            # AIAgent expects the bare model name (e.g. "glm-5.1"), not the
            # provider-prefixed form. Provider was already used above to
            # resolve api_key + base_url; the prefix would otherwise be
            # forwarded verbatim to the OpenAI client and rejected with
            # `1211 Unknown Model`.
            "model": model_only,
            **runtime_kwargs,
            "max_iterations": int(os.getenv("HERMES_MAX_ITERATIONS", "30")),
            "quiet_mode": True,
            "verbose_logging": False,
            "session_id": str(session_id),
            "platform": platform_key,
            "user_id": str(getattr(source, "user_id", "") or "") if source else "",
            "user_name": str(getattr(source, "user_name", "") or "") if source else "",
            "chat_id": str(getattr(source, "chat_id", "") or "") if source else "",
            "chat_name": str(getattr(source, "chat_name", "") or "") if source else "",
            "chat_type": str(getattr(source, "chat_type", "") or "") if source else "",
            "tool_progress_callback": _tool_progress_event_callback,
            "stream_delta_callback": _stream_delta_event_callback if event_sink is not None else None,
            "reasoning_callback": _reasoning_event_callback if event_sink is not None else None,
            "tool_gen_callback": _tool_gen_event_callback if event_sink is not None else None,
        }
        if enabled_toolsets is not None:
            agent_kwargs["enabled_toolsets"] = enabled_toolsets
        if fallback_model:
            agent_kwargs["fallback_model"] = fallback_model

        agent = AIAgent(**agent_kwargs)
        try:
            run_kwargs: dict[str, Any] = {
                "user_message": user_text,
                "task_id": str(session_id),
            }
            if conversation_history is not None:
                run_kwargs["conversation_history"] = conversation_history
            result = agent.run_conversation(**run_kwargs)
        finally:
            if clear_session_vars is not None and session_tokens is not None:
                clear_session_vars(session_tokens)
            try:
                if hasattr(agent, "cleanup"):
                    agent.cleanup()
            except Exception:
                pass

    return (result or {}).get("final_response", "") or ""


def _resolve_enabled_toolsets(
    config: dict[str, Any],
    platform_key: str,
    *,
    platform_tools_resolver: Any,
) -> Optional[list[str]]:
    """Resolve profile toolsets without dropping core non-Feishu abilities.

    A plain Hermes Feishu gateway defaults to the composite ``hermes-feishu``
    toolset, which includes web/search/browser/file/etc. During multitenant
    UAT it is common to add ``platform_toolsets.feishu`` only for Feishu user
    token helpers; treating that list as a hard replacement makes the agent
    look competent inside Feishu but unable to search the web.

    Default mode therefore merges explicit profile entries with the platform
    default. Set ``multitenancy.toolsets_mode: explicit`` or
    ``HERMES_MULTITENANCY_TOOLSETS_MODE=explicit`` to preserve the old strict
    replacement behavior for providers that need a smaller schema.
    """
    explicit = (config.get("platform_toolsets") or {}).get(platform_key)
    explicit_toolsets = _normalize_toolset_list(explicit)
    mode = _toolsets_mode(config)

    if explicit_toolsets and mode in {"explicit", "strict", "replace"}:
        logger.info(
            "[multitenancy] platform_toolsets explicit mode for %s: %s",
            platform_key, explicit_toolsets,
        )
        return explicit_toolsets

    default_toolsets: list[str] = []
    if platform_tools_resolver is not None:
        resolver_config = config
        if explicit_toolsets:
            import copy

            resolver_config = copy.deepcopy(config)
            platform_toolsets = resolver_config.get("platform_toolsets")
            if isinstance(platform_toolsets, dict):
                platform_toolsets.pop(platform_key, None)
        try:
            try:
                resolved = platform_tools_resolver(
                    resolver_config,
                    platform_key,
                    include_default_mcp_servers=("no_mcp" not in explicit_toolsets),
                )
            except TypeError:
                resolved = platform_tools_resolver(resolver_config, platform_key)
            default_toolsets = _normalize_toolset_list(resolved)
        except Exception as exc:
            logger.warning(
                "[multitenancy] _get_platform_tools failed for %s: %s",
                platform_key, exc,
            )

    if explicit_toolsets:
        merged = sorted(set(default_toolsets) | set(explicit_toolsets))
        logger.info(
            "[multitenancy] platform_toolsets merged for %s: explicit=%s default=%s merged=%s",
            platform_key, explicit_toolsets, default_toolsets, merged,
        )
        return merged

    return default_toolsets or None


def _normalize_toolset_list(value: Any) -> list[str]:
    """Return a sorted list of non-empty string toolset names."""
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, (set, tuple)):
        items = list(value)
    else:
        return []
    return sorted({str(item).strip() for item in items if str(item).strip()})


def _toolsets_mode(config: dict[str, Any]) -> str:
    """Return multitenancy toolset resolution mode."""
    env_mode = os.getenv("HERMES_MULTITENANCY_TOOLSETS_MODE")
    if env_mode:
        return env_mode.strip().lower()
    plugin_cfg = config.get("multitenancy") or {}
    if isinstance(plugin_cfg, dict):
        mode = plugin_cfg.get("toolsets_mode")
        if mode:
            return str(mode).strip().lower()
    return "merge_default"
