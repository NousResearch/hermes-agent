"""Shared PydanticAI runtime helpers for investment assistant agents."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchSettings:
    web_enabled: bool = True
    max_searches: int = 5
    max_fetches: int = 8
    require_sources_for_must_consider: bool = True
    thinking_effort: str = "high"
    web_search_mode: Literal["auto", "local", "native", "off"] = "auto"
    web_search_context_size: Literal["low", "medium", "high"] | None = "low"
    web_fetch_mode: Literal["auto", "local", "native", "off"] = "local"
    web_fetch_timeout: int = 30
    web_fetch_retries: int = 3
    web_fetch_max_content_length: int = 50_000
    request_timeout: float = 420.0
    trace_enabled: bool = False
    trace_max_chars: int = 600


def create_pydantic_agent(
    *,
    output_type: type[Any],
    instructions: str,
    agent_kind: str,
    output_retries: int = 2,
    enable_web_search: bool = False,
    enable_web_fetch: bool = False,
    agent_skill_names: list[str] | None = None,
    research_overrides: dict[str, Any] | None = None,
):
    """Create a PydanticAI Agent with lazy imports and optional research capabilities."""

    version = ensure_pydantic_ai_available()

    from pydantic_ai import Agent

    settings = research_settings()
    if research_overrides:
        settings = _with_research_overrides(settings, research_overrides)
    model_config = load_model_config()
    thinking_enabled = bool(settings.thinking_effort and settings.thinking_effort.lower() != "none")
    prefer_openai_responses = enable_web_search or enable_web_fetch or thinking_enabled
    model, model_runtime = _create_model(
        model_config,
        prefer_openai_responses=prefer_openai_responses,
    )
    capabilities, capabilities_status = _research_capabilities(
        enable_web_search=enable_web_search,
        enable_web_fetch=enable_web_fetch,
        enable_provider_web_tools=model_runtime["api_mode"] == "openai_responses",
        settings=settings,
    )
    agent_skills_status: dict[str, Any] | None = None
    if agent_skill_names:
        from .skill_runtime import create_agent_skills_capability, pydantic_ai_skills_status

        capabilities.append(create_agent_skills_capability(agent_skill_names))
        agent_skills_status = {
            **pydantic_ai_skills_status(),
            "requested_skills": list(agent_skill_names),
        }
    kwargs: dict[str, Any] = {
        "output_type": output_type,
        "instructions": instructions,
        "output_retries": output_retries,
        "model_settings": _agent_model_settings(settings),
    }
    if capabilities:
        kwargs["capabilities"] = capabilities

    try:
        agent = Agent(model, **kwargs)
    except TypeError as exc:
        if "capabilities" not in kwargs:
            raise
        capabilities_status = {
            **capabilities_status,
            "enabled": False,
            "error": f"Agent constructor rejected capabilities: {exc}",
        }
        kwargs.pop("capabilities", None)
        agent = Agent(model, **kwargs)

    runtime = {
        "available": True,
        "mode": f"pydantic_ai_{agent_kind}",
        "package_version": version,
        "model": model_runtime["model"],
        "configured_model": model_config["model"],
        "api_mode": model_runtime["api_mode"],
        "base_url": model_config["base_url"],
        "capabilities": capabilities_status,
        "agent_skills": agent_skills_status,
        "model_settings": kwargs["model_settings"],
    }
    return agent, model_config, runtime


def pydantic_ai_status() -> dict[str, object]:
    """Return dependency/config status without claiming fallback capability."""

    try:
        version = ensure_pydantic_ai_available()
    except Exception as exc:
        return {
            "available": False,
            "mode": "pydantic_ai_unavailable",
            "reason": str(exc),
        }

    status = {
        "available": True,
        "mode": "pydantic_ai_agent",
        "package_version": version,
        "reason": "PydanticAI is configured as the only portfolio-map generation path.",
    }
    try:
        model_config = load_model_config()
    except Exception as exc:
        status["config_error"] = str(exc)
        return status
    status["model"] = model_config["model"]
    status["base_url"] = model_config["base_url"]
    return status


def ensure_pydantic_ai_available() -> str:
    if importlib.util.find_spec("pydantic_ai") is None:
        try:
            from tools.lazy_deps import ensure

            ensure("investment.pydantic_ai", prompt=False)
        except Exception as exc:
            raise RuntimeError(
                "pydantic_ai is required for portfolio-map generation and "
                "deterministic fallback is disabled. Install pydantic-ai or "
                "enable Hermes lazy installs."
            ) from exc

    if importlib.util.find_spec("pydantic_ai") is None:
        raise RuntimeError(
            "pydantic_ai is required for portfolio-map generation and deterministic fallback is disabled."
        )
    for dist_name in ("pydantic-ai", "pydantic-ai-slim"):
        try:
            return importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def load_model_config() -> dict[str, str]:
    config = _read_hermes_config()
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    model = (
        os.getenv("INVESTMENT_ASSISTANT_MODEL")
        or str(model_cfg.get("default") or "").strip()
        or os.getenv("OPENAI_MODEL")
        or ""
    )
    base_url = (
        os.getenv("INVESTMENT_ASSISTANT_BASE_URL")
        or str(model_cfg.get("base_url") or "").strip()
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    api_key = (
        os.getenv("INVESTMENT_ASSISTANT_API_KEY")
        or str(model_cfg.get("api_key") or "").strip()
        or os.getenv("OPENAI_API_KEY")
        or _read_hermes_env_key("OPENAI_API_KEY")
    )
    if not model:
        raise RuntimeError(
            "No model is configured for the PydanticAI investment assistant. "
            "Set model.default in ~/.hermes/config.yaml or INVESTMENT_ASSISTANT_MODEL."
        )
    if not api_key:
        raise RuntimeError(
            "No API key is configured for the PydanticAI investment assistant. "
            "Set OPENAI_API_KEY or INVESTMENT_ASSISTANT_API_KEY."
        )
    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }


def research_settings() -> ResearchSettings:
    config = _read_hermes_config()
    model_settings = _investment_assistant_model_settings(config)
    return ResearchSettings(
        web_enabled=_setting_bool(model_settings, "web_enabled", "IA_RESEARCH_WEB_ENABLED", True),
        max_searches=max(0, _setting_int(model_settings, "max_searches", "IA_RESEARCH_MAX_SEARCHES", 5)),
        max_fetches=max(0, _setting_int(model_settings, "max_fetches", "IA_RESEARCH_MAX_FETCHES", 8)),
        require_sources_for_must_consider=_setting_bool(
            model_settings,
            "require_sources_for_must_consider",
            "IA_RESEARCH_REQUIRE_SOURCES_FOR_MUST_CONSIDER",
            True,
        ),
        thinking_effort=_setting_str(
            model_settings,
            "thinking_effort",
            "IA_RESEARCH_THINKING_EFFORT",
            "high",
        ),
        web_search_mode=_tool_mode(
            _setting_str(model_settings, "web_search_mode", "IA_RESEARCH_WEB_SEARCH_MODE", "auto"),
            default="auto",
        ),
        web_search_context_size=_search_context_size(
            _setting_str(
                model_settings,
                "web_search_context_size",
                "IA_RESEARCH_WEB_SEARCH_CONTEXT_SIZE",
                "low",
            )
        ),
        web_fetch_mode=_tool_mode(
            _setting_str(model_settings, "web_fetch_mode", "IA_RESEARCH_WEB_FETCH_MODE", "local"),
            default="local",
        ),
        web_fetch_timeout=max(
            1,
            _setting_int(
                model_settings,
                "web_fetch_timeout",
                "IA_RESEARCH_WEB_FETCH_TIMEOUT",
                30,
            ),
        ),
        web_fetch_retries=max(
            0,
            _setting_int(
                model_settings,
                "web_fetch_retries",
                "IA_RESEARCH_WEB_FETCH_RETRIES",
                3,
            ),
        ),
        web_fetch_max_content_length=max(
            1_000,
            _setting_int(
                model_settings,
                "web_fetch_max_content_length",
                "IA_RESEARCH_WEB_FETCH_MAX_CONTENT_LENGTH",
                50_000,
            ),
        ),
        request_timeout=max(
            1.0,
            _setting_float(
                model_settings,
                "request_timeout",
                "IA_RESEARCH_REQUEST_TIMEOUT",
                420.0,
            ),
        ),
        trace_enabled=_setting_bool(model_settings, "trace_enabled", "IA_PYDANTIC_TRACE", False),
        trace_max_chars=max(
            80,
            _setting_int(model_settings, "trace_max_chars", "IA_PYDANTIC_TRACE_MAX_CHARS", 600),
        ),
    )


def usage_metadata(result: Any) -> dict[str, Any]:
    usage = getattr(result, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    if hasattr(usage, "__dict__"):
        return dict(usage.__dict__)
    return {"repr": repr(usage)}


def pydantic_event_stream_handler(agent_kind: str):
    """Return an event stream logger for PydanticAI internals when trace is enabled."""

    settings = research_settings()
    if not settings.trace_enabled:
        return None

    request_count = 0

    async def handler(ctx: Any, events: Any) -> None:
        nonlocal request_count
        request_count += 1
        request_no = request_count
        LOGGER.info(
            "IA_PYDANTIC_TRACE agent=%s request=%s start run_id=%s retry=%s max_retries=%s",
            agent_kind,
            request_no,
            getattr(ctx, "run_id", ""),
            getattr(ctx, "retry", ""),
            getattr(ctx, "max_retries", ""),
        )
        try:
            async for event in events:
                summary = _summarize_pydantic_event(event, settings.trace_max_chars)
                if summary is None:
                    continue
                LOGGER.info(
                    "IA_PYDANTIC_TRACE agent=%s request=%s event=%s %s",
                    agent_kind,
                    request_no,
                    getattr(event, "event_kind", type(event).__name__),
                    summary,
                )
        except Exception:
            LOGGER.exception("IA_PYDANTIC_TRACE agent=%s request=%s handler failed", agent_kind, request_no)
            raise
        finally:
            LOGGER.info("IA_PYDANTIC_TRACE agent=%s request=%s end", agent_kind, request_no)

    return handler


def log_model_retry(agent_kind: str, reason: Exception | str) -> None:
    settings = research_settings()
    if not settings.trace_enabled:
        return
    LOGGER.info(
        "IA_PYDANTIC_TRACE agent=%s output_validator_retry reason=%s",
        agent_kind,
        _truncate_text(str(reason), settings.trace_max_chars),
    )


def _summarize_pydantic_event(event: Any, max_chars: int) -> str | None:
    kind = getattr(event, "event_kind", "")
    part = getattr(event, "part", None)
    if kind in {"function_tool_call", "output_tool_call"} and part is not None:
        args = ""
        if hasattr(part, "args_as_json_str"):
            try:
                args = part.args_as_json_str()
            except Exception:
                args = repr(getattr(part, "args", ""))
        return (
            f"tool={getattr(part, 'tool_name', '')} call_id={getattr(part, 'tool_call_id', '')} "
            f"args_valid={getattr(event, 'args_valid', None)} args={_truncate_text(args, max_chars)}"
        )
    if kind in {"function_tool_result", "output_tool_result"} and part is not None:
        content = getattr(part, "content", "")
        return (
            f"tool={getattr(part, 'tool_name', '')} call_id={getattr(part, 'tool_call_id', '')} "
            f"part_kind={getattr(part, 'part_kind', '')} outcome={getattr(part, 'outcome', '')} "
            f"content={_truncate_text(_content_preview(content), max_chars)}"
        )
    if kind == "builtin_tool_call":
        native_part = getattr(event, "part", None)
        return (
            f"tool={getattr(native_part, 'tool_name', '')} call_id={getattr(native_part, 'tool_call_id', '')} "
            f"args={_tool_part_args_preview(native_part, max_chars)}"
        )
    if kind == "builtin_tool_result":
        native_result = getattr(event, "result", None)
        return (
            f"tool={getattr(native_result, 'tool_name', '')} call_id={getattr(native_result, 'tool_call_id', '')} "
            f"part_kind={getattr(native_result, 'part_kind', '')} outcome={getattr(native_result, 'outcome', '')} "
            f"content={_truncate_text(_content_preview(getattr(native_result, 'content', '')), max_chars)}"
        )
    if kind in {"part_start", "part_end"} and part is not None:
        return (
            f"index={getattr(event, 'index', '')} part_kind={getattr(part, 'part_kind', '')} "
            f"tool={getattr(part, 'tool_name', '')} args={_tool_part_args_preview(part, max_chars)}"
        )
    if kind == "part_delta":
        if not _trace_part_deltas_enabled():
            return None
        delta = getattr(event, "delta", None)
        delta_kind = getattr(delta, "part_delta_kind", "")
        if delta_kind == "tool_call":
            args_delta = getattr(delta, "args_delta", None)
            return (
                f"index={getattr(event, 'index', '')} delta_kind=tool_call "
                f"tool_name_delta={_truncate_text(str(getattr(delta, 'tool_name_delta', '') or ''), max_chars)} "
                f"call_id={getattr(delta, 'tool_call_id', '') or getattr(delta, 'tool_call_id_delta', '') or ''} "
                f"args_delta_len={len(str(args_delta or ''))} "
                f"args_delta={_truncate_text(_content_preview(args_delta), max_chars)}"
            )
        return (
            f"index={getattr(event, 'index', '')} delta_kind={delta_kind} "
            f"content={_truncate_text(_content_preview(getattr(delta, 'content_delta', '')), max_chars)}"
        )
    if kind == "final_result":
        return f"tool={getattr(event, 'tool_name', '')} call_id={getattr(event, 'tool_call_id', '')}"
    return _truncate_text(repr(event), max_chars)


def _trace_part_deltas_enabled() -> bool:
    return str(os.getenv("IA_PYDANTIC_TRACE_PART_DELTAS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _content_preview(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "; ".join(_content_preview(item) for item in content[:3])
    if isinstance(content, dict):
        return repr({key: content[key] for key in list(content)[:8]})
    return repr(content)


def _tool_part_args_preview(part: Any, max_chars: int) -> str:
    if not hasattr(part, "args_as_json_str"):
        return ""
    try:
        args = part.args_as_json_str()
    except Exception:
        args = repr(getattr(part, "args", ""))
    return _truncate_text(args, max_chars)


def _truncate_text(value: str, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _create_model(
    model_config: dict[str, str],
    *,
    prefer_openai_responses: bool = False,
) -> tuple[Any, dict[str, str]]:
    model_name = _model_name_for_agent(
        model_config["model"],
        prefer_openai_responses=prefer_openai_responses,
    )
    if ":" in model_name:
        _prime_provider_env(model_config)
        return model_name, {
            "model": model_name,
            "api_mode": "openai_responses"
            if _is_openai_responses_model_name(model_name)
            else "provider_string",
        }

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(
        base_url=model_config["base_url"],
        api_key=model_config["api_key"],
    )
    if prefer_openai_responses and _is_openai_base_url(model_config["base_url"]):
        return OpenAIResponsesModel(model_name, provider=provider), {
            "model": f"openai-responses:{model_name}",
            "api_mode": "openai_responses",
        }
    return OpenAIChatModel(model_name, provider=provider), {
        "model": model_name,
        "api_mode": "openai_chat",
    }


def _model_name_for_agent(model_name: str, *, prefer_openai_responses: bool) -> str:
    value = str(model_name or "").strip()
    if not prefer_openai_responses:
        return value
    if value.startswith("openai-responses:"):
        return value
    for prefix in ("openai:", "openai-chat:"):
        if value.startswith(prefix):
            return "openai-responses:" + value.split(":", 1)[1]
    return value


def _is_openai_responses_model_name(model_name: str) -> bool:
    return str(model_name or "").startswith("openai-responses:")


def _is_openai_base_url(base_url: str) -> bool:
    normalized = str(base_url or "").strip().lower().rstrip("/")
    return normalized in {
        "https://api.openai.com/v1",
        "https://api.openai.com",
    }


def _prime_provider_env(model_config: dict[str, str]) -> None:
    if model_config.get("api_key"):
        os.environ.setdefault("OPENAI_API_KEY", model_config["api_key"])
    if model_config.get("base_url"):
        os.environ.setdefault("OPENAI_BASE_URL", model_config["base_url"])


def _research_capabilities(
    *,
    enable_web_search: bool,
    enable_web_fetch: bool,
    enable_provider_web_tools: bool = True,
    settings: ResearchSettings | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    settings = settings or research_settings()
    capabilities: list[Any] = []
    status: dict[str, Any] = {
        "enabled": False,
        "thinking_effort": settings.thinking_effort,
        "web_enabled": settings.web_enabled,
        "web_search": False,
        "web_fetch": False,
        "local_fallback": True,
        "provider_web_tools": enable_provider_web_tools,
        "web_search_mode": settings.web_search_mode,
        "web_search_context_size": settings.web_search_context_size,
        "web_fetch_mode": settings.web_fetch_mode,
    }
    if not settings.web_enabled and not settings.thinking_effort:
        return capabilities, status

    try:
        from pydantic_ai.capabilities import Thinking, WebFetch, WebSearch
    except Exception as exc:
        status["error"] = f"pydantic_ai capabilities unavailable: {exc}"
        return capabilities, status

    if settings.thinking_effort and settings.thinking_effort.lower() != "none":
        capabilities.append(Thinking(effort=settings.thinking_effort))
        status["enabled"] = True

    if (
        settings.web_enabled
        and enable_web_search
        and settings.web_search_mode != "off"
        and settings.max_searches > 0
    ):
        search_kwargs = _web_search_kwargs(settings, enable_provider_web_tools)
        capabilities.append(
            _capability_with_research_extras(
                lambda: WebSearch(**search_kwargs),
                "WebSearch(local='duckduckgo')",
            )
        )
        status["enabled"] = True
        status["web_search"] = True
        status["max_searches"] = settings.max_searches

    if (
        settings.web_enabled
        and enable_web_fetch
        and settings.web_fetch_mode != "off"
        and settings.max_fetches > 0
    ):
        fetch_kwargs = _web_fetch_kwargs(settings)
        capabilities.append(
            _capability_with_research_extras(
                lambda: WebFetch(**fetch_kwargs),
                "WebFetch(local=True)",
            )
        )
        status["enabled"] = True
        status["web_fetch"] = True
        status["max_fetches"] = settings.max_fetches

    return capabilities, status


def _agent_model_settings(settings: ResearchSettings) -> dict[str, Any]:
    model_settings: dict[str, Any] = {}
    if settings.request_timeout > 0:
        model_settings["timeout"] = settings.request_timeout
    return model_settings


def _web_search_kwargs(settings: ResearchSettings, enable_provider_web_tools: bool) -> dict[str, Any]:
    if settings.web_search_mode == "local":
        return {"native": False, "local": "duckduckgo"}
    native_extras: dict[str, Any] = {}
    if settings.web_search_context_size:
        native_extras["search_context_size"] = settings.web_search_context_size
    if settings.max_searches > 0:
        native_extras["max_uses"] = settings.max_searches
    if settings.web_search_mode == "native":
        return {"native": True, "local": False, **native_extras}
    kwargs: dict[str, Any] = {"local": "duckduckgo"}
    if not enable_provider_web_tools:
        kwargs["native"] = False
    else:
        kwargs.update(native_extras)
    return kwargs


def _web_fetch_kwargs(settings: ResearchSettings) -> dict[str, Any]:
    if settings.web_fetch_mode == "native":
        return {"native": True, "local": False}
    # OpenAI Responses supports native WebSearchTool, but not native
    # WebFetchTool. Keep fetch local by default even when search is native.
    return {"native": False, "local": _local_web_fetch_tool(settings)}


def _local_web_fetch_tool(settings: ResearchSettings) -> Any:
    from pydantic_ai.common_tools.web_fetch import web_fetch_tool

    tool = web_fetch_tool(
        max_content_length=settings.web_fetch_max_content_length,
        allow_local_urls=False,
        timeout=settings.web_fetch_timeout,
        headers={
            "Accept": "text/markdown, text/html;q=0.9, application/json;q=0.8, */*;q=0.5",
            "User-Agent": "HermesInvestmentAssistant/0.1 (+local research fetch)",
        },
    )
    tool.max_retries = settings.web_fetch_retries
    tool.timeout = settings.web_fetch_timeout + 5
    return tool


def _capability_with_research_extras(factory, label: str) -> Any:
    try:
        return factory()
    except Exception as exc:
        if not _is_missing_pydantic_optional_group(exc):
            raise
        _install_pydantic_ai_research_extras(label, exc)
    try:
        return factory()
    except Exception as retry_exc:
        raise RuntimeError(
            f"{label} still cannot be initialized after installing PydanticAI research extras: "
            f"{retry_exc}"
        ) from retry_exc


def _is_missing_pydantic_optional_group(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "optional group" in message
        and "pydantic-ai" in message
        and "pip install" in message
    )


def _install_pydantic_ai_research_extras(label: str, original_error: Exception) -> None:
    try:
        from tools.lazy_deps import ensure

        ensure("investment.pydantic_ai", prompt=False, force=True)
    except Exception as install_exc:
        raise RuntimeError(
            f"{label} requires PydanticAI research extras, but automatic lazy install failed. "
            f"Original error: {original_error}. Install error: {install_exc}"
        ) from install_exc


def _read_hermes_config() -> dict[str, Any]:
    try:
        from hermes_constants import get_hermes_config_path

        path = get_hermes_config_path()
    except Exception:
        path = Path.home() / ".hermes" / "config.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _investment_assistant_model_settings(config: dict[str, Any]) -> dict[str, Any]:
    plugin_config = config.get("investment_assistant", {}) if isinstance(config, dict) else {}
    if not isinstance(plugin_config, dict):
        return {}
    settings = plugin_config.get("model_settings") or plugin_config.get("modelsetting") or {}
    return settings if isinstance(settings, dict) else {}


def _read_hermes_env_key(name: str) -> str:
    try:
        from hermes_constants import get_hermes_env_path

        path = get_hermes_env_path()
    except Exception:
        path = Path.home() / ".hermes" / ".env"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return ""
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip().strip('"').strip("'")
        return value
    return ""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name) or default).strip())
    except ValueError:
        return default


def _setting_str(settings: dict[str, Any], key: str, env_name: str, default: str) -> str:
    env_value = os.getenv(env_name)
    if env_value is not None:
        return env_value.strip() or default
    value = settings.get(key, default)
    return str(value or default).strip() or default


def _setting_bool(settings: dict[str, Any], key: str, env_name: str, default: bool) -> bool:
    env_value = os.getenv(env_name)
    if env_value is not None:
        return env_value.strip().lower() not in {"0", "false", "no", "off"}
    value = settings.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _setting_int(settings: dict[str, Any], key: str, env_name: str, default: int) -> int:
    env_value = os.getenv(env_name)
    value = env_value if env_value is not None else settings.get(key, default)
    try:
        return int(str(value).strip())
    except ValueError:
        return default


def _setting_float(settings: dict[str, Any], key: str, env_name: str, default: float) -> float:
    env_value = os.getenv(env_name)
    value = env_value if env_value is not None else settings.get(key, default)
    try:
        return float(str(value).strip())
    except ValueError:
        return default


def _tool_mode(value: str, *, default: Literal["auto", "local", "native", "off"]) -> Literal[
    "auto",
    "local",
    "native",
    "off",
]:
    normalized = str(value or default).strip().lower().replace("-", "_")
    aliases = {
        "builtin": "native",
        "provider": "native",
        "provider_native": "native",
        "false": "off",
        "disabled": "off",
        "disable": "off",
        "none": "off",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in {"auto", "local", "native", "off"}:
        return normalized  # type: ignore[return-value]
    return default


def _search_context_size(value: str) -> Literal["low", "medium", "high"] | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "none", "off", "false", "0"}:
        return None
    if normalized in {"low", "medium", "high"}:
        return normalized  # type: ignore[return-value]
    return "low"


def _with_research_overrides(settings: ResearchSettings, overrides: dict[str, Any]) -> ResearchSettings:
    allowed = set(ResearchSettings.__dataclass_fields__)
    values: dict[str, Any] = {}
    for key, value in (overrides or {}).items():
        if key not in allowed or value is None:
            continue
        if key in {
            "max_searches",
            "max_fetches",
            "web_fetch_timeout",
            "web_fetch_retries",
            "web_fetch_max_content_length",
        }:
            try:
                values[key] = max(0, int(value))
            except (TypeError, ValueError):
                continue
        elif key == "request_timeout":
            try:
                values[key] = max(1.0, float(value))
            except (TypeError, ValueError):
                continue
        elif key == "web_search_context_size":
            values[key] = _search_context_size(str(value))
        elif key in {"web_search_mode", "web_fetch_mode"}:
            values[key] = _tool_mode(str(value), default=getattr(settings, key))
        else:
            values[key] = value
    return replace(settings, **values) if values else settings
