"""Hermes plugin that bridges to a running Vibe-Trading API service."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

TOOLSET = "vibe-trading"
DEFAULT_BASE_URL = "http://192.168.1.58:8899"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_AGENT_TIMEOUT_SECONDS = 180.0
DEFAULT_AGENT_POLL_SECONDS = 3.0


def _base_url() -> str:
    return os.getenv("VIBE_TRADING_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _timeout_seconds() -> float:
    raw = os.getenv("VIBE_TRADING_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def _agent_timeout_seconds() -> float:
    raw = os.getenv("VIBE_TRADING_AGENT_TIMEOUT_SECONDS", str(DEFAULT_AGENT_TIMEOUT_SECONDS))
    try:
        return max(10.0, float(raw))
    except ValueError:
        return DEFAULT_AGENT_TIMEOUT_SECONDS


def _agent_poll_seconds() -> float:
    raw = os.getenv("VIBE_TRADING_AGENT_POLL_SECONDS", str(DEFAULT_AGENT_POLL_SECONDS))
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_AGENT_POLL_SECONDS


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _request_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
) -> str:
    """Call Vibe-Trading and return a JSON string for Hermes tool output."""
    url = f"{_base_url()}/{path.lstrip('/')}"
    if query:
        clean_query = {k: v for k, v in query.items() if v is not None}
        if clean_query:
            url = f"{url}?{urllib.parse.urlencode(clean_query)}"

    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=body, headers=headers, method=method.upper())

    try:
        with urllib.request.urlopen(request, timeout=_timeout_seconds()) as response:
            raw = response.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return _json_dumps({"success": True, "status": getattr(response, "status", 200)})
            try:
                return _json_dumps(json.loads(raw))
            except json.JSONDecodeError:
                return _json_dumps({"success": True, "text": raw})
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        return _json_dumps({
            "success": False,
            "error_type": "HTTPError",
            "status": exc.code,
            "error": detail,
            "url": url,
        })
    except Exception as exc:
        return _json_dumps({
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "url": url,
        })


def _vibe_health(args: dict[str, Any], **kwargs) -> str:
    return _request_json("GET", "/health")


def _latest_assistant_message(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            content = _clean_agent_answer(str(message.get("content") or ""))
            if content:
                return content
    return None


def _clean_agent_answer(content: str) -> str:
    text = content.strip()
    return re.sub(r"^<think>\s*</think>\s*", "", text, flags=re.IGNORECASE).strip()


def _vibe_agent_ask(question: str, *, title: str, instruction: str | None = None) -> str:
    clean_question = question.strip()
    if not clean_question:
        return _json_dumps({"success": False, "error": "question is required"})

    create_payload = json.loads(_request_json("POST", "/sessions", {"title": title}))
    if create_payload.get("success") is False:
        return _json_dumps(create_payload)
    session_id = str(create_payload.get("session_id") or "").strip()
    if not session_id:
        return _json_dumps({
            "success": False,
            "error": "Vibe-Trading did not return session_id",
            "response": create_payload,
        })

    content = clean_question if not instruction else f"{instruction.strip()}\n\n用户原始问题：\n{clean_question}"
    send_payload = json.loads(_request_json(
        "POST",
        f"/sessions/{urllib.parse.quote(session_id, safe='')}/messages",
        {"content": content},
    ))
    if send_payload.get("success") is False:
        send_payload.setdefault("session_id", session_id)
        return _json_dumps(send_payload)
    attempt_id = send_payload.get("attempt_id")

    deadline = time.monotonic() + _agent_timeout_seconds()
    last_messages: Any = None
    while time.monotonic() < deadline:
        messages_payload = json.loads(_request_json(
            "GET",
            f"/sessions/{urllib.parse.quote(session_id, safe='')}/messages",
            query={"limit": 50},
        ))
        last_messages = messages_payload
        answer = _latest_assistant_message(messages_payload)
        if answer:
            return _json_dumps({
                "success": True,
                "session_id": session_id,
                "attempt_id": attempt_id,
                "answer": answer,
            })
        time.sleep(_agent_poll_seconds())

    return _json_dumps({
        "success": False,
        "error": "Timed out waiting for Vibe-Trading Agent response",
        "session_id": session_id,
        "attempt_id": attempt_id,
        "last_messages": last_messages,
    })


ASHARE_AKSHARE_INSTRUCTION = """请作为 Vibe-Trading A股投资分析 Agent 处理以下问题，并直接输出最终中文分析报告。

第一阶段约束：
- 优先使用免费 AKShare 数据源和 Vibe-Trading 内置 A股能力。
- 不要默认调用 Tushare 或 QVeris；只有用户明确要求时才使用它们。
- 如果无法获取实时或最新数据，请明确说明数据时效和限制。
- 最终回答不要输出 <minimax:tool_call>、<invoke>、工具 XML 或代码块式工具调用标记。

请尽量输出结构化报告，包含：
1. 当前行情状态
2. 技术指标和趋势判断
3. 买入、加仓、卖出或观望条件
4. 止损位和止盈目标
5. 风险收益比
6. 关键风险与反向信号
7. 操作建议总结
8. 免责声明：仅供研究参考，不构成投资建议
"""


def _vibe_ask(args: dict[str, Any], **kwargs) -> str:
    return _vibe_agent_ask(
        str(args.get("question", "")),
        title=args.get("title") or "Vibe-Trading Ask",
    )


def _vibe_ask_ashare(args: dict[str, Any], **kwargs) -> str:
    return _vibe_agent_ask(
        str(args.get("question", "")),
        title=args.get("title") or "A股自然语言分析",
        instruction=ASHARE_AKSHARE_INSTRUCTION,
    )


def _vibe_list_skills(args: dict[str, Any], **kwargs) -> str:
    return _request_json("GET", "/skills")


def _vibe_list_swarm_presets(args: dict[str, Any], **kwargs) -> str:
    return _request_json("GET", "/swarm/presets")


def _vibe_run_swarm(args: dict[str, Any], **kwargs) -> str:
    payload = {
        "preset_name": args.get("preset_name"),
        "variables": args.get("variables") or {},
    }
    return _request_json("POST", "/swarm/runs", payload)


def _vibe_get_swarm_run(args: dict[str, Any], **kwargs) -> str:
    run_id = str(args.get("run_id", "")).strip()
    if not run_id:
        return _json_dumps({"success": False, "error": "run_id is required"})
    return _request_json("GET", f"/swarm/runs/{urllib.parse.quote(run_id, safe='')}")


def _vibe_create_session(args: dict[str, Any], **kwargs) -> str:
    payload = {
        "title": args.get("title") or "",
        "config": args.get("config"),
    }
    return _request_json("POST", "/sessions", payload)


def _vibe_send_message(args: dict[str, Any], **kwargs) -> str:
    session_id = str(args.get("session_id", "")).strip()
    content = str(args.get("content", "")).strip()
    if not session_id:
        return _json_dumps({"success": False, "error": "session_id is required"})
    if not content:
        return _json_dumps({"success": False, "error": "content is required"})
    return _request_json(
        "POST",
        f"/sessions/{urllib.parse.quote(session_id, safe='')}/messages",
        {"content": content},
    )


def _vibe_get_run_result(args: dict[str, Any], **kwargs) -> str:
    run_id = str(args.get("run_id", "")).strip()
    if not run_id:
        return _json_dumps({"success": False, "error": "run_id is required"})
    return _request_json("GET", f"/runs/{urllib.parse.quote(run_id, safe='')}")


def _vibe_list_runs(args: dict[str, Any], **kwargs) -> str:
    limit = args.get("limit", 20)
    return _request_json("GET", "/runs", query={"limit": limit})


VIBE_HEALTH_SCHEMA = {
    "name": "vibe_health",
    "description": "Check whether the Vibe-Trading API service is reachable and healthy.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

VIBE_ASK_SCHEMA = {
    "name": "vibe_ask",
    "description": (
        "Forward a natural-language finance or trading question directly to the "
        "Vibe-Trading Agent and return the final assistant report."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "User's original natural-language question.",
            },
            "title": {
                "type": "string",
                "description": "Optional Vibe-Trading session title.",
            },
        },
        "required": ["question"],
    },
}

VIBE_ASK_ASHARE_SCHEMA = {
    "name": "vibe_ask_ashare",
    "description": (
        "Use this first for A-share or stock questions from Feishu, including buy/sell "
        "points, stop-loss, take-profit, risk, technical analysis, sector analysis, "
        "and strategy questions. It forwards natural language to Vibe-Trading Agent "
        "with free AKShare-first instructions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "User's original A-share natural-language question.",
            },
            "title": {
                "type": "string",
                "description": "Optional Vibe-Trading session title.",
            },
        },
        "required": ["question"],
    },
}

VIBE_LIST_SKILLS_SCHEMA = {
    "name": "vibe_list_skills",
    "description": "List Vibe-Trading finance skills, including A-share analysis skills.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

VIBE_LIST_SWARM_PRESETS_SCHEMA = {
    "name": "vibe_list_swarm_presets",
    "description": "List Vibe-Trading multi-agent team presets such as investment_committee and risk_committee.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

VIBE_RUN_SWARM_SCHEMA = {
    "name": "vibe_run_swarm",
    "description": (
        "Run a Vibe-Trading multi-agent team. Use this for investment committee, "
        "risk committee, quant strategy, and A-share research desk analysis."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "preset_name": {
                "type": "string",
                "description": "Swarm preset name, e.g. investment_committee, risk_committee, quant_strategy_desk.",
            },
            "variables": {
                "type": "object",
                "description": "Preset variables, e.g. {'target': '600519.SH', 'market': 'A股'}.",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["preset_name", "variables"],
    },
}

VIBE_GET_SWARM_RUN_SCHEMA = {
    "name": "vibe_get_swarm_run",
    "description": "Fetch a Vibe-Trading swarm run result by run_id.",
    "parameters": {
        "type": "object",
        "properties": {"run_id": {"type": "string", "description": "Swarm run identifier."}},
        "required": ["run_id"],
    },
}

VIBE_CREATE_SESSION_SCHEMA = {
    "name": "vibe_create_session",
    "description": "Create a Vibe-Trading session for natural-language finance analysis.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Session title."},
            "config": {"type": "object", "description": "Optional session config."},
        },
        "required": [],
    },
}

VIBE_SEND_MESSAGE_SCHEMA = {
    "name": "vibe_send_message",
    "description": (
        "Send a natural-language request to a Vibe-Trading session. Use this for "
        "A-share buy/sell research, backtest requests, ST risk checks, and sector analysis."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "Vibe-Trading session identifier."},
            "content": {"type": "string", "description": "Natural-language analysis request."},
        },
        "required": ["session_id", "content"],
    },
}

VIBE_GET_RUN_RESULT_SCHEMA = {
    "name": "vibe_get_run_result",
    "description": "Fetch a Vibe-Trading run result by run_id, including backtest and analysis outputs.",
    "parameters": {
        "type": "object",
        "properties": {"run_id": {"type": "string", "description": "Run identifier."}},
        "required": ["run_id"],
    },
}

VIBE_LIST_RUNS_SCHEMA = {
    "name": "vibe_list_runs",
    "description": "List recent Vibe-Trading runs with summary fields.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of runs to return.",
                "default": 20,
                "minimum": 1,
                "maximum": 100,
            }
        },
        "required": [],
    },
}


def register(ctx):
    """Register Vibe-Trading bridge tools."""
    tools = [
        ("vibe_ask", VIBE_ASK_SCHEMA, _vibe_ask),
        ("vibe_ask_ashare", VIBE_ASK_ASHARE_SCHEMA, _vibe_ask_ashare),
        ("vibe_health", VIBE_HEALTH_SCHEMA, _vibe_health),
        ("vibe_list_skills", VIBE_LIST_SKILLS_SCHEMA, _vibe_list_skills),
        ("vibe_list_swarm_presets", VIBE_LIST_SWARM_PRESETS_SCHEMA, _vibe_list_swarm_presets),
        ("vibe_run_swarm", VIBE_RUN_SWARM_SCHEMA, _vibe_run_swarm),
        ("vibe_get_swarm_run", VIBE_GET_SWARM_RUN_SCHEMA, _vibe_get_swarm_run),
        ("vibe_create_session", VIBE_CREATE_SESSION_SCHEMA, _vibe_create_session),
        ("vibe_send_message", VIBE_SEND_MESSAGE_SCHEMA, _vibe_send_message),
        ("vibe_get_run_result", VIBE_GET_RUN_RESULT_SCHEMA, _vibe_get_run_result),
        ("vibe_list_runs", VIBE_LIST_RUNS_SCHEMA, _vibe_list_runs),
    ]
    for name, schema, handler in tools:
        ctx.register_tool(
            name=name,
            toolset=TOOLSET,
            schema=schema,
            handler=handler,
            description=schema["description"],
        )
    logger.info("vibe-trading plugin: registered %s tools", len(tools))
