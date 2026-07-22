"""
LLM-powered Chinese fact extraction for holographic memory.

Replaces English-only regex patterns with a configurable LLM API call
that handles Chinese naturally.

Config under plugins.hermes-memory-store in config.yaml:

  auto_extract: true                    # 开启会话结束自动提取
  extract_llm: auto                     # auto | off | regex
  extract_llm_model: glm-4.5-flash      # 模型名
  extract_llm_endpoint: https://api.z.ai/api/paas/v4/chat/completions
                                        # API 完整地址（含 /chat/completions）
  extract_llm_api_key_env: GLM_API_KEY  # 存放 API Key 的环境变量名

换模型/换供应商只需要改这 4 行配置，不动代码。
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompt (Chinese)
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM_PROMPT = """你是一个记忆提取助手。从以下用户对话中，提取有价值的事实。

只提取以下三类事实，每类单独判断：

1. **user_pref** — 用户的偏好、习惯、使用方式
   例：我持有恒生红利ETF / 我喜欢稳健型投资 / 我平时用微信聊天

2. **project** — 用户的决策、选择、变更记录
   例：我决定定投创业板 / 我把短债换成长城短债了 / 我开通了港股通

3. **tool** — 工具、配置、技术选型相关
   例：我用了 DeepSeek 的 API / TTSKILL_HOME 设在 AppData 下 / 这个用 Python 写的

返回格式（JSON 数组，每个元素包含 fact 和 category）：
[
  {"fact": "用户持有恒生红利ETF", "category": "user_pref"},
  {"fact": "用户决定定投创业板指数基金", "category": "project"}
]

注意事项：
- fact 用简洁的中文陈述句，以第三人称"用户"开头
- 只提取明确表达或可合理推断的事实，不确定的不提取
- 每个事实一句话，不超过 80 字
- **普通对话内容（"你好"、"吃了没"、"谢谢"等）不要提取**
- 如果当前对话没有值得提取的事实，直接返回空数组 []
- 不要编造"""


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def _call_llm(
    messages: List[Dict[str, str]],
    model: str,
    endpoint: str,
    api_key_env: str,
    timeout: int = 15,
) -> str | None:
    """Call an OpenAI-compatible chat-completion API."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        logger.warning("提取失败：环境变量 %s 未设置", api_key_env)
        return None

    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1024,
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("LLM 提取 API 调用失败: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_facts(
    user_messages: List[str],
    *,
    model: str = "glm-4.5-flash",
    endpoint: str = "https://api.z.ai/api/paas/v4/chat/completions",
    api_key_env: str = "GLM_API_KEY",
) -> List[Dict[str, Any]]:
    """Extract facts from user messages via LLM.

    Args:
        user_messages: 本轮回话的用户消息列表
        model: 模型名（如 "glm-4.5-flash", "deepseek-v4-flash"）
        endpoint: API 完整地址（含 /chat/completions）
        api_key_env: 存放 API Key 的环境变量名

    Returns:
        [{"fact": "...", "category": "user_pref"}, ...]
    """
    if not user_messages:
        return []

    dialogue = "\n".join(
        line.strip() for line in user_messages
        if isinstance(line, str) and len(line.strip()) >= 5
    )
    if not dialogue or len(dialogue) < 10:
        return []

    # 截断到 2000 字符，控制 token 消耗
    if len(dialogue) > 2000:
        dialogue = dialogue[-2000:]

    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"以下是用户本轮对话的内容：\n{dialogue}"},
    ]

    raw = _call_llm(messages, model=model, endpoint=endpoint, api_key_env=api_key_env)
    if not raw:
        return []

    # 解析 JSON 响应
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        facts = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("提取结果解析失败: %s", raw[:200])
        return []

    if not isinstance(facts, list):
        return []

    valid_categories = {"user_pref", "project", "tool"}
    validated = []
    for item in facts:
        if isinstance(item, dict) and "fact" in item and "category" in item:
            cat = str(item["category"])
            validated.append({
                "fact": str(item["fact"])[:200],
                "category": cat if cat in valid_categories else "general",
            })
    return validated
