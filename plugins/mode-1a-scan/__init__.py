"""
Mode 1A 自动扫描插件 — 同步追加异质视角问题

架构:
  每轮 Agent 回复 → transform_llm_output hook 同步拦截
  → Flash 判断是否策略性回复
  → 是 → 生成一个异质视角问题 → 追加到回复末尾
  → 否 → 原样放行

与结论锚点的区别:
  - 同步（改输出后返回）而非异步 fire-and-forget
  - 不验证事实，只生成问题
  - 轻量：一次 Flash 调用，max_tokens=200
"""

import json
import logging
import os
import re
import time

import requests

# ══ 加载标记 ══
try:
    with open("/tmp/mode_1a_scan_loaded", "w") as f:
        f.write(f"loaded at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
except Exception:
    pass

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
SCAN_TIMEOUT = 8  # 短回复，Flash 很快
MIN_LENGTH = 80    # 短于此跳过（确认/闲聊）

# ── 跳过模式（纯流程/确认/闲聊） ──
SKIP_PATTERNS = [
    r'^[好可以行对是的嗯]+[，。！]?$',           # 纯确认
    r'^收到[，。！]?$',
    r'^```',                                    # 代码块开始
    r'^📁\s',                                   # 文件归档声明
    r'^✅\s验证',                               # 验证声明
    r'^orphans:',                               # orphans 标记
    r'^Fri\s\w{3}\s\d{2}',                      # 日期行
    r'^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}',         # 时间戳
]

# ═══════════════════════════════════════════════
# Flash 调用
# ═══════════════════════════════════════════════

SCAN_SYSTEM = """你是异质视角问题生成器。

分析这段 AI 回复：
1. 判断：是否包含策略性分析、判断、建议、方案设计？
   - 日常问答、流程确认、闲聊 → 返回 NO
   - 分析/判断/建议/方案 → 返回 YES
2. 如果是 YES：生成一个杨旸（中海油IT架构师，偏好系统化思维和反面验证）不太可能问自己的异质视角问题。
   - 偏"他没看到的盲区""反过来的假设""不同角色的视角"
   - 短问题，15字以内
   - 不要说教，不要建议，只问问题

输出格式（严格遵守）：
如果非策略性 → 只输出: NO
如果是策略性 → 只输出问题本身（不要引号，不要前缀）"""


def _call_flash(system_prompt: str, user_content: str) -> str | None:
    """调用 Flash API，返回生成的问题或 None"""
    if not DEEPSEEK_KEY:
        return None
    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.3,  # 轻微变化避免重复
                "max_tokens": 200,
            },
            timeout=SCAN_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning(f"mode-1a-scan: Flash HTTP {resp.status_code}")
            return None
        body = resp.json()
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip()
    except requests.Timeout:
        logger.warning("mode-1a-scan: Flash timeout")
        return None
    except Exception as e:
        logger.warning(f"mode-1a-scan: Flash error: {e}")
        return None


def _is_strategic(response_text: str) -> bool:
    """快速预判：跳过明显的非策略回复"""
    text = response_text.strip()
    if len(text) < MIN_LENGTH:
        return False
    # 跳过纯确认/流程声明
    for pattern in SKIP_PATTERNS:
        if re.match(pattern, text):
            return False
    # 跳过以日期/时间戳开头的行（通常不是分析内容）
    if re.match(r'^\d{4}-\d{2}-\d{2}|^Fri\s|^orphans:', text):
        return False
    return True


# ═══════════════════════════════════════════════
# Hook 入口
# ═══════════════════════════════════════════════

def register(ctx):
    """插件入口——注册 transform_llm_output hook"""
    ctx.register_hook("transform_llm_output", on_transform_llm_output)
    logger.info("mode-1a-scan: registered")

def on_transform_llm_output(response_text, session_id=None, model=None, platform=None, **kwargs):
    """同步扫描：判断策略性 → 生成异质视角问题 → 追加"""
    start = time.time()

    # 快速跳过
    if not _is_strategic(response_text):
        return response_text

    # 用回复的后 1500 字符做判断（前文可能有上下文噪音，后段是结论/建议）
    snippet = response_text[-2000:] if len(response_text) > 2000 else response_text

    result = _call_flash(SCAN_SYSTEM, snippet)

    elapsed = time.time() - start

    if result is None:
        # Flash 失败 → 静默跳过，不阻塞回复
        logger.warning(f"mode-1a-scan: Flash failed ({elapsed:.1f}s), passing through")
        return response_text

    result = result.strip()

    if result.upper() == "NO" or result == "否":
        logger.debug(f"mode-1a-scan: non-strategic ({elapsed:.1f}s)")
        return response_text

    # 是策略性回复 → 追加问题
    # 清理可能的多余输出（Flash 有时会加解释）
    if len(result) > 50:
        # 太长，可能不是纯问题，尝试提取第一句
        result = result.split("。")[0].split("\n")[0].strip()

    enhanced = response_text.rstrip() + f"\n\n_反面：{result}_"
    logger.info(f"mode-1a-scan: appended question ({elapsed:.1f}s): {result}")
    return enhanced
