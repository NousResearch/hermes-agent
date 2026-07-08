"""
上下文注射提醒插件 — 在 Agent 回复末尾注入 checklist 提醒 (v1)

设计：第十人审计通过的 v2 设计文档
  ~/hermes-local/yangyang/设计文档/上下文注射提醒系统_设计_v2_2026-06-27.md

架构：
  LLM 回复 → transform_llm_output hook
    → Flash 语义检测（classify scenes）
    → 检查连续抑制计数
    → 多场景合并为一行
    → 追加到回复末尾
    → 返回修改后的文本（或 None=不变）

三层安全网（T0b/T10）：
  ① Flash 不可达 → 静默跳过，原样返回
  ② injection_state.json 损坏 → 重建空状态，继续
  ③ injector 内部异常 → try/except 全包，原样返回
"""

import json
import logging
import os
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
FLASH_MODEL = "deepseek-chat"  # V3 = fast model for classification
FLASH_TIMEOUT = 8               # seconds — fast model, 8s is generous
STATE_DIR = os.path.expanduser("~/.hermes/state")
STATE_FILE = os.path.join(STATE_DIR, "injection_state.json")
LOG_FILE = os.path.expanduser("~/.hermes/logs/injector.log")
MIN_RESPONSE_LENGTH = 40        # short replies skip classification
MAX_CONSECUTIVE = 5             # suppress after 5 consecutive hits

# ═══════════════════════════════════════════════
# 场景定义
# ═══════════════════════════════════════════════

SCENE_DEFS = {
    "SCRIPT": "writing or modifying shell/Python scripts",
    "SKILL": "creating or modifying Hermes skills",
    "CONFIG": "changing gateway configuration, config.yaml, or systemd units",
    "CRON": "creating or modifying cron jobs",
    "FILE": "creating new tool files or script files",
}

# ═══════════════════════════════════════════════
# 合并规则（硬编码 if-else，不建动态引擎）
# ═══════════════════════════════════════════════

MERGE_PATTERNS = [
    # 排序：先长后短，优先匹配更具体的组合
    (frozenset(["SCRIPT", "CONFIG", "CRON"]), "动脚本/config/cron前"),
    (frozenset(["SCRIPT", "FILE", "CRON"]), "动脚本/文件/cron前"),
    (frozenset(["SCRIPT", "CONFIG"]), "动脚本/config前"),
    (frozenset(["SCRIPT", "CRON"]), "动脚本/cron前"),
    (frozenset(["SCRIPT", "FILE"]), "动脚本/新建文件前"),
    (frozenset(["CONFIG", "CRON"]), "改配置/建cron前"),
    (frozenset(["FILE", "CRON"]), "建文件/cron前"),
    (frozenset(["SKILL", "SCRIPT"]), "动skill/脚本前"),
]

SINGLE_REMINDER = {
    "SCRIPT": "动脚本前：find已有代码 → which验证依赖 → pre_flight_check",
    "FILE":   "新建文件前：find搜已有 → 确认无现成方案 → pre_flight",
    "SKILL":  "动skill前：翻设计指南 → 查已有skill → 四步门控",
    "CONFIG": "改config前：备份+at回退 → YAML语法 → restart验证",
    "CRON":   "建cron前：验证脚本路径 → 声明成本+刹车(0E) → 更新清单(§2E)",
}

EXPANDED_TAIL = "：find→验证→pre_flight→at回退→翻指南"

# ═══════════════════════════════════════════════
# 状态管理
# ═══════════════════════════════════════════════

def _load_state():
    """加载连续计数状态。损坏→返回空字典。"""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        _log_error(f"state file corrupt, resetting: {e}")
    return {}


def _save_state(state):
    """保存状态到磁盘。异常→只记日志不阻断。"""
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        _log_error(f"failed to save state: {e}")


def _log_error(msg):
    """非阻塞错误日志。"""
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {msg}\n")
    except Exception:
        pass  # 日志写失败也不阻断


def _log_info(msg):
    """信息日志。"""
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass


# ═══════════════════════════════════════════════
# Flash 分类
# ═══════════════════════════════════════════════

def _classify_scenes(response_text):
    """
    调用 DeepSeek Flash 分类回复文本。
    返回: set of scene IDs, e.g. {"SCRIPT", "CRON"} 或空 set
    异常: 返回空 set（静默降级）
    """
    if not DEEPSEEK_KEY:
        _log_error("DEEPSEEK_API_KEY not set — skipping injection")
        return set()

    prompt = (
        "You are a scene classifier for an AI agent. "
        "Given the agent's reply below, determine which operational scenes are active.\n\n"
        "Rules:\n"
        "- SCRIPT: agent is about to write, modify, or debug a shell/Python script\n"
        "- SKILL: agent is about to create or modify a Hermes skill\n"
        "- CONFIG: agent is about to change config.yaml, gateway config, or systemd units\n"
        "- CRON: agent is about to create or modify a cron job\n"
        "- FILE: agent is about to create a new tool/script file\n"
        "- If none of these apply, return NONE\n\n"
        "The agent's reply may mention multiple operations. Return ALL matching scenes.\n"
        "If the agent is only talking ABOUT past scripts (回顾/查看/讨论), return NONE.\n"
        "If the agent says it WILL do an operation (即将/马上/让我), classify accordingly.\n\n"
        "Reply with ONLY a JSON object, no other text:\n"
        '{"scenes": ["SCRIPT", "CRON"]}  or  {"scenes": []}\n\n'
        f"Agent reply:\n{response_text[-3000:]}"  # last 3000 chars = likely the clearest signal
    )

    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": FLASH_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.0,
            },
            timeout=FLASH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()

        # Extract JSON from response (may have markdown fences)
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        scenes = set(result.get("scenes", []))

        # Filter to known scenes only
        valid = scenes & set(SCENE_DEFS.keys())
        return valid

    except requests.Timeout:
        _log_error("Flash classify timeout — skipping injection")
        return set()
    except requests.RequestException as e:
        _log_error(f"Flash classify HTTP error: {e}")
        return set()
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        _log_error(f"Flash classify parse error: {e} | raw={content[:200] if 'content' in dir() else 'N/A'}")
        return set()
    except Exception as e:
        _log_error(f"Flash classify unexpected error: {e}")
        return set()


# ═══════════════════════════════════════════════
# 合并 + 计数
# ═══════════════════════════════════════════════

def _build_reminder(scenes):
    """从场景集合构建合并后的提醒行。"""
    if not scenes:
        return None

    scenes_set = frozenset(scenes)

    # 第一步：尝试多场景合并
    for pattern, prefix in MERGE_PATTERNS:
        if scenes_set == pattern:
            return f"💡 {prefix}{EXPANDED_TAIL}"

    # 第二步：单场景
    if len(scenes_set) == 1:
        scene = list(scenes_set)[0]
        return f"💡 {scene}: {SINGLE_REMINDER.get(scene, '')}"

    # 第三步：未匹配的组合 → 通用尾部
    prefix = "动" + "/".join(sorted(scenes_set)) + "前"
    return f"💡 {prefix}{EXPANDED_TAIL}"


def _check_consecutive_limit(scenes):
    """
    检查连续抑制：
    - 每个场景独立计数
    - 达到 MAX_CONSECUTIVE → 抑制
    - 命中时计数+1，未命中时清零
    返回: 仍需注射的场景集合
    """
    state = _load_state()
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    active = set()

    for scene in scenes:
        entry = state.get(scene, {"count": 0})
        count = entry.get("count", 0)

        if count >= MAX_CONSECUTIVE:
            # 已达上限——首次触发时告警
            if count == MAX_CONSECUTIVE:
                _log_info(f"SCENE {scene} suppressed at count={count} — WeChat alert should fire")
            # 仍然计数（保持抑制状态），但不注射
            state[scene] = {"count": count + 1, "last_seen": now}
            continue

        active.add(scene)
        state[scene] = {"count": count + 1, "last_seen": now}

    # 清零未命中的场景计数
    for scene in list(state.keys()):
        if scene not in scenes:
            del state[scene]

    _save_state(state)
    return active


# ═══════════════════════════════════════════════
# Hook 入口
# ═══════════════════════════════════════════════

def register(ctx):
    """注册 transform_llm_output hook"""
    ctx.register_hook("transform_llm_output", _inject_reminder)
    _log_info("context_injection: registered v1 hook")
    # 落盘标记
    with open("/tmp/context_injection_loaded", "w") as f:
        f.write(f"loaded at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")


def _inject_reminder(response_text, session_id=None, model=None, platform=None, **kwargs):
    """
    transform_llm_output hook 入口。

    返回:
      - 修改后的文本（含注射行）→ gateway 用此文本替换原始回复
      - None → 原样通过

    安全网：
      - 任何异常 → 返回 None（原样通过，不阻断消息）
    """
    try:
        # Guard 1: skip cron/internal sessions
        if not session_id or str(session_id).startswith("cron_"):
            return None

        # Guard 2: skip short responses
        if not response_text or not isinstance(response_text, str):
            return None
        if len(response_text.strip()) < MIN_RESPONSE_LENGTH:
            return None

        # Guard 3: skip if already has injection (prevent double-injection)
        if "💡 动" in response_text and ("前：" in response_text or "前:" in response_text):
            return None

        # ═══ 分类 ═══
        scenes = _classify_scenes(response_text)
        if not scenes:
            return None

        # ═══ 连续抑制 ═══
        active = _check_consecutive_limit(scenes)
        if not active:
            return None

        # ═══ 构建提醒 ═══
        reminder = _build_reminder(active)
        if not reminder:
            return None

        # ═══ 注射 ═══
        modified = response_text + "\n\n" + reminder
        _log_info(f"INJECTED: scenes={sorted(active)} | session={session_id} | len={len(modified)}")
        return modified

    except Exception as e:
        # 最后的保险——任何未预见的异常都不阻断消息
        _log_error(f"injector crashed: {e}")
        return None
