"""Hermes Achievements dashboard plugin backend.

Mounted at /api/plugins/hermes-achievements/ by Hermes dashboard.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from hermes_constants import get_hermes_home
except ImportError:
    import os as _os
    def get_hermes_home() -> Path:  # type: ignore[misc]
        val = (_os.environ.get("HERMES_HOME") or "").strip()
        return Path(val) if val else Path.home() / ".hermes"

try:
    from fastapi import APIRouter
except Exception:  # Allows local unit tests without dashboard dependencies.
    class APIRouter:  # type: ignore
        def get(self, *_args, **_kwargs):
            return lambda fn: fn
        def post(self, *_args, **_kwargs):
            return lambda fn: fn

router = APIRouter()

SNAPSHOT_TTL_SECONDS = 120
_SCAN_LOCK = threading.Lock()
_SNAPSHOT_CACHE: Optional[Dict[str, Any]] = None
_SNAPSHOT_CACHE_AT = 0
_SCAN_STATUS: Dict[str, Any] = {
    "state": "idle",
    "started_at": None,
    "finished_at": None,
    "last_error": None,
    "last_duration_ms": None,
    "run_count": 0,
}

ERROR_RE = re.compile(r"\b(error|failed|failure|traceback|exception|permission denied|not found|eaddrinuse|already in use|timed out|blocked)\b", re.I)
PORT_RE = re.compile(r"\b(port\s+)?(3000|5173|8000|8080|9119)\b.*\b(in use|already|taken|eaddrinuse)\b|\beaddrinuse\b", re.I)
INSTALL_RE = re.compile(r"\b(npm|pnpm|yarn|pip|uv)\b.*\b(install|add)\b", re.I)
SUCCESS_RE = re.compile(r"\b(success|passed|built|compiled|done|exit_code[\"']?\s*[:=]\s*0|verified|ok)\b", re.I)
FILE_RE = re.compile(r"(?:/home/|~/?|\./|/mnt/)[\w./-]+\.(?:py|js|ts|tsx|jsx|css|html|md|json|yaml|yml|svg|sql|sh)")

TIER_NAMES = ["Copper", "Silver", "Gold", "Diamond", "Olympian"]


def tiers(values: List[int]) -> List[Dict[str, Any]]:
    return [{"name": name, "threshold": threshold} for name, threshold in zip(TIER_NAMES, values)]


def req(metric: str, gte: int) -> Dict[str, Any]:
    return {"metric": metric, "gte": gte}


ACHIEVEMENTS: List[Dict[str, Any]] = [
    # Agent Autonomy — mostly best-session feats
    {"id": "let_him_cook", "name": "Let Him Cook", "description": "Let Hermes run a serious autonomous tool chain in one session.", "category": "Agent Autonomy", "kind": "best_session", "icon": "flame", "threshold_metric": "max_tool_calls_in_session", "tiers": tiers([200, 500, 1200, 3000, 8000])},
    {"id": "autonomous_avalanche", "name": "Autonomous Avalanche", "description": "Accumulate a lifetime avalanche of Hermes tool calls across sessions.", "category": "Agent Autonomy", "kind": "lifetime", "icon": "avalanche", "threshold_metric": "total_tool_calls", "tiers": tiers([1000, 3000, 8000, 20000, 50000])},
    {"id": "toolchain_maxxer", "name": "Toolchain Maxxer", "description": "Use a wide spread of distinct Hermes tools in one session.", "category": "Agent Autonomy", "kind": "best_session", "icon": "nodes", "threshold_metric": "max_distinct_tools_in_session", "tiers": tiers([18, 28, 45, 70, 100])},
    {"id": "full_send", "name": "Full Send", "description": "Terminal, files, and web/browser all get involved in one real run.", "category": "Agent Autonomy", "kind": "multi_condition", "icon": "rocket", "requirements": [req("max_terminal_calls_in_session", 180), req("max_file_tool_calls_in_session", 120), req("max_web_browser_calls_in_session", 60)]},
    {"id": "subagent_commander", "name": "Subagent Commander", "description": "Coordinate delegated agent work.", "category": "Agent Autonomy", "kind": "lifetime", "icon": "branch", "threshold_metric": "total_delegate_calls", "tiers": tiers([5, 40, 100, 1000, 5000])},
    {"id": "background_process_enjoyer", "name": "Background Process Enjoyer", "description": "Start or control enough long-running processes to deserve the title.", "category": "Agent Autonomy", "kind": "lifetime", "icon": "daemon", "threshold_metric": "total_process_calls", "tiers": tiers([300, 800, 2000, 6000, 15000])},
    {"id": "cron_necromancer", "name": "Cron Necromancer", "description": "Raise scheduled autonomous jobs from the dead.", "category": "Agent Autonomy", "kind": "lifetime", "icon": "clock", "threshold_metric": "total_cron_calls", "tiers": tiers([1000, 3000, 8000, 20000, 50000])},

    # Debugging Chaos — higher thresholds + multi-condition events
    {"id": "red_text_connoisseur", "name": "Red Text Connoisseur", "description": "Encounter enough errors to develop a palate for red text.", "category": "Debugging Chaos", "kind": "lifetime", "icon": "warning", "threshold_metric": "total_errors", "tiers": tiers([1500, 4000, 10000, 25000, 75000])},
    {"id": "stack_trace_sommelier", "name": "Stack Trace Sommelier", "description": "Taste tracebacks by the flight, not by the sip.", "category": "Debugging Chaos", "kind": "lifetime", "icon": "wine", "threshold_metric": "traceback_events", "tiers": tiers([300, 1000, 3000, 8000, 20000])},
    {"id": "actually_read_the_logs", "name": "Actually Read The Logs", "description": "Inspect logs repeatedly instead of guessing.", "category": "Debugging Chaos", "kind": "lifetime", "icon": "scroll", "threshold_metric": "log_read_events", "tiers": tiers([1000, 3000, 8000, 20000, 50000])},
    {"id": "port_3000_taken", "name": "Port 3000 Is Taken", "description": "Discover dev-server port conflict patterns enough times to become numb.", "category": "Debugging Chaos", "kind": "lifetime", "icon": "plug", "secret": True, "threshold_metric": "port_conflict_events", "tiers": tiers([15, 40, 100, 300, 1000])},
    {"id": "permission_denied_any_percent", "name": "Permission Denied Any%", "description": "Speedrun into permission walls.", "category": "Debugging Chaos", "kind": "lifetime", "icon": "lock", "secret": True, "threshold_metric": "permission_denied_events", "tiers": tiers([25, 75, 200, 600, 1500])},
    {"id": "dependency_hell_tourist", "name": "Dependency Hell Tourist", "description": "Package installs fail, then somehow life continues.", "category": "Debugging Chaos", "kind": "multi_condition", "icon": "package_skull", "requirements": [req("install_error_events", 25), req("install_success_events", 10)]},
    {"id": "the_fix_was_restarting", "name": "The Fix Was Restarting It", "description": "Restart after enough error clusters to call it a technique.", "category": "Debugging Chaos", "kind": "multi_condition", "icon": "restart", "requirements": [req("restart_after_error_events", 50), req("total_errors", 4000)]},
    {"id": "forgot_the_env_var", "name": "Forgot The Env Var", "description": "Auth or configuration failed because an environment variable was missing.", "category": "Debugging Chaos", "kind": "lifetime", "icon": "key", "secret": True, "threshold_metric": "env_var_error_events", "tiers": tiers([5000, 15000, 40000, 100000, 250000])},
    {"id": "yaml_colon_incident", "name": "YAML Colon Incident", "description": "Configuration syntax bites back.", "category": "Debugging Chaos", "kind": "lifetime", "icon": "colon", "secret": True, "threshold_metric": "yaml_error_events", "tiers": tiers([1000, 3000, 8000, 20000, 50000])},
    {"id": "docker_name_collision", "name": "Docker Name Collision", "description": "A container name already exists. Of course it does.", "category": "Debugging Chaos", "kind": "lifetime", "icon": "container", "secret": True, "threshold_metric": "docker_conflict_events", "tiers": tiers([75, 200, 600, 1500, 4000])},

    # Vibe Coding
    {"id": "supposed_to_be_quick", "name": "This Was Supposed To Be Quick", "description": "A tiny ask becomes an entire expedition.", "category": "Vibe Coding", "kind": "best_session", "icon": "melting_clock", "threshold_metric": "max_messages_in_session", "tiers": tiers([300, 600, 1200, 2500, 6000])},
    {"id": "one_more_small_change", "name": "One More Small Change", "description": "Make enough file edits in one session to invalidate the phrase small change.", "category": "Vibe Coding", "kind": "best_session", "icon": "pencil", "threshold_metric": "max_file_tool_calls_in_session", "tiers": tiers([150, 400, 1000, 3000, 8000])},
    {"id": "vibe_architect", "name": "Vibe Architect", "description": "Touch a broad surface area in one project session.", "category": "Vibe Coding", "kind": "best_session", "icon": "blueprint", "threshold_metric": "max_files_touched_in_session", "tiers": tiers([300, 700, 1500, 4000, 10000])},
    {"id": "pixel_goblin", "name": "Pixel Goblin", "description": "Do sustained frontend, CSS, SVG, or visual tuning.", "category": "Vibe Coding", "kind": "lifetime", "icon": "pixel", "threshold_metric": "frontend_activity_events", "tiers": tiers([20000, 50000, 120000, 300000, 800000])},
    {"id": "ship_first_ask_later", "name": "Ship First, Ask Later", "description": "Git activity after a serious tool chain.", "category": "Vibe Coding", "kind": "multi_condition", "icon": "ship", "requirements": [req("git_events", 50), req("max_tool_calls_in_session", 500)]},
    {"id": "css_exorcist", "name": "CSS Exorcist", "description": "Cast repeated styling demons out of the interface.", "category": "Vibe Coding", "kind": "lifetime", "icon": "spark_cursor", "threshold_metric": "css_activity_events", "tiers": tiers([10000, 30000, 80000, 200000, 500000])},
    {"id": "one_character_fix", "name": "One Character Fix", "description": "A tiny edit after a pile of errors. Painful. Beautiful.", "category": "Vibe Coding", "kind": "multi_condition", "icon": "needle", "secret": True, "requirements": [req("tiny_patch_after_errors_events", 5), req("total_errors", 4000)]},

    # Hermes Native
    {"id": "skillsmith", "name": "Skillsmith", "description": "Work with Hermes skills enough to leave fingerprints.", "category": "Hermes Native", "kind": "lifetime", "icon": "hammer_scroll", "threshold_metric": "skill_events", "tiers": tiers([5000, 15000, 40000, 100000, 250000])},
    {"id": "skill_issue_skill_created", "name": "Skill Issue? Skill Created.", "description": "Create or patch durable procedures instead of repeating yourself.", "category": "Hermes Native", "kind": "lifetime", "icon": "anvil", "threshold_metric": "skill_manage_events", "tiers": tiers([25, 75, 200, 600, 1500])},
    {"id": "memory_keeper", "name": "Memory Keeper", "description": "Persist durable knowledge with memory or Mnemosyne.", "category": "Hermes Native", "kind": "lifetime", "icon": "crystal", "threshold_metric": "memory_events", "tiers": tiers([100, 300, 1000, 3000, 8000])},
    {"id": "memory_palace", "name": "Memory Palace", "description": "Build a serious durable-memory trail.", "category": "Hermes Native", "kind": "lifetime", "icon": "palace", "threshold_metric": "memory_write_events", "tiers": tiers([100, 300, 1000, 3000, 8000])},
    {"id": "context_dragon", "name": "Context Dragon", "description": "Brush against compression, huge context, or token pressure repeatedly.", "category": "Hermes Native", "kind": "lifetime", "icon": "dragon", "threshold_metric": "context_events", "tiers": tiers([5000, 15000, 40000, 100000, 250000])},
    {"id": "gateway_dweller", "name": "Gateway Dweller", "description": "Live through gateway-connected Hermes workflows.", "category": "Hermes Native", "kind": "lifetime", "icon": "antenna", "threshold_metric": "gateway_events", "tiers": tiers([5000, 15000, 40000, 100000, 250000])},
    {"id": "plugin_goblin", "name": "Plugin Goblin", "description": "Use or develop plugins enough that the dashboard notices.", "category": "Hermes Native", "kind": "lifetime", "icon": "puzzle", "threshold_metric": "plugin_events", "tiers": tiers([1000, 3000, 8000, 20000, 50000])},
    {"id": "rollback_wizard", "name": "Rollback Wizard", "description": "Invoke rollback/checkpoint recovery magic.", "category": "Hermes Native", "kind": "lifetime", "icon": "rewind", "secret": True, "threshold_metric": "rollback_events", "tiers": tiers([500, 1500, 4000, 10000, 25000])},

    # Research/Web
    {"id": "rabbit_hole_certified", "name": "Rabbit Hole Certified", "description": "Search or extract enough web content to qualify as a research spiral.", "category": "Research/Web", "kind": "lifetime", "icon": "spiral", "threshold_metric": "total_web_calls", "tiers": tiers([400, 1200, 3000, 8000, 20000])},
    {"id": "citation_goblin", "name": "Citation Goblin", "description": "Extract enough web pages to become a tiny librarian.", "category": "Research/Web", "kind": "lifetime", "icon": "quote", "threshold_metric": "total_web_extract_calls", "tiers": tiers([100, 300, 1000, 3000, 8000])},
    {"id": "docs_archaeologist", "name": "Docs Archaeologist", "description": "Dig through documentation sources over and over.", "category": "Research/Web", "kind": "lifetime", "icon": "compass", "threshold_metric": "docs_activity_events", "tiers": tiers([5000, 15000, 40000, 100000, 250000])},
    {"id": "browser_possession", "name": "Browser Possession", "description": "Possess a browser through automation repeatedly.", "category": "Research/Web", "kind": "lifetime", "icon": "browser", "threshold_metric": "browser_calls", "tiers": tiers([75, 200, 600, 1500, 4000])},

    # Tool Mastery
    {"id": "terminal_goblin", "name": "Terminal Goblin", "description": "Spend serious time in shell-land.", "category": "Tool Mastery", "kind": "lifetime", "icon": "terminal", "threshold_metric": "total_terminal_calls", "tiers": tiers([750, 2000, 6000, 15000, 50000])},
    {"id": "patch_wizard", "name": "Patch Wizard", "description": "Bend files to your will with targeted patches.", "category": "Tool Mastery", "kind": "lifetime", "icon": "wand", "threshold_metric": "total_patch_calls", "tiers": tiers([250, 750, 2000, 6000, 15000])},
    {"id": "file_archaeologist", "name": "File Archaeologist", "description": "Dig through the filesystem with reads and searches.", "category": "Tool Mastery", "kind": "lifetime", "icon": "folder", "threshold_metric": "total_file_reads_searches", "tiers": tiers([750, 2000, 6000, 15000, 50000])},
    {"id": "image_whisperer", "name": "Image Whisperer", "description": "Use image generation or vision tools enough for visual work.", "category": "Tool Mastery", "kind": "lifetime", "icon": "eye", "threshold_metric": "image_vision_calls", "tiers": tiers([100, 300, 1000, 3000, 8000])},
    {"id": "voice_of_the_machine", "name": "Voice Of The Machine", "description": "Use text-to-speech or voice tooling repeatedly.", "category": "Tool Mastery", "kind": "lifetime", "icon": "wave", "threshold_metric": "tts_calls", "tiers": tiers([10, 30, 100, 300, 800])},

    # Model Lore
    {"id": "model_hopper", "name": "Model Hopper", "description": "Switch or inspect providers/models enough to count as a habit.", "category": "Model Lore", "kind": "lifetime", "icon": "swap", "threshold_metric": "model_events", "tiers": tiers([10000, 30000, 80000, 200000, 500000])},
    {"id": "openrouter_enjoyer", "name": "OpenRouter Enjoyer", "description": "Route model work through OpenRouter repeatedly.", "category": "Model Lore", "kind": "lifetime", "icon": "router", "threshold_metric": "openrouter_events", "tiers": tiers([250, 750, 2000, 6000, 15000])},
    {"id": "codex_conjurer", "name": "Codex Conjurer", "description": "Summon Codex-flavored assistance often enough for a ritual.", "category": "Model Lore", "kind": "lifetime", "icon": "codex", "threshold_metric": "codex_events", "tiers": tiers([500, 1500, 4000, 10000, 25000])},
    {"id": "multi_model_mage", "name": "Multi-Model Mage", "description": "Use a real spread of distinct model names across Hermes history.", "category": "Model Lore", "kind": "lifetime", "icon": "prism", "threshold_metric": "distinct_model_count", "tiers": tiers([10, 20, 40, 80, 160])},
    {"id": "five_model_flight", "name": "Five-Model Flight", "description": "Try at least five distinct LLMs instead of marrying the first model that answers.", "category": "Model Lore", "kind": "lifetime", "icon": "prism", "threshold_metric": "distinct_model_count", "tiers": tiers([5, 10, 20, 40, 80])},
    {"id": "provider_polyglot", "name": "Provider Polyglot", "description": "Use models from multiple providers across Hermes history.", "category": "Model Lore", "kind": "lifetime", "icon": "swap", "threshold_metric": "distinct_provider_count", "tiers": tiers([2, 3, 5, 8, 12])},
    {"id": "model_sommelier", "name": "Model Sommelier", "description": "Taste enough model/provider conversations to develop preferences.", "category": "Model Lore", "kind": "lifetime", "icon": "wine", "threshold_metric": "model_events", "tiers": tiers([250, 750, 2000, 6000, 15000])},
    {"id": "claude_confidant", "name": "Claude Confidant", "description": "Bring Claude-flavored reasoning into the workflow repeatedly.", "category": "Model Lore", "kind": "lifetime", "icon": "quote", "threshold_metric": "claude_events", "tiers": tiers([50, 150, 500, 1500, 4000])},
    {"id": "gemini_cartographer", "name": "Gemini Cartographer", "description": "Map enough Gemini-related workflows to know the terrain.", "category": "Model Lore", "kind": "lifetime", "icon": "compass", "threshold_metric": "gemini_events", "tiers": tiers([50, 150, 500, 1500, 4000])},
    {"id": "open_weights_pilgrim", "name": "Open Weights Pilgrim", "description": "Actually chat with local/open-weight models through Hermes session metadata.", "category": "Model Lore", "kind": "lifetime", "icon": "terminal", "threshold_metric": "local_model_chat_sessions", "tiers": tiers([1, 3, 10, 30, 100])},

    # Workflow Intelligence
    {"id": "toolset_cartographer", "name": "Toolset Cartographer", "description": "Navigate Hermes toolsets deliberately instead of treating tools as a blur.", "category": "Hermes Native", "kind": "lifetime", "icon": "compass", "threshold_metric": "toolset_events", "tiers": tiers([20, 60, 200, 600, 1500])},
    {"id": "config_surgeon", "name": "Config Surgeon", "description": "Operate on real config files, manifests, env files, and dashboard settings without flinching.", "category": "Hermes Native", "kind": "lifetime", "icon": "key", "threshold_metric": "config_events", "tiers": tiers([100, 300, 1000, 3000, 10000])},
    {"id": "rebase_acrobat", "name": "Rebase Acrobat", "description": "Handle real git history surgery: rebase, conflict, merge, fetch, push.", "category": "Vibe Coding", "kind": "lifetime", "icon": "branch", "threshold_metric": "git_history_events", "tiers": tiers([10, 30, 100, 300, 800])},
    {"id": "test_suite_tamer", "name": "Test Suite Tamer", "description": "Run enough verification commands that green text becomes part of the ritual.", "category": "Tool Mastery", "kind": "lifetime", "icon": "daemon", "threshold_metric": "test_events", "tiers": tiers([100, 300, 800, 2400, 6000])},
    {"id": "screenshot_hunter", "name": "Screenshot Hunter", "description": "Capture, inspect, and polish visual proof instead of just claiming it works.", "category": "Tool Mastery", "kind": "lifetime", "icon": "eye", "threshold_metric": "screenshot_events", "tiers": tiers([50, 150, 500, 1500, 5000])},

    # Lifestyle
    {"id": "marathon_operator", "name": "Marathon Operator", "description": "Accumulate a serious number of Hermes sessions.", "category": "Lifestyle", "kind": "lifetime", "icon": "marathon", "threshold_metric": "session_count", "tiers": tiers([75, 200, 500, 1500, 5000])},
    {"id": "weekend_warrior", "name": "Weekend Warrior", "description": "Run Hermes on weekends enough times to make it a lifestyle.", "category": "Lifestyle", "kind": "lifetime", "icon": "calendar", "threshold_metric": "weekend_sessions", "tiers": tiers([25, 75, 200, 600, 1500])},
    {"id": "night_shift_operator", "name": "Night Shift Operator", "description": "Run sessions during gremlin hours repeatedly.", "category": "Lifestyle", "kind": "lifetime", "icon": "moon", "threshold_metric": "night_sessions", "tiers": tiers([25, 75, 200, 600, 1500])},
    {"id": "cache_hit_appreciator", "name": "Cache Hit Appreciator", "description": "Notice or benefit from prompt/cache behavior.", "category": "Lifestyle", "kind": "lifetime", "icon": "cache", "secret": True, "threshold_metric": "cache_events", "tiers": tiers([100, 300, 1000, 3000, 8000])},
]


def state_path() -> Path:
    return get_hermes_home() / "plugins" / "hermes-achievements" / "state.json"


# ---------------------------------------------------------------------------
# Chinese (zh) translations for achievement names, descriptions, and categories.
# The frontend reads *_zh fields when the dashboard locale is set to "zh".
# ---------------------------------------------------------------------------

CATEGORY_ZH: Dict[str, str] = {
    "Agent Autonomy": "智能体自主",
    "Debugging Chaos": "调试混沌",
    "Vibe Coding": "氛围编程",
    "Hermes Native": "Hermes 原生",
    "Research/Web": "研究/网络",
    "Tool Mastery": "工具精通",
    "Model Lore": "模型传说",
    "Lifestyle": "生活方式",
}

ACHIEVEMENT_ZH: Dict[str, Dict[str, str]] = {
    # Agent Autonomy
    "let_him_cook": {"name": "放手去做", "description": "让 Hermes 在单次会话中运行一整套严肃的自主工具链。"},
    "autonomous_avalanche": {"name": "自主雪崩", "description": "跨会话累积海量 Hermes 工具调用。"},
    "toolchain_maxxer": {"name": "工具链达人", "description": "在单次会话中使用大量不同的 Hermes 工具。"},
    "full_send": {"name": "全力输出", "description": "终端、文件和网页/浏览器在同一轮任务中全部上阵。"},
    "subagent_commander": {"name": "子任务指挥官", "description": "协调委派给子智能体的工作。"},
    "background_process_enjoyer": {"name": "后台进程爱好者", "description": "启动或控制足够多的长时间运行进程。"},
    "cron_necromancer": {"name": "定时任务亡灵法师", "description": "唤醒足够多的定时自动任务。"},
    # Debugging Chaos
    "red_text_connoisseur": {"name": "红色文本鉴赏家", "description": "遇到足够多的错误，培养出对红色文本的品味。"},
    "stack_trace_sommelier": {"name": "堆栈追踪侍酒师", "description": "不是小口品尝异常，而是一杯杯地品味堆栈追踪。"},
    "actually_read_the_logs": {"name": "居然去读日志了", "description": "反复检查日志而不是靠猜测。"},
    "port_3000_taken": {"name": "3000 端口被占了", "description": "发现开发服务器端口冲突的次数多到麻木。"},
    "permission_denied_any_percent": {"name": "权限拒绝速通", "description": "速通撞上权限墙。"},
    "dependency_hell_tourist": {"name": "依赖地狱游客", "description": "包安装失败了，然后生活还得继续。"},
    "the_fix_was_restarting": {"name": "修好了，靠重启", "description": "在足够多的错误聚集后重启，把它变成一种技巧。"},
    "forgot_the_env_var": {"name": "忘了设环境变量", "description": "因为缺少环境变量导致认证或配置失败。"},
    "yaml_colon_incident": {"name": "YAML 冒号事件", "description": "配置语法反咬一口。"},
    "docker_name_collision": {"name": "Docker 命名冲突", "description": "容器名已存在。当然了。"},
    # Vibe Coding
    "supposed_to_be_quick": {"name": "说好的很快搞定", "description": "一个小需求变成了一整场远征。"},
    "one_more_small_change": {"name": "再改一个小地方", "description": "在单次会话中做了足够多的文件编辑，让'小改动'这个词失去意义。"},
    "vibe_architect": {"name": "氛围架构师", "description": "在一次项目会话中触及足够广的代码范围。"},
    "pixel_goblin": {"name": "像素小妖精", "description": "持续进行前端、CSS、SVG 或视觉微调。"},
    "ship_first_ask_later": {"name": "先发布，后提问", "description": "在一轮严肃的工具链操作后有 Git 活动。"},
    "css_exorcist": {"name": "CSS 驱魔师", "description": "反复从界面中驱除样式恶魔。"},
    "one_character_fix": {"name": "一个字符的修复", "description": "在一堆错误之后做了一个微小的编辑。痛苦。优美。"},
    # Hermes Native
    "skillsmith": {"name": "技能铁匠", "description": "与 Hermes 技能打交道多到留下指纹。"},
    "skill_issue_skill_created": {"name": "技能不足？技能创建。", "description": "创建或修补持久化流程，而不是重复自己。"},
    "memory_keeper": {"name": "记忆守护者", "description": "用 memory 或 Mnemosyne 持久化知识。"},
    "memory_palace": {"name": "记忆宫殿", "description": "建立一条严肃的持久记忆链。"},
    "context_dragon": {"name": "上下文巨龙", "description": "反复接触上下文压缩、巨大上下文或 token 压力。"},
    "gateway_dweller": {"name": "网关居民", "description": "经历足够多的网关连接的 Hermes 工作流。"},
    "plugin_goblin": {"name": "插件小精灵", "description": "使用或开发插件多到被仪表盘注意到。"},
    "rollback_wizard": {"name": "回滚巫师", "description": "施展回滚/检查点恢复魔法。"},
    "toolset_cartographer": {"name": "工具集制图师", "description": "有目的地导航 Hermes 工具集，而不是把工具当一团模糊的东西。"},
    "config_surgeon": {"name": "配置外科医生", "description": "对真实的配置文件、清单、环境文件和仪表盘设置动手术，毫不退缩。"},
    # Research/Web
    "rabbit_hole_certified": {"name": "兔子洞认证", "description": "搜索或提取足够多的网页内容，有资格称为一次研究漩涡。"},
    "citation_goblin": {"name": "引用小精灵", "description": "提取足够多的网页，成为一个小图书管理员。"},
    "docs_archaeologist": {"name": "文档考古学家", "description": "反复挖掘文档源。"},
    "browser_possession": {"name": "浏览器附身", "description": "通过自动化反复操控浏览器。"},
    # Tool Mastery
    "terminal_goblin": {"name": "终端小妖精", "description": "在 Shell 世界中投入大量时间。"},
    "patch_wizard": {"name": "补丁巫师", "description": "用精准的补丁让文件服从你的意志。"},
    "file_archaeologist": {"name": "文件考古学家", "description": "通过读取和搜索反复挖掘文件系统。"},
    "image_whisperer": {"name": "图像低语者", "description": "使用图像生成或视觉工具进行足够多的视觉工作。"},
    "voice_of_the_machine": {"name": "机器之声", "description": "反复使用文字转语音或语音工具。"},
    "test_suite_tamer": {"name": "测试套件驯服者", "description": "运行足够多的验证命令，让绿色文本成为仪式的一部分。"},
    "screenshot_hunter": {"name": "截图猎人", "description": "捕获、检查和打磨视觉证据，而不是只说'能用'。"},
    # Model Lore
    "model_hopper": {"name": "模型跳跃者", "description": "切换或检查 provider/模型的次数多到成为习惯。"},
    "openrouter_enjoyer": {"name": "OpenRouter 爱好者", "description": "反复通过 OpenRouter 路由模型工作。"},
    "codex_conjurer": {"name": "Codex 召唤师", "description": "召唤 Codex 风格的辅助多到像在做仪式。"},
    "multi_model_mage": {"name": "多模型法师", "description": "在 Hermes 历史中使用过大量不同的模型名称。"},
    "five_model_flight": {"name": "五模型飞行", "description": "尝试至少五个不同的 LLM，而不是跟第一个回复的模型结婚。"},
    "provider_polyglot": {"name": "Provider 多语者", "description": "在 Hermes 历史中使用过多个 provider 的模型。"},
    "model_sommelier": {"name": "模型侍酒师", "description": "品尝足够多的模型/provider 对话，培养出偏好。"},
    "claude_confidant": {"name": "Claude 知己", "description": "反复将 Claude 风格的推理引入工作流。"},
    "gemini_cartographer": {"name": "Gemini 制图师", "description": "绘制足够多的 Gemini 相关工作流，了解这片地形。"},
    "open_weights_pilgrim": {"name": "开放权重朝圣者", "description": "通过 Hermes 会话元数据与本地/开放权重模型交谈。"},
    "rebase_acrobat": {"name": "变基杂技演员", "description": "处理真实的 Git 历史手术：变基、冲突、合并、拉取、推送。"},
    # Lifestyle
    "marathon_operator": {"name": "马拉松操作员", "description": "累积大量 Hermes 会话。"},
    "weekend_warrior": {"name": "周末战士", "description": "在周末运行 Hermes 的次数多到成为一种生活方式。"},
    "night_shift_operator": {"name": "夜班操作员", "description": "在深夜时段反复运行会话。"},
    "cache_hit_appreciator": {"name": "缓存命中鉴赏家", "description": "注意到或受益于 prompt/缓存行为。"},
}

# Secret achievement reveal text (zh)
SECRET_REVEAL_ZH = "隐藏成就：在 Hermes 检测到你会话历史中的第一个相关行为之前保持隐藏。"


def snapshot_path() -> Path:
    return get_hermes_home() / "plugins" / "hermes-achievements" / "scan_snapshot.json"


def checkpoint_path() -> Path:
    return get_hermes_home() / "plugins" / "hermes-achievements" / "scan_checkpoint.json"


def load_state() -> Dict[str, Any]:
    path = state_path()
    if not path.exists():
        return {"unlocks": {}}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"unlocks": {}}


def save_state(state: Dict[str, Any]) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    return value


def load_snapshot() -> Optional[Dict[str, Any]]:
    path = snapshot_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def save_snapshot(data: Dict[str, Any]) -> None:
    path = snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True))


def load_checkpoint() -> Dict[str, Any]:
    path = checkpoint_path()
    if not path.exists():
        return {"schema_version": 1, "generated_at": 0, "sessions": {}}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data.setdefault("schema_version", 1)
            data.setdefault("generated_at", 0)
            data.setdefault("sessions", {})
            if isinstance(data.get("sessions"), dict):
                return data
    except Exception:
        pass
    return {"schema_version": 1, "generated_at": 0, "sessions": {}}


def save_checkpoint(data: Dict[str, Any]) -> None:
    path = checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True))


def session_fingerprint(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "last_active": meta.get("last_active"),
        "started_at": meta.get("started_at"),
        "model": meta.get("model"),
        "title": meta.get("title") or meta.get("preview") or "Untitled",
    }


def _cache_is_fresh(now: int) -> bool:
    return _SNAPSHOT_CACHE is not None and (now - _SNAPSHOT_CACHE_AT) <= SNAPSHOT_TTL_SECONDS


def _is_snapshot_stale(snapshot: Optional[Dict[str, Any]], now: Optional[int] = None) -> bool:
    if not isinstance(snapshot, dict):
        return True
    ts = int(snapshot.get("generated_at") or 0)
    current = int(now or time.time())
    if ts <= 0:
        return True
    return (current - ts) > SNAPSHOT_TTL_SECONDS


def _scan_status_payload(now: Optional[int] = None) -> Dict[str, Any]:
    current = int(now or time.time())
    snap = _SNAPSHOT_CACHE if isinstance(_SNAPSHOT_CACHE, dict) else None
    generated_at = int((snap or {}).get("generated_at") or 0) if snap else 0
    return {
        "state": _SCAN_STATUS.get("state", "idle"),
        "started_at": _SCAN_STATUS.get("started_at"),
        "finished_at": _SCAN_STATUS.get("finished_at"),
        "last_error": _SCAN_STATUS.get("last_error"),
        "last_duration_ms": _SCAN_STATUS.get("last_duration_ms"),
        "run_count": _SCAN_STATUS.get("run_count", 0),
        "ttl_seconds": SNAPSHOT_TTL_SECONDS,
        "snapshot_generated_at": generated_at or None,
        "snapshot_age_seconds": (current - generated_at) if generated_at else None,
        "snapshot_stale": _is_snapshot_stale(snap, current),
    }


def _tool_name_from_call(call: Any) -> Optional[str]:
    if not isinstance(call, dict):
        return None
    fn = call.get("function") or {}
    return call.get("name") or fn.get("name")


def _content(msg: Dict[str, Any]) -> str:
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content)
    except Exception:
        return str(content)


def _count_tool(tool_names: List[str], *needles: str) -> int:
    lowered = [name.lower() for name in tool_names]
    return sum(1 for name in lowered if any(needle in name for needle in needles))


def model_provider(model_name: str) -> Optional[str]:
    name = (model_name or "").strip().lower()
    if not name or name == "none":
        return None
    if "/" in name:
        return name.split("/", 1)[0]
    for provider in ["openai", "anthropic", "google", "gemini", "mistral", "meta", "qwen", "deepseek", "xai", "nous", "ollama", "groq", "openrouter", "codex"]:
        if provider in name:
            return "google" if provider == "gemini" else provider
    return name.split(":", 1)[0].split("-", 1)[0]


def is_local_model_name(model_name: str) -> bool:
    name = (model_name or "").strip().lower()
    if not name or name == "none":
        return False
    local_markers = ["ollama", "llama.cpp", "localhost", "127.0.0.1", "local/", "local:", "gguf", "vllm-local"]
    return any(marker in name for marker in local_markers)


def analyze_messages(session_id: str, title: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    tool_names: Set[str] = set()
    tool_sequence: List[str] = []
    files_touched: Set[str] = set()
    full_text_parts: List[str] = []
    error_count = 0

    for msg in messages:
        text = _content(msg)
        full_text_parts.append(text)
        if msg.get("tool_name"):
            name = str(msg["tool_name"])
            tool_names.add(name)
            # Tool result rows name the tool that already appeared in the assistant tool_calls.
            # Keep it for distinct-tool detection, but do not double-count it as a new call.
            if msg.get("role") != "tool":
                tool_sequence.append(name)
        for call in msg.get("tool_calls") or []:
            name = _tool_name_from_call(call)
            if name:
                tool_names.add(name)
                tool_sequence.append(name)
        if ERROR_RE.search(text):
            error_count += 1
        blob = text
        if msg.get("tool_calls"):
            blob += " " + json.dumps(msg.get("tool_calls"), default=str)
        files_touched.update(FILE_RE.findall(blob))

    full_text = "\n".join(full_text_parts)
    lower = full_text.lower()
    terminal_calls = _count_tool(tool_sequence, "terminal")
    web_calls = _count_tool(tool_sequence, "web_search", "web_extract")
    web_extract_calls = _count_tool(tool_sequence, "web_extract")
    browser_calls = _count_tool(tool_sequence, "browser")
    web_browser_calls = web_calls + browser_calls
    patch_calls = _count_tool(tool_sequence, "patch")
    file_reads_searches = _count_tool(tool_sequence, "read_file", "search_files")
    file_tool_calls = _count_tool(tool_sequence, "read_file", "write_file", "patch", "search_files")
    delegate_calls = _count_tool(tool_sequence, "delegate_task")
    process_calls = _count_tool(tool_sequence, "process") + len(re.findall(r"background\s*=\s*true", full_text, re.I))
    cron_calls = _count_tool(tool_sequence, "cronjob")
    image_vision_calls = _count_tool(tool_sequence, "image", "vision")
    tts_calls = _count_tool(tool_sequence, "tts", "text_to_speech")
    skill_events = _count_tool(tool_sequence, "skill") + len(re.findall(r"\bskill", lower))
    skill_manage_events = _count_tool(tool_sequence, "skill_manage")
    memory_events = _count_tool(tool_sequence, "memory", "mnemosyne")
    memory_write_events = _count_tool(tool_sequence, "mnemosyne_remember", "memory")

    return {
        "session_id": session_id,
        "title": title or "Untitled session",
        "message_count": len(messages),
        "tool_call_count": len(tool_sequence),
        "tool_names": tool_names,
        "distinct_tool_count": len(tool_names),
        "error_count": error_count,
        "terminal_calls": terminal_calls,
        "web_calls": web_calls,
        "web_extract_calls": web_extract_calls,
        "browser_calls": browser_calls,
        "web_browser_calls": web_browser_calls,
        "patch_calls": patch_calls,
        "file_reads_searches": file_reads_searches,
        "file_tool_calls": file_tool_calls,
        "files_touched_count": len(files_touched),
        "delegate_calls": delegate_calls,
        "process_calls": process_calls,
        "cron_calls": cron_calls,
        "image_vision_calls": image_vision_calls,
        "tts_calls": tts_calls,
        "skill_events": skill_events,
        "skill_manage_events": skill_manage_events,
        "memory_events": memory_events,
        "memory_write_events": memory_write_events,
        "port_conflict": bool(PORT_RE.search(full_text)),
        "port_conflict_events": 1 if PORT_RE.search(full_text) else 0,
        "traceback_events": len(re.findall(r"traceback|exception", full_text, re.I)),
        "log_read_events": len(re.findall(r"gateway\.log|errors\.log|agent\.log|/api/logs|\blogs\b", full_text, re.I)),
        "permission_denied_events": len(re.findall(r"permission denied|eacces|operation not permitted", full_text, re.I)),
        "install_error_events": 1 if INSTALL_RE.search(full_text) and ERROR_RE.search(full_text) else 0,
        "install_success_events": 1 if INSTALL_RE.search(full_text) and SUCCESS_RE.search(full_text) else 0,
        "restart_after_error_events": 1 if error_count and re.search(r"\brestart|reload|kill|start\b", full_text, re.I) else 0,
        "env_var_error_events": len(re.findall(r"missing .*env|api key|environment variable|not configured|unauthorized|auth", full_text, re.I)),
        "yaml_error_events": len(re.findall(r"yaml|yml|colon|parse error", full_text, re.I)) if ERROR_RE.search(full_text) else 0,
        "docker_conflict_events": len(re.findall(r"docker.*(name|container).*already|container name conflict|Conflict\. The container", full_text, re.I)),
        "frontend_activity_events": len(re.findall(r"\.(css|svg|tsx|jsx)|frontend|tailwind|react", full_text, re.I)),
        "css_activity_events": len(re.findall(r"\.css|tailwind|style|className|visual", full_text, re.I)),
        "git_events": len(re.findall(r"\bgit\s+(commit|push|merge|rebase|status|diff)", full_text, re.I)),
        "tiny_patch_after_errors_events": 1 if error_count >= 5 and re.search(r"one character|single character|typo", full_text, re.I) else 0,
        "context_events": len(re.findall(r"compress|context window|token|cache", full_text, re.I)),
        "gateway_events": len(re.findall(r"gateway|discord|telegram|slack|api_server", full_text, re.I)),
        "plugin_events": len(re.findall(r"plugin|dashboard-plugins|__HERMES_PLUGIN|manifest\.json", full_text, re.I)),
        "rollback_events": len(re.findall(r"rollback|checkpoint", full_text, re.I)),
        "docs_activity_events": len(re.findall(r"docs|documentation|docusaurus|README", full_text, re.I)),
        "model_events": len(re.findall(r"model|provider|openrouter|codex|gemini|claude|anthropic|openai|mistral|qwen|deepseek|llama|ollama|vllm|gguf", full_text, re.I)),
        "openrouter_events": len(re.findall(r"openrouter", full_text, re.I)),
        "codex_events": len(re.findall(r"codex", full_text, re.I)),
        "claude_events": len(re.findall(r"claude|anthropic", full_text, re.I)),
        "gemini_events": len(re.findall(r"gemini|google ai|google model", full_text, re.I)),
        "local_model_events": len(re.findall(r"ollama|llama\.cpp|gguf|vllm|local model|open[- ]weight|open weights", full_text, re.I)),
        "toolset_events": len(re.findall(r"toolset|enabled_toolsets|browser tool|terminal tool|file tool|web tool", full_text, re.I)),
        "config_events": len(re.findall(r"config\.ya?ml|\b[a-z0-9_-]+config\.(?:js|ts|json|ya?ml)|\.env(?:\b|\.)|manifest\.json|settings\.json|pyproject\.toml|package\.json", full_text, re.I)),
        "git_history_events": len(re.findall(r"\bgit\s+(rebase|merge|fetch|pull|push|tag|checkout)|merge conflict|conflict\s*\(|rebase --continue", full_text, re.I)),
        "test_events": len(re.findall(r"pytest|unittest|vitest|playwright|npm test|pnpm test|node --check|py_compile|tests? passed|\bOK\b", full_text, re.I)),
        "screenshot_events": len(re.findall(r"screenshot|playwright|vision_analyze|browser_vision|\.png|image data", full_text, re.I)),
        "release_events": len(re.findall(r"\bgit\s+tag|release|version bump|changelog|publish|pushed? tag", full_text, re.I)),
        "cache_events": len(re.findall(r"cache hit|prompt caching|cache_read", full_text, re.I)),
        "model_names": set(),
    }


def evaluate_tiered(definition: Dict[str, Any], aggregate: Dict[str, Any]) -> Dict[str, Any]:
    metric = definition["threshold_metric"]
    progress = int(aggregate.get(metric, 0) or 0)
    tiers_list = sorted(definition.get("tiers", []), key=lambda t: t["threshold"])
    achieved = [t for t in tiers_list if progress >= t["threshold"]]
    next_tiers = [t for t in tiers_list if progress < t["threshold"]]
    tier = achieved[-1]["name"] if achieved else None
    next_tier = next_tiers[0]["name"] if next_tiers else None
    next_threshold = next_tiers[0]["threshold"] if next_tiers else (tiers_list[-1]["threshold"] if tiers_list else 1)
    current_threshold = achieved[-1]["threshold"] if achieved else 0
    denom = max(1, next_threshold - current_threshold)
    pct = 100 if not next_tiers and achieved else max(0, min(99, math.floor(((progress - current_threshold) / denom) * 100)))
    unlocked = bool(achieved)
    discovered = bool(progress > 0)
    state = "unlocked" if unlocked else ("secret" if definition.get("secret") and not discovered else "discovered")
    return {"unlocked": unlocked, "discovered": discovered or not definition.get("secret"), "state": state, "tier": tier, "progress": progress, "next_tier": next_tier, "next_threshold": next_threshold, "progress_pct": pct}


def evaluate_requirements(definition: Dict[str, Any], aggregate: Dict[str, Any]) -> Dict[str, Any]:
    requirements = definition.get("requirements", [])
    if not requirements:
        return {"unlocked": False, "discovered": not definition.get("secret"), "state": "secret" if definition.get("secret") else "discovered", "tier": None, "progress": 0, "next_tier": None, "next_threshold": 1, "progress_pct": 0}
    parts = []
    any_progress = False
    complete = True
    for requirement in requirements:
        value = int(aggregate.get(requirement["metric"], 0) or 0)
        threshold = int(requirement.get("gte", 1))
        any_progress = any_progress or value > 0
        complete = complete and value >= threshold
        parts.append(min(1.0, value / max(1, threshold)))
    pct = math.floor((sum(parts) / len(parts)) * 100)
    state = "unlocked" if complete else ("secret" if definition.get("secret") and not any_progress else "discovered")
    return {"unlocked": complete, "discovered": any_progress or not definition.get("secret"), "state": state, "tier": None, "progress": pct, "next_tier": None, "next_threshold": 100, "progress_pct": 100 if complete else min(99, pct)}


def evaluate_boolean(definition: Dict[str, Any], aggregate: Dict[str, Any]) -> Dict[str, Any]:
    # Backward-compatible helper for old tests/definitions. New catalog avoids simple booleans.
    unlocked = bool(aggregate.get(definition["metric"]))
    return {"unlocked": unlocked, "discovered": True, "state": "unlocked" if unlocked else "discovered", "tier": None, "progress": 1 if unlocked else 0, "next_tier": None, "next_threshold": 1, "progress_pct": 100 if unlocked else 0}


METRIC_LABELS = {
    "max_tool_calls_in_session": "tool calls in one session",
    "max_distinct_tools_in_session": "distinct Hermes tools used in one session",
    "max_terminal_calls_in_session": "terminal calls in one session",
    "max_file_tool_calls_in_session": "file/search/patch calls in one session",
    "max_web_browser_calls_in_session": "web search/extract or browser calls in one session",
    "max_messages_in_session": "messages in one session",
    "max_files_touched_in_session": "files touched in one session",
    "total_delegate_calls": "lifetime delegate_task calls",
    "total_process_calls": "lifetime background process operations",
    "total_cron_calls": "lifetime scheduled-job operations",
    "total_errors": "error/failed/traceback messages observed",
    "traceback_events": "traceback or exception mentions",
    "log_read_events": "log inspections",
    "port_conflict_events": "dev-server port conflict detections",
    "permission_denied_events": "permission-denied errors",
    "install_error_events": "package-install failures",
    "install_success_events": "successful package installs after package work",
    "restart_after_error_events": "restart/reload actions after error clusters",
    "env_var_error_events": "missing auth/config/environment-variable events",
    "yaml_error_events": "YAML/config parse incidents",
    "docker_conflict_events": "Docker/container-name conflicts",
    "frontend_activity_events": "frontend/CSS/SVG/React activity mentions",
    "css_activity_events": "CSS, styling, Tailwind, or className activity",
    "git_events": "git workflow commands",
    "tiny_patch_after_errors_events": "tiny typo-style fixes after error clusters",
    "skill_events": "Hermes skill mentions or tool use",
    "skill_manage_events": "skill_manage create/patch/delete operations",
    "memory_events": "memory or Mnemosyne tool events",
    "memory_write_events": "durable memory writes",
    "context_events": "context, compression, token, or cache-pressure mentions",
    "gateway_events": "gateway/API/chat-platform activity",
    "plugin_events": "dashboard plugin development or usage signals",
    "rollback_events": "rollback/checkpoint recovery mentions",
    "docs_activity_events": "documentation/README/docs activity",
    "model_events": "model/provider-related activity",
    "openrouter_events": "OpenRouter mentions",
    "codex_events": "Codex mentions",
    "cache_events": "prompt-cache/cache-hit mentions",
    "total_web_calls": "lifetime web_search/web_extract calls",
    "total_web_extract_calls": "lifetime web_extract calls",
    "browser_calls": "lifetime browser automation calls",
    "total_tool_calls": "lifetime Hermes tool calls",
    "total_terminal_calls": "lifetime terminal calls",
    "total_patch_calls": "lifetime targeted patch edits",
    "total_file_reads_searches": "lifetime read_file/search_files calls",
    "image_vision_calls": "image generation or vision tool calls",
    "tts_calls": "text-to-speech or voice tool calls",
    "distinct_model_count": "distinct model names seen in session metadata",
    "distinct_provider_count": "distinct model providers inferred from session metadata",
    "claude_events": "Claude/Anthropic model mentions",
    "gemini_events": "Gemini/Google model mentions",
    "local_model_events": "local/open-weight model mentions",
    "local_model_chat_sessions": "Hermes sessions whose model metadata is local/open-weight",
    "toolset_events": "toolset or tool-family mentions",
    "config_events": "configuration/environment/manifest activity",
    "git_history_events": "git history operations such as rebase, merge, fetch, push, or tag",
    "test_events": "test/check/verification command mentions",
    "screenshot_events": "screenshot, Playwright, PNG, or vision-inspection activity",
    "release_events": "release, version, publish, or git tag events",
    "session_count": "Hermes sessions",
    "weekend_sessions": "sessions started on weekends",
    "night_sessions": "sessions started late night or before dawn",
}


METRIC_LABELS_ZH = {
    "agent_full_loop_events": "\u81ea\u4e3b\u5faa\u73af\u4e8b\u4ef6",
    "agent_full_loop_sessions": "\u81ea\u4e3b\u5faa\u73af\u4f1a\u8bdd",
    "tool_call_events": "\u5de5\u5177\u8c03\u7528\u4e8b\u4ef6",
    "terminal_events": "\u7ec8\u7aef\u4e8b\u4ef6",
    "debugging_events": "\u8c03\u8bd5\u4e8b\u4ef6",
    "planning_events": "\u89c4\u5212\u4e8b\u4ef6",
    "bug_hunt_events": "\u6355\u86ce\u4e8b\u4ef6",
    "tiny_patch_after_errors_events": "\u9519\u8bef\u540e\u7684\u5c0f\u4fee\u590d",
    "skill_events": "\u6280\u80fd/\u5de5\u5177\u4f7f\u7528",
    "skill_manage_events": "\u6280\u80fd\u7ba1\u7406\u64cd\u4f5c",
    "memory_events": "\u8bb0\u5fc6/\u5de5\u5177\u4e8b\u4ef6",
    "memory_write_events": "\u6301\u4e45\u8bb0\u5fc6\u5199\u5165",
    "context_events": "\u4e0a\u4e0b\u6587/\u7f13\u5b58\u4e8b\u4ef6",
    "gateway_events": "\u7f51\u5173/API \u6d3b\u52a8",
    "plugin_events": "\u63d2\u4ef6\u5f00\u53d1/\u4f7f\u7528",
    "rollback_events": "\u56de\u6eda/\u68c0\u67e5\u70b9\u6062\u590d",
    "docs_activity_events": "\u6587\u6863\u6d3b\u52a8",
    "model_events": "\u6a21\u578b/\u63d0\u4f9b\u5546\u6d3b\u52a8",
    "openrouter_events": "OpenRouter \u6d3b\u52a8",
    "codex_events": "Codex \u6d3b\u52a8",
    "cache_events": "\u7f13\u5b58\u547d\u4e2d",
    "total_web_calls": "\u7f51\u7edc\u8c03\u7528\u603b\u6570",
    "total_web_extract_calls": "web_extract \u8c03\u7528\u603b\u6570",
    "browser_calls": "\u6d4f\u89c8\u5668\u81ea\u52a8\u5316\u8c03\u7528",
    "total_tool_calls": "\u5de5\u5177\u8c03\u7528\u603b\u6570",
    "total_terminal_calls": "\u7ec8\u7aef\u8c03\u7528\u603b\u6570",
    "total_patch_calls": "\u8865\u4e01\u7f16\u8f91\u603b\u6570",
    "total_file_reads_searches": "\u6587\u4ef6\u8bfb\u53d6/\u641c\u7d22\u603b\u6570",
    "image_vision_calls": "\u56fe\u50cf/\u89c6\u89c9\u5de5\u5177\u8c03\u7528",
    "tts_calls": "\u8bed\u97f3\u5408\u6210\u8c03\u7528",
    "distinct_model_count": "\u4f7f\u7528\u8fc7\u7684\u4e0d\u540c\u6a21\u578b\u6570",
    "distinct_provider_count": "\u4f7f\u7528\u8fc7\u7684\u4e0d\u540c\u63d0\u4f9b\u5546\u6570",
    "claude_events": "Claude/Anthropic \u6d3b\u52a8",
    "gemini_events": "Gemini/Google \u6d3b\u52a8",
    "local_model_events": "\u672c\u5730/\u5f00\u6e90\u6a21\u578b\u6d3b\u52a8",
    "local_model_chat_sessions": "\u672c\u5730\u6a21\u578b\u4f1a\u8bdd\u6570",
    "toolset_events": "\u5de5\u5177\u96c6\u6d3b\u52a8",
    "config_events": "\u914d\u7f6e\u6d3b\u52a8",
    "git_history_events": "git \u5386\u53f2\u64cd\u4f5c",
    "test_events": "\u6d4b\u8bd5/\u9a8c\u8bc1\u6d3b\u52a8",
    "screenshot_events": "\u622a\u56fe/\u89c6\u89c9\u68c0\u67e5",
    "release_events": "\u53d1\u5e03/\u7248\u672c\u6d3b\u52a8",
    "session_count": "Hermes \u4f1a\u8bdd\u6570",
    "weekend_sessions": "\u5468\u672b\u4f1a\u8bdd",
    "night_sessions": "\u6df1\u591c\u4f1a\u8bdd",
    "css_activity_events": "CSS/\u6837\u5f0f/Tailwind \u6d3b\u52a8",
    "docker_conflict_events": "Docker/\u5bb9\u5668\u540d\u51b2\u7a81",
    "env_var_error_events": "\u73af\u5883\u53d8\u91cf/\u8ba4\u8bc1\u914d\u7f6e\u7f3a\u5931",
    "frontend_activity_events": "\u524d\u7aef/CSS/SVG/React \u6d3b\u52a8",
    "git_events": "git \u5de5\u4f5c\u6d41\u547d\u4ee4",
    "install_error_events": "\u5305\u5b89\u88c5\u5931\u8d25",
    "install_success_events": "\u5305\u5b89\u88c5\u6210\u529f",
    "log_read_events": "\u65e5\u5fd7\u67e5\u770b",
    "max_distinct_tools_in_session": "\u5355\u6b21\u4f1a\u8bdd\u4f7f\u7528\u7684\u4e0d\u540c\u5de5\u5177\u6570",
    "max_file_tool_calls_in_session": "\u5355\u6b21\u4f1a\u8bdd\u6587\u4ef6/\u641c\u7d22/\u8865\u4e01\u8c03\u7528\u6570",
    "max_files_touched_in_session": "\u5355\u6b21\u4f1a\u8bdd\u64cd\u4f5c\u7684\u6587\u4ef6\u6570",
    "max_messages_in_session": "\u5355\u6b21\u4f1a\u8bdd\u6d88\u606f\u6570",
    "max_terminal_calls_in_session": "\u5355\u6b21\u4f1a\u8bdd\u7ec8\u7aef\u8c03\u7528\u6570",
    "max_tool_calls_in_session": "\u5355\u6b21\u4f1a\u8bdd\u5de5\u5177\u8c03\u7528\u6570",
    "max_web_browser_calls_in_session": "\u5355\u6b21\u4f1a\u8bdd\u7f51\u7edc/\u6d4f\u89c8\u5668\u8c03\u7528\u6570",
    "permission_denied_events": "\u6743\u9650\u62d2\u7edd\u9519\u8bef",
    "port_conflict_events": "\u7aef\u53e3\u51b2\u7a81",
    "restart_after_error_events": "\u9519\u8bef\u540e\u91cd\u542f/\u91cd\u8f7d",
    "total_cron_calls": "\u5b9a\u65f6\u4efb\u52a1\u64cd\u4f5c\u603b\u6570",
    "total_delegate_calls": "delegate_task \u8c03\u7528\u603b\u6570",
    "total_errors": "\u9519\u8bef/\u5931\u8d25\u6d88\u606f\u603b\u6570",
    "total_process_calls": "\u540e\u53f0\u8fdb\u7a0b\u64cd\u4f5c\u603b\u6570",
    "traceback_events": "\u5f02\u5e38/\u5806\u6808\u8ddf\u8e2a",
    "yaml_error_events": "YAML/\u914d\u7f6e\u89e3\u6790\u9519\u8bef",
}

TIER_NAME_ZH = {"Copper": "\u94dc", "Silver": "\u94f6", "Gold": "\u91d1", "Diamond": "\u94bb\u77f3", "Olympian": "\u5965\u6797\u5339\u65af"}


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def metric_label_zh(metric: str) -> str:
    return METRIC_LABELS_ZH.get(metric, METRIC_LABELS.get(metric, metric.replace("_", " ")))


def criteria_zh_for(definition: Dict[str, Any]) -> str:
    if definition.get("secret") and definition.get("state") == "secret":
        return "\u9690\u85cf\uff1a\u5177\u4f53\u89e6\u53d1\u6761\u4ef6\u5c06\u5728 Hermes \u68c0\u6d4b\u5230\u7b2c\u4e00\u4e2a\u5339\u914d\u4fe1\u53f7\u540e\u663e\u793a\u3002\u7ee7\u7eed\u4f7f\u7528 Hermes \u8fdb\u884c\u8c03\u8bd5\u3001\u5de5\u5177\u8c03\u7528\u3001\u8bb0\u5fc6\u3001\u6280\u80fd\u3001\u63d2\u4ef6\u548c\u6a21\u578b\u5de5\u4f5c\u6d41\u4ee5\u89e3\u9501\u3002"
    if "threshold_metric" in definition:
        tiers_list = sorted(definition.get("tiers", []), key=lambda t: t["threshold"])
        if not tiers_list:
            return "\u8981\u6c42\uff1a\u5728\u5339\u914d\u7684\u5de5\u4f5c\u6d41\u4e2d\u4f7f\u7528 Hermes\u3002"
        metric = metric_label_zh(definition["threshold_metric"])
        ladder = ", ".join(f"{TIER_NAME_ZH.get(t['name'], t['name'])} {t['threshold']}" for t in tiers_list)
        return f"\u8981\u6c42\uff1a{metric}\u3002\u7b49\u7ea7\u9636\u68af\uff1a{ladder}\u3002"
    requirements = definition.get("requirements") or []
    if requirements:
        parts = [f"{metric_label_zh(r['metric'])} \u2265 {int(r.get('gte', 1))}" for r in requirements]
        return "\u8981\u6c42\uff1a" + "\uff1b".join(parts) + "\u3002"
    return "\u8981\u6c42\uff1a\u5b8c\u6210\u5339\u914d\u7684 Hermes \u884c\u4e3a\u3002"


def criteria_for(definition: Dict[str, Any]) -> str:
    if definition.get("secret") and definition.get("state") == "secret":
        return "Secret: exact requirement hidden until Hermes sees the first matching signal. Keep using Hermes across debugging, tools, memory, skills, plugins, and model workflows to reveal it."
    secret_prefix = ""
    if "threshold_metric" in definition:
        tiers_list = sorted(definition.get("tiers", []), key=lambda t: t["threshold"])
        if not tiers_list:
            return secret_prefix + "Requirement: use Hermes in the matching workflow."
        metric = metric_label(definition["threshold_metric"])
        ladder = ", ".join(f"{t['name']} {t['threshold']}" for t in tiers_list)
        return secret_prefix + f"Requirement: {metric}. Tier ladder: {ladder}."
    requirements = definition.get("requirements") or []
    if requirements:
        parts = [f"{metric_label(r['metric'])} ≥ {int(r.get('gte', 1))}" for r in requirements]
        return secret_prefix + "Requirement: " + "; ".join(parts) + "."
    return secret_prefix + "Requirement: complete the matching Hermes behavior."


def display_achievement(item: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(item)
    aid = clean.get("id", "")
    zh = ACHIEVEMENT_ZH.get(aid, {})
    cat_zh = CATEGORY_ZH.get(clean.get("category", ""), clean.get("category", ""))
    if clean.get("state") == "secret":
        return {**clean, "name": "???", "description": "Secret achievement: hidden until Hermes detects the first relevant behavior in your session history.", "criteria": criteria_for(clean), "criteria_zh": criteria_zh_for(clean), "icon": "secret", "name_zh": "???", "description_zh": SECRET_REVEAL_ZH, "category_zh": cat_zh}
    clean["criteria"] = criteria_for(clean)
    clean["criteria_zh"] = criteria_zh_for(clean)
    clean["name_zh"] = zh.get("name", clean.get("name", ""))
    clean["description_zh"] = zh.get("description", clean.get("description", ""))
    clean["category_zh"] = cat_zh
    return clean


def scan_sessions(
    limit: Optional[int] = None,
    progress_callback: Optional[Any] = None,
    progress_every: int = 250,
) -> Dict[str, Any]:
    """Scan Hermes sessions and build per-session achievement stats.

    ``limit=None`` (the default) scans the ENTIRE session history. Prior
    versions capped this at 200, which silently reduced achievement totals
    to ~2% of history on long-running installs and made lifetime badges
    unreachable. SQLite's ``LIMIT -1`` means "unlimited"; we map ``None``
    and non-positive values to ``-1`` so callers get the full catalog.

    Warm scans stay cheap: the checkpoint cache stores per-session stats
    keyed by ``(started_at, last_active)`` and only re-analyzes sessions
    whose fingerprint changed. Cold scans on large histories (thousands
    of sessions) take tens of seconds to several minutes; ``evaluate_all``
    runs them on a background thread so the dashboard UI never blocks on
    the first request.

    ``progress_callback(partial_sessions, scanned_so_far, total)`` — when
    provided, fires every ``progress_every`` sessions with the sessions
    analyzed so far and progress counters. Background scans use this to
    publish intermediate snapshots so a long cold scan surfaces badges
    incrementally on each dashboard refresh instead of going all-at-once
    at the end.
    """
    try:
        from hermes_state import SessionDB
    except Exception as exc:
        return {"sessions": [], "aggregate": {}, "error": f"Could not import SessionDB: {exc}", "scan_meta": {"mode": "failed", "sessions_total": 0, "sessions_rescanned": 0, "sessions_reused": 0}}

    checkpoint = load_checkpoint()
    previous_sessions = checkpoint.get("sessions") if isinstance(checkpoint.get("sessions"), dict) else {}
    reused = 0
    rescanned = 0

    # SQLite treats LIMIT -1 as "no limit". Map None / <=0 to -1 so the
    # full session history flows through unless the caller explicitly
    # requests a small sample (e.g. a smoke test).
    db_limit = -1 if (limit is None or limit <= 0) else int(limit)

    db = SessionDB()
    try:
        sessions_meta = db.list_sessions_rich(limit=db_limit, include_children=True, project_compression_tips=False)
        total_sessions = len(sessions_meta)
        sessions: List[Dict[str, Any]] = []
        checkpoint_sessions: Dict[str, Any] = {}
        for idx, meta in enumerate(sessions_meta, start=1):
            sid = meta.get("id")
            if not sid:
                continue
            fp = session_fingerprint(meta)
            cached = previous_sessions.get(sid) if isinstance(previous_sessions, dict) else None
            cached_stats = cached.get("stats") if isinstance(cached, dict) else None
            cached_fp = cached.get("fingerprint") if isinstance(cached, dict) else None

            if isinstance(cached_stats, dict) and cached_fp == fp:
                stats = dict(cached_stats)
                reused += 1
            else:
                messages = db.get_messages(sid)
                stats = analyze_messages(sid, meta.get("title") or meta.get("preview") or "Untitled", messages)
                rescanned += 1

            stats["session_id"] = sid
            stats["title"] = meta.get("title") or meta.get("preview") or stats.get("title") or "Untitled"
            stats["started_at"] = meta.get("started_at")
            stats["last_active"] = meta.get("last_active")
            stats["source"] = meta.get("source")
            if meta.get("model"):
                stats.setdefault("model_names", set())
                if isinstance(stats["model_names"], set):
                    stats["model_names"].add(str(meta.get("model")))
                elif isinstance(stats["model_names"], list):
                    if str(meta.get("model")) not in stats["model_names"]:
                        stats["model_names"].append(str(meta.get("model")))
                else:
                    stats["model_names"] = {str(meta.get("model"))}

            sessions.append(stats)
            checkpoint_sessions[sid] = {"fingerprint": fp, "stats": _json_safe(stats)}

            if progress_callback is not None and progress_every > 0 and (idx % progress_every == 0) and idx < total_sessions:
                try:
                    progress_callback(list(sessions), idx, total_sessions)
                except Exception:
                    # Progress callbacks are advisory — a broken publisher
                    # must never abort the scan itself.
                    pass

        save_checkpoint({
            "schema_version": 1,
            "generated_at": int(time.time()),
            "sessions": checkpoint_sessions,
        })
    finally:
        close = getattr(db, "close", None)
        if close:
            close()
    return {
        "sessions": sessions,
        "aggregate": aggregate_stats(sessions),
        "scan_meta": {
            "mode": "incremental" if reused > 0 else "full",
            "sessions_total": len(sessions),
            "sessions_rescanned": rescanned,
            "sessions_reused": reused,
            "sessions_scanned_so_far": len(sessions),
            "sessions_expected_total": total_sessions,
        },
    }


def aggregate_stats(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {
        "session_count": len(sessions),
        "max_tool_calls_in_session": 0,
        "max_distinct_tools_in_session": 0,
        "max_messages_in_session": 0,
        "max_terminal_calls_in_session": 0,
        "max_file_tool_calls_in_session": 0,
        "max_web_calls_in_session": 0,
        "max_web_browser_calls_in_session": 0,
        "max_files_touched_in_session": 0,
        "total_errors": 0,
        "total_tool_calls": 0,
        "total_terminal_calls": 0,
        "total_web_calls": 0,
        "total_web_extract_calls": 0,
        "total_patch_calls": 0,
        "total_file_reads_searches": 0,
        "total_delegate_calls": 0,
        "total_process_calls": 0,
        "total_cron_calls": 0,
        "browser_calls": 0,
        "image_vision_calls": 0,
        "tts_calls": 0,
        "distinct_model_count": 0,
        "distinct_provider_count": 0,
        "local_model_chat_sessions": 0,
        "weekend_sessions": 0,
        "night_sessions": 0,
    }
    sum_keys = [
        "traceback_events", "log_read_events", "port_conflict_events", "permission_denied_events", "install_error_events", "install_success_events", "restart_after_error_events", "env_var_error_events", "yaml_error_events", "docker_conflict_events", "frontend_activity_events", "css_activity_events", "git_events", "tiny_patch_after_errors_events", "skill_events", "skill_manage_events", "memory_events", "memory_write_events", "context_events", "gateway_events", "plugin_events", "rollback_events", "docs_activity_events", "model_events", "openrouter_events", "codex_events", "claude_events", "gemini_events", "local_model_events", "toolset_events", "config_events", "git_history_events", "test_events", "screenshot_events", "release_events", "cache_events",
    ]
    for key in sum_keys:
        agg[key] = 0

    model_names: Set[str] = set()
    provider_names: Set[str] = set()
    for s in sessions:
        agg["max_tool_calls_in_session"] = max(agg["max_tool_calls_in_session"], s.get("tool_call_count", 0))
        agg["max_distinct_tools_in_session"] = max(agg["max_distinct_tools_in_session"], s.get("distinct_tool_count", 0))
        agg["max_messages_in_session"] = max(agg["max_messages_in_session"], s.get("message_count", 0))
        agg["max_terminal_calls_in_session"] = max(agg["max_terminal_calls_in_session"], s.get("terminal_calls", 0))
        agg["max_file_tool_calls_in_session"] = max(agg["max_file_tool_calls_in_session"], s.get("file_tool_calls", 0))
        agg["max_web_calls_in_session"] = max(agg["max_web_calls_in_session"], s.get("web_calls", 0))
        agg["max_web_browser_calls_in_session"] = max(agg["max_web_browser_calls_in_session"], s.get("web_browser_calls", 0))
        agg["max_files_touched_in_session"] = max(agg["max_files_touched_in_session"], s.get("files_touched_count", 0))
        agg["total_errors"] += s.get("error_count", 0)
        agg["total_tool_calls"] += s.get("tool_call_count", 0)
        agg["total_terminal_calls"] += s.get("terminal_calls", 0)
        agg["total_web_calls"] += s.get("web_calls", 0)
        agg["total_web_extract_calls"] += s.get("web_extract_calls", 0)
        agg["total_patch_calls"] += s.get("patch_calls", 0)
        agg["total_file_reads_searches"] += s.get("file_reads_searches", 0)
        agg["total_delegate_calls"] += s.get("delegate_calls", 0)
        agg["total_process_calls"] += s.get("process_calls", 0)
        agg["total_cron_calls"] += s.get("cron_calls", 0)
        agg["browser_calls"] += s.get("browser_calls", 0)
        agg["image_vision_calls"] += s.get("image_vision_calls", 0)
        agg["tts_calls"] += s.get("tts_calls", 0)
        for key in sum_keys:
            agg[key] += s.get(key, 0)
        model_names.update(s.get("model_names") or set())
        session_models = s.get("model_names") or set()
        for model_name in session_models:
            provider = model_provider(str(model_name))
            if provider:
                provider_names.add(provider)
        if any(is_local_model_name(str(model_name)) for model_name in session_models):
            agg["local_model_chat_sessions"] += 1
        if s.get("started_at"):
            try:
                lt = time.localtime(float(s.get("started_at")))
                if lt.tm_wday >= 5:
                    agg["weekend_sessions"] += 1
                if lt.tm_hour < 6 or lt.tm_hour >= 23:
                    agg["night_sessions"] += 1
            except Exception:
                pass
    agg["distinct_model_count"] = len({m for m in model_names if m and m != "None"})
    agg["distinct_provider_count"] = len(provider_names)
    return agg


def evaluate_definition(definition: Dict[str, Any], aggregate: Dict[str, Any]) -> Dict[str, Any]:
    if "threshold_metric" in definition:
        return evaluate_tiered(definition, aggregate)
    if "requirements" in definition:
        return evaluate_requirements(definition, aggregate)
    return evaluate_boolean(definition, aggregate)


def evidence_for(definition: Dict[str, Any], sessions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not sessions:
        return None
    metric = definition.get("threshold_metric")
    metric_to_session_key = {
        "max_tool_calls_in_session": "tool_call_count",
        "max_distinct_tools_in_session": "distinct_tool_count",
        "max_messages_in_session": "message_count",
        "max_terminal_calls_in_session": "terminal_calls",
        "max_file_tool_calls_in_session": "file_tool_calls",
        "max_web_calls_in_session": "web_calls",
        "max_web_browser_calls_in_session": "web_browser_calls",
        "max_files_touched_in_session": "files_touched_count",
    }
    if metric in metric_to_session_key:
        key = metric_to_session_key[metric]
        s = max(sessions, key=lambda x: x.get(key, 0))
        return {"session_id": s.get("session_id"), "title": s.get("title"), "value": s.get(key, 0)}
    return None


def _compute_from_scan(scan: Dict[str, Any], *, is_partial: bool = False) -> Dict[str, Any]:
    """Evaluate every achievement definition against a scan result.

    Used by ``compute_all`` for finished scans AND by the background
    progress callback for partial, in-flight snapshots. ``is_partial=True``
    skips persisting ``state.json`` unlocks — we don't want to record an
    "unlock time" based on half a scan that a later session might shift.
    """
    aggregate = scan.get("aggregate", {})
    state = load_state() if not is_partial else {"unlocks": {}}
    unlocks = state.setdefault("unlocks", {})
    now = int(time.time())
    evaluated = []
    for definition in ACHIEVEMENTS:
        result = evaluate_definition(definition, aggregate)
        unlock_id = definition["id"]
        if not is_partial and result["unlocked"] and unlock_id not in unlocks:
            unlocks[unlock_id] = {"unlocked_at": now, "first_tier": result.get("tier"), "evidence": evidence_for(definition, scan.get("sessions", []))}
        item = {**definition, **result}
        if result["unlocked"]:
            item["unlocked_at"] = unlocks.get(unlock_id, {}).get("unlocked_at")
            item["evidence"] = unlocks.get(unlock_id, {}).get("evidence") or evidence_for(definition, scan.get("sessions", []))
        evaluated.append(display_achievement(item))
    if not is_partial:
        save_state(state)
    unlocked = [a for a in evaluated if a["unlocked"]]
    discovered = [a for a in evaluated if a.get("state") == "discovered"]
    secret = [a for a in evaluated if a.get("state") == "secret"]
    return {
        "achievements": evaluated,
        "sessions": scan.get("sessions", []),
        "aggregate": aggregate,
        "scan_meta": scan.get("scan_meta", {}),
        "error": scan.get("error"),
        "unlocked_count": len(unlocked),
        "discovered_count": len(discovered),
        "secret_count": len(secret),
        "total_count": len(evaluated),
        "generated_at": now,
    }


def compute_all(progress_callback: Optional[Any] = None, progress_every: int = 250) -> Dict[str, Any]:
    scan = scan_sessions(progress_callback=progress_callback, progress_every=progress_every)
    return _compute_from_scan(scan, is_partial=False)


_BACKGROUND_SCAN_THREAD: Optional[threading.Thread] = None
_BACKGROUND_SCAN_LOCK = threading.Lock()


def _build_pending_snapshot(now: int) -> Dict[str, Any]:
    """Placeholder payload used while the first-ever scan is still running.

    Returns a structurally-complete response so the dashboard UI can render
    an empty achievement list + spinner without special-casing "no data yet".
    """
    evaluated = [display_achievement({**d, **{"unlocked": False, "discovered": False, "state": "secret" if d.get("secret") else "discovered", "progress": 0, "progress_pct": 0, "next_tier": (d.get("tiers") or [{}])[0].get("name"), "next_threshold": (d.get("tiers") or [{}])[0].get("threshold", 1), "tier": None}}) for d in ACHIEVEMENTS]
    return {
        "achievements": evaluated,
        "sessions": [],
        "aggregate": {},
        "scan_meta": {"mode": "pending", "sessions_total": 0, "sessions_rescanned": 0, "sessions_reused": 0},
        "error": None,
        "unlocked_count": 0,
        "discovered_count": sum(1 for a in evaluated if a.get("state") == "discovered"),
        "secret_count": sum(1 for a in evaluated if a.get("state") == "secret"),
        "total_count": len(evaluated),
        "generated_at": now,
    }


def _run_scan_and_update_cache(publish_partial_snapshots: bool = True) -> None:
    """Execute a scan + snapshot update. Called synchronously or from a thread.

    When ``publish_partial_snapshots=True`` (the default for background
    scans), the scanner periodically publishes an in-progress snapshot to
    ``_SNAPSHOT_CACHE`` so each dashboard refresh during a long cold scan
    shows more progress — badges unlock incrementally as sessions stream
    in, instead of staying at zero for minutes and then jumping to the
    final state. Synchronous /rescan callers pass ``False`` because they
    block on the full result anyway.
    """
    global _SNAPSHOT_CACHE, _SNAPSHOT_CACHE_AT
    with _SCAN_LOCK:
        started = int(time.time())
        _SCAN_STATUS["state"] = "running"
        _SCAN_STATUS["started_at"] = started
        _SCAN_STATUS["last_error"] = None

        def _publish_partial(partial_sessions, scanned_so_far, total):
            global _SNAPSHOT_CACHE, _SNAPSHOT_CACHE_AT
            try:
                partial_scan = {
                    "sessions": partial_sessions,
                    "aggregate": aggregate_stats(partial_sessions),
                    "scan_meta": {
                        "mode": "in_progress",
                        "sessions_total": scanned_so_far,
                        "sessions_rescanned": 0,
                        "sessions_reused": 0,
                        "sessions_scanned_so_far": scanned_so_far,
                        "sessions_expected_total": total,
                    },
                }
                partial = _compute_from_scan(partial_scan, is_partial=True)
                # Keep the cache in the 'stale' TTL regime by NOT bumping
                # _SNAPSHOT_CACHE_AT to "now". The UI treats partial
                # results as stale so it keeps polling /scan-status and
                # sees the final snapshot when the scan finishes. In-flight
                # partials are visible but are never mistaken for finished.
                _SNAPSHOT_CACHE = _json_safe(partial)
                _SNAPSHOT_CACHE_AT = 0
            except Exception:
                # Intermediate publication is best-effort; don't kill the scan.
                pass

        callback = _publish_partial if publish_partial_snapshots else None
        try:
            computed = compute_all(progress_callback=callback)
            _SNAPSHOT_CACHE = _json_safe(computed)
            _SNAPSHOT_CACHE_AT = int(_SNAPSHOT_CACHE.get("generated_at") or int(time.time()))
            save_snapshot(_SNAPSHOT_CACHE)
            _SCAN_STATUS["state"] = "idle"
        except Exception as exc:
            _SCAN_STATUS["state"] = "failed"
            _SCAN_STATUS["last_error"] = str(exc)
        finally:
            _SCAN_STATUS["finished_at"] = int(time.time())
            _SCAN_STATUS["last_duration_ms"] = int((_SCAN_STATUS["finished_at"] - started) * 1000)
            _SCAN_STATUS["run_count"] = int(_SCAN_STATUS.get("run_count", 0)) + 1


def _start_background_scan() -> None:
    """Kick off a scan in a daemon thread if one isn't already running.

    Idempotent: concurrent callers see the in-flight thread and return
    immediately. The thread updates ``_SNAPSHOT_CACHE`` on completion so
    subsequent ``/achievements`` requests see fresh data. While running,
    it also publishes partial snapshots every ~250 sessions so the UI
    reflects incremental progress on long cold scans.
    """
    global _BACKGROUND_SCAN_THREAD
    with _BACKGROUND_SCAN_LOCK:
        existing = _BACKGROUND_SCAN_THREAD
        if existing is not None and existing.is_alive():
            return
        thread = threading.Thread(
            target=_run_scan_and_update_cache,
            kwargs={"publish_partial_snapshots": True},
            name="hermes-achievements-scan",
            daemon=True,
        )
        _BACKGROUND_SCAN_THREAD = thread
        thread.start()


def evaluate_all(force: bool = False) -> Dict[str, Any]:
    """Return the current achievements payload.

    Behavior matrix:

    * Fresh in-memory cache → return it instantly.
    * Stale on-disk snapshot → load it, kick a background rescan, return
      the stale data (UI decorates it with ``is_stale=True``).
    * No snapshot yet (first-ever run) → kick a background scan, return
      an empty-but-valid "pending" payload so the UI can render a spinner
      without blocking.
    * ``force=True`` (manual /rescan) → run synchronously, block the
      caller, replace the cache.

    Warm scans stay cheap (the checkpoint cache reuses per-session stats).
    Cold scans on 8000+ session databases take minutes; the background
    thread prevents that from ever blocking the dashboard request path.
    """
    global _SNAPSHOT_CACHE, _SNAPSHOT_CACHE_AT
    now = int(time.time())

    if not force and _cache_is_fresh(now):
        return _SNAPSHOT_CACHE or {}

    # Lazy-load persisted snapshot from disk so fresh process starts
    # don't have to wait for a scan to serve cached data.
    if _SNAPSHOT_CACHE is None:
        persisted = load_snapshot()
        if isinstance(persisted, dict):
            generated_at = int(persisted.get("generated_at") or 0)
            _SNAPSHOT_CACHE = persisted
            _SNAPSHOT_CACHE_AT = generated_at or now

    if force:
        # Manual /rescan — block the caller, synchronous scan path.
        # No partial publishing: the caller is waiting for the final result.
        _run_scan_and_update_cache(publish_partial_snapshots=False)
        if _SNAPSHOT_CACHE is not None:
            return _SNAPSHOT_CACHE
        # Scan failed with no prior cache — surface empty payload.
        return _build_pending_snapshot(now)

    # Non-force path: serve whatever we have and refresh in background.
    if _SNAPSHOT_CACHE is not None:
        if not _cache_is_fresh(now):
            _start_background_scan()
        return _SNAPSHOT_CACHE

    # First-ever run on this machine — no snapshot yet. Kick off a scan
    # and return a pending placeholder. The UI polls /scan-status and
    # re-fetches /achievements when the scan completes.
    _start_background_scan()
    return _build_pending_snapshot(now)


@router.get("/achievements")
async def achievements():
    data = evaluate_all()
    payload = {k: data[k] for k in ["achievements", "unlocked_count", "discovered_count", "secret_count", "total_count", "error", "generated_at"] if k in data}
    payload["is_stale"] = _is_snapshot_stale(data)
    payload["scan_meta"] = {
        **(data.get("scan_meta") or {}),
        "status": _scan_status_payload(),
    }
    return payload


@router.get("/scan-status")
async def scan_status():
    return _scan_status_payload()


@router.get("/recent-unlocks")
async def recent_unlocks():
    data = evaluate_all()
    return sorted([a for a in data["achievements"] if a["unlocked"]], key=lambda a: a.get("unlocked_at") or 0, reverse=True)[:20]


@router.get("/sessions/{session_id}/badges")
async def session_badges(session_id: str):
    data = evaluate_all()
    session = next((s for s in data["sessions"] if s["session_id"] == session_id), None)
    if not session:
        return {"session_id": session_id, "badges": []}
    aggregate = aggregate_stats([session])
    badges = []
    for definition in ACHIEVEMENTS:
        result = evaluate_definition(definition, aggregate)
        if result["unlocked"]:
            badges.append(display_achievement({**definition, **result}))
    return {"session_id": session_id, "badges": badges}


@router.post("/rescan")
async def rescan():
    return {"ok": True, **evaluate_all(force=True)}


@router.post("/reset-state")
async def reset_state():
    global _SNAPSHOT_CACHE, _SNAPSHOT_CACHE_AT
    save_state({"unlocks": {}})
    _SNAPSHOT_CACHE = None
    _SNAPSHOT_CACHE_AT = 0
    _SCAN_STATUS["state"] = "idle"
    _SCAN_STATUS["started_at"] = None
    _SCAN_STATUS["finished_at"] = None
    _SCAN_STATUS["last_error"] = None
    _SCAN_STATUS["last_duration_ms"] = None
    try:
        snapshot_path().unlink(missing_ok=True)
    except Exception:
        pass
    try:
        checkpoint_path().unlink(missing_ok=True)
    except Exception:
        pass
    return {"ok": True}
