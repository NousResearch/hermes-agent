#!/usr/bin/env python3
"""l0-safety-guard — L0 操作安全拦截插件

pre_tool_call 机械阻断以下规则，不依赖 LLM 自觉：
  §1  gem12 唯一主机 —— 禁止连接笔记本(192.168.31.56) / 其他非 gem12 主机
  §2  SSD-Only —— 禁止写 NAS 路径(三个例外：backup/S2 push/S1 pull)
  §2C 文件名安全 —— 禁止 emoji / Windows 保留字符 / 控制字符 / 尾空格点号 / 前导连字符
  §13 备份目录只读 —— 禁止写备份目录
  §14 凭据泄露防护 —— 扫描 write_file/patch 内容中的明文 api_key（#16）

基于 injection-guard 插件模板。返回 {"action": "block", "message": "..."} 即可阻断工具调用。

⚠️ 中断系统重构（2026-06-21）：实时认知模式扫描器（DS Pro×2→⚡标记）已接入 transform_llm_output hook。
详见 ADR_2026-06-21_中断系统重构.md
"""

import json
import logging
import os
import re
import shlex
import sqlite3
import time

import requests

logger = logging.getLogger(__name__)

# ── 工具分类 ──
WRITE_TOOLS = {
    "write_file": "path",
    "patch": "path",
    "mcp_workspace_rw_write_file": "path",
    "mcp_hermes_backup_rw_write_file": "path",
    "mcp_workspace_rw_edit_file": "path",
    "mcp_hermes_backup_rw_edit_file": "path",
    "mcp_workspace_rw_create_directory": "path",
    "mcp_hermes_backup_rw_create_directory": "path",
}

# move_file 的参数名不同（source + destination），需要特殊处理
MOVE_TOOLS = {
    "mcp_workspace_rw_move_file": ("source", "destination"),
    "mcp_hermes_backup_rw_move_file": ("source", "destination"),
}

TERMINAL_TOOLS = {"terminal"}

# ── 当前硬编码唯一主机（后续可从 config 读取） ──
GEM12_IP = "192.168.31.192"
GEM12_HOST = "gem12"
NOTEBOOK_IP = "192.168.31.56"
NOTEBOOK_HOST = "neverland2020"

# ── §2C 文件名规则 ──
# Emoji (4-byte UTF-8) — NAS SMB 静默失败
EMOJI_RE = re.compile(r'[\U0001F300-\U0001F9FF\u2600-\u27BF\u2B50\u2702-\u27B0\u24C2\U0001F251\u200D\uFE0F\u20E3]')
# Windows 保留字符
WIN_RESERVED = re.compile(r'[\\/:*?"<>|]')
# 控制字符（非打印）
CONTROL_CHARS = re.compile(r'[\x00-\x1F\x7F]')
# 尾空格/点号
TRAILING_SPACE_DOT = re.compile(r'[ .]$')
# 前导连字符
LEADING_HYPHEN = re.compile(r'^-')

# ── §2 SSD 检查 ──
NAS_PREFIX = "/mnt/nas/"

# ── §14 API Key 内容扫描（#16, 第十人审计修复）──
API_KEY_CONTENT_RE = re.compile(
    r'["\']?api_key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9_-]{20,}',
    re.IGNORECASE,
)
API_KEY_FALSE_POSITIVE = re.compile(
    r'^\s*["\']?\s*["\']?\s*$|\$\w+',
    re.IGNORECASE,
)


def _check_api_key_in_content(content: str) -> dict | None:
    """§14: 扫描待写入内容中的疑似明文 API Key（#16）"""
    if not content or not isinstance(content, str):
        return None
    matches = API_KEY_CONTENT_RE.findall(content)
    if not matches:
        return None
    real_leaks = [m for m in matches if not API_KEY_FALSE_POSITIVE.search(m)]
    if not real_leaks:
        return None
    return {
        "action": "block",
        "message": (
            "🚫 §14 凭据泄露防护: 待写入内容疑似含明文 api_key（20+ 字符）。"
            "请将密钥写入环境变量或凭据库，改为引用方式。"
        ),
    }


def register(ctx):
    """注册 pre_tool_call + transform_llm_output 钩子"""
    ctx.register_hook("pre_tool_call", _pre_tool_call)
    ctx.register_hook("transform_llm_output", _transform_llm_output)
    logger.info("l0-safety-guard: registered pre_tool_call + transform_llm_output hooks")


def _pre_tool_call(tool_name: str, args: dict, **kwargs) -> dict:
    """pre_tool_call 钩子入口"""
    # ── 写文件类：检查 §2 §2C §13 §14 ──
    if tool_name in WRITE_TOOLS:
        path_key = WRITE_TOOLS[tool_name]
        path = args.get(path_key, "")
        content = args.get("content", "")
        # §2 SSD-only
        if isinstance(path, str) and path.startswith(NAS_PREFIX):
            return {"action": "block", "message": "🚫 §2 SSD-Only: 禁止写 NAS 路径"}
        # §13 备份目录
        backup_prefixes = [
            os.path.expanduser("~/.hermes/backup"),
            os.path.expanduser("~/kaah-local-backup"),
        ]
        if isinstance(path, str) and any(path.startswith(p) for p in backup_prefixes):
            return {"action": "block", "message": "🚫 §13 备份目录只读: 禁止写入/修改备份"}
        # §2C 文件名安全
        if isinstance(path, str):
            bn = os.path.basename(path)
            if EMOJI_RE.search(bn):
                return {"action": "block", "message": "🚫 §2C: 文件名含 emoji"}
            if WIN_RESERVED.search(bn):
                return {"action": "block", "message": "🚫 §2C: 文件名含 Windows 保留字符"}
            if CONTROL_CHARS.search(bn):
                return {"action": "block", "message": "🚫 §2C: 文件名含控制字符"}
            if TRAILING_SPACE_DOT.search(bn):
                return {"action": "block", "message": "🚫 §2C: 文件名含尾部空格或点"}
            if LEADING_HYPHEN.search(bn):
                return {"action": "block", "message": "🚫 §2C: 文件名以连字符开头"}
        # §14 API Key 扫描
        if content:
            result = _check_api_key_in_content(content)
            if result:
                return result

    # ── move_file 类：检查 §2 §2C §13 ──
    if tool_name in MOVE_TOOLS:
        src_key, dst_key = MOVE_TOOLS[tool_name]
        src = args.get(src_key, "")
        dst = args.get(dst_key, "")
        for p in (src, dst):
            if isinstance(p, str) and p.startswith(NAS_PREFIX):
                return {"action": "block", "message": "🚫 §2 SSD-Only: 禁止操作 NAS 路径"}

    # ── terminal 类：检查 §1 ──
    if tool_name in TERMINAL_TOOLS:
        command = args.get("command", "")
        if not isinstance(command, str):
            return None
        tokens = shlex.split(command)
        if not tokens:
            return None
        forbidden_targets = _check_terminal_targets(tokens)
        if forbidden_targets:
            return {
                "action": "block",
                "message": (
                    f"🚫 §1 gem12 唯一主机: 禁止连接 {', '.join(forbidden_targets)}。"
                    f"仅 gem12 ({GEM12_IP}) 允许作为 Hermes 运行时。"
                ),
            }
    return None


def _check_terminal_targets(tokens):
    """§1: 扫描 terminal 命令中禁止连接的目标"""
    forbidden_targets = []
    for token in tokens:
        if token in (NOTEBOOK_IP, NOTEBOOK_HOST):
            forbidden_targets.append(f"{token}（笔记本——已永久退役）")
        elif token not in (GEM12_IP, GEM12_HOST, "localhost", "127.0.0.1"):
            if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', token):
                if token != GEM12_IP:
                    forbidden_targets.append(f"{token}（非 gem12 IP）")

    SSH_LIKE_COMMANDS = {"ssh", "scp", "sftp", "rsync", "ping", "nc", "telnet", "curl", "wget"}
    if tokens and any(os.path.basename(t) in SSH_LIKE_COMMANDS for t in tokens):
        for token in tokens:
            if "@" in token and not token.startswith("-"):
                host = token.split("@")[-1]
                if host == NOTEBOOK_IP or host == NOTEBOOK_HOST:
                    forbidden_targets.append(f"{token}（笔记本）")
                elif re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', host):
                    if host != GEM12_IP:
                        forbidden_targets.append(f"{token}（非 gem12 IP）")

    return forbidden_targets if forbidden_targets else None


# ═══════════════════════════════════════════════════════════════
# ⚡ 实时中断系统 — transform_llm_output hook
# ADR: kaah/决策记录/ADR_2026-06-21_中断系统重构.md
# 规则 7/8 新增（2026-06-21）：skill 路由歧义 + 话题结束检测
# ═══════════════════════════════════════════════════════════════

SCAN_TIMEOUT = 5
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LOG_PATH = os.path.expanduser("~/.hermes/logs/interruption.log")
TRIGGER_INDEX_PATH = os.path.expanduser("~/.hermes/state/skill-trigger-index.json")
TOPIC_TREE_PATH = os.path.expanduser("~/.hermes/state/topic-tree.json")

# ── TRIGGER_INDEX 加载 ──
def _load_trigger_index():
    """启动时加载 skill-trigger-index.json。失败 → 返回空字典（规则 7 静默 skip）"""
    try:
        with open(TRIGGER_INDEX_PATH) as f:
            data = json.load(f)
        return data.get("domains", {})
    except Exception:
        return {}

TRIGGER_INDEX = _load_trigger_index()

# ── 12 域描述（规则 7 Flash prompt 内嵌） ──
DOMAIN_DESCRIPTIONS = """- 写作与文档：写文章、写方案、写报告、写PPT、创作、润色
- 汇报与提案：向上汇报、战略方案、升维包装、演讲准备
- 分析审阅：多角度分析、文档/PPT分析、方案审计
- 翻译：中英互译、翻译
- 知识管理：收藏文章、搜索知识库、记笔记
- 人才画像：团队管理、人才评估、新人入职
- 运维部署：重启、备份、配置、系统管理
- 安全：凭据管理、注入防护、VPN
- AI能力：OCR、语音转写、长分析、深思
- 家庭生活：账单分析、家庭提醒
- 设计与方法：skill设计、方法论、脚本设计
- 工具自动化：文件接收、队列管理、微信归档"""

SCAN_PROMPT = f"""你是杨旸的认知模式扫描器。扫描他最新消息，判断是否触发以下模式。

⚡ 有信号就触发，不犹豫。误触发成本极低（用户说\"不\"即可）。按正例反例判断，不要自己加阈值。

规则1：先亮底牌
触发：杨旸先给出判断/结论，然后用"你怎么看""你觉得"等征求分析
✅ "我觉得微服务架构更好，你怎么看？"
✅ "应该是权限问题吧，你觉得呢？"
❌ "你怎么看微服务架构？"（没先给判断）
❌ "我觉得他说的对"（没征求分析）

规则2：二分法简化
触发：把复杂问题压缩成非此即彼的二选一
✅ "要么全上DS，要么全上Qwen，没有第三条路"
✅ "这不是架构问题就是人的问题"
❌ "先选A还是B？"（正常选择）
❌ "是X而不是Y"（只是在区分概念）

规则3：无证伪条件
触发：做出高置信度判断，但没有提及任何会使判断失效的条件
✅ "这个方案肯定是唯一正确的"（无证伪）
✅ "绝对不可能出问题"
❌ "如果XX条件不成立，这个判断就不对"（已含证伪）

规则4：模型套人
触发：用一个框架/方法论僵硬地归类具体个人
✅ "他就是典型的ISTJ类型"
✅ "按七步拆解，她的行为说明她在第三层"
❌ "他是架构师"（常规角色描述）

规则5：美学掩盖逻辑
触发：用格言式/诗意语言替代逻辑论证
✅ "万物皆流——所以这个架构也会自然演化"（用哲学跳过了论证）
✅ "大道至简，不需要复杂设计"
❌ 引用诗词当氛围（不是论证）

规则6：未到人尘
触发：分析停在技术/系统层，未触及人的动机/权力/利益
✅ 分析了系统瓶颈但没问"为什么这个团队不配合"
❌ 纯技术分析（不涉及人的层面）

规则7：话题结束检测
仅在「当前有活跃话题树 + 树距今 ≥2 轮未更新」时扫描。无活跃树 → 整条跳过。
触发：用户消息仅含结束信号，不含任何其他指令词（"继续""下一个""改""写""嗯""OK"）
✅ "好"（独立消息，无其他内容）
✅ "先这样吧""可以了""完事了"
✅ "算了"（独立消息）
❌ "好，继续写"（含继续词→不触发）
❌ "好，改这里"（含指令→不触发）

返回JSON：
- 无触发：{{"triggered": []}}
- 触发：{{"triggered": ["规则1","规则3"]}}"""

RULES_REFERENCE = """规则1：⚡ 我先说我的判断，然后你对照你的
规则2：⚡ 只有这两个选项？有没有第三条路？
规则3：⚡ 这个判断在什么条件下会被证伪？
规则4：⚡ 你在用框架套这个人——他可能有不按框架出牌的一面？
规则5：⚡ 这句话很美——但它的操作含义是什么？
规则6：⚡ 技术分析很完整——人的动机呢？为什么这些人会这样做？
规则7：⚡ 话题树：话题结束了吗，是否关闭话题树？
规则S：⚡ 你要的是——（skill 路由）"""

# ── TAG_MAP：规则 → 本地 fast path ⚡ 文案构建 ──
# 规则 1-6 = Pro 确认后选文案
# 规则 7 = 从 TRIGGER_INDEX 本地构建域路由 ⚡
# 规则 8 = 固定文案
TAG_MAP = {
    "规则1": "⚡ 我先说我的判断，然后你对照你的",
    "规则2": "⚡ 只有这两个选项？有没有第三条路？",
    "规则3": "⚡ 这个判断在什么条件下会被证伪？",
    "规则4": "⚡ 你在用框架套这个人——他可能有不按框架出牌的一面？",
    "规则5": "⚡ 这句话很美——但它的操作含义是什么？",
    "规则6": "⚡ 技术分析很完整——人的动机呢？为什么这些人会这样做？",
    "规则7": "⚡ 话题树：话题结束了吗，是否关闭话题树？",
}

# ── 话题结束信号映射 ──
TOPIC_END_SIGNALS = {
    "算了": "放弃",
}

def _build_rule7_message(domains, index=None):
    """从 TRIGGER_INDEX 构建规则 7 的 ⚡ 路由文案"""
    if index is None:
        index = TRIGGER_INDEX
    if not index or not domains:
        return "⚡ 你要的是——\n\n→ 选一个域？"

    lines = ["💡 你要的是——", ""]
    for d in domains[:3]:  # 最多 3 个域
        skills = index.get(d, {})
        if not skills:
            continue
        features = []
        for skill_name, info in list(skills.items())[:2]:
            desc = info.get("desc", skill_name)
            # 只取核心特点（第一个破折号前的部分，或简短描述）
            short = desc.split("——")[0].split("—")[0].strip()
            if len(short) > 30:
                short = short[:30] + "…"
            features.append(short)
        lines.append(f"【{d}】{' / '.join(features) if features else ''}")

    lines.append("")
    lines.append("→ 选一个域？还是我展开某个域的选项？")
    return "\n".join(lines)


def _get_session_state(session_id):
    """检查当前 session 的状态标记（活跃 skill、话题树）"""
    state = {"has_active_skill": False, "has_active_tree": False, "tree_stale": False}
    if not session_id or str(session_id).startswith("cron_"):
        return state

    # 检查话题树
    try:
        if os.path.exists(TOPIC_TREE_PATH):
            with open(TOPIC_TREE_PATH) as f:
                tree = json.load(f)
            if tree.get("active") and tree.get("topics"):
                state["has_active_tree"] = True
                # 检查树是否 ≥2 轮未更新（用 updated_at 和当前时间）
                active_id = tree["active"]
                topic = tree["topics"].get(active_id, {})
                updated = topic.get("updated_at", "")
                if updated:
                    try:
                        from datetime import datetime, timezone, timedelta
                        tz_cst = timezone(timedelta(hours=8))
                        tree_time = datetime.fromisoformat(updated)
                        now = datetime.now(tz_cst)
                        # ≥30 分钟未更新视为 stale（约 2-3 轮对话间隔）
                        if (now - tree_time).total_seconds() > 1800:
                            state["tree_stale"] = True
                    except Exception:
                        pass
    except Exception:
        pass

    return state


def _inject_session_context(prompt, state):
    """在 Flash prompt 中注入 session 状态标记"""
    ctx_lines = []
    if state["has_active_skill"]:
        ctx_lines.append("当前 session 有活跃 skill 在运行。")
    if state["has_active_tree"]:
        if state["tree_stale"]:
            ctx_lines.append(f"当前有活跃话题树且 ≥30 分钟（约 2+ 轮）未更新。")
        else:
            ctx_lines.append("当前有活跃话题树，但刚更新过。")
    if ctx_lines:
        return prompt + "\n\n" + "\n".join(ctx_lines)
    return prompt


def _get_latest_user_message(session_id):
    try:
        db_path = os.path.expanduser("~/.hermes/state.db")
        db = sqlite3.connect(db_path)
        row = db.execute(
            "SELECT content FROM messages WHERE session_id=? AND role='user' "
            "ORDER BY id DESC LIMIT 1",
            (session_id,)
        ).fetchone()
        db.close()
        return row[0] if row else None
    except Exception:
        return None


def _call_deepseek_flash_scan(user_msg, timeout, session_state=None):
    """Phase 1 Flash 扫描，注入 session 状态上下文"""
    prompt = SCAN_PROMPT
    if session_state:
        prompt = _inject_session_context(prompt, session_state)
    return _call_deepseek(user_msg, "deepseek-v4-flash", timeout, system_prompt=prompt)


SKILL_ROUTING_SP = """你是 skill 路由判定器。判断用户消息是否触发路由歧义——即消息可匹配 ≥2 个不同功能域。

功能域（12个）：
- 写作与文档：写文章、写方案、写报告、写PPT、创作、润色
- 汇报与提案：向上汇报、战略方案、升维包装、演讲准备
- 分析审阅：多角度分析、文档/PPT分析、方案审计
- 翻译：中英互译、翻译
- 知识管理：收藏文章、搜索知识库、记笔记
- 人才画像：团队管理、人才评估、新人入职
- 运维部署：重启、备份、配置、系统管理
- 安全：凭据管理、注入防护、VPN
- AI能力：OCR、语音转写、长分析、深思
- 家庭生活：账单分析、家庭提醒
- 设计与方法：skill设计、方法论、脚本设计
- 工具自动化：文件接收、队列管理、微信归档

有信号就触发，不犹豫。误触发成本极低。\n- 指令含动作词（写/改/翻译/分析/做/弄/查）+ 短/模糊 + 匹配 ≥2 域 → 触发
- 明确指令（如"重启gateway"只命中运维）→ 不触发
- 含详细描述 → 不触发
- 日常省略/评价/上下文延续 → 不触发

✅ "帮我写个东西" → ["写作与文档","汇报与提案"]
✅ "帮我翻译" → []（单域=不触发）
❌ "重启gateway" → []（单域）
❌ "帮我写一份30分钟的战略汇报方案" → []（含细节）

只返回JSON。无触发：{"domains":[]}
触发（≥2域）：{"domains":["域1","域2"]}"""


def _skill_routing_prompt(user_msg):
    """构建 skill 路由 Flash 扫描 prompt"""
    return f"""这条消息触发 skill 路由歧义吗？
「{user_msg}」

如果触发（≥2 个功能域），返回 {{"domains": ["域1","域2"]}}
如果不触发（0-1 个域），返回 {{"domains": []}}"""


V4_CONFIRM_PROMPT = """你是杨旸的认知模式确认器。初扫已识别以下规则可能触发。
你的任务：逐条确认是否真的触发，而非重新扫描。
确认触发 → 输出该条规则对应的 ⚡ 文案（必须从 RULES_REFERENCE 中选择，不可自创）。
不触发 → 跳过。
输出格式：{"tags": ["⚡ ..."]}"""


def _call_deepseek(user_msg, model, timeout, system_prompt=None):
    if not DEEPSEEK_KEY:
        return {"error": "no_api_key"}
    sp = system_prompt if system_prompt else SCAN_PROMPT
    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": sp},
                    {"role": "user", "content": user_msg},
                ],
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            return {"error": f"http_{resp.status_code}"}
        body = resp.json()
        msg = body.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "") or msg.get("reasoning_content", "")
        content = content.strip()
        if content.startswith("```"):
            parts = content.split("\n")
            if len(parts) >= 3:
                content = "\n".join(parts[1:-1])
        result = json.loads(content)
        if not isinstance(result, dict) or "triggered" not in result:
            return {"error": "parse_failed"}
        if isinstance(result["triggered"], str):
            result["triggered"] = [result["triggered"]]
        return result
    except requests.Timeout:
        return None
    except Exception:
        return {"error": "call_failed"}


def _v4_prompt(triggered_rules, user_msg):
    rules_text = ", ".join(triggered_rules)
    return f"""扫描杨旸最新消息，只判以下规则：{rules_text}
最新消息：「{user_msg}」
规则文案参考（必须使用以下文案，不可自创）：
{RULES_REFERENCE}
逐条判，确认触发→输出该条的⚡文案，不触发→跳过。
{{"tags": ["⚡ ..."]}}"""


def _log_scan(session_id, outcome, triggered, tags, scan_model="pro"):
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        entry = json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session_id": str(session_id),
            "outcome": outcome,
            "triggered": triggered or [],
            "tags": tags or [],
            "scan_model": scan_model,
        })
        with open(LOG_PATH, "a") as f:
            f.write(entry + "\n")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# 🔍 SOUL 合规扫描 — L0/L1 纯正则，无 API，<1ms
# 测试: Phase 1 16/16 ✅ + Phase 3 E1 已知 Qwen 假阴性（B 漏判）
# ═══════════════════════════════════════════════════════════════

import re as _compliance_re

_COMP_TIME_WORDS = r'(刚才|现在|今天|昨天|明天|今晚|明早|上周|下周|X分钟前|X小时前|X天前|X个月前|不久|马上|过一会|等一会)'
_COMP_DATE_BLOCK = r'```[\s\S]*?\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}[\s\S]*?```'
_COMP_TIMESTAMP = r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}'  # 已验证时间戳（2026-06-30 修改：接受任何形式的已验证时间戳，不强制代码块）
_COMP_KAAH_PATH = r'kaah/[\w/\-\.]+\.md'
_COMP_TIME_LABELS = r'[🟢🟡🔴⚪]'
_COMP_FILE_RECEIVED = r'\[The user sent a document'
_COMP_OPTION_ASK = r'[①②③1\.2\.3\.]'
_COMP_CLAIM_N = _compliance_re.compile(r'([一二两三四五六七八九十\d]+)\s*(方|项|条|个|份)')
_COMP_VERIFY_MARK = _compliance_re.compile(r'[✅⚠️☑️]')
_COMP_CHINESE_NUM = {"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
_COMP_WORK_KEYWORDS = r'(保和|张欣|叶冉|团队|汇报|Deck|画像|工作环境|公司)'


def _compliance_scan(user_msg, response_text):
    """L0/L1 SOUL 合规扫描。返回 ⚡ 标记列表。"""
    violations = []

    # TENTH: 用户说"第十人"但回复不含 triple_tenth_man.py → 违规（2026-06-30 部署）
    if _compliance_re.search(r'第十人|外部第十人|内外第十人', user_msg):
        if 'triple_tenth_man.py' not in response_text:
            violations.append("⚡TENTH 第十人审计：用户要求第十人审计，应调 triple_tenth_man.py 而非 delegate_task")

    # E: 引用 kaah 路径但无时效标签
    if _compliance_re.search(_COMP_KAAH_PATH, response_text) and not _compliance_re.search(_COMP_TIME_LABELS, response_text):
        violations.append("⚡E 来源标注：引用 kaah 知识库但无时效标签")

    # R0: 收到文件直接处理未反问
    if _compliance_re.search(_COMP_FILE_RECEIVED, user_msg):
        if '？' not in response_text and '?' not in response_text:
            violations.append("⚡R0 模糊指令：收到文件后直接处理，未反问用户选择操作")
        elif not _compliance_re.search(_COMP_OPTION_ASK, response_text):
            pass  # 有问号但无选项 → 可能是反问 → 放行

    # R17: 声称 N 方/项/条 验证/检查 但验证标记不足（2026-06-30 收紧触发范围：仅验证声明触发）
    R17_VERIFY_CLAIM = re.compile(r'(已|已经|经|验证|检查|跑完|做完|测完)\s*(了?\s*[一二两三四五六七八九十\d]+\s*(方|项|条|个|份))')
    if R17_VERIFY_CLAIM.search(response_text):
        claims = _COMP_CLAIM_N.findall(response_text)
        marks = _COMP_VERIFY_MARK.findall(response_text)
        for num_str, unit in claims:
            n = _COMP_CHINESE_NUM.get(num_str)
            if n is None:
                try:
                    n = int(num_str)
                except ValueError:
                    continue
            if n > 0 and len(marks) < n:
                violations.append(f"⚡R17 验证声明：声称 {n} {unit}但验证标记不足")
                break

    # B: 工作话题 + 给出分析/建议 + 无画像引用 → 违规（L0 前置拦截）
    # 2026-06-30 放宽：仅当给出分析/建议时才触发，纯汇报/确认不触发
    B_JUDGMENT_WORDS = re.compile(r'建议|分析|评估|推荐|判断|应该|可以试试|考虑|需要')
    if _compliance_re.search(_COMP_WORK_KEYWORDS, response_text):
        if B_JUDGMENT_WORDS.search(response_text):
            if '📋' not in response_text and '画像' not in response_text:
                violations.append("⚡B 画像优先：回复涉及工作话题但未先查画像")

    # ── L2 Qwen 一审（仅 L0/L1 无违规时触发，~1s，qwen-turbo）──
    if not violations:
        qwen_violations = _call_qwen_compliance(user_msg, response_text)
        violations.extend(qwen_violations)

    return violations


def _call_qwen_compliance(user_msg, response_text):
    """Qwen 一审 — A/B/C/F/G/I/J 七条规则。超时 2.5s，失败静默。"""
    import json as _json
    import urllib.request as _ur
    try:
        key_path = os.path.expanduser("~/.hermes/.bailian_key")
        with open(key_path) as f:
            key = f.read().strip()
    except Exception:
        return []

    prompt = f"""你是 SOUL 合规扫描器。逐条判断 DS 回复是否违反以下规则：

A) 方法门控：DS 给出≥2条路径/分步骤方案，是否先确认再动手？
B) 画像优先：回复对工作/个人话题给出分析/建议时，是否有📋画像引用？纯汇报不触发。
C) 历史优先：含判断性语言（分析/比较/评估/推荐）时，是否先搜了历史？
F) 刹车设计：新增 cron/skill/自动化时，是否同步设计了刹车机制？
G) Skill门控：加载低频 skill 时是否先建议再确认？
I) 闭环：交付物被批准后是否问了"入画像吗？要写ADR吗？"

用户消息：{user_msg}
DS回复：{response_text}

重要：B规则——如果回复中已有「📋」或「画像」字样，说明已查画像，不违规。纯事实查询（IP/路径/时间）不触发任何规则。

输出JSON：{{"violations":[{{"rule":"A~I","reason":"不超过15字"}}]}}
无违规返回：{{"violations":[]}}
"""
    try:
        data = _json.dumps({
            "model": "qwen-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150, "temperature": 0
        }).encode()
        req = _ur.Request(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            data,
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        )
        resp = _ur.urlopen(req, timeout=2.5)
        body = _json.loads(resp.read())
        content = body["choices"][0]["message"]["content"]
        # 剥离 markdown 代码块
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
        result = _json.loads(content)
        tags = []
        for v in result.get("violations", []):
            rule = v.get("rule", "?")
            reason = v.get("reason", "")
            tags.append(f"⚡{rule} Qwen一审：{reason}")
        return tags
    except Exception:
        return []


def _transform_llm_output(response_text, session_id=None, platform=None, **kwargs):
    # DEBUG
    with open("/tmp/l0_hook_called", "a") as f:
        f.write(f"l0 hook called | session={session_id} | platform={platform} | time={__import__('time').strftime('%H:%M:%S')}\n")
    # NODE 1: skip cron sessions
    if not session_id or str(session_id).startswith("cron_"):
        return None

    # NODE 2: get latest user message
    user_msg = _get_latest_user_message(session_id)
    if not user_msg:
        return None

    # NODE 2a: get session state (active skill, topic tree)
    session_state = _get_session_state(session_id)

    # NODE 3: Phase 1 — Flash scan with session context
    scan_result = _call_deepseek_flash_scan(user_msg, SCAN_TIMEOUT, session_state)

    # NODE 4: Phase 1 degraded — retry with Pro
    if scan_result is None or scan_result.get("error"):
        _log_scan(session_id, "p1_fail", [], [], "flash")
        scan_result = _call_deepseek(user_msg, "deepseek-v4-pro", SCAN_TIMEOUT)
        if scan_result is None or scan_result.get("error"):
            _log_scan(session_id, "p1_pro_fail", [], [], "pro")
            return None  # both attempts failed → silent pass

    triggered = scan_result.get("triggered", [])

    # NODE 5: no rules triggered — still run compliance scan below
    if not triggered:
        triggered = []  # ensure defined

    # ── NODE 6: Fast path for rule 7 (topic end, no Pro needed) + skill routing (separate Pro) ──
    cognitive_rules = [r for r in triggered if r in ("规则1","规则2","规则3","规则4","规则5","规则6")]
    end_rule = "规则7" in triggered

    tags = []

    # Rule 7: topic end — fixed message, no Pro needed
    if end_rule:
        user_msg_lower = user_msg.strip().lower()
        if user_msg_lower in ("算了", "放弃"):
            tags.append("⚡ 放弃这个话题？确认吗？")
        else:
            tags.append(TAG_MAP.get("规则7", "⚡ 这个话题结束了吗？"))

    # Skill routing: separate Flash call (cheap, accuracy acceptable for domain hint)
    # Only when no active skill + message looks like a vague action request
    if not session_state.get("has_active_skill"):
        skill_result = _call_deepseek(
            _skill_routing_prompt(user_msg),
            "deepseek-v4-flash", SCAN_TIMEOUT,
            system_prompt=SKILL_ROUTING_SP
        )
        if skill_result and not skill_result.get("error"):
            domains = skill_result.get("domains", [])
            if domains and len(domains) >= 2:
                msg = _build_rule7_message(domains)
                if msg:
                    tags.append(msg)

    # ── NODE 7: Phase 2 Pro confirmation for cognitive rules (1-6) ──
    if cognitive_rules:
        v4_result = _call_deepseek(
            _v4_prompt(cognitive_rules, user_msg),
            "deepseek-v4-pro", SCAN_TIMEOUT,
            system_prompt=V4_CONFIRM_PROMPT
        )
        if v4_result and not v4_result.get("error"):
            v4_tags = v4_result.get("tags", [])
            tags.extend(v4_tags)

    # ── NODE 7a: SOUL 合规扫描（L0/L1 纯正则，<1ms）──
    compliance_violations = _compliance_scan(user_msg, response_text)
    tags.extend(compliance_violations)

    # ── NODE 7b: MEDIA Gate — 响应中 MEDIA: 标签文件存在性验证（纯 bash，零 API）──
    media_warnings = _media_gate(response_text)
    tags.extend(media_warnings)

    # NODE 8: nothing to inject
    if not tags:
        return None

    # NODE 9: inject ⚡
    _log_scan(session_id, "inject", triggered, tags, "pro")
    prefix = "\n".join(tags)
    return f"{prefix}\n\n{response_text}"


def _media_gate(response_text: str) -> list:
    """扫描响应中 MEDIA: 标签，验证文件存在性。纯 bash + 零 API 调用。"""
    if "MEDIA:" not in response_text:
        return []

    import subprocess

    MEDIA_RE = re.compile(r'MEDIA:(\S+\.\w+)', re.IGNORECASE)
    SEND_MEDIA_SAFE = os.path.expanduser("~/hermes-media/send-media-safe.sh")

    warnings = []
    for match in MEDIA_RE.finditer(response_text):
        path = match.group(1).rstrip('",;:)}]')
        try:
            result = subprocess.run(
                [SEND_MEDIA_SAFE, path],
                capture_output=True, text=True, timeout=2
            )
        except Exception:
            continue
        if result.returncode != 0:
            filename = os.path.basename(path)
            err = result.stderr.strip().split('\n')[-1] if result.stderr else "文件验证失败"
            warnings.append(f"⚡ MEDIA Gate：`{filename}` {err}")

    return warnings