"""
结论锚点插件 — 异步输出侧事实验证 (v7)

架构:
  每轮 Pro 回复后 → hook 返回 None（让回复直接通过）
  → 后台 threading.Thread:
      ① Flash 主张提取
      ② Flash 直读 MEMORY.md 全文，逐条判断矛盾
      ③ 矛盾？→ send_weixin_direct() 追加 ⚡
      ④ 失败？→ 写 pending_alert 文件（下一轮注入）

v7 变更: 废除检索层（N5 embedding 0/10, N5.5 Flash+grep 1/10 双杀）
         改为 Flash 直读 MEMORY 全文做矛盾判断（N6 实测 10/10 召回）
"""

import asyncio  # 仅用于 asyncio.run() 调用 async send_weixin_direct
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MEMORY_PATH = os.path.expanduser("~/.hermes/memories/MEMORY.md")
PENDING_ALERT_DIR = os.path.expanduser("~/.hermes/pending_alerts")
VERIFY_TIMEOUT = 30  # Flash 全量 MEMORY 验证超时（55K chars ≈ 3-5s）
MIN_RESPONSE_LENGTH = 50  # 短回复跳过验证

# ── MEMORY.md 缓存 ──
_MEMORY_CACHE: str | None = None
_MEMORY_CACHE_TIME: float = 0
_MEMORY_CACHE_TTL = 300  # 5 分钟缓存


def _load_memory() -> str:
    """加载 MEMORY.md，5 分钟缓存"""
    global _MEMORY_CACHE, _MEMORY_CACHE_TIME
    now = time.time()
    if _MEMORY_CACHE is not None and (now - _MEMORY_CACHE_TIME) < _MEMORY_CACHE_TTL:
        return _MEMORY_CACHE
    try:
        with open(MEMORY_PATH) as f:
            _MEMORY_CACHE = f.read()
        _MEMORY_CACHE_TIME = now
        logger.info(f"conclusion-anchor: loaded MEMORY.md ({len(_MEMORY_CACHE)} chars)")
    except Exception:
        logger.warning("conclusion-anchor: failed to load MEMORY.md")
        _MEMORY_CACHE = ""
    return _MEMORY_CACHE


def _call_flash(system_prompt: str, user_content: str, timeout: int = VERIFY_TIMEOUT) -> dict:
    """调用 DeepSeek Flash API"""
    if not DEEPSEEK_KEY:
        return {"error": "no_api_key"}
    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",  # Flash = deepseek-chat (fast model)
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0,
                "max_tokens": 500,
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            return {"error": f"http_{resp.status_code}"}
        body = resp.json()
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"content": content, "model": body.get("model", "")}
    except requests.Timeout:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════
# Phase 1: 验证流水线
# ═══════════════════════════════════════════════

CLAIM_EXTRACTION_SP = """你是事实主张提取器。分析以下 Agent 回复，提取其中的事实性主张（factual claims）。

**事实性主张 = 可被验证真假的具体声明。**
✅ 提取: "P2-1 已完成" / "杨旸偏好极简风格" / "DS V4 上下文 128K" / "gem12 IP 是 192.168.31.192"
❌ 不提取: 建议/意见("建议用 Flash") / 推测("可能是配置问题") / 礼貌用语 / 引用标记 / 会话状态标记（orphans:N、队列状态、本轮扫描结果、💡标记、📊声明等）——这些是实时状态，不是永久事实

只返回 JSON 数组。无主张时返回空数组。
格式: [{"claim": "主张文本", "context": "原文上下文（50字以内）"}]

Agent 回复：
---
{response_text}
---"""


def _extract_claims(response_text: str) -> list[dict]:
    """Flash 主张提取 → [{claim, context}, ...]"""
    prompt = CLAIM_EXTRACTION_SP.replace("{response_text}", response_text)
    result = _call_flash("你是一个精确的事实主张提取器。", prompt, 15)
    if result.get("error"):
        logger.warning(f"conclusion-anchor: claim extraction failed: {result['error']}")
        return []

    try:
        content = result.get("content", "[]")
        # 提取 JSON 数组
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        if json_match:
            claims = json.loads(json_match.group())
            if isinstance(claims, list) and len(claims) > 0:
                logger.info(f"conclusion-anchor: extracted {len(claims)} claims")
                return claims
    except (json.JSONDecodeError, ValueError):
        logger.warning(f"conclusion-anchor: failed to parse claims JSON: {content[:200]}")
    return []

# ═══════════════════════════════════════════════
# B+: transient 过滤（2026-06-30 P1部署）
# ═══════════════════════════════════════════════
TRANSIENT_PATTERNS = [
    r"orphans\s*:\s*\d+",
    r"队列\s*\d+\s*件",
    r"队列\s*\[#\d+\]",
    r"^💡\s*队列",
    r"^💡\s*\d+\s*[个件]",
    r"^💡\s*以上矛盾",
    r"^\d+\s*[件个条项次]",
    r"^✅\s*队列",
    r"^📊\s*数据量",
    r"^🔍\s*来源",
]
TRANSIENT_KEYWORDS = ["本轮", "当前会话", "当前轮次", "扫描结果", "检查结果", "刚刚", "刚才"]

def _is_transient(claim_text: str) -> bool:
    """判断 claim 是否为会话瞬时状态（不参与 MEMORY 矛盾验证）"""
    for p in TRANSIENT_PATTERNS:
        if re.search(p, claim_text):
            return True
    # 启发式：短文本 + 数字 + 量词
    if re.search(r"\d+\s*[件个条项次]", claim_text) and len(claim_text) < 50:
        return True
    if any(kw in claim_text for kw in TRANSIENT_KEYWORDS):
        return True
    return False


# ═══════════════════════════════════════════════
# P1: 专有名词校验（2026-06-30 部署）
# ═══════════════════════════════════════════════

_ENTITY_WHITELIST: dict | None = None
_ENTITY_WHITELIST_PATH = os.path.expanduser("~/.hermes/config/entity_whitelist.yaml")


def _load_entity_whitelist() -> dict:
    """加载专有名词白名单（懒加载 + 5 分钟缓存）"""
    global _ENTITY_WHITELIST
    import time as _time
    now = _time.time()
    if _ENTITY_WHITELIST is not None:
        if now - _ENTITY_WHITELIST.get("_loaded_at", 0) < 300:
            return _ENTITY_WHITELIST
    try:
        import yaml
        with open(_ENTITY_WHITELIST_PATH) as f:
            _ENTITY_WHITELIST = yaml.safe_load(f) or {}
        _ENTITY_WHITELIST["_loaded_at"] = now
        total = sum(len(v) for k, v in _ENTITY_WHITELIST.items()
                    if isinstance(v, list) and not k.startswith("_"))
        logger.info(f"conclusion-anchor: loaded entity whitelist ({total} entities)")
    except Exception as e:
        logger.warning(f"conclusion-anchor: failed to load entity whitelist: {e}")
        _ENTITY_WHITELIST = {"_loaded_at": now}
    return _ENTITY_WHITELIST


def _word_match(pattern: str, text: str) -> bool:
    """自定义词边界匹配——entity前后不能是字母/数字/连字符（防止子串误匹配）
    
    例如：'plus' 不会匹配 'qwen-plus' 内部的 'plus'
    """
    for m in re.finditer(re.escape(pattern), text, re.IGNORECASE):
        start, end = m.start(), m.end()
        left_ok = (start == 0 or not re.match(r'[a-zA-Z0-9-]', text[start - 1]))
        right_ok = (end == len(text) or not re.match(r'[a-zA-Z0-9-]', text[end]))
        if left_ok and right_ok:
            return True
    return False


def _entity_check(claim_text: str, memory_text: str) -> bool:
    """检查 claim 中的专有名词是否与 MEMORY 记录冲突（v7：词边界匹配）

    Returns: True = 发现硬证据冲突（可用于升级 HARD）
    """
    whitelist = _load_entity_whitelist()
    if not whitelist:
        return False

    aliases_map = whitelist.get("aliases", {})

    # 收集 claim 中出现的所有已知专名（词边界匹配）
    claim_entities = set()
    for category in ("model_names", "service_names", "person_names", "project_names"):
        for entity in whitelist.get(category, []):
            if _word_match(entity, claim_text):
                claim_entities.add(entity)
            for alias in aliases_map.get(entity, []):
                if _word_match(alias, claim_text):
                    claim_entities.add(entity)

    if not claim_entities:
        return False

    # 断言词：claim中这些词后面的专名才是"被断言的"
    assertion_pre = (
        r'(?:是|为|等于|默认[用走]?|应该[用走]?|必须[用走]?'
        r'|走|用|切[换到]?|改成了?|改成?了?|用的?是|改叫|也有|也走|换成)'
    )
    assertion_patterns = [
        assertion_pre + r'\s*[\'"]?\s*{entity}',
        r'{entity}\s*(?:是|为|作为|被用作|被选为|也有|也走|换成)',
    ]

    for entity in claim_entities:
        if len(entity) <= 1:
            continue

        all_names = {entity}
        all_names.update(aliases_map.get(entity, []))

        # 检查 claim 中是否在断言此实体
        has_assertion = any(
            re.search(p.format(entity=re.escape(name)), claim_text)
            for p in assertion_patterns
            for name in all_names
        )
        if not has_assertion:
            continue

        # 检查 MEMORY 中是否有此实体的记录
        entity_in_memory = any(
            _word_match(name, memory_text)
            for name in all_names
        )

        if not entity_in_memory:
            # MEMORY 中没有此实体 → 检查同类别是否有其他实体
            for category in ("model_names", "service_names", "person_names", "project_names"):
                cat_entities = set(whitelist.get(category, []))
                if entity in cat_entities:
                    other_entities = cat_entities - {entity}
                    for other in other_entities:
                        all_other_names = {other}
                        all_other_names.update(aliases_map.get(other, []))
                        if any(_word_match(n, memory_text) for n in all_other_names):
                            logger.info(
                                f"conclusion-anchor: P1 entity conflict: "
                                f"claim='{entity}' vs memory='{other}' [{category}]"
                            )
                            return True
                    break
    return False


STATE_DB_PATH = os.path.expanduser("~/.hermes/state.db")


def _context_filter(claims: list[dict], session_id: str) -> list[dict]:
    """全 session 上下文消歧：claim 出现在 session 中 → 跳过 D3
    
    纯字符串匹配，零 API 成本。<100ms 查询 SQLite。
    """
    if not session_id:
        return claims

    try:
        con = sqlite3.connect(STATE_DB_PATH)
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT content FROM messages WHERE session_id = ? AND role IN ('user', 'assistant')"
            " AND content IS NOT NULL ORDER BY timestamp",
            (session_id,)
        ).fetchall()
        con.close()
    except Exception as e:
        logger.warning(f"conclusion-anchor: context filter DB error: {e}")
        return claims

    if not rows:
        return claims

    # 拼接全 session 文本（去重防重复匹配）
    session_text = " ".join(r["content"] for r in rows if r["content"])

    before = len(claims)
    filtered = []
    for c in claims:
        claim_text = c.get("claim", "")
        # 短 claim（<15字）跳过——太短容易假匹配
        if len(claim_text) < 15:
            filtered.append(c)
            continue
        # claim 文本出现在 session 中 → 跳过
        if claim_text in session_text:
            logger.info(f"conclusion-anchor: context filter skipped: {claim_text[:80]}")
            continue
        filtered.append(c)

    after = len(filtered)
    if before != after:
        logger.info(f"conclusion-anchor: context filter {before}→{after} claims")
    return filtered


FULLTEXT_VERIFY_PROMPT = """你是事实一致性校验器。下方是 MEMORY.md —— 关于杨旸及其环境的权威知识库。

对每条 CLAIM，判断矛盾级别（D3 三级判定，2026-06-30 部署）：
- "HARD_CONTRADICTION" = 与 MEMORY 记录的永久事实（配置、偏好、IP、身份信息）明确相反 → 需推⚡
- "SOFT_MISMATCH"    = 与 MEMORY 中的行为规则/约定不一致 → 记日志不推送
- "CONSISTENT"       = 一致 → 静默

⚠️ 核心原则:
- MEMORY 中的"规则"（如"应该怎么做""不要写什么"）不是永久事实。违反规则 = SOFT_MISMATCH。
- 仅当 claim 与 MEMORY 记录的永久事实明确相反时 = HARD_CONTRADICTION。
- 已过滤的请求不会到达这里——如果你看到会话标记（orphans:N、队列N件等），判 CONSISTENT。

返回所有主张的 JSON 数组:
[{{"claim": "原主张文本", "verdict": "HARD_CONTRADICTION|SOFT_MISMATCH|CONSISTENT", "explanation": "一句话"}}]

MEMORY.md:
---
{memory_text}
---

待验证主张 (CLAIMS):
{claims_json}"""


def _verify_claims_fulltext(claims: list[dict], session_id: str = None) -> list[dict]:
    """v9: B+ 预过滤 + 上下文消歧 + D3 三级判定 + P1 实体校验（2026-06-30 部署）"""
    # B+ 预过滤：移除 transient 会话状态标记
    before_filter = len(claims)
    claims = [c for c in claims if not _is_transient(c.get("claim", ""))]
    after_filter = len(claims)
    if before_filter != after_filter:
        logger.info(f"conclusion-anchor: B+ filtered {before_filter}→{after_filter} claims")

    # 上下文消歧：全 session 字符串匹配——claim 在对话中刚出现过 → 跳过
    claims = _context_filter(claims, session_id)

    memory = _load_memory()
    if not memory or not claims:
        return []

    # 构建 claims JSON
    claims_json = json.dumps(
        [{"id": i, "claim": c.get("claim", ""), "context": c.get("context", "")}
         for i, c in enumerate(claims)],
        ensure_ascii=False,
        indent=2,
    )

    prompt = FULLTEXT_VERIFY_PROMPT.format(
        memory_text=memory,
        claims_json=claims_json,
    )

    result = _call_flash(
        "你是一个精确的事实一致性校验器。只返回与 MEMORY.md 矛盾的主张。",
        prompt,
        VERIFY_TIMEOUT,
    )

    if result.get("error"):
        logger.warning(f"conclusion-anchor: fulltext verify failed: {result['error']}")
        return []

    try:
        content = result.get("content", "[]")
        logger.info(f"conclusion-anchor: Flash raw verify response ({len(content)} chars): {content[:300]}")
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                contradictions = [
                    {
                        "claim": c.get("claim", ""),
                        "explanation": c.get("explanation", ""),
                    }
                    for c in parsed
                    if c.get("verdict") == "HARD_CONTRADICTION"
                ]
                # P1: 实体校验增强 — 对非 HARD 的 claim 也做专名冲突检测
                entity_conflicts = []
                for c in parsed:
                    if c.get("verdict") != "HARD_CONTRADICTION":
                        claim_text = c.get("claim", "")
                        if _entity_check(claim_text, memory):
                            entity_conflicts.append({
                                "claim": claim_text,
                                "explanation": f"专有名词冲突: {c.get('explanation', 'D3未检测到')}",
                            })
                            logger.info(f"conclusion-anchor: P1 entity check found missed conflict: {claim_text[:80]}")
                if entity_conflicts:
                    contradictions.extend(entity_conflicts)
                    logger.warning(f"conclusion-anchor: P1 added {len(entity_conflicts)} entity-based contradictions")

                if contradictions:
                    logger.warning(f"conclusion-anchor: {len(contradictions)} HARD_CONTRADICTION(s) found, pushing ⚡")
                # Log SOFT_MISMATCH for observability (不推送，仅日志)
                soft_mismatches = [c for c in parsed if c.get("verdict") == "SOFT_MISMATCH"]
                if soft_mismatches:
                    logger.info(f"conclusion-anchor: {len(soft_mismatches)} SOFT_MISMATCH(es) logged (not pushed)")
                    for sm in soft_mismatches:
                        logger.info(f"  SOFT: {sm.get('claim','')[:80]} → {sm.get('explanation','')[:80]}")
                return contradictions
    except (json.JSONDecodeError, ValueError):
        logger.warning(f"conclusion-anchor: failed to parse verify JSON: {content[:200]}")

    return []


def _format_contradiction_alert(contradictions: list[dict]) -> str:
    """格式化 ⚡ 消息"""
    n = len(contradictions)
    lines = [f"⚡ 事实锚点：上一条回复存在 {n} 处可能矛盾\n"]
    for i, c in enumerate(contradictions, 1):
        claim = c.get("claim", "") if isinstance(c, dict) else str(c)
        lines.append(f'{i}. "{claim}"')
        if isinstance(c, dict) and c.get("explanation"):
            lines.append(f"   → {c['explanation']}")
        lines.append("")
    lines.append("💡 以上矛盾可能影响后续对话，建议确认后纠正。")
    return "\n".join(lines)


def _write_pending_alert(session_id: str, chat_id: str, alert_msg: str) -> str:
    """写入 pending_alert 文件 → 返回路径"""
    os.makedirs(PENDING_ALERT_DIR, exist_ok=True)
    ts = int(time.time())
    path = os.path.join(PENDING_ALERT_DIR, f"{session_id}_{ts}.json")
    data = {
        "session_id": session_id,
        "chat_id": chat_id,
        "alert_msg": alert_msg,
        "attempts": 0,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    return path


def _pop_pending_alert(session_id: str) -> dict | None:
    """读取最早未投递的 pending_alert 并清理"""
    if not os.path.exists(PENDING_ALERT_DIR):
        return None
    files = sorted(
        [f for f in os.listdir(PENDING_ALERT_DIR) if f.startswith(session_id)],
        key=lambda x: os.path.getmtime(os.path.join(PENDING_ALERT_DIR, x)),
    )
    for fname in files:
        path = os.path.join(PENDING_ALERT_DIR, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            os.remove(path)
            return data
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════
# Hook 主入口
# ═══════════════════════════════════════════════

def _verify_in_background(response_text: str, session_id: str, chat_id: str):
    """后台验证任务（在独立线程中运行）— v7: 主张提取→全文直读验证→⚡投递"""
    try:
        # Step 1: 主张提取
        claims = _extract_claims(response_text)
        if not claims:
            return

        # Step 2: 全文直读验证（v9: B+ 消歧 + D3 判定 + P1 实体校验）
        contradictions = _verify_claims_fulltext(claims, session_id)
        if not contradictions:
            return

        alert_msg = _format_contradiction_alert(contradictions)

        # Step 3: Priority 1 — 直发
        try:
            from gateway.platforms.weixin import send_weixin_direct
            result = asyncio.run(asyncio.wait_for(
                send_weixin_direct(
                    extra={},
                    token=None,
                    chat_id=chat_id,
                    message=alert_msg,
                ),
                timeout=5.0,
            ))
            if isinstance(result, dict) and result.get("success"):
                logger.info(f"conclusion-anchor: ⚡ sent to {chat_id}")
                return
            logger.warning(f"conclusion-anchor: direct send failed: {result}")
        except Exception as e:
            logger.warning(f"conclusion-anchor: direct send exception: {e}")

        # Step 4: Priority 2 — 写 pending_alert
        alert_path = _write_pending_alert(session_id, chat_id, alert_msg)
        logger.warning(f"conclusion-anchor: ⚡ queued to {alert_path}")

    except Exception as e:
        logger.error(f"conclusion-anchor: background verify crashed: {e}", exc_info=True)


def register(ctx):
    """注册 transform_llm_output hook"""
    # 落盘标记：证明 register() 被执行
    with open("/tmp/conclusion_anchor_loaded", "w") as f:
        f.write(f"loaded at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    ctx.register_hook("transform_llm_output", _verify_conclusions)
    logger.info("conclusion-anchor: registered v7 hook (fulltext verify)")


def _verify_conclusions(response_text, session_id=None, model=None, platform=None, **kwargs):
    """transform_llm_output hook 入口（同步，fire-and-forget 后台线程）"""
    # DEBUG: 每次调用落盘
    with open("/tmp/conclusion_hook_called", "a") as f:
        f.write(f"hook called | session={session_id} | platform={platform} | len={len(str(response_text))} | time={time.strftime('%H:%M:%S')}\n")

    # Guard 1: skip cron/internal sessions
    if not session_id or str(session_id).startswith("cron_"):
        with open("/tmp/conclusion_hook_called", "a") as f:
            f.write(f"  → BLOCKED at guard1: session={session_id}\n")
        return None

    # Guard 2: skip non-weixin platforms
    if platform != "weixin":
        with open("/tmp/conclusion_hook_called", "a") as f:
            f.write(f"  → BLOCKED at guard2: platform={platform}\n")
        return None

    # Guard 3: skip short responses
    if not response_text or not isinstance(response_text, str) or len(response_text.strip()) < MIN_RESPONSE_LENGTH:
        with open("/tmp/conclusion_hook_called", "a") as f:
            f.write(f"  → BLOCKED at guard3: len={len(str(response_text)) if response_text else 'None'}\n")
        return None

    # Guard 4: inject pending alert from previous turn (if any)
    try:
        pending = _pop_pending_alert(session_id) if session_id else None
        if pending and isinstance(pending, dict):
            alert = pending.get("alert_msg", "")
            if alert and isinstance(response_text, str):
                logger.info(f"conclusion-anchor: injecting pending ⚡ for session {session_id}")
                return response_text + "\n\n---\n" + alert
    except Exception:
        pass

    try:
        # Get chat_id from session context
        from gateway.session_context import get_session_env
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID")
        if not chat_id:
            return None

        # Fire-and-forget: start background verification in a daemon thread
        threading.Thread(
            target=_verify_in_background,
            args=(str(response_text), str(session_id), str(chat_id)),
            daemon=True,
        ).start()
    except Exception as e:
        logger.error(f"conclusion-anchor: hook setup failed: {e}", exc_info=True)

    return None  # let original response through unchanged
