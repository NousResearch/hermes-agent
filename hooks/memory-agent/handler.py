"""
Auto-extract key facts from each conversation and store in Holographic.
Mandatory entity linking — every fact gets at least one entity.
Keyword → entity_type routing (holographic-design spec).
Entity relations auto-built on co-occurrence (Phase3).

V2 升级 (2026-05-10):
- 扩大触发词（FACT_KEYWORDS 3x）
- 多 fact 提取（每 session 不限 1 条）
- 重要性评分（过滤噪音）
- 扩展 entity patterns（14类全覆盖）
- 智能分类路由
"""

import logging
import re
import sqlite3
import numpy as np
from pathlib import Path
from itertools import combinations

logger = logging.getLogger("hooks.memory-agent")

HERMES_HOME = Path.home() / ".hermes"
STATE_DB = HERMES_HOME / "state.db"
MEMORY_STORE_DB = HERMES_HOME / "memory_store.db"

VEC_DIM = 1024
_RNG = np.random.default_rng(42)

# 重要性评分阈值
IMPORTANCE_THRESHOLD = 0.35
# 每条 fact 最大字符数
FACT_MAX_CHARS = 600
# 每 session 最大提取 fact 数
MAX_FACTS_PER_SESSION = 8


def _random_hrr_vector(seed: int = None) -> np.ndarray:
    """Generate a random normalized HRR vector (1024-dim)."""
    rng = _RNG if seed is None else np.random.default_rng(seed)
    v = rng.standard_normal(VEC_DIM).astype(np.float64)
    v = v / (np.linalg.norm(v) + 1e-10)
    return v


def _circular_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution via FFT (frequency domain)."""
    A = np.fft.fft(a)
    B = np.fft.fft(b)
    return np.fft.ifft(A * B).real.astype(np.float64)


def _get_entity_vector(conn, entity_name: str) -> np.ndarray:
    """Get or create HRR vector for an entity."""
    cur = conn.execute(
        "SELECT vector FROM entity_vectors WHERE entity_name = ?", (entity_name,)
    )
    row = cur.fetchone()
    if row and row[0]:
        return np.frombuffer(row[0], dtype=np.float64)
    v = _random_hrr_vector()
    conn.execute(
        "INSERT OR IGNORE INTO entity_vectors (entity_name, vector) VALUES (?, ?)",
        (entity_name, v.tobytes())
    )
    return v


def _compose_fact_hrr(conn, entity_names: list[str]) -> np.ndarray | None:
    """Compose HRR vector for a fact by binding all its entity vectors."""
    if not entity_names:
        return None
    result = None
    for name in entity_names:
        v = _get_entity_vector(conn, name)
        if result is None:
            result = v
        else:
            result = _circular_convolve(result, v)
    if result is not None:
        norm = np.linalg.norm(result) + 1e-10
        result = (result / norm).astype(np.float64)
    return result


def _ensure_entity_vectors_table(conn) -> None:
    """Create entity_vectors table if not exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_vectors (
            entity_name TEXT PRIMARY KEY,
            vector BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


# ─────────────────────────────────────────────────────────────
# Extended entity patterns (14 types from holographic-design)
# ─────────────────────────────────────────────────────────────
ENTITY_PATTERNS = {
    "user": [
        "张哥", "用户", "老板", "你", "本人", "我", "咱们",
        "client", "customer", "user",
    ],
    "preference": [
        "偏好", "喜欢", "不喜欢", "希望", "不希望", "习惯",
        "微信不打扰", "简洁", "重点前置", "优先表格",
        "不想要", "受不了", "满意", "不错", "挺好的",
    ],
    "project": [
        "金融", "量化", "视频创作", "ai_quant", "hermes-upgrade",
        "hermes-backup", "罗马", "抖音", "B站",
    ],
    "workflow": [
        "Pipeline", "pipeline", "流程", "自动化", "复盘cron",
        "金融Pipeline", "cron", "调度", "定时任务",
    ],
    "tool": [
        "browser", "terminal", "cron", "hook", "skill", "pytesseract",
        "edge-tts", "kanban", "memory-agent", "hermes-startup",
        "dispatcher", "agent-browser", "FFmpeg", "git", "sqlite-vec",
        "fact_store", "hermes", "gateway",
    ],
    "platform": [
        "微信", "飞书", "telegram", "discord", "weixin", "feishu",
        "WeChat", "微博", "小红书", "抖音",
    ],
    "format": [
        "Excel", "表格", "报告格式", "彩色方案", "字幕", "配音",
        "飞书表格", "图表", "图示",
    ],
    "concept": [
        "工具链", "原理", "方法论", "架构", "设计",
        "HRR", "FTS5", "sqlite-vec", "RRF", "KNN",
        "召回", "写入", "检索", "检索",
    ],
    "pattern": [
        "成功", "修复", "解决", "搞定", "跑通了", "workaround",
        "好了", "完成", "搞定", "没问题",
    ],
    "error": [
        "错误", "失败", "bug", "崩溃", "断了", "连不上",
        "矛盾", "失效", "卡", "超时", "报错", "异常",
        "segmentation", "timeout", "crash", "failed",
    ],
    "decision": [
        "决策", "拍板", "结论", "最终决定", "用这个", "不改了",
        "就这样", "行", "可以", "按这个来", "就这么定了",
        "以后都用这个", "以后用这个",
    ],
    "todo": [
        "待做", "需要做", "下一步", "还没做", "未完成",
        "要做", "待办", "计划", "准备做",
    ],
    "idea": [
        "想法", "灵感", "创意", "机会", "可以试试", "也许",
        "要是能", "要不", "也许可以", "感觉可以",
    ],
    "session": [
        "会话", "对话", "讨论", "聊", "聊到",
    ],
}


# ─────────────────────────────────────────────────────────────
# Expanded FACT_KEYWORDS (3x original)
# ─────────────────────────────────────────────────────────────
FACT_KEYWORDS = [
    # ── 确认类 ──
    "记住", "记住了", "明白了", "懂了", "知道了",
    "行", "可以", "好的", "好", "没问题",
    "这个好", "这个不错", "挺好的", "不错",
    "按这个来", "就这样", "就这么定了",
    "你说得对", "说得对", "有道理",
    "以后都用这个", "以后用这个", "以后就",
    # ── 偏好类 ──
    "偏好", "喜欢", "不喜欢", "讨厌", "希望",
    "不希望", "习惯", "要", "不要", "别",
    "一定", "必须", "不要", "微信不打扰",
    # ── 指示类 ──
    "以后", "下次", "将来", "记得",
    "要记住", "别忘了", "应该", "得",
    # ── 配置/设置类 ──
    "配置", "设置", "改成", "改成",
    "activate", "enabled", "disabled",
    # ── 重要性/强调 ──
    "重要", "关键", "核心", "必须记住",
    # ── 待办/计划类 ──
    "待做", "需要做", "下一步", "还没做", "未完成",
    "要做", "待办", "计划", "准备做",
    # ── 错误/问题类 ──
    "错误", "失败", "bug", "崩溃", "断了", "连不上",
    "报错", "异常", "超时", "卡", "失效", "矛盾",
    # ── 想法/创意类 ──
    "想法", "灵感", "创意", "机会", "也许",
    "要是能", "要不", "也许可以", "感觉可以",
    # ── 决策/结论类 ──
    "决策", "拍板", "结论", "最终决定",
    "用这个", "不改了", "就这样",
    # ── 成功/修复类 ──
    "成功", "修复", "解决", "搞定", "跑通了", "完成",
    "好了", "没问题", "搞定",
    # ── 英文 ──
    "remember", "user said", "配置", "setting",
]


# ─────────────────────────────────────────────────────────────
# 重要性评分
# ─────────────────────────────────────────────────────────────
def _importance_score(text: str) -> float:
    """Score how important a piece of content is (0.0–1.0).

    Higher = more worth storing as a fact.
    """
    score = 0.0
    text_lower = text.lower()

    # 否定指令（表示用户明确要求什么）
    if re.search(r"不要|别|不准|不能|不要", text):
        score += 0.35
    # 确认类（表示接受了某个信息）
    if re.search(r"记住了|明白了|懂了|行|可以|没问题|就这样", text):
        score += 0.30
    # 未来指示（表示以后要用什么方式）
    if re.search(r"以后|下次|将来|记得|要记住|别忘", text):
        score += 0.35
    # 配置/参数类（数字、路径、配置项）
    if re.search(r"(0x[0-9a-fA-F]|https?://|/[^/\s]+/|:\d+|password|token|key|api)", text):
        score += 0.25
    # 编号列表（1. 2. A. B. 或 一、二、三、）
    if re.search(r"\d+[.、)）]|[一二三四][、.)]", text):
        score += 0.20
    # 长度适中（太短没信息量，太长是叙述）
    if 30 <= len(text) <= 500:
        score += 0.15
    # 情绪词（表示态度）
    if re.search(r"喜欢|讨厌|受不了|满意|偏好", text):
        score += 0.30
    # 明确要求（必须、一定）
    if re.search(r"必须|一定|不准", text):
        score += 0.35
    # 结论词
    if re.search(r"结论是|决定是|就这样|不改了", text):
        score += 0.40
    # 问题+解决方案（同时有错误和解决）
    if re.search(r"错误|失败|bug|崩溃", text) and re.search(r"修复|解决|搞定|好了", text):
        score += 0.45

    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────
# 内容分块（多 fact 提取）
# ─────────────────────────────────────────────────────────────
def _split_into_chunks(text: str) -> list[str]:
    """Split session text into meaningful chunks for multi-fact extraction.

    Splits by:
    1. Double newlines (paragraphs)
    2. Numbered lists (1. 2. / A. B.)
    3. Sentence boundaries (。！？)
    """
    if not text or len(text) < 15:
        return [text] if text else []

    chunks = []

    # 先按空行分段落
    paragraphs = re.split(r"\n\s*\n", text)
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 段落太短，直接用
        if len(para) <= FACT_MAX_CHARS:
            chunks.append(para)
            continue

        # 段落太长，按编号/序号拆分
        sub_parts = re.split(r"(?:\n|(?=\d+[.、])|(?=[A-Z][.、]))", para)
        for part in sub_parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= FACT_MAX_CHARS:
                chunks.append(part)
            else:
                # 再按句号拆分
                sentences = re.split(r"[。！？.!?]", part)
                current = ""
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    if len(current) + len(s) <= FACT_MAX_CHARS:
                        current = (current + "。" + s).strip() if current else s
                    else:
                        if current:
                            chunks.append(current)
                        current = s
                if current:
                    chunks.append(current)

    return [c for c in chunks if len(c) >= 10]


# ─────────────────────────────────────────────────────────────
# 实体提取（增强版）
# ─────────────────────────────────────────────────────────────
def _extract_entities(text: str) -> list[tuple[str, str]]:
    """Detect entities in text. Returns list of (entity_name, entity_type)."""
    found = []
    seen = set()
    for ent_type, keywords in ENTITY_PATTERNS.items():
        for kw in keywords:
            if kw in text and kw not in seen:
                found.append((kw, ent_type))
                seen.add(kw)
    return found


# ─────────────────────────────────────────────────────────────
# 智能分类路由
# ─────────────────────────────────────────────────────────────
def _route_category(content: str) -> str:
    """Route content to category based on expanded keyword rules."""
    # 偏好类
    pref_kw = ["偏好", "喜欢", "不喜欢", "讨厌", "希望", "习惯", "受不了",
               "简洁", "优先表格", "不想要", "满意", "挺好的"]
    if any(kw in content for kw in pref_kw):
        return "preference"

    # 决策/结论类（优先级最高）
    dec_kw = ["决策", "拍板", "结论", "最终决定", "用这个", "不改了",
              "就这样", "行", "可以", "按这个来", "就这么定了",
              "记住了", "明白了", "懂了"]
    if any(kw in content for kw in dec_kw):
        return "decision"

    # 待办类
    todo_kw = ["待做", "需要做", "下一步", "还没做", "未完成",
                "要做", "待办", "计划", "准备做"]
    if any(kw in content for kw in todo_kw):
        return "todo"

    # 错误类
    err_kw = ["错误", "失败", "bug", "崩溃", "断了", "连不上",
              "报错", "异常", "超时", "卡", "失效", "segmentation"]
    if any(kw in content for kw in err_kw):
        return "error"

    # 修复/成功类
    pat_kw = ["成功", "修复", "解决", "搞定", "跑通了", "好了", "没问题"]
    if any(kw in content for kw in pat_kw):
        return "pattern"

    # 想法类
    idea_kw = ["想法", "灵感", "创意", "机会", "也许", "要是能", "要不"]
    if any(kw in content for kw in idea_kw):
        return "idea"

    # 项目/工作流类
    proj_kw = ["金融", "量化", "视频", "Pipeline", "pipeline", "自动化",
               "cron", "定时任务", "调度", "罗马", "抖音", "B站"]
    if any(kw in content for kw in proj_kw):
        return "project" if any(kw in content for kw in ["金融", "量化", "罗马", "抖音", "B站"]) else "workflow"

    return "general"


def _ensure_entity(cursor, name: str, ent_type: str = "unknown") -> int:
    """Ensure entity exists in DB, return entity_id. Uses existing entity if already present."""
    cursor.execute("SELECT entity_id FROM entities WHERE name = ?", (name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute(
        "INSERT OR IGNORE INTO entities (name, entity_type) VALUES (?, ?)",
        (name, ent_type)
    )
    cursor.execute("SELECT entity_id FROM entities WHERE name = ?", (name,))
    row = cursor.fetchone()
    return row[0] if row else None


def _store_fact(cursor, content: str, category: str, tags: str, entity_ids: list[int],
                entity_names: list[str] = None) -> int | None:
    """Store a fact, link to entities, and compose HRR vector."""
    if not entity_ids or not content:
        return None
    try:
        hrr_bytes = None
        if entity_names:
            hrr = _compose_fact_hrr(cursor, entity_names)
            if hrr is not None:
                hrr_bytes = hrr.tobytes()

        cursor.execute("""
            INSERT INTO facts (content, category, tags, trust_score, hrr_vector)
            VALUES (?, ?, ?, 0.5, ?)
        """, (content, category, tags, hrr_bytes))
        fact_id = cursor.lastrowid
        for eid in entity_ids:
            cursor.execute(
                "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                (fact_id, eid)
            )
        return fact_id
    except sqlite3.IntegrityError:
        return None


def _store_entity_relations(cursor, entity_names: list[str], fact_id: int) -> None:
    """Auto-build co-occurrence relations between entities in the same fact."""
    if len(entity_names) < 2:
        return
    try:
        for a, b in combinations(sorted(set(entity_names)), 2):
            cursor.execute("""
                INSERT OR IGNORE INTO entity_relations
                (from_entity, to_entity, relation_type, source_fact_id)
                VALUES (?, ?, 'co-occurrence', ?)
            """, (a, b, fact_id))
    except Exception as e:
        logger.warning("Failed to store entity relations: %s", e)


def _store_git_change_fact(session_id: str, change_info: dict) -> None:
    """Store significant git changes as a fact with entity links."""
    if change_info["status"] == "clean" or not change_info.get("changed"):
        return

    significant_patterns = [
        "config.yaml", "SOUL.md", "BOOT.md",
        "hooks/", "cron/jobs.json",
        "profiles/", "skills/",
        ".gitignore",
    ]
    significant = [
        c for c in change_info["changed"]
        if any(p in c["path"] for p in significant_patterns)
    ]
    if not significant:
        return

    try:
        conn = sqlite3.connect(str(MEMORY_STORE_DB))
        cur = conn.cursor()

        hermes_backup_id = _ensure_entity(cur, "hermes-backup", "project")
        session_eid = _ensure_entity(cur, session_id[:8], "session")
        entity_ids = [eid for eid in [hermes_backup_id, session_eid] if eid]
        entity_names = ["hermes-backup", session_id[:8]]

        content = (f"Git changed during session {session_id[:8]}: "
                   + ", ".join(f"{c['status']}:{c['path']}" for c in significant))

        _ensure_entity_vectors_table(cur)

        fid = _store_fact(cur, content, "tool", "git,hermes-backup", entity_ids, entity_names)

        if fid:
            _store_entity_relations(cur, entity_names, fid)
            logger.info("Stored git change fact %d for session %s", fid, session_id[:8])

        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("Could not store git change fact: %s", e)


def _process_session(session_id: str) -> int:
    """Process a session, extract MULTIPLE facts, store to Holographic.

    V2 changes:
    - Split session into chunks → extract up to MAX_FACTS_PER_SESSION facts
    - Score each chunk for importance → skip below threshold
    - Each chunk gets its own category + entity extraction
    """
    messages = _get_session_messages(session_id, limit=60)
    if not messages:
        return 0

    # 合并 user 消息
    user_content = " ".join(
        m["content"][:500] for m in messages
        if m["role"] == "user" and m["content"]
    )

    if not user_content:
        return 0

    # 检查是否有关键词（快速过滤）
    has_keyword = any(kw.lower() in user_content.lower() for kw in FACT_KEYWORDS)
    importance = _importance_score(user_content)
    if not has_keyword and importance < IMPORTANCE_THRESHOLD:
        logger.info("Session %s: no important facts found (score=%.2f)", session_id[:8], importance)
        return 0

    # 分块
    chunks = _split_into_chunks(user_content)
    if not chunks:
        return 0

    stored = 0
    try:
        conn = sqlite3.connect(str(MEMORY_STORE_DB))
        cur = conn.cursor()
        _ensure_entity_vectors_table(cur)

        for chunk in chunks[:MAX_FACTS_PER_SESSION]:
            chunk = chunk.strip()
            if len(chunk) < 15:
                continue

            chunk_score = _importance_score(chunk)
            if chunk_score < IMPORTANCE_THRESHOLD:
                continue

            # 每个 chunk 独立提取实体和分类
            entities = list(set(_extract_entities(chunk)))
            entity_names = [name for name, _ in entities]
            entity_ids = []
            for name, ent_type in entities:
                eid = _ensure_entity(cur, name, ent_type)
                if eid:
                    entity_ids.append(eid)

            # fallback
            if not entity_ids:
                zge_id = _ensure_entity(cur, "张哥", "user")
                if zge_id:
                    entity_ids.append(zge_id)
                entity_names.append("张哥")
                logger.warning("Chunk in session %s: no entities, fallback to 张哥", session_id[:8])

            if not entity_ids:
                continue

            category = _route_category(chunk)
            tags = ",".join(set(ent_type for _, ent_type in entities)) if entities else "general"

            fid = _store_fact(cur, chunk, category, tags, entity_ids, entity_names)
            if fid:
                stored += 1
                _store_entity_relations(cur, entity_names, fid)
                logger.info(
                    "Stored fact %d (score=%.2f) for session %s (category=%s)",
                    fid, chunk_score, session_id[:8], category
                )
            # 已存在的 fact 会因 UNIQUE constraint 返回 None，不会报错

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Failed to store facts: %s", e)

    return stored


async def handle(event_type: str, context: dict) -> None:
    """Gateway hook: agent:end"""
    session_id = context.get("session_id")
    if not session_id:
        logger.warning("agent:end received but no session_id in context")
        return

    logger.info("memory-agent processing session %s", session_id[:8])

    # Git 变更检测
    git_info = _check_git_changes_since()
    if git_info["status"] == "dirty":
        _store_git_change_fact(session_id, git_info)

    # 多 fact 提取
    count = _process_session(session_id)
    if count > 0:
        logger.info("memory-agent: stored %d facts from session %s", count, session_id[:8])
    else:
        logger.info("memory-agent: no new facts from session %s", session_id[:8])


# ─────────────────────────────────────────────────────────────
# Git 检测（保持不变）
# ─────────────────────────────────────────────────────────────
def _check_git_changes_since() -> dict:
    """Get git diff of uncommitted changes in ~/.hermes."""
    import subprocess
    hermes_home = Path.home() / ".hermes"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(hermes_home), capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return {"status": "no git repo", "changed": []}
        current_hash = result.stdout.strip()

        result = subprocess.run(
            ["git", "diff", "--name-status", "HEAD"],
            cwd=str(hermes_home), capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return {"status": "diff failed", "changed": []}

        changes = result.stdout.strip()
        if not changes:
            return {"status": "clean", "hash": current_hash, "changed": []}

        lines = changes.splitlines()
        changed_files = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) == 2:
                status, filepath = parts
                changed_files.append({"status": status, "path": filepath})

        return {"status": "dirty", "hash": current_hash, "changed": changed_files}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "changed": []}
    except Exception as e:
        return {"status": f"error: {e}", "changed": []}


# ─────────────────────────────────────────────────────────────
# 辅助：读取 session 消息（保持不变）
# ─────────────────────────────────────────────────────────────
def _get_session_messages(session_id: str, limit: int = 60) -> list[dict]:
    """Fetch recent messages for a session from state.db."""
    try:
        conn = sqlite3.connect(str(STATE_DB))
        conn.row_factory = sqlite3.Row
        cur = conn.execute("""
            SELECT role, content FROM messages
            WHERE session_id = ? AND content IS NOT NULL
            ORDER BY timestamp DESC LIMIT ?
        """, (session_id, limit))
        rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("Could not read session messages: %s", e)
        return []
