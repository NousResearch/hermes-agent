"""
Hermes pre_llm_call plugin — self-contained skill router with 5-layer pipeline.

Zero external dependencies. BM25 + trigger phrase index + regex rules +
Levenshtein fuzzy fallback + task gate — all in ~350 lines.
LLM sees only top-3 skills, not 332. 2-10ms per query.

Architecture (community-verified, 2026-06-15):
  on_session_start → scan SKILL.md files → build in-memory index + load triggers
  pre_llm_call  →
    1. Task Gate     → skip small talk ("thanks", "ok")
    2. Trigger Match → O(1) exact phrase lookup
    3. Rule Check    → regex priority rules (48 rules default)
    4. BM25 Routing  → TF-IDF keyword ranking
    5. Levenshtein   → typo-tolerant correction (when BM25 returns nothing)
    All fail         → empty context (AI uses built-in tools)
"""

import json
import math
import os
import re
import sys
import threading
from pathlib import Path

# ── Configuration ──

_SKILL_DIRS_CACHE = None
_RouteIndex = None
_RouteLock = threading.Lock()
_RulesCache = None
_TriggerIndex = None
_TriggerLock = threading.Lock()

# ── Skill directory detection ──

def _get_skill_dirs() -> list[str]:
    global _SKILL_DIRS_CACHE
    if _SKILL_DIRS_CACHE is not None:
        return _SKILL_DIRS_CACHE

    home = Path.home()
    local = os.environ.get("LOCALAPPDATA", "") if sys.platform == "win32" else ""
    hermes_home = os.environ.get("HERMES_HOME", "")

    dirs = [
        str(home / ".claude" / ".skills-store" / "skills"),
        str(home / ".claude" / ".skills-store" / "agents"),
        str(home / ".claude" / "skills"),
        str(home / ".claude" / "agents"),
        str(home / ".claude" / ".agents" / "skills"),
        str(home / ".hermes" / "skills"),
        str(home / ".hermes" / "agents"),
    ]
    if local:
        dirs.append(str(Path(local) / "hermes" / "skills"))
        dirs.append(str(Path(local) / "hermes-agent" / "skills"))
    if hermes_home:
        dirs.append(str(Path(hermes_home) / "skills"))
        dirs.append(str(Path(hermes_home) / "agents"))

    _SKILL_DIRS_CACHE = sorted(set(d for d in dirs if Path(d).is_dir()))
    return _SKILL_DIRS_CACHE


# ── Skill file discovery ──

def _find_skill_files(dirs: list[str]) -> list[tuple[str, str, str]]:
    """Scan directories for SKILL.md and .md agent files.
    Returns list of (name, description, search_text).
    """
    seen = set()
    skills = []

    for d in dirs:
        dp = Path(d)
        if not dp.exists():
            continue
        for item in sorted(dp.iterdir()):
            name = None
            description = ""
            filepath = None

            if item.is_dir():
                smd = item / "SKILL.md"
                if smd.exists():
                    filepath = smd
            elif item.is_file() and item.suffix == ".md":
                filepath = item

            if filepath is None:
                continue

            try:
                text = filepath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            meta, body = _parse_frontmatter(text)
            name = meta.get("name", filepath.stem)
            description = meta.get("description", "")

            if name in seen or len(text.strip()) < 20:
                continue
            seen.add(name)

            search_text = f"{name} {description} {body[:500]}".lower()
            skills.append((name, description, search_text))

    return skills


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from skill text."""
    text = text.strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                import yaml
                meta = yaml.safe_load(parts[1]) or {}
            except Exception:
                meta = {}
            return meta, parts[2].strip()
    return {}, text


# ── Tokenizer ──

# Frequently used Chinese tech terms — when found, treated as whole words
# (not bigram-split). Just bigram fallback covers the rest — this list
# only needs to cover the most common query terms.
_CN_TECH_TERMS = [
    # Performance / 性能
    "性能", "优化", "加速", "缓存", "延迟", "响应",
    # Security / 安全
    "安全", "漏洞", "注入", "扫描", "审计", "加密", "认证",
    # Deploy / 部署
    "部署", "发布", "上线", "回滚", "配置", "环境",
    # Database / 数据库
    "数据库", "查询", "索引", "事务", "备份", "恢复",
    # Frontend / 前端
    "前端", "组件", "页面", "路由", "渲染", "样式",
    # Backend / 后端
    "后端", "接口", "服务", "网关", "队列", "消息",
    # Testing / 测试
    "测试", "单元", "集成", "覆盖", "回归", "自动化",
    # Code quality / 代码质量
    "代码", "重构", "审查", "规范", "错误", "调试",
    # General tech / 通用技术
    "文件", "搜索", "检索", "构建", "编译", "打包", "安装",
    "容器", "镜像", "编排", "监控", "日志", "告警",
    "网络", "代理", "域名", "协议", "链接",
    "架构", "设计", "文档", "模板", "工具",
]


def _tokenize_set(text: str) -> set[str]:
    tokens = set()
    tokens.update(re.findall(r"[a-z0-9]{2,}", text.lower()))

    # Chinese tech terms first (whole-word match, more precise)
    lower = text.lower()
    for term in _CN_TECH_TERMS:
        if term in lower:
            tokens.add(term)

    # Then bigram fallback for everything else
    tokens.update(re.findall(r"[一-鿿]{2,6}", text.lower()))
    return tokens


def _tokenize_list(text: str) -> list[str]:
    text_lower = text.lower()
    tokens = re.findall(r"[a-z0-9]{2,}", text_lower)

    # Chinese tech terms as whole words
    for term in _CN_TECH_TERMS:
        if term in text_lower:
            tokens.append(term)

    tokens.extend(re.findall(r"[一-鿿]{2,6}", text_lower))
    return tokens


# ── BM25 Keyword Index ──

class _KeywordIndex:
    """Minimal BM25 keyword index. No numpy needed."""

    def __init__(self, skills: list[tuple[str, str, str]]):
        self.skills = skills
        # search_text already includes name + description + body[:500]
        self._doc_tokens = [_tokenize_list(st) for _, _, st in skills]
        # Boost: name + description tokens get 2x weight (they're in search_text
        # once to begin with; appending once more = 2x total)
        for i, (name, desc, _) in enumerate(skills):
            boost_text = f"{name} {desc}".lower()
            self._doc_tokens[i].extend(_tokenize_list(boost_text))
        self._doc_token_sets = [set(t) for t in self._doc_tokens]
        self._doc_lens = [len(t) for t in self._doc_tokens]
        self._avgdl = sum(self._doc_lens) / max(len(skills), 1)
        self._inverted = {}
        for i, token_set in enumerate(self._doc_token_sets):
            for t in token_set:
                self._inverted.setdefault(t, []).append(i)

        N = len(skills)
        self._idf = {}
        for term, docs in self._inverted.items():
            self._idf[term] = math.log(1 + (N - len(docs) + 0.5) / (len(docs) + 0.5))

    def query(self, text: str, top_k: int = 3,
              k1: float = 1.2, b: float = 0.75) -> list[tuple[str, float]]:
        q_tokens = _tokenize_set(text)
        if not q_tokens:
            return []

        N = len(self.skills)
        scores = [0.0] * N
        avgdl = max(self._avgdl, 1)

        for qt in q_tokens:
            idf = self._idf.get(qt, 0)
            if idf == 0:
                continue
            for doc_idx in self._inverted.get(qt, []):
                tf = self._doc_tokens[doc_idx].count(qt)
                dl = self._doc_lens[doc_idx]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / avgdl)
                scores[doc_idx] += idf * numerator / max(denominator, 0.01)

        ranked = sorted(
            [(self.skills[i][0], scores[i]) for i in range(N) if scores[i] > 0],
            key=lambda x: -x[1],
        )[:top_k]
        return ranked


# ── Rule Engine ──

def _load_rules() -> list[dict]:
    global _RulesCache
    if _RulesCache is not None:
        return _RulesCache
    rules_path = Path.home() / ".neuro-skill" / "rules.json"
    if rules_path.exists():
        try:
            _RulesCache = json.loads(rules_path.read_text(encoding="utf-8"))
            return _RulesCache
        except Exception:
            pass
    _RulesCache = []
    return _RulesCache


def _check_rules(query: str) -> str | None:
    rules = _load_rules()
    for rule in rules:
        pattern = rule.get("pattern", "")
        skill = rule.get("skill", "")
        if pattern and skill:
            try:
                if re.search(pattern, query, re.IGNORECASE):
                    return skill
            except re.error:
                pass
    return None


# ── Layer 1: Task Gate ──

# Chinese + English small-talk / acknowledgment phrases that should
# NOT trigger routing. These are conversational, not task-oriented.
_TASK_GATE_SKIP = re.compile(
    r"^(谢谢|好的|嗯|哦|OK|好|知道了|了解|明白|收到|thanks?|okay|got it|"
    r"well done|great|nice|awesome|cool|讲得.*好|说得.*好|不错|可以|行|"
    r"不客气|没关系|没.?事|不用.?谢|再见|拜拜|bye|hello|hi|hey|"
    r"早上好|下午好|晚上好|good morning|good afternoon|good evening|"
    r"晚安|good night|好梦|哈哈+|嘻嘻|嘿嘿|呵呵+|"
    r"测试一下|test|测试)",
    re.IGNORECASE
)


def _is_task_query(text: str) -> bool:
    """Returns False for small talk / ack — router should skip these."""
    # Strip punctuation for matching
    t = re.sub(r"[!！。，,\.\s]+$", "", text.strip()).lower()
    if len(t) <= 1:
        return False
    if _TASK_GATE_SKIP.fullmatch(t):
        return False
    if re.fullmatch(r"[\U0001F300-\U0001F9FF\U00002700-\U000027BF\W]+", t):
        return False
    return True


# ── Layer 2: Trigger Phrase Index ──

def _load_trigger_index():
    """Load trigger phrase → skill_name mapping from trigger-index.json."""
    global _TriggerIndex
    with _TriggerLock:
        if _TriggerIndex is not None:
            return
        _TriggerIndex = {}
        # Search same directory as this file
        trigger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "trigger-index.json")
        if not os.path.isfile(trigger_path):
            # Fallback: check cwd
            trigger_path = "trigger-index.json"
        if not os.path.isfile(trigger_path):
            return  # No trigger file — skip this layer
        try:
            with open(trigger_path, encoding="utf-8") as f:
                idx = json.load(f)
            for skill_name, phrases in idx.items():
                for phrase in phrases:
                    _TriggerIndex[phrase.lower().strip()] = skill_name
        except Exception as e:
            import logging
            logging.getLogger("neuro_skill").warning(
                "Failed to load trigger-index.json: %s", e
            )


def _trigger_match(query: str) -> str | None:
    """O(1) exact trigger phrase lookup. Returns skill name or None."""
    if _TriggerIndex is None:
        return None
    q = query.lower().strip()
    # Direct lookup
    if q in _TriggerIndex:
        return _TriggerIndex[q]
    # Substring match (for longer queries containing trigger phrases)
    for phrase, skill in sorted(_TriggerIndex.items(),
                                 key=lambda x: -len(x[0])):  # longest first
        if phrase in q:
            return skill
    return None


# ── Layer 5: Levenshtein Fuzzy Fallback ──

def _levenshtein(s1: str, s2: str) -> int:
    """Edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,        # insertion
                curr[j] + 1,             # deletion
                prev[j] + (c1 != c2),   # substitution
            ))
        prev = curr
    return prev[-1]


def _fuzzy_keyword_correction(query: str, known_terms: set[str],
                               max_dist: int = 2) -> str:
    """Return query with misspelled keywords corrected via edit distance.

    Only corrects terms that look like English words (>= 3 chars).
    Each term is compared against known terms; the closest match within
    max_dist is used as replacement.
    """
    words = re.findall(r"[a-z]{3,}", query.lower())
    if not words:
        return query

    corrected = query.lower()
    for w in set(words):
        if w in known_terms:
            continue
        best = None
        best_dist = max_dist + 1
        for t in known_terms:
            d = _levenshtein(w, t)
            if d < best_dist:
                best_dist = d
                best = t
                if d == 1:
                    break  # early exit — good enough
        if best:
            corrected = corrected.replace(w, best)

    return corrected


# ── Hook Handlers ──

def on_session_start(**kwargs):
    """Scan skills + load trigger index. ~500ms for 332 skills, one-time."""
    global _RouteIndex
    with _RouteLock:
        if _RouteIndex is not None:
            return
        try:
            dirs = _get_skill_dirs()
            skills = _find_skill_files(dirs)
            _RouteIndex = _KeywordIndex(skills)
            _load_trigger_index()
        except Exception:
            import logging
            logging.getLogger("neuro_skill").warning(
                "Failed to build skill index", exc_info=True
            )


def pre_llm_call(**kwargs) -> dict:
    """5-layer routing pipeline. Returns {"context": str} for Hermes."""
    global _RouteIndex, _TriggerIndex

    user_message = kwargs.get("user_message", "")
    if not user_message or not user_message.strip():
        return {}

    # ── Layer 0: Task Gate ──
    if not _is_task_query(user_message):
        return {}

    # Build index on first call (fallback)
    if _RouteIndex is None:
        try:
            on_session_start()
        except Exception:
            return {}

    if _RouteIndex is None:
        return {}

    # ── Layer 1: Trigger Phrase Exact Match ──
    trigger_skill = _trigger_match(user_message)
    if trigger_skill:
        return {"context": (
            f"[Top 3 skills for this query]\n"
            f"  (trigger-matched)\n"
            f"  1. {trigger_skill} (1.000)\n"
            f"\n"
            f"Trigger phrase matched — use this skill."
        )}

    # ── Layer 2: Rule Check ──
    rule_match = _check_rules(user_message)
    if rule_match:
        return {"context": (
            f"[Top 3 skills for this query]\n"
            f"  (Rule: {rule_match})\n"
            f"  1. {rule_match} (1.000)\n"
            f"\n"
            f"Rule-matched — use this skill unless the query clearly needs something else."
        )}

    # ── Layer 3: BM25 Routing ──
    results = []
    try:
        results = _RouteIndex.query(user_message, top_k=3)
    except Exception:
        pass

    # ── Layer 4: Levenshtein Fallback ──
    if not results:
        known = set(_RouteIndex._inverted.keys())
        corrected = _fuzzy_keyword_correction(user_message, known)
        if corrected != user_message.lower():
            try:
                results = _RouteIndex.query(corrected, top_k=3)
            except Exception:
                pass

    # ── No match — let AI use built-in tools ──
    if not results:
        return {"context": "[Top 3 skills for this query]\n  (no strong match — use built-in tools)"}

    # ── Build context block ──
    lines = ["[Top 3 skills for this query]"]
    for i, (name, score) in enumerate(results):
        lines.append(f"  {i+1}. {name} ({score:.3f})")
    lines.append("")
    lines.append("If none match, fall back to built-in tools.")

    return {"context": "\n".join(lines)}
