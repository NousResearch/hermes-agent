"""Block 选项化 — reason 分类 + 选项生成 + 消息格式化 + 回复解析.

当 Worker 调用 ``kanban_block(reason="...")`` 时，Gateway 的 notifier
自动解析 reason 文本，生成结构化选项卡片推送给用户。用户回复数字
或自定义文本即可决策，系统自动 unblock 任务。

纯函数模块，零副作用。集成点:
  - 推送侧: ``kanban_watchers.py`` 的 blocked 消息构造分支
  - 回复侧: ``kanban_watchers.py`` 的 ``_maybe_handle_block_reply``

Reference: DESIGN.md §3, §6.3-6.4
           block-options-requirement.md (M3-PM)

Reviewer fixes applied (t_79b91e39):
  - P1-1: Added access_key / config_missing / 无法确定 keywords
  - P1-2: review-required: prefix detection (skip 选项化)
  - P1-3: Bearer token + DB URL credential masking patterns
  - P2-1: between A and B extraction pattern
  - P2-2: T6 ambiguous label/value consistency fix
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class BlockOption:
    """一个可选选项."""
    label: str  # 显示文本, 如 "确认 5432 ⭐"
    value: str  # 语义值, 如 "确认" (不含 ⭐)
    is_recommended: bool = False


@dataclass
class BlockOptionsResult:
    """classify + build 的完整结果."""
    template: str  # "T1-choice", "T2-confirm", ..., "T0-generic"
    options: list[BlockOption] = field(default_factory=list)
    recommendation: Optional[str] = None  # Worker 建议/评估
    footer: str = ""  # 如 "💡 选择 [1] 后请在本地配置"
    security_hint: str = ""  # 凭证类安全提示


@dataclass
class ReplyParseResult:
    """用户回复的解析结果."""
    action: str  # "option" | "custom" | "ignore" | "skip" | "cancel"
    option_text: Optional[str] = None  # 选中的选项文本
    option_index: Optional[int] = None  # 选中选项的 0-based 索引
    custom_text: str = ""  # 自定义文本 (action == "custom")


# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

_TEMPLATES = {
    # 匹配优先级决定渲染顺序; T3 有安全提升规则
    "T1_choice": {
        "keywords": [
            r"还是", r"\bor\b", r"选择", r"或是",
            r"pick\s+(one|either)", r"which\s+one", r"哪个",
        ],
        "extract_options": True,
    },
    "T2_confirm": {
        "keywords": [
            r"确认", r"approve", r"是否", r"批准", r"检查",
            r"\breview\b", r"\bcheck\b", r"验证", r"verify",
        ],
        "extract_options": False,
    },
    "T3_credential": {
        "keywords": [
            r"credential", r"token", r"密码", r"password",
            r"api\s*key", r"secret", r"密钥",
            # P1-1 fix: added access_key
            r"access\s*key", r"accesskey",
            r"权限", r"permission", r"auth",
            r"登录", r"login", r"signin",
            # P1-1 fix: added config missing patterns
            r"配置文件缺失", r"missing\s+config",
            r"config.*missing", r"配置.*缺失",
            # DB / connection-string identifiers (signal credential config)
            r"db_url", r"database_url", r"connection_string",
            r"postgres(?:ql)?://", r"mysql://", r"mongodb://", r"redis://",
        ],
        "extract_options": False,
        "security_elevated": True,
    },
    "T4_tech_choice": {
        "keywords": [
            r"选型", r"评估", r"比较", r"compare", r"evaluate",
            r"技术方案", r"tech\s*stack", r"框架",
            r"which\s+(?:framework|library|tool|service)",
            r"trade[- ]?off", r"权衡", r"vs\b",
        ],
        "extract_options": True,
    },
    "T5_dependency": {
        "keywords": [
            r"等待", r"waiting", r"wait\s+for",
            r"依赖", r"dependency", r"depends?\s+on",
            r"blocked\s+by", r"前置条件",
            r"外部", r"external", r"third[- ]party",
        ],
        "extract_options": False,
    },
    "T6_ambiguous": {
        "keywords": [
            r"不明确", r"歧义", r"ambiguous", r"unclear",
            # P1-1 fix: added 无法确定 / 不确定 / uncertain / unsure
            r"无法确定", r"不确定", r"uncertain", r"unsure",
            r"什么意思", r"what.*mean",
            r"如何理解", r"how.*interpret",
        ],
        "extract_options": False,
    },
}

# 匹配优先级顺序 (T3 在安全提升检查中特殊处理)
_MATCH_ORDER = [
    "T2_confirm",
    "T1_choice",
    "T4_tech_choice",
    "T3_credential",
    "T5_dependency",
    "T6_ambiguous",
]

# 推荐标记关键词
_RECOMMEND_KEYWORDS = [r"推荐", r"建议", r"prefer", r"recommend", r"优势", r"倾向"]
_REJECT_KEYWORDS = [r"不建议", r"don'?t", r"不推荐"]

# review-required prefix (P1-2 fix: skip 选项化 for standard review blocks)
_REVIEW_REQUIRED_PREFIX = "review-required:"


# ---------------------------------------------------------------------------
# Reason classifier
# ---------------------------------------------------------------------------


def classify_block_reason(reason: str) -> str:
    """对 block reason 分类, 返回模板 ID.

    Returns one of: "T1_choice", "T2_confirm", "T3_credential",
    "T4_tech_choice", "T5_dependency", "T6_ambiguous", "T0_generic".

    匹配规则:
      1. T3 安全提升: 凭证关键词优先于其他所有模板
      2. 否则按 _MATCH_ORDER 顺序匹配, 第一个命中生效
      3. 无匹配 -> T0_generic
    """
    if not reason or not reason.strip():
        return "T0_generic"

    lower = reason.lower()

    # Step 1: 安全提升 — T3 凭证关键词优先
    t3_keywords = _TEMPLATES["T3_credential"]["keywords"]
    if any(re.search(p, lower) for p in t3_keywords):
        return "T3_credential"

    # Step 2: 按优先级顺序匹配
    for tpl_key in _MATCH_ORDER:
        if tpl_key == "T3_credential":
            continue  # 已在 Step 1 处理
        keywords = _TEMPLATES[tpl_key]["keywords"]
        if any(re.search(p, lower) for p in keywords):
            return tpl_key

    return "T0_generic"


# ---------------------------------------------------------------------------
# Option builders
# ---------------------------------------------------------------------------


def _detect_recommendation(reason: str, option_a: str, option_b: str) -> Optional[int]:
    """从 reason 文本中检测推荐项.

    Returns 0 (推荐 A) 或 1 (推荐 B) 或 None.
    """
    lower = reason.lower()
    if any(re.search(p, lower) for p in _REJECT_KEYWORDS):
        return None

    for i, opt in enumerate([option_a, option_b]):
        if not opt:
            continue
        # "推荐 X" / "X 有优势"
        for kw in _RECOMMEND_KEYWORDS:
            if opt.lower() in lower and re.search(kw, lower):
                return i

    return None


def _extract_options_from_reason(reason: str) -> tuple[Optional[str], Optional[str]]:
    """从 reason 中提取 A/B 选项文本.

    用于 T1 (二选一) 和 T4 (技术选型).
    Returns (option_a, option_b). 解析失败返回 (None, None).
    """
    patterns = [
        # "A 还是 B" / "A or B" / "A vs B"
        r"(.{1,40}?)\s*(?:还是|\bor\b|vs\.?)\s*(.{1,40}?)(?:[？?。.！!]|$)",
        # "选择 A 还是 B" / "pick A or B"
        r"(?:选择|pick|选)\s*(.{1,40}?)\s*(?:还是|\bor\b|vs\.?)\s*(.{1,40})",
        # P2-1 fix: "choose between A and B" / "在 A 和 B 之间选择"
        r"(?:between|在)\s+([\w\u4e00-\u9fff][^？?。.！!和]{0,40}?)\s+(?:and|和)\s+([\w\u4e00-\u9fff][^？?。.！!]{0,40}?)(?:[？?。.！!]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, reason, re.IGNORECASE)
        if m:
            a = m.group(1).strip().strip("\"'`()")
            b = m.group(2).strip().strip("\"'`()")
            # Strip recommendation keywords from the option text so
            # "Use Redis（推荐）" -> "Use Redis" and "推荐 Redis 还是 Memcached"
            # doesn't capture "推荐 Redis" as A.
            for kw in _RECOMMEND_KEYWORDS:
                a = re.sub(rf"[\s（(]?{kw}[\s）)]?", "", a).strip()
                b = re.sub(rf"[\s（(]?{kw}[\s）)]?", "", b).strip()
            if a and b and a.lower() != b.lower():
                return a, b
    return None, None


def _extract_recommendation_text(reason: str) -> Optional[str]:
    """提取 reason 中的建议/评估文本用于额外展示."""
    patterns = [
        r"(?:建议|推荐|建议使用|评估)[：:]\s*(.{2,80})",
        r"(?:Worker\s+建议|worker\s+suggestion)[：:]\s*(.{2,80})",
    ]
    for pat in patterns:
        m = re.search(pat, reason, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def build_block_options(reason: str, template: str) -> BlockOptionsResult:
    """根据分类结果生成选项列表.

    Args:
        reason: 原始 block reason 文本.
        template: classify_block_reason() 返回的模板 ID.

    Returns:
        BlockOptionsResult 包含模板类型和选项列表.
    """
    options: list[BlockOption] = []
    recommendation = None
    footer = ""
    security_hint = ""

    if template == "T1_choice":
        a, b = _extract_options_from_reason(reason)
        if a and b:
            rec_idx = _detect_recommendation(reason, a, b)
            options = [
                BlockOption(
                    label=f"{a} {'⭐' if rec_idx == 0 else ''}",
                    value=a,
                    is_recommended=(rec_idx == 0),
                ),
                BlockOption(
                    label=f"{b} {'⭐' if rec_idx == 1 else ''}",
                    value=b,
                    is_recommended=(rec_idx == 1),
                ),
                BlockOption(label="让我补充", value="补充"),
            ]
        else:
            # T1 提取失败 -> 降级为 T6
            return build_block_options(reason, "T6_ambiguous")

    elif template == "T2_confirm":
        has_reject = any(re.search(p, reason.lower()) for p in _REJECT_KEYWORDS)
        options = [
            BlockOption(label="确认", value="确认", is_recommended=not has_reject),
            BlockOption(label="拒绝 + 原因", value="拒绝"),
            BlockOption(label="需要更多信息", value="需要更多信息"),
        ]

    elif template == "T3_credential":
        has_default = bool(re.search(r"默认|default", reason.lower()))
        has_skip = bool(re.search(r"跳过|skip", reason.lower()))
        options = [
            BlockOption(
                label="我来配置",
                value="我来配置",
                is_recommended=(not has_default and not has_skip),
            ),
            BlockOption(label="暂时跳过此任务", value="跳过", is_recommended=has_skip),
            BlockOption(label="用环境默认值", value="用默认值", is_recommended=has_default),
        ]
        footer = "💡 选择 [1] 后请在本地配置文件中添加，不要在群聊发送密钥"
        security_hint = "凭证类 block — 推送已脱敏"

    elif template == "T4_tech_choice":
        a, b = _extract_options_from_reason(reason)
        if a and b:
            rec_idx = _detect_recommendation(reason, a, b)
            options = [
                BlockOption(label=a, value=a, is_recommended=(rec_idx == 0)),
                BlockOption(label=b, value=b, is_recommended=(rec_idx == 1)),
                BlockOption(label="由 worker 推荐决定", value="推荐"),
            ]
        else:
            # T4 提取失败 -> 降级为 T6
            return build_block_options(reason, "T6_ambiguous")

    elif template == "T5_dependency":
        has_skip = bool(re.search(r"跳过|skip", reason.lower()))
        has_alt = bool(re.search(r"替代|alternative", reason.lower()))
        options = [
            BlockOption(
                label="继续等待",
                value="继续等待",
                is_recommended=(not has_skip and not has_alt),
            ),
            BlockOption(label="跳过此依赖", value="跳过", is_recommended=has_skip),
            BlockOption(label="替代方案", value="替代", is_recommended=has_alt),
        ]

    elif template == "T6_ambiguous":
        has_expand = bool(re.search(r"扩大|更多", reason.lower()))
        has_shrink = bool(re.search(r"聚焦|收窄", reason.lower()))
        # P2-2 fix: label/value now consistent
        options = [
            BlockOption(
                label="请详细说明可选项",
                value="扩大",
                is_recommended=has_expand,
            ),
            BlockOption(
                label="由 worker 推荐决定",
                value="推荐",
                is_recommended=has_shrink,
            ),
            BlockOption(
                label="保持现状",
                value="保持现状",
                is_recommended=(not has_expand and not has_shrink),
            ),
        ]

    else:  # T0_generic (兜底)
        options = [
            BlockOption(label="继续（我知道了）", value="继续", is_recommended=True),
            BlockOption(label="暂停此任务", value="暂停"),
            BlockOption(label="取消此任务", value="取消"),
        ]

    recommendation = _extract_recommendation_text(reason)

    return BlockOptionsResult(
        template=template.replace("_", "-"),  # "T1_choice" -> "T1-choice"
        options=options,
        recommendation=recommendation,
        footer=footer,
        security_hint=security_hint,
    )


# ---------------------------------------------------------------------------
# Message formatter
# ---------------------------------------------------------------------------

_SEPARATOR = "━" * 28


def format_options_message(
    *,
    task_title: str,
    block_reason: str,
    task_id: str,
    result: BlockOptionsResult,
    max_reason_len: int = 200,
) -> str:
    """将 BlockOptionsResult 格式化为推送消息.

    格式 (per DESIGN.md §3.4 / requirement §5):
      ⏸ 阻塞: {task_title}
         {reason摘要}
      ━━━━━━━━━━━━━━━━━━━━━━━━
      Worker 评估: {recommendation}

        [1] 选项1 ⭐
        [2] 选项2
        [3] 选项3
      ━━━━━━━━━━━━━━━━━━━━━━━━
      {footer}
      回复数字即可，或直接输入自定义方案
    """
    reason_display = block_reason[:max_reason_len]
    if len(block_reason) > max_reason_len:
        reason_display += "..."

    lines: list[str] = []
    lines.append(f"⏸ 阻塞: {task_title[:50]}")
    lines.append(f"   {reason_display}")
    lines.append(_SEPARATOR)

    if result.recommendation:
        lines.append(f"Worker 评估: {result.recommendation}")
        lines.append("")

    for i, opt in enumerate(result.options, 1):
        rec = " ⭐" if opt.is_recommended else ""
        lines.append(f"  [{i}] {opt.label}{rec}")

    lines.append(_SEPARATOR)

    if result.footer:
        lines.append(result.footer)

    lines.append("回复数字即可，或直接输入自定义方案")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Credential masking
# ---------------------------------------------------------------------------

_CREDENTIAL_PATTERNS = [
    (r"(token\s*[=:]\s*)\S{8,}", r"\1***"),
    (r"(password\s*[=:]\s*)\S{8,}", r"\1***"),
    (r"(api[_-]?key\s*[=:]\s*)\S{8,}", r"\1***"),
    (r"(secret\s*[=:]\s*)\S{8,}", r"\1***"),
    (r"(密钥\s*[=:]\s*)\S{8,}", r"\1***"),
    # P1-3 fix: Bearer token pattern
    (r"(Bearer\s+)\S{20,}", r"\1***"),
    # P1-3 fix: DB URL / connection string pattern
    (
        r"(db_url|database_url|connection_string)\s*[=:]\s*\S+@\S+",
        r"\1=***@***",
    ),
    # P1-3 fix: postgres/mysql scheme URLs
    (
        r"(postgres(?:ql)?|mysql|mongodb|redis)://\S+:\S+@",
        r"\1://***:***@",
    ),
]


def mask_credentials(text: str) -> str:
    """脱敏 reason 中的凭证值.

    仅用于推送消息; 原始 reason 保留在 DB 中.
    """
    if not text:
        return text
    result = text
    for pattern, replacement in _CREDENTIAL_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


# ---------------------------------------------------------------------------
# Reply parser
# ---------------------------------------------------------------------------


def parse_user_reply(text: str, num_options: int) -> ReplyParseResult:
    """解析用户对 block 选项的回复.

    规则 (per requirement §2.3):
      - 纯数字 1-N -> 匹配对应选项
      - 数字 > N -> 作为自定义文本处理
      - "跳过" / "skip" -> action=skip
      - "取消" / "cancel" -> action=cancel
      - "取消" + 文本 -> action=cancel, custom_text 带原因
      - 空白/纯空格 -> action=ignore
      - 其他文本 -> action=custom
    """
    if not text or not text.strip():
        return ReplyParseResult(action="ignore")

    stripped = text.strip()
    lower = stripped.lower()

    # 跳过
    if lower in {"跳过", "skip"}:
        return ReplyParseResult(action="skip", option_text="跳过")

    # 取消
    if lower in {"取消", "cancel"}:
        return ReplyParseResult(action="cancel", option_text="取消")
    if lower.startswith(("取消", "cancel")):
        prefix = "取消" if lower.startswith("取消") else "cancel"
        extra = stripped[len(prefix):].strip()
        if extra:
            return ReplyParseResult(action="cancel", option_text=f"取消: {extra}")
        return ReplyParseResult(action="cancel", option_text="取消")

    # 纯数字
    if stripped.isdigit():
        idx = int(stripped)
        if 1 <= idx <= num_options:
            return ReplyParseResult(action="option", option_index=idx - 1)
        # 超出范围 -> 自定义
        return ReplyParseResult(action="custom", custom_text=stripped)

    # 其他文本 -> 自定义
    return ReplyParseResult(action="custom", custom_text=stripped)


# ---------------------------------------------------------------------------
# Reply resolution (integrates with kanban DB)
# ---------------------------------------------------------------------------


def resolve_block_reply(
    reply: str,
    reason: str,
) -> dict[str, Any]:
    """Resolve a user *reply* against a block *reason* into an unblock plan.

    Enhanced version of notification_preferences.resolve_block_action that
    uses the template-based classifier and richer option generation.

    Returns:
        dict with keys:
        - ``unblock`` (bool): whether to unblock the task
        - ``comment`` (str): the comment text to write
        - ``option_text`` (str): the selected option text
        - ``error`` (str, optional): error message if not unblocking
    """
    if not reply or not reply.strip():
        return {"unblock": False, "error": "空回复"}

    # Use the enhanced classifier + option builder
    masked_reason = mask_credentials(reason) if reason else ""
    template = classify_block_reason(masked_reason)
    result = build_block_options(masked_reason, template)
    parsed = parse_user_reply(reply, len(result.options))

    if parsed.action == "ignore":
        return {"unblock": False, "error": "空回复"}

    if parsed.action == "option" and parsed.option_index is not None:
        opt = result.options[parsed.option_index]
        return {
            "unblock": True,
            "comment": f"用户选择: [{parsed.option_index + 1}] {opt.value}",
            "option_text": opt.value,
        }

    if parsed.action == "skip":
        return {
            "unblock": True,
            "comment": f"用户选择: 跳过",
            "option_text": "跳过",
        }

    if parsed.action == "cancel":
        return {
            "unblock": True,
            "comment": f"用户选择: {parsed.option_text}",
            "option_text": parsed.option_text or "取消",
        }

    # custom text
    text = parsed.custom_text
    return {
        "unblock": True,
        "comment": f"用户指定: {text}",
        "option_text": text,
    }


# ---------------------------------------------------------------------------
# Adapter integration — maybe_process_block_reply
# ---------------------------------------------------------------------------


def maybe_process_block_reply(
    message_text: str,
    board: Any = None,
    user_id: str = "",
) -> bool:
    """Check if a message is a block-reply, without actually processing it.

    This is a lightweight check used by adapters to decide whether to
    short-circuit the normal LLM conversation path. The actual unblock
    logic lives in ``kanban_watchers._maybe_handle_block_reply`` which
    has access to the pending-block registry and DB connection.

    Args:
        message_text: The raw message text from the user.
        board: Unused (kept for API compatibility with the task spec).
        user_id: Unused (kept for API compatibility).

    Returns:
        True if the message *looks like* a block reply (bare digit 1-9
        or short text that could be an option). This is intentionally
        permissive — the actual handler does the authoritative check
        with TTL + pending-registry context.
    """
    if not message_text or not message_text.strip():
        return False
    stripped = message_text.strip()
    # Bare digit 1-9 is the primary fast path
    if stripped.isdigit() and 1 <= int(stripped) <= 9:
        return True
    # Short keywords that map to block actions
    lower = stripped.lower()
    if lower in {"跳过", "skip", "取消", "cancel"}:
        return True
    return False


# P0 fix (reviewer t_79b91e39): bare-digit messages in group chat
# ("port 5432", "PR #123", "status 404") must NOT be auto-treated as
# block replies.  A message is a real block reply only when:
#   1. The chat has a *pending block invite* (the notifier pushed a
#      numbered option block to this chat within the TTL), AND
#   2. The message text is a bare digit 1-N (where N is the invite's
#      option count) or a recognized block keyword ("跳过", "skip",
#      "取消", "cancel").
# The TTL bounds the risk if a pending invite was missed.
#
# The storage lives on the GatewayRunner (``runner._pending_block_invites``,
# a ``{session_key: InviteRecord}`` dict).  This module is pure functions
# only — the runner calls :func:`register_block_invite` / :func:`consume_block_invite`
# passing its own dict.  We don't reach into the runner here.
_BLOCK_REPLY_TTL_SECONDS = 30 * 60  # 30 minutes


def _evict_stale_invite(store: dict, session_key: str, now_ts: float) -> None:
    """Drop a single invite if it's past TTL. Mutates *store* in place."""
    rec = store.get(session_key)
    if not rec:
        return
    if now_ts - float(rec.get("created_at") or 0) > _BLOCK_REPLY_TTL_SECONDS:
        store.pop(session_key, None)


def register_block_invite(
    store: dict,
    session_key: str,
    task_id: str,
    reason: str,
    num_options: int,
    *,
    now_ts: Optional[float] = None,
) -> None:
    """Record that the notifier just pushed a block options card to *session_key*.

    The invite is the bridge between push-side and reply-side: without it
    a bare "3" in a group chat can't safely trigger unblock.  Storage
    is provided by the caller (typically ``runner._pending_block_invites``)
    so this library stays decoupled from the runner's lifecycle.

    Args:
        store: The dict the runner uses to track invites; mutated in place.
        session_key: Platform-scoped chat identifier (same as the
            ``_quick_key`` the rest of the gateway uses).
        task_id: The kanban task this invite refers to.
        reason: The full block reason (so the reply handler can
            re-classify for option indexing).
        num_options: How many numbered options the user saw (used to
            bound the digit range — "1" is only valid if there are ≥1
            options).
        now_ts: Override current time (for tests).
    """
    if not isinstance(store, dict):
        return
    ts = float(now_ts) if now_ts is not None else __import__("time").time()
    store[session_key] = {
        "task_id": str(task_id),
        "reason": str(reason or ""),
        "num_options": int(num_options),
        "created_at": ts,
    }


def consume_block_invite(
    store: dict,
    session_key: str,
    *,
    now_ts: Optional[float] = None,
) -> Optional[dict]:
    """Atomically read + drop the pending block invite for *session_key*.

    Returns ``None`` if no invite exists, or the invite is older than
    ``_BLOCK_REPLY_TTL_SECONDS``.  TTL eviction is per-call (we don't
    sweep) — stale entries are dropped lazily on next read.
    """
    if not isinstance(store, dict):
        return None
    rec = store.get(session_key)
    if not rec:
        return None
    ts = float(now_ts) if now_ts is not None else __import__("time").time()
    if ts - float(rec.get("created_at") or 0) > _BLOCK_REPLY_TTL_SECONDS:
        store.pop(session_key, None)
        return None
    store.pop(session_key, None)
    return rec


def lookup_block_invite(
    store: dict,
    session_key: str,
    *,
    now_ts: Optional[float] = None,
) -> Optional[dict]:
    """Non-destructive read of a pending invite (used for diagnostic/test)."""
    if not isinstance(store, dict):
        return None
    rec = store.get(session_key)
    if not rec:
        return None
    ts = float(now_ts) if now_ts is not None else __import__("time").time()
    if ts - float(rec.get("created_at") or 0) > _BLOCK_REPLY_TTL_SECONDS:
        return None
    return dict(rec)


def is_block_reply_text(text: str, num_options: int) -> bool:
    """Authoritative check: is *text* a valid block reply for a card with
    *num_options* options?

    Unlike :func:`maybe_process_block_reply` (which is a permissive
    heuristic for adapter short-circuit), this function answers "would
    we actually consume this?" by:

    - Bare digit must be in 1..num_options (not just 1..9 — bound it
      to what the user actually saw so a bare "7" isn't accepted when
      only 3 options were shown).
    - Recognized keywords are always valid.
    - Other text is *not* treated as a block reply here (it would be
      "custom" and the caller should fall through to the LLM path —
      which is what the user actually wants when typing free-form).

    The companion function :func:`resolve_block_reply` still accepts
    free-form text inside a 1-9 fast-path; this one is the *strict*
    guard for the no-LLM-shortcut path.
    """
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if stripped.isdigit():
        n = int(stripped)
        return 1 <= n <= max(0, int(num_options))
    lower = stripped.lower()
    return lower in {"跳过", "skip", "取消", "cancel"}


# ---------------------------------------------------------------------------
# Push-side helper — build the options suffix for a blocked message
# ---------------------------------------------------------------------------


def build_options_suffix(
    reason: str,
    *,
    enabled: bool = True,
    mask_creds: bool = True,
) -> str:
    """Build the numbered option block to append to a blocked push message.

    This is the convenience wrapper used by the notifier push path.
    When *enabled* is False (the ``block.auto_options`` config switch)
    returns an empty string — the caller appends nothing.

    When *mask_creds* is True, the reason is passed through
    :func:`mask_credentials` before classification.
    """
    if not enabled or not reason:
        return ""

    safe_reason = mask_credentials(reason) if mask_creds else reason
    template = classify_block_reason(safe_reason)
    result = build_block_options(safe_reason, template)

    lines = [_SEPARATOR]
    if result.recommendation:
        lines.append(f"Worker 评估: {result.recommendation}")
        lines.append("")

    for i, opt in enumerate(result.options, 1):
        rec = " ⭐" if opt.is_recommended else ""
        lines.append(f"  [{i}] {opt.label}{rec}")

    lines.append(_SEPARATOR)

    if result.footer:
        lines.append(result.footer)

    lines.append("回复数字即可，或直接输入自定义方案")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BLOCK_OPTIONS_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "auto_options": True,  # block.auto_options in the user-facing config
    "mask_credentials": True,
    "max_reason_len": 200,
    # When True, the reply-side hook in ``_handle_message`` consumes
    # bare-digit / "跳过" / "取消" replies from a chat that received a
    # block options card and short-circuits the LLM.  Disable to fall
    # back to "user must type /kanban unblock <id> <decision>".
    "auto_reply": True,
    # Pending-invite TTL in seconds.  After this many seconds a
    # block options card expires and the chat's invite is dropped —
    # bare digits from that chat stop being interpreted as block
    # decisions, even if the user is slow to type.
    "invite_ttl_seconds": 30 * 60,
}


def _cfg_get_with_aliases(section: str, default: dict) -> dict:
    """Read *section* from config; honor ``block.<key>`` aliases for top-level.

    The M3 spec exposes the config as ``block.auto_options`` (top-level
    under the ``block:`` key in user config), but internally the
    module has always stored it under ``block_options``.  This helper
    accepts both shapes: ``cfg_get("block_options", {})`` first, then
    ``cfg_get("block", {})`` as a fallback so existing config files
    work and the spec's canonical key takes precedence.
    """
    merged: dict = {}
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config() or {}
        for key in (section, "block"):
            if not isinstance(cfg, dict):
                break
            sub = cfg_get(cfg, key, default={})
            if isinstance(sub, dict):
                merged.update(sub)
    except Exception:
        pass
    if not merged:
        return default
    return {**default, **merged}


def load_block_options_config() -> dict[str, Any]:
    """加载 block_options 配置.

    读取 ``~/.hermes/config.yaml`` 中的 ``block_options:`` 节
    （兼容 ``block:`` 节作为别名）. 缺失时使用 ``BLOCK_OPTIONS_DEFAULTS``.

    Recognised keys (all optional):

    - ``enabled`` (bool, default ``True``) — 整个 block_options 模块的开关
    - ``auto_options`` (bool, default ``True``) — 推送时是否自动附加选项卡
    - ``auto_reply`` (bool, default ``True``) — 回复时是否自动消费选项
    - ``mask_credentials`` (bool, default ``True``) — 推送前是否脱敏
    - ``max_reason_len`` (int, default 200) — 推送时 reason 显示截断
    - ``invite_ttl_seconds`` (int, default 1800) — pending-invite 寿命

    Spec key name: ``block.auto_options: true`` is canonical.
    """
    return _cfg_get_with_aliases("block_options", dict(BLOCK_OPTIONS_DEFAULTS))


def is_block_options_enabled() -> bool:
    """检查 block_options 功能是否启用 (主开关).

    If *auto_options* is False the module is still considered
    "enabled" (mask_credentials + reply interception still work), but
    the push side stops appending the option block.  Use this helper
    to gate any feature on the master switch.
    """
    cfg = load_block_options_config()
    return bool(cfg.get("enabled", True))


def is_auto_options_enabled() -> bool:
    """检查是否自动在 push 侧追加选项卡 (block.auto_options).

    Independent from :func:`is_block_options_enabled` so the
    ``auto_options`` flag can be turned off without disabling
    credential masking or the reply hook.
    """
    cfg = load_block_options_config()
    return bool(cfg.get("auto_options", True))


def is_auto_reply_enabled() -> bool:
    """检查是否在 reply 侧自动消费 block 回复 (block.auto_reply)."""
    cfg = load_block_options_config()
    return bool(cfg.get("auto_reply", True))


def get_invite_ttl_seconds() -> int:
    """Return the configured invite TTL in seconds (default 1800)."""
    cfg = load_block_options_config()
    try:
        return max(1, int(cfg.get("invite_ttl_seconds", _BLOCK_REPLY_TTL_SECONDS)))
    except Exception:
        return _BLOCK_REPLY_TTL_SECONDS


def is_review_required(reason: str) -> bool:
    """Check if a block reason is a review-required block (P1-2 fix).

    Review-required blocks use the standard ``review-required:`` prefix
    and should NOT be subject to 选项化 — they need human eyes, not
    numbered options.
    """
    if not reason:
        return False
    return reason.lstrip().lower().startswith(_REVIEW_REQUIRED_PREFIX)


__all__ = [
    # 数据模型
    "BlockOption",
    "BlockOptionsResult",
    "ReplyParseResult",
    # 分类 + 生成
    "classify_block_reason",
    "build_block_options",
    "format_options_message",
    "build_options_suffix",
    # 安全
    "mask_credentials",
    "is_review_required",
    # 回复解析
    "parse_user_reply",
    "resolve_block_reply",
    "maybe_process_block_reply",
    "is_block_reply_text",
    "register_block_invite",
    "consume_block_invite",
    "lookup_block_invite",
    "_BLOCK_REPLY_TTL_SECONDS",
    # 配置
    "is_block_options_enabled",
    "load_block_options_config",
]
