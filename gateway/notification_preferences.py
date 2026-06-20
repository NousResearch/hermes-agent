"""User notification floor preferences (Layer 2 of the notification policy).

Implements the per-user override layer described in ``DESIGN.md`` §6.3
(notification-policy board).  Layer 1 (``classify_event_severity`` in
``kanban_watchers.py``) assigns a base severity (P0/P1/P2) to each
incoming kanban event; this module lets the user *demote* events that
would otherwise push — e.g. set the floor to ``"quiet"`` to suppress
most notifications, or override individual event types to a lower
severity.

Configuration file: ``~/.hermes/notification_preferences.yaml``

Data model::

    notification:
      floor: verbose          # verbose(P1+) | normal(P0) | quiet
      overrides:              # optional, per-event-type severity override
        task_completed: P0
        task_crashed: P2

Floor semantics — given an event with severity ``S``:

* ``verbose``  — push anything at P1 or P0.  (Default; preserves the
  original behaviour where every P0/P1 fires a notification.)
* ``normal``   — push only P0.  P1 events are deferred (aggregated by
  M2 or simply dropped, depending on caller).
* ``quiet``    — push almost nothing.  Only P0 events survive.

Overrides take precedence over the floor: if ``task_completed`` is
explicitly overridden to ``P2`` then even at ``verbose`` floor it stays
silent.  Conversely, an override from P1 to ``P0`` promotes an event
above what the floor alone would have allowed.

The module never raises on missing / invalid config — failures fall
back to the safest defaults (floor ``"verbose"``, no overrides) with a
warning log so a typo in the YAML can't take down the notification
pipeline.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger("gateway.run")

# Paths can be passed as either ``str`` (convenient for shell callers) or
# ``Path`` (preferred from Python callers).
_PathLike = Union[str, os.PathLike, Path]

# ---------------------------------------------------------------------------
# Defaults — preserved exactly so callers can treat them as sentinels
# ---------------------------------------------------------------------------

DEFAULT_FLOOR = "verbose"

# Allowed floor values.  Anything outside this set falls back to DEFAULT_FLOOR
# with a warning (per the task constraint "配置文件不存在时用默认值，不报错").
VALID_FLOORS = frozenset({"verbose", "normal", "quiet"})

# Allowed override severity values.  Same fallback rule applies.
VALID_SEVERITIES = frozenset({"P0", "P1", "P2"})

# Path of the user-level config file.  Override via env for tests.
_DEFAULT_CONFIG_PATH = Path("~/.hermes/notification_preferences.yaml").expanduser()


def _config_path() -> Path:
    """Resolve the notification_preferences.yaml path.

    Honours ``HERMES_NOTIFICATION_PREFERENCES`` for tests / power users;
    otherwise returns ``~/.hermes/notification_preferences.yaml``.
    """
    override = os.environ.get("HERMES_NOTIFICATION_PREFERENCES")
    if override:
        return Path(override).expanduser()
    return _DEFAULT_CONFIG_PATH


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------


def _load_raw_config(path: Optional[_PathLike] = None) -> dict:
    """Read + parse the YAML config.

    Returns an empty dict on any of:
    * file does not exist
    * file is empty
    * YAML is malformed
    * top-level structure isn't a mapping

    A warning is logged in the malformed cases; the missing-file case is
    silent because it's the documented default.

    *path* accepts str, os.PathLike, or ``Path`` — internally normalised
    to ``Path``.
    """
    p: Path
    if path is None:
        p = _config_path()
    elif isinstance(path, Path):
        p = path
    else:
        p = Path(path).expanduser()
    try:
        if not p.exists():
            return {}
    except Exception as e:
        logger.warning("notification_preferences: cannot stat %s: %s", p, e)
        return {}

    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("notification_preferences: cannot read %s: %s", p, e)
        return {}

    if not text.strip():
        # Empty file — treat as no overrides, no warning.
        return {}

    try:
        import yaml  # local import: keeps the module import-cheap for tests
    except ImportError:
        logger.warning("notification_preferences: PyYAML not installed; using defaults")
        return {}

    try:
        raw = yaml.safe_load(text)
    except Exception as e:
        logger.warning("notification_preferences: YAML parse failed (%s); using defaults", e)
        return {}

    if not isinstance(raw, dict):
        # Top-level must be a mapping; otherwise the schema is wrong.
        if raw is not None:
            logger.warning(
                "notification_preferences: top-level must be a mapping, got %s; using defaults",
                type(raw).__name__,
            )
        return {}

    return raw


def _extract_notification_block(raw: dict) -> dict:
    """Pull the ``notification:`` sub-block out of the parsed YAML.

    Missing or wrong-typed ``notification`` key returns ``{}`` — the rest
    of the config is irrelevant to this module.
    """
    block = raw.get("notification")
    if not isinstance(block, dict):
        return {}
    return block


# ---------------------------------------------------------------------------
# Public API — Layer 2 functions
# ---------------------------------------------------------------------------


def load_user_floor(path: Optional[_PathLike] = None) -> str:
    """Return the user's notification floor (``"verbose"`` by default).

    Reads ``~/.hermes/notification_preferences.yaml`` (or *path* when
    provided) and returns the ``notification.floor`` value.  Falls back
    to ``DEFAULT_FLOOR`` (``"verbose"``) when:
    * the file doesn't exist
    * the file is empty / malformed
    * the ``notification`` block is missing or wrong-typed
    * the ``floor`` value isn't one of ``VALID_FLOORS``
    """
    raw = _load_raw_config(path)
    block = _extract_notification_block(raw)
    floor = block.get("floor")
    if isinstance(floor, str) and floor in VALID_FLOORS:
        return floor
    if floor is not None:
        logger.warning(
            "notification_preferences: invalid floor %r; using default %r",
            floor, DEFAULT_FLOOR,
        )
    return DEFAULT_FLOOR


def load_user_overrides(path: Optional[_PathLike] = None) -> dict[str, str]:
    """Return ``{event_type: severity}`` from ``notification.overrides``.

    Filters out malformed entries (non-string keys, non-string values,
    values outside ``VALID_SEVERITIES``) and logs a single warning for
    each so a single bad row never sinks the whole config.  Returns an
    empty dict when the file is missing / malformed / has no
    ``overrides`` block.
    """
    raw = _load_raw_config(path)
    block = _extract_notification_block(raw)
    overrides_raw = block.get("overrides")
    if not isinstance(overrides_raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, val in overrides_raw.items():
        if not isinstance(key, str) or not isinstance(val, str):
            logger.warning(
                "notification_preferences: override entry %r -> %r has wrong type; skipped",
                key, val,
            )
            continue
        if val not in VALID_SEVERITIES:
            logger.warning(
                "notification_preferences: override %s -> %r is not a valid severity; skipped",
                key, val,
            )
            continue
        out[key] = val
    return out


def should_push(
    event_severity: str,
    user_floor: str,
    overrides: dict[str, str],
    *,
    event_type: Optional[str] = None,
) -> bool:
    """Decide whether an event at ``event_severity`` should push to the user.

    Logic:
      1. If *event_type* is provided AND *overrides* has an entry for it,
         the override REPLACES ``event_severity`` before step 2.
      2. Apply the *user_floor* filter:

         * ``"verbose"`` — push at P0 or P1
         * ``"normal"``  — push only at P0
         * ``"quiet"``   — push nothing (caller decides whether P0 is
           still sacred; this function honours the user's wish)

      Unknown floors fall back to the verbose behaviour so the caller
      never silently drops notifications because of a typo.
    """
    # 1. Apply overrides.
    effective = event_severity
    if event_type and isinstance(overrides, dict):
        ov = overrides.get(event_type)
        if ov in VALID_SEVERITIES:
            effective = ov

    # 2. Apply floor.
    floor = user_floor if user_floor in VALID_FLOORS else DEFAULT_FLOOR
    if floor == "verbose":
        return effective in ("P0", "P1")
    if floor == "normal":
        return effective == "P0"
    if floor == "quiet":
        return False
    # Shouldn't reach here — but stay safe.
    return effective in ("P0", "P1")


def effective_severity(
    event: dict,
    base_severity: str,
    overrides: dict[str, str],
) -> str:
    """Return the user-effective severity for *event*.

    Looks up ``event``'s ``"kind"`` (or ``"event_type"``) in
    *overrides*; if a valid override exists it replaces *base_severity*,
    otherwise *base_severity* is returned untouched.

    Accepts either ``event["kind"]`` (matching the rest of the kanban
    pipeline) or ``event["event_type"]`` (for caller convenience).  When
    neither is present, the base severity is returned unchanged.
    """
    if not isinstance(event, dict) or not isinstance(overrides, dict):
        return base_severity
    event_type = event.get("event_type") or event.get("kind")
    if not isinstance(event_type, str):
        return base_severity
    ov = overrides.get(event_type)
    if isinstance(ov, str) and ov in VALID_SEVERITIES:
        return ov
    return base_severity


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------


def load_all_preferences(path: Optional[_PathLike] = None) -> dict[str, Any]:
    """Bundle ``floor`` + ``overrides`` into one dict for hot-path callers.

    Returns ``{"floor": <str>, "overrides": <dict[str, str]>}`` after
    applying the same defaults and fallbacks as the individual loaders.
    """
    return {
        "floor": load_user_floor(path),
        "overrides": load_user_overrides(path),
    }


# ---------------------------------------------------------------------------
# M3: Block 选项化 — reason 分类器 + 选项模板 + 推送格式
# ---------------------------------------------------------------------------
#
# DESIGN.md §4 Block 选项化.  When a worker blocks a task, the notifier
# appends a numbered option block to the push message so the user can
# reply with a digit (or custom text) to unblock.
#
# The classifier is deterministic keyword matching (DESIGN §9: "规则匹配 +
# 兜底，确定性强，不依赖 LLM").  Six categories + one fallback:

BLOCK_REASON_CATEGORIES = [
    "二选一",
    "确认类",
    "凭证",
    "技术选型",
    "依赖等待",
    "不明确",
]
BLOCK_FALLBACK_CATEGORY = "兜底"

# Keyword tables — the first matching category wins (order matters:
# more specific categories come first).
_BLOCK_REASON_KEYWORDS: list[tuple[str, list[str]]] = [
    ("二选一", ["还是", " or ", "选择"]),
    ("确认类", ["确认", "approve", "是否", "confirm"]),
    ("凭证", ["credential", "token", "密码", "secret", "api key", "apikey"]),
    ("技术选型", ["选型", "评估", "比较", "vs", "对比"]),
    ("依赖等待", ["等待", "依赖", "waiting", "depends"]),
    ("不明确", ["不明确", "歧义", "ambiguous", "unclear"]),
]

# Fixed option lists per category (DESIGN §4 表格).  The first option is
# marked ⭐ (recommended) in the rendered output.
BLOCK_OPTION_TEMPLATES: dict[str, list[str]] = {
    "二选一": ["选项A", "选项B", "让我补充"],
    "确认类": ["确认", "拒绝（说明原因）", "需要更多信息"],
    "凭证": ["我来配置", "暂时跳过", "用默认值"],
    "技术选型": ["方案A", "方案B", "由 worker 推荐"],
    "依赖等待": ["继续等待", "跳过依赖", "替代方案"],
    "不明确": ["扩大范围", "缩小范围", "保持现状"],
    BLOCK_FALLBACK_CATEGORY: ["继续", "暂停", "取消"],
}

# Unicode separator line used in the push format (DESIGN §4 推送格式).
_BLOCK_SEP = "━" * 28


def classify_block_reason(reason: str) -> str:
    """Classify a block *reason* string into one of six templates or fallback.

    Returns one of the keys in :data:`BLOCK_OPTION_TEMPLATES`.  Pure
    keyword matching — no LLM, no DB.  Empty / ``None`` / non-string
    input returns the fallback category.

    >>> classify_block_reason("使用 5432 还是 3306？")
    '二选一'
    >>> classify_block_reason("需要确认端口")
    '确认类'
    >>> classify_block_reason("missing API token")
    '凭证'
    >>> classify_block_reason("继续工作")
    '兜底'
    """
    if not reason or not isinstance(reason, str):
        return BLOCK_FALLBACK_CATEGORY
    text = " " + reason.lower() + " "  # pad so " or " matches standalone
    for category, keywords in _BLOCK_REASON_KEYWORDS:
        for kw in keywords:
            if kw.lower() in text:
                return category
    return BLOCK_FALLBACK_CATEGORY


def block_options_for_reason(reason: str) -> list[str]:
    """Return the fixed option list for a block *reason*.

    Convenience wrapper: classify → template lookup.  Never raises;
    unknown categories return the fallback list.
    """
    cat = classify_block_reason(reason)
    return BLOCK_OPTION_TEMPLATES.get(cat) or BLOCK_OPTION_TEMPLATES[BLOCK_FALLBACK_CATEGORY]


def build_block_options(reason: str, *, enabled: bool = True) -> str:
    """Render the numbered option block to append to a blocked push message.

    Output follows DESIGN §4 推送格式::

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          [1] 确认 ⭐
          [2] 拒绝（说明原因）
          [3] 需要更多信息
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        回复数字即可

    When *enabled* is ``False`` (the ``block.auto_options`` config switch)
    returns an empty string — the caller simply appends nothing.
    """
    if not enabled:
        return ""
    options = block_options_for_reason(reason)
    lines = [_BLOCK_SEP]
    for i, opt in enumerate(options, 1):
        star = " ⭐" if i == 1 else ""
        lines.append(f"  [{i}] {opt}{star}")
    lines.append(_BLOCK_SEP)
    lines.append("回复数字即可")
    return "\n".join(lines)


# -----------------------------------------------------------------------
# M3 §4: Reply parsing — number/text reply → unblock action
# -----------------------------------------------------------------------


def parse_block_reply(reply: str) -> dict[str, Any]:
    """Parse a user reply to a block-options message.

    Returns a dict describing the intended action:

    * Bare digit in option range → ``{"action": "select", "index": N}``
    * Any other non-empty text   → ``{"action": "custom", "text": "..."}`
    * Empty / whitespace         → ``{"action": "none"}``

    The caller pairs this with :func:`resolve_block_action` to produce
    the final unblock comment + decision.

    >>> parse_block_reply("1")
    {'action': 'select', 'index': 1}
    >>> parse_block_reply("3")
    {'action': 'select', 'index': 3}
    >>> parse_block_reply("用 8080 端口")
    {'action': 'custom', 'text': '用 8080 端口'}
    >>> parse_block_reply("")
    {'action': 'none'}
    """
    if not reply or not isinstance(reply, str):
        return {"action": "none"}
    stripped = reply.strip()
    if not stripped:
        return {"action": "none"}
    # Bare positive integer → option selection.
    if stripped.isdigit():
        return {"action": "select", "index": int(stripped)}
    return {"action": "custom", "text": stripped}


def resolve_block_action(
    reply: str,
    reason: str,
) -> dict[str, Any]:
    """Resolve a user *reply* against a block *reason* into an unblock plan.

    Combines :func:`classify_block_reason` + :func:`parse_block_reply`:

    * Number reply in range → ``{"unblock": True, "comment": "用户选择: [N] <opt>", "option_text": "<opt>"}``
    * Number out of range   → ``{"unblock": False, "error": "选项超出范围 (1-N)"}``
    * Text reply            → ``{"unblock": True, "comment": "用户指定: <text>", "option_text": "<text>"}``
    * Empty                 → ``{"unblock": False, "error": "空回复"}``

    The caller writes *comment* as a board comment and calls
    ``unblock_task`` when ``unblock`` is ``True``.

    >>> r = resolve_block_action("1", "使用 5432 还是 3306")
    >>> r["unblock"]
    True
    >>> "选项A" in r["option_text"]
    True
    """
    parsed = parse_block_reply(reply)
    options = block_options_for_reason(reason)

    if parsed["action"] == "none":
        return {"unblock": False, "error": "空回复"}

    if parsed["action"] == "select":
        idx = parsed["index"]
        if 1 <= idx <= len(options):
            opt = options[idx - 1]
            return {
                "unblock": True,
                "comment": f"用户选择: [{idx}] {opt}",
                "option_text": opt,
            }
        return {
            "unblock": False,
            "error": f"选项超出范围 (1-{len(options)})",
        }

    # custom text
    text = parsed["text"]
    return {
        "unblock": True,
        "comment": f"用户指定: {text}",
        "option_text": text,
    }


__all__ = [
    "DEFAULT_FLOOR",
    "VALID_FLOORS",
    "VALID_SEVERITIES",
    "load_user_floor",
    "load_user_overrides",
    "should_push",
    "effective_severity",
    "load_all_preferences",
    # M3: Block 选项化
    "BLOCK_REASON_CATEGORIES",
    "BLOCK_FALLBACK_CATEGORY",
    "BLOCK_OPTION_TEMPLATES",
    "classify_block_reason",
    "block_options_for_reason",
    "build_block_options",
    "parse_block_reply",
    "resolve_block_action",
]