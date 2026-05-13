from __future__ import annotations

import re
from typing import Any


BACKPACK_SYSTEM_VERSION = "v0"

STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "the",
    "to",
    "with",
    "where",
}

ACTION_TERMS = {
    "check",
    "debug",
    "debugging",
    "diagnose",
    "edit",
    "explain",
    "find",
    "fix",
    "implement",
    "inspect",
    "locate",
    "open",
    "patch",
    "read",
    "run",
    "search",
    "summarize",
    "test",
    "use",
    "validate",
    "verify",
}

OBJECT_TERMS = {
    "backpack",
    "browser",
    "bug",
    "command",
    "edge",
    "error",
    "failure",
    "file",
    "agent",
    "history",
    "image",
    "log",
    "metadata",
    "package",
    "parser",
    "readme",
    "screenshot",
    "session",
    "skill",
    "test",
    "tool",
    "todo",
    "structure",
}

UNCERTAINTY_TERMS = {"choose", "option", "unclear", "unsure"}

ACTION_PHRASES = (
    "看一下",
    "查一下",
    "查",
    "查找",
    "搜索",
    "读一下",
    "打开",
    "修一下",
    "调试",
    "运行",
    "跑测试",
    "验证",
    "检查",
    "分析",
    "实现",
    "执行",
    "调查",
)

OBJECT_PHRASES = (
    "报错",
    "测试",
    "失败",
    "函数",
    "文件",
    "子 agent",
    "日志",
    "截图",
    "图片",
    "工具",
    "技能",
    "任务",
    "需求",
    "方案",
    "edge case",
)

WORKFLOW_PHRASES = (
    "先找原因",
    "先问我",
    "任务清单",
    "之前我们",
    "怎么处理",
    "继续跑",
    "继续修",
    "继续查",
)

UNCERTAINTY_PHRASES = (
    "not sure",
    "which option",
    "what should",
    "how to handle",
    "should we",
    "不确定",
    "不清楚",
    "怎么办",
    "要不要",
    "该不该",
    "选哪个",
    "判断一下",
)


def build_backpack_candidate_hints(request: str, valid_tool_names: set[str] | None, limit: int = 5) -> str:
    if not isinstance(request, str) or not request.strip():
        return ""
    if not should_build_backpack_candidate_hints(request):
        return ""
    names = set(valid_tool_names or set())
    catalog: list[dict[str, Any]] = []
    if "tool_backpack" in names:
        catalog.extend(_tool_catalog())
    if "skill_backpack" in names:
        catalog.extend(_skill_catalog())
    return build_candidate_hints(request, catalog, limit=limit)


def should_build_backpack_candidate_hints(request: str) -> bool:
    if not isinstance(request, str) or not request.strip():
        return False
    text = request.strip().lower()
    tokens = _tokens(text)

    has_file_path = bool(re.search(r"\b[\w./-]+\.(md|json|py|txt|yaml|yml|toml|log)\b", text))
    has_url = bool(re.search(r"https?://", text))
    has_error_shape = bool(re.search(r"\b(error|exception|traceback|failed|failure)\b", text)) or _has_any_phrase(text, ("报错", "失败", "异常"))

    action_score = len(tokens & ACTION_TERMS) + _phrase_count(text, ACTION_PHRASES)
    object_score = len(tokens & OBJECT_TERMS) + _phrase_count(text, OBJECT_PHRASES)
    workflow_score = _phrase_count(text, WORKFLOW_PHRASES)
    uncertainty_score = len(tokens & UNCERTAINTY_TERMS) + _phrase_count(text, UNCERTAINTY_PHRASES)

    score = 0
    score += action_score * 2
    score += object_score * 2
    score += workflow_score * 3
    if uncertainty_score and (object_score or action_score or workflow_score or has_file_path or has_url or has_error_shape):
        score += uncertainty_score * 3
    score += 4 if has_file_path else 0
    score += 4 if has_url else 0
    score += 4 if has_error_shape else 0

    if len(tokens) <= 2 and not _contains_cjk(text) and score < 4:
        return False
    return score >= 4


def build_candidate_hints(request: str, catalog: list[dict[str, Any]], limit: int = 5) -> str:
    grouped = _build_grouped_candidate_hints(request, catalog, limit=limit)
    if grouped:
        return grouped

    ranked = _rank(request, catalog)
    if not ranked:
        return ""
    return _format_grouped_hints(
        [
            {
                "title": "fallback candidates",
                "reason": "no specialized group matched; these are the highest-ranked candidates.",
                "items": ranked[:limit],
                "lines": _fallback_reason_lines(ranked[:limit]),
            }
        ]
    )


def _build_grouped_candidate_hints(request: str, catalog: list[dict[str, Any]], limit: int = 5) -> str:
    request_text = request.lower()
    request_tokens = _tokens(request)
    groups = []

    if _is_file_inspection_request(request_text, request_tokens):
        search_first = _is_file_search_request(request_text, request_tokens)
        names = ["search_files", "read_file"] if search_first else ["read_file", "search_files"]
        tools = _available_items(catalog, "tool", names)
        if tools:
            groups.append(
                {
                    "title": "local file inspection",
                    "reason": "user asked to search local files or repository content."
                    if search_first
                    else "user asked to read a local file.",
                    "items": tools,
                    "lines": _tool_reason_lines(
                        tools,
                        {
                            "read_file": "read the known file path.",
                            "search_files": "locate the file if the path is ambiguous.",
                        },
                    ),
                }
            )

    if _is_debug_failure_request(request_text, request_tokens):
        skills = _available_items(catalog, "skill", ["systematic-debugging"])
        if skills:
            groups.append(
                {
                    "title": "debugging workflow",
                    "reason": "user described a failing test or error.",
                    "items": skills,
                    "lines": ["1. skill.systematic-debugging - diagnose the failure before changing code."],
                }
            )
        tools = _available_items(catalog, "tool", ["search_files", "read_file", "terminal"])
        if tools:
            groups.append(
                {
                    "title": "repo diagnosis tools",
                    "reason": "debugging usually needs locating files, reading code, and running targeted checks.",
                    "items": tools,
                    "lines": _tool_reason_lines(
                        tools,
                        {
                            "search_files": "find related tests, source files, or symbols.",
                            "read_file": "inspect relevant source or test files.",
                            "terminal": "run the targeted test or reproduction command.",
                        },
                    ),
                }
            )

    if _is_implementation_request(request_text, request_tokens):
        skills = _available_items(catalog, "skill", ["test-driven-development"])
        if skills:
            groups.append(
                {
                    "title": "implementation workflow",
                    "reason": "user asked for a code change or feature implementation.",
                    "items": skills,
                    "lines": ["1. skill.test-driven-development - write or update the failing test before implementation."],
                }
            )
        tools = _available_items(catalog, "tool", ["search_files", "read_file", "patch", "terminal"])
        if tools:
            groups.append(
                {
                    "title": "repo edit tools",
                    "reason": "implementation usually needs locating files, reading code, patching, and verifying.",
                    "items": tools,
                    "lines": _tool_reason_lines(
                        tools,
                        {
                            "search_files": "locate relevant files and symbols.",
                            "read_file": "inspect current implementation before editing.",
                            "patch": "apply the source change.",
                            "terminal": "run tests or build verification.",
                        },
                    ),
                }
            )

    if _is_web_lookup_request(request_text, request_tokens):
        tools = _available_items(catalog, "tool", ["web_extract", "web_search"])
        if tools:
            groups.append(
                {
                    "title": "web lookup",
                    "reason": "user provided a URL or asked for current web information.",
                    "items": tools,
                    "lines": _tool_reason_lines(
                        tools,
                        {
                            "web_extract": "read the provided URL directly.",
                            "web_search": "search the web when the exact page is unknown.",
                        },
                    ),
                }
            )

    if not groups:
        return ""
    return _format_grouped_hints(_limit_groups(groups, limit))


def _limit_groups(groups: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    remaining = max(1, limit)
    limited = []
    for group in groups:
        if remaining <= 0 or len(limited) >= 3:
            break
        items = group["items"][:remaining]
        if not items:
            continue
        next_group = dict(group)
        next_group["items"] = items
        next_group["lines"] = group["lines"][: len(items)]
        limited.append(next_group)
        remaining -= len(items)
    return limited


def _available_items(catalog: list[dict[str, Any]], kind: str, names: list[str]) -> list[dict[str, Any]]:
    by_name = {
        str(item.get("name") or str(item.get("id") or "").split(".", 1)[-1]): item
        for item in catalog
        if item.get("kind") == kind
    }
    return [by_name[name] for name in names if name in by_name]


def _tool_reason_lines(items: list[dict[str, Any]], reasons: dict[str, str]) -> list[str]:
    lines = []
    for index, item in enumerate(items, start=1):
        name = str(item.get("name") or str(item.get("id") or "").split(".", 1)[-1])
        lines.append(f"{index}. tool.{name} - {reasons.get(name, 'use when relevant to the task.')}")
    return lines


def _fallback_reason_lines(items: list[dict[str, Any]]) -> list[str]:
    lines = []
    for index, item in enumerate(items, start=1):
        description = str(item.get("description") or "").splitlines()[0].strip()
        if len(description) > 90:
            description = description[:87].rstrip() + "..."
        suffix = f" - {description}" if description else ""
        lines.append(f"{index}. {item['id']}{suffix}")
    return lines


def _format_grouped_hints(groups: list[dict[str, Any]]) -> str:
    lines = [
        f"Backpack candidate hints ({BACKPACK_SYSTEM_VERSION}):",
        "Hints only; select explicitly through tool_backpack or skill_backpack.",
    ]
    for index, group in enumerate(groups, start=1):
        items = group["items"]
        if not items:
            continue
        lines.extend(
            [
                f"Group {index}: {group['title']}",
                f"reason: {group['reason']}",
            ]
        )
        lines.extend(f"select: {selector}" for selector in _group_selectors(items))
        kinds = {item.get("kind") for item in items}
        if kinds == {"tool"}:
            lines.append("tools:")
        elif kinds == {"skill"}:
            lines.append("skills:")
        else:
            lines.append("candidates:")
        lines.extend(group["lines"])
    return "\n".join(lines)


def _group_selectors(items: list[dict[str, Any]]) -> list[str]:
    tool_names = [
        str(item.get("name") or str(item.get("id") or "").split(".", 1)[-1])
        for item in items
        if item.get("kind") == "tool"
    ]
    skill_names = [
        str(item.get("name") or str(item.get("id") or "").split(".", 1)[-1])
        for item in items
        if item.get("kind") == "skill"
    ]
    selectors = []
    if tool_names:
        selectors.append(f"tool_backpack select {','.join(tool_names)}")
    selectors.extend(f"skill_backpack select {name}" for name in skill_names)
    return selectors


def _is_file_inspection_request(request_text: str, request_tokens: set[str]) -> bool:
    if not _has_file_object(request_text, request_tokens):
        return False
    if _is_file_search_request(request_text, request_tokens):
        return True
    if request_tokens & {"explain", "look", "open", "read", "says", "summarize", "understand", "inspect"}:
        return True
    return _has_any_phrase(request_text, ("读", "打开", "看一下", "解释"))


def _is_file_search_request(request_text: str, request_tokens: set[str]) -> bool:
    return bool(request_tokens & {"find", "search", "locate"}) or _has_any_phrase(request_text, ("搜索", "查找"))


def _is_debug_failure_request(request_text: str, request_tokens: set[str]) -> bool:
    return bool(
        request_tokens
        & {"broken", "cause", "debug", "diagnose", "error", "exception", "failing", "fails", "failure", "traceback", "unexpected"}
    ) or "stack trace" in request_text or _has_any_phrase(request_text, ("报错", "失败", "异常", "找原因"))


def _is_implementation_request(request_text: str, request_tokens: set[str]) -> bool:
    return bool(request_tokens & {"fix", "implement", "change", "edit", "patch"}) or _has_any_phrase(
        request_text, ("修", "改", "实现")
    )


def _is_web_lookup_request(request_text: str, request_tokens: set[str]) -> bool:
    if re.search(r"https?://", request_text):
        return True
    if _is_implementation_request(request_text, request_tokens):
        return False
    return bool(request_tokens & {"web", "lookup"})


def _tool_catalog() -> list[dict[str, Any]]:
    try:
        from tools.tool_backpack import TOOL_SHORT_DESCRIPTIONS, _all_index_tool_names
    except Exception:
        return []
    return [
        {
            "id": f"tool.{name}",
            "kind": "tool",
            "name": name,
            "description": TOOL_SHORT_DESCRIPTIONS.get(name, "tool"),
        }
        for name in _all_index_tool_names()
    ]


def _skill_catalog() -> list[dict[str, Any]]:
    try:
        from tools.skill_backpack import _all_skill_entries
    except Exception:
        return []
    items = []
    for entry in _all_skill_entries():
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        items.append(
            {
                "id": f"skill.{name}",
                "kind": "skill",
                "name": name,
                "description": str(entry.get("description") or ""),
            }
        )
    return items


def _rank(request: str, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    request_text = request.lower()
    request_tokens = _tokens(request)
    ranked = []
    for item in catalog:
        item_text = " ".join([str(item.get("name") or ""), str(item.get("description") or "")])
        item_tokens = _tokens(item_text)
        score = len(request_tokens & item_tokens)
        score += _action_score(item, item_tokens, request_tokens, request_text)
        if score:
            ranked.append((score, str(item.get("id") or ""), item))
    ranked.sort(key=lambda entry: (-entry[0], entry[1]))
    return [item for _score, _item_id, item in ranked]


def _tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in STOP_WORDS:
            continue
        tokens.add(token)
        if len(token) > 3 and token.endswith("s"):
            tokens.add(token[:-1])
    return tokens


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def _has_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _phrase_count(text: str, phrases: tuple[str, ...]) -> int:
    return sum(1 for phrase in phrases if phrase in text)


def _action_score(item: dict[str, Any], item_tokens: set[str], request_tokens: set[str], request_text: str) -> float:
    score = 0.0
    item_id = str(item.get("id") or "")
    kind = item.get("kind")
    is_backpack_management_request = "backpack" in request_tokens and bool(
        request_tokens & {"availability", "hidden", "index", "indexes", "manage", "manager", "state", "update"}
    )
    is_file_object = _has_file_object(request_text, request_tokens)
    if kind == "tool" and is_file_object and request_tokens & {"explain", "look", "open", "read", "says", "summarize", "understand", "inspect"}:
        if item_tokens & {"read", "contents"} or item_id in {"tool.read_file", "tool.read"}:
            score += 13.0
    if kind == "tool" and is_file_object and _has_any_phrase(request_text, ("读", "打开", "看一下", "解释")):
        if item_tokens & {"read", "contents"} or item_id in {"tool.read_file", "tool.read"}:
            score += 13.0
    if request_tokens & {"find", "search", "locate"}:
        if item_id in {"tool.search_files", "tool.grep", "tool.glob"}:
            score += 7.0
    if request_tokens & {"broken", "cause", "debug", "diagnose", "failing", "fails", "failure", "unexpected"} or "stack trace" in request_text:
        if item_id == "skill.systematic-debugging":
            score += 10.0
    if _has_any_phrase(request_text, ("报错", "失败", "异常", "找原因")):
        if item_id == "skill.systematic-debugging":
            score += 10.0
    if request_tokens & {"fix", "implement", "change", "edit"}:
        if item_id == "skill.test-driven-development":
            score += 7.0
        if item_id in {"tool.patch", "tool.apply_patch"}:
            score += 5.0
    if _has_any_phrase(request_text, ("修", "改", "实现")):
        if item_id == "skill.test-driven-development":
            score += 7.0
        if item_id in {"tool.patch", "tool.apply_patch"}:
            score += 5.0
    if request_tokens & {"verify", "validate", "validation", "command", "run"} and not is_backpack_management_request:
        if item_id in {"tool.terminal", "tool.bash"}:
            score += 7.0
    if _has_any_phrase(request_text, ("跑测试", "运行", "验证")) and not is_backpack_management_request:
        if item_id in {"tool.terminal", "tool.bash"}:
            score += 7.0
    if request_tokens & {"choose", "clarify", "clarifying", "option", "preference", "unclear", "unsure"} or "not sure" in request_text:
        if item_id == "tool.clarify":
            score += 10.0
    if _has_any_phrase(request_text, ("不确定", "不清楚", "选哪个", "先问我", "要不要", "该不该")):
        if item_id == "tool.clarify":
            score += 10.0
    if request_tokens & {"conversation", "earlier", "history", "previous", "session", "sessions"} or "said before" in request_text:
        if item_id == "tool.session_search":
            score += 10.0
    if _has_any_phrase(request_text, ("之前我们", "之前说", "历史")):
        if item_id == "tool.session_search":
            score += 10.0
    if request_tokens & {"todo", "task", "tasks", "list", "track", "tracked"} or "next steps" in request_text:
        if item_id == "tool.todo":
            score += 10.0
    if _has_any_phrase(request_text, ("任务清单", "任务列表", "列个任务")):
        if item_id == "tool.todo":
            score += 10.0
    if request_tokens & {"browser", "browse", "navigate", "page", "web"} or re.search(r"https?://", request_text):
        if item_id == "tool.browser_navigate":
            score += 8.0
    if request_tokens & {"image", "picture", "screenshot", "visual", "vision"}:
        if item_id in {"tool.vision_analyze", "tool.browser_vision"}:
            score += 8.0
    if request_tokens & {"delegate", "focused", "parallel", "subagent"}:
        if item_id == "tool.delegate_task":
            score += 10.0
    if is_backpack_management_request and item_id in {"skill.backpack-manager", "skill.skill-backpack"}:
        score += 12.0
    return score


def _has_file_object(request_text: str, request_tokens: set[str]) -> bool:
    if re.search(r"\b[\w./-]+\.(md|json|py|txt|yaml|yml|toml)\b", request_text):
        return True
    return bool(request_tokens & {"readme", "protocol", "manifest"})


def _selector(item: dict[str, Any]) -> str:
    name = item.get("name") or str(item.get("id") or "").split(".", 1)[-1]
    if item.get("kind") == "skill":
        return f"skill_backpack select {name}"
    return f"tool_backpack select {name}"
