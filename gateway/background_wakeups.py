"""Compact wake-up registry for background worker routing.

Keeps the main Feishu session thin while letting /background workers wake the
right tool lanes and skill bundles for high-frequency work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

from agent.wake_manifest import build_wake_manifest
from hermes_cli.config import get_config_path
from hermes_constants import get_hermes_home


@dataclass(frozen=True)
class BackgroundWakeupPlan:
    """Resolved wake-up plan for one background task."""

    route_names: tuple[str, ...]
    enabled_toolsets: tuple[str, ...]
    skill_names: tuple[str, ...]
    route_families: tuple[str, ...] = ()
    wrapper_commands: tuple[str, ...] = ()
    owner_work_plan: OwnerWorkDispatchPlan | None = None
    execution_plan: ExecutionPlan | None = None
    match_details: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        parts: list[str] = []
        if self.route_names:
            parts.append("routes=" + ", ".join(self.route_names))
        if self.skill_names:
            parts.append("skills=" + ", ".join(self.skill_names))
        return " | ".join(parts)

    @property
    def explainability_summary(self) -> str:
        if not self.match_details:
            return ""
        return "; ".join(self.match_details)


@dataclass(frozen=True)
class ForegroundWakeSuggestion:
    """Foreground nudge when a thin parent should surface a dormant lane."""

    route_names: tuple[str, ...]
    suggested_commands: tuple[str, ...]
    missing_toolsets: tuple[str, ...]
    skill_names: tuple[str, ...]
    work_shape_reasons: tuple[str, ...] = ()
    match_details: tuple[str, ...] = ()

    @property
    def explainability_summary(self) -> str:
        if not self.match_details:
            return ""
        return "; ".join(self.match_details)


@dataclass(frozen=True)
class SpecialistReceiptBinding:
    """Truthful route→specialist binding for runtime receipt emission."""

    route_role: str
    target_agent_id: str
    route_names: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeReceiptContract:
    """Receipt contract for runtime paths.

    binding_kind='entity' means a real specialist alias binding exists and the
    receipt may update a workbench. binding_kind='route' means this is a
    route-scoped ledger entry only and must not pretend an entity accepted it.
    """

    route_role: str
    target_agent_id: str
    route_names: tuple[str, ...]
    binding_kind: str


@dataclass(frozen=True)
class OwnerWorkDispatchUnit:
    """One Hank-facing work unit resolved into an owner + route."""

    owner: str
    work_class: str
    route_name: str
    explicit_owner: bool = False


@dataclass(frozen=True)
class OwnerWorkDispatchConflict:
    """Structured owner/work mismatch — never silently reroute this."""

    owner: str
    work_class: str
    allowed_work_classes: tuple[str, ...]

    @property
    def summary(self) -> str:
        allowed = ", ".join(self.allowed_work_classes)
        return f"owner '{self.owner}' only supports [{allowed}], not '{self.work_class}'"


@dataclass(frozen=True)
class UnsupportedOwnerRequest:
    """Known governance alias that exists conceptually but is not live-routable."""

    owner: str


@dataclass(frozen=True)
class OwnerWorkDispatchPlan:
    """Hank-facing dispatch interpretation before lane/tool expansion."""

    explicit_owner: str | None
    work_units: tuple[OwnerWorkDispatchUnit, ...]
    conflict: OwnerWorkDispatchConflict | None = None
    unsupported_owner: UnsupportedOwnerRequest | None = None

    @property
    def route_names(self) -> tuple[str, ...]:
        return tuple(unit.route_name for unit in self.work_units)

    @property
    def needs_delegation_planner(self) -> bool:
        return self.conflict is None and self.unsupported_owner is None and len(self.work_units) > 1


@dataclass(frozen=True)
class ExecutionPlanUnit:
    """One ordered execution unit for multi-work routing."""

    unit_index: int
    owner: str
    work_class: str
    route_name: str
    explicit_owner: bool
    depends_on: tuple[int, ...] = ()
    parallelizable: bool = False
    input_contract: str = ""
    output_contract: str = ""
    merge_strategy: str = ""
    review_required: bool = True


@dataclass(frozen=True)
class ExecutionPlan:
    """Machine-readable execution plan for worker-side orchestration."""

    units: tuple[ExecutionPlanUnit, ...]
    dispatch_policy: str = "serial"


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        item = str(raw or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _normalize(prompt: str) -> str:
    return " ".join(str(prompt or "").lower().replace("_", " ").replace("-", " ").split())


def _normalize_match_term(value: str) -> str:
    return _normalize(value)


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


_WRAPPER_HISTORY_WINDOW_DAYS = 30
_WRAPPER_FULL_HISTORY_WEIGHT_SAMPLE_COUNT = 3.0


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_sample_count(value: Any) -> float:
    parsed = _coerce_float(value)
    if parsed is None:
        return 0.0
    return max(0.0, parsed)


def _route_history_row(summary: Mapping[str, Mapping[str, Any]], route_name: str) -> Mapping[str, Any] | None:
    route_key = str(route_name or "").strip().lower()
    if not route_key:
        return None
    for candidate in (route_key, route_key.replace("_", "-"), route_key.replace("-", "_")):
        row = summary.get(candidate)
        if row is not None:
            return row
    return None


def _wrapper_history_weight(sample_count: float) -> float:
    if sample_count <= 0:
        return 0.0
    return min(1.0, float(sample_count) / _WRAPPER_FULL_HISTORY_WEIGHT_SAMPLE_COUNT)


def _route_history_sort_signal(summary: Mapping[str, Mapping[str, Any]], route_name: str) -> tuple[bool, float, str]:
    """Return sample-weighted route history for wrapper suggestion ordering."""

    row = _route_history_row(summary, route_name)
    if row is None:
        return (False, 0.0, "")

    sample_count = _coerce_sample_count(row.get("history_effective_sample_count_30d"))
    if sample_count <= 0:
        sample_count = _coerce_sample_count(row.get("history_sample_count_30d"))
    if sample_count <= 0:
        return (False, 0.0, "")

    route_signal = _coerce_float(row.get("route_signal_score_30d"))
    worker_signal = _coerce_float(row.get("worker_outcome_signal_30d"))
    sort_signal = (route_signal or 0.0) * _wrapper_history_weight(sample_count)
    worker_count = _coerce_sample_count(row.get("worker_outcome_effective_sample_count_30d"))
    if worker_count <= 0:
        worker_count = _coerce_sample_count(row.get("worker_outcome_count_30d"))
    explanation = (
        f"route_signal={route_signal or 0.0:.3g};"
        f"worker_signal={worker_signal or 0.0:.3g};"
        f"effective_samples={sample_count:.3g};effective_worker_samples={worker_count:.3g}"
    )
    return (True, sort_signal, explanation)


def _route_history_summary() -> Mapping[str, Mapping[str, Any]]:
    try:
        from tools.skill_usage import summarize_route_usage

        return summarize_route_usage(window_days=_WRAPPER_HISTORY_WINDOW_DAYS)
    except Exception:
        return {}


def _append_match_detail(match_details: list[str], route: str, reason: str) -> None:
    detail = f"{route}<= {reason}".replace("<= ", "<=")
    if detail not in match_details:
        match_details.append(detail)


def _find_matching_terms(text: str, needles: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        needle for needle in needles if needle and needle in text
    )


_SEMANTIC_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "be", "but", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "please", "so", "that", "the", "this", "to", "with",
    "帮", "帮我", "请", "一下", "一个", "这个", "那个", "的", "了", "和", "与", "把",
})


def _semantic_tokens(value: str) -> frozenset[str]:
    normalized = _normalize_match_term(value)
    tokens = [
        token.strip(".,:;!?()[]{}'\"")
        for token in normalized.split()
    ]
    return frozenset(
        token for token in tokens
        if len(token) >= 2 and token not in _SEMANTIC_STOP_WORDS
    )


def _best_route_example_match(text: str, examples: Sequence[str]) -> tuple[str, float] | None:
    prompt_tokens = _semantic_tokens(text)
    if not prompt_tokens:
        return None

    best_example = ""
    best_score = 0.0
    for raw_example in examples:
        example = str(raw_example or "").strip()
        normalized_example = _normalize_match_term(example)
        if not normalized_example:
            continue
        if normalized_example in text:
            return (example, 1.0)
        example_tokens = _semantic_tokens(example)
        if len(example_tokens) < 3:
            continue
        overlap = prompt_tokens & example_tokens
        if len(overlap) < 3:
            continue
        score = len(overlap) / max(1, min(len(prompt_tokens), len(example_tokens)))
        if score > best_score:
            best_score = score
            best_example = example

    if best_score >= 0.45:
        return (best_example, round(best_score, 3))
    return None


def _iter_occurrences(text: str, needle: str) -> tuple[tuple[int, int], ...]:
    if not needle:
        return ()
    spans: list[tuple[int, int]] = []
    start = text.find(needle)
    while start >= 0:
        spans.append((start, start + len(needle)))
        start = text.find(needle, start + 1)
    return tuple(spans)


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return not (left[1] <= right[0] or left[0] >= right[1])


def _find_match_spans(text: str, needles: Sequence[str]) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for raw in needles:
        needle = _normalize_match_term(raw)
        if not needle:
            continue
        spans.extend(_iter_occurrences(text, needle))
    return tuple(sorted(set(spans)))


def _first_match_index(
    text: str,
    needles: Sequence[str],
    *,
    blocked_spans: Sequence[tuple[int, int]] = (),
) -> int | None:
    best: int | None = None
    for raw in needles:
        needle = _normalize_match_term(raw)
        if not needle:
            continue
        for span in _iter_occurrences(text, needle):
            if any(_spans_overlap(span, blocked) for blocked in blocked_spans):
                continue
            if best is None or span[0] < best:
                best = span[0]
    return best


def _detect_owner_matches(text: str) -> tuple[tuple[int, str], ...]:
    matches: list[tuple[int, str]] = []
    for owner, aliases in _OWNER_ALIASES.items():
        for raw in aliases:
            needle = _normalize_match_term(raw)
            if not needle:
                continue
            for span in _iter_occurrences(text, needle):
                matches.append((span[0], owner))
    return tuple(sorted(matches))


def _detect_explicit_owner(text: str) -> str | None:
    owner_matches = _detect_owner_matches(text)
    owners = tuple(_dedupe(owner for _, owner in owner_matches))
    if len(owners) != 1:
        return None
    return owners[0]


def _detect_work_class_matches(text: str) -> tuple[tuple[int, str], ...]:
    feishu_doc_spans = _find_match_spans(text, _FEISHU_DOC_KEYWORDS)
    google_doc_spans = _find_match_spans(text, _GOOGLE_DOC_KEYWORDS)
    html_spans = _find_match_spans(text, _HTML_KEYWORDS)

    matches: list[tuple[int, int, str]] = []
    work_indices = {
        "research": _first_match_index(text, _RESEARCH_KEYWORDS),
        "feishu_doc": _first_match_index(text, _FEISHU_DOC_KEYWORDS),
        "google_doc": _first_match_index(text, _GOOGLE_DOC_KEYWORDS),
        "document": _first_match_index(
            text,
            _DOCUMENT_KEYWORDS,
            blocked_spans=feishu_doc_spans + google_doc_spans,
        ),
        "html": _first_match_index(text, _HTML_KEYWORDS),
        "ppt": _first_match_index(text, _PPT_KEYWORDS, blocked_spans=html_spans),
        "repo": _first_match_index(text, _REPO_KEYWORDS),
    }

    for order, work_class in enumerate(_WORK_CLASS_ORDER):
        index = work_indices.get(work_class)
        if index is None:
            continue
        matches.append((index, order, work_class))

    return tuple((index, work_class) for index, _, work_class in sorted(matches))


def _detect_work_classes(text: str) -> tuple[str, ...]:
    return tuple(work_class for _, work_class in _detect_work_class_matches(text))


def _build_unsupported_owner_plan(owner: str) -> OwnerWorkDispatchPlan:
    return OwnerWorkDispatchPlan(
        explicit_owner=owner,
        work_units=(),
        unsupported_owner=UnsupportedOwnerRequest(owner=owner),
    )


def _build_owner_dispatch_plan(
    owner: str,
    work_classes: Sequence[str],
    *,
    explicit_owner: bool,
) -> OwnerWorkDispatchPlan:
    if owner in _GOVERNANCE_ONLY_OWNERS:
        return _build_unsupported_owner_plan(owner)

    allowed_work_classes = tuple(sorted(_OWNER_ALLOWED_WORK_CLASSES[owner]))
    normalized_work_classes = tuple(work_classes)
    if normalized_work_classes:
        for work_class in normalized_work_classes:
            if work_class not in _OWNER_ALLOWED_WORK_CLASSES[owner]:
                return OwnerWorkDispatchPlan(
                    explicit_owner=owner,
                    work_units=(),
                    conflict=OwnerWorkDispatchConflict(
                        owner=owner,
                        work_class=work_class,
                        allowed_work_classes=allowed_work_classes,
                    ),
                )
        return OwnerWorkDispatchPlan(
            explicit_owner=owner if explicit_owner else None,
            work_units=tuple(
                OwnerWorkDispatchUnit(
                    owner=owner,
                    work_class=work_class,
                    route_name=_WORK_CLASS_TO_ROUTE[work_class],
                    explicit_owner=explicit_owner,
                )
                for work_class in normalized_work_classes
            ),
        )

    default_work_class = _OWNER_DEFAULT_WORK_CLASS[owner]
    return OwnerWorkDispatchPlan(
        explicit_owner=owner if explicit_owner else None,
        work_units=(
            OwnerWorkDispatchUnit(
                owner=owner,
                work_class=default_work_class,
                route_name=_OWNER_DEFAULT_ROUTE[owner],
                explicit_owner=explicit_owner,
            ),
        ),
    )


def _assign_owner_to_work_unit(
    work_index: int,
    owner_matches: Sequence[tuple[int, str]],
) -> str | None:
    if not owner_matches:
        return None
    preceding = [owner for index, owner in owner_matches if index <= work_index]
    if preceding:
        return preceding[-1]
    return owner_matches[0][1]


def resolve_owner_work_dispatch(prompt: str) -> OwnerWorkDispatchPlan | None:
    """Resolve Hank-facing owner/work dispatch before lane tool expansion."""

    normalized = _normalize(prompt)
    owner_matches = _detect_owner_matches(normalized)
    work_matches = _detect_work_class_matches(normalized)
    work_classes = tuple(work_class for _, work_class in work_matches)
    explicit_owner = _detect_explicit_owner(normalized)

    if not owner_matches and not work_matches:
        return None

    if explicit_owner is not None:
        return _build_owner_dispatch_plan(
            explicit_owner,
            work_classes,
            explicit_owner=True,
        )

    if work_matches and owner_matches:
        work_units: list[OwnerWorkDispatchUnit] = []
        for work_index, work_class in work_matches:
            assigned_owner = _assign_owner_to_work_unit(work_index, owner_matches)
            if assigned_owner in _GOVERNANCE_ONLY_OWNERS:
                return _build_unsupported_owner_plan(assigned_owner)
            if assigned_owner in _ROUTABLE_OWNERS:
                if work_class not in _OWNER_ALLOWED_WORK_CLASSES[assigned_owner]:
                    return OwnerWorkDispatchPlan(
                        explicit_owner=assigned_owner,
                        work_units=(),
                        conflict=OwnerWorkDispatchConflict(
                            owner=assigned_owner,
                            work_class=work_class,
                            allowed_work_classes=tuple(sorted(_OWNER_ALLOWED_WORK_CLASSES[assigned_owner])),
                        ),
                    )
                work_units.append(
                    OwnerWorkDispatchUnit(
                        owner=assigned_owner,
                        work_class=work_class,
                        route_name=_WORK_CLASS_TO_ROUTE[work_class],
                        explicit_owner=True,
                    )
                )
                continue

            mapped_owner = _WORK_CLASS_TO_OWNER[work_class]
            work_units.append(
                OwnerWorkDispatchUnit(
                    owner=mapped_owner,
                    work_class=work_class,
                    route_name=_WORK_CLASS_TO_ROUTE[work_class],
                    explicit_owner=False,
                )
            )

        return OwnerWorkDispatchPlan(
            explicit_owner=None,
            work_units=tuple(work_units),
        )

    if owner_matches:
        first_owner = owner_matches[0][1]
        return _build_owner_dispatch_plan(first_owner, (), explicit_owner=True)

    return OwnerWorkDispatchPlan(
        explicit_owner=None,
        work_units=tuple(
            OwnerWorkDispatchUnit(
                owner=_WORK_CLASS_TO_OWNER[work_class],
                work_class=work_class,
                route_name=_WORK_CLASS_TO_ROUTE[work_class],
                explicit_owner=False,
            )
            for work_class in work_classes
        ),
    )


_FEISHU_WORK_BASE_TOOLSETS = (
    "clarify",
    "file",
    "memory",
    "session_search",
    "skills",
    "terminal",
    "todo",
)


def _resolve_default_toolsets(platform: str, default_toolsets: Sequence[str] | None) -> list[str]:
    """Resolve the worker-lane baseline toolsets without platform expansion.

    Background wake-up routing receives already-resolved platform toolsets at
    runtime. Re-expanding them through ``_get_platform_tools`` here leaks
    platform cockpit tools (for example Feishu document/comment tools and the
    dispatcher-only kanban toolset) into worker lanes and can starve /repo of
    local inspection tools. Treat the special ``hermes-feishu-work`` sentinel as
    the lean worker baseline used by Feishu route tests; otherwise preserve the
    caller-provided toolsets exactly and let concrete routes append what they
    need.
    """

    toolset_names = _dedupe(default_toolsets or ())
    if platform == "feishu" and (not toolset_names or "hermes-feishu-work" in toolset_names):
        expanded: list[str] = []
        for toolset_name in toolset_names or ["hermes-feishu-work"]:
            if toolset_name == "hermes-feishu-work":
                expanded.extend(_FEISHU_WORK_BASE_TOOLSETS)
            else:
                expanded.append(toolset_name)
        return _dedupe(expanded)
    return toolset_names


_INFO_KEYWORDS = (
    "搜集", "收集", "资料", "信息", "来源", "source", "sources", "fact", "facts",
    "scan", "gather", "collect", "landscape",
)
_RESEARCH_KEYWORDS = (
    "研究", "research", "行业", "赛道", "market", "industry", "trend", "trends", "竞品", "格局",
    "对标", "机会", "风险", "市场",
)
_MULTI_AGENT_KEYWORDS = (
    "多 agent", "多agent", "多个 agent", "多个agent", "并行", "分头", "拆给", "fan out",
    "parallel", "workers", "多个 worker", "多个worker", "开分身", "子 agent", "子agent",
)
_LONG_RUNNING_SCOPE_KEYWORDS = (
    "完整", "完整的", "全面", "系统", "深入", "deep dive", "deep-dive", "comprehensive",
    "full", "full brief", "full report", "end to end", "end-to-end", "从0到1", "整版",
)
_SYNTHESIS_KEYWORDS = (
    "综合结论", "综合判断", "结论", "建议", "判断", "takeaway", "takeaways", "synthesis",
    "recommendation", "recommendations", "brief", "report",
)
_REPO_KEYWORDS = (
    "github", "repository", "repo", "codebase", "pull request", "pr", "issue triage", "repo stats",
    "仓库", "代码库", "pr review", "github pr",
)
_SELF_GOVERNANCE_REPO_SUBJECT_KEYWORDS = (
    "hermes", "lira", "route", "routing", "router", "dispatcher", "wrapper", "worker", "lane",
    "background wakeups", "wake manifest", "receipt", "session lifecycle", "memory governance",
    "routing governance", "自我治理", "路由", "分派", "分诊", "调度", "协议",
)
_SELF_GOVERNANCE_REPO_SIGNAL_KEYWORDS = (
    "体系化审查", "审查", "检查", "制定", "提升计划", "优化空间", "机制提升", "机制设计",
    "本身机制", "多文件", "文件", "代码", "代码库", "codebase", "repo", "repository",
    "protocol", "protocols", "validator", "script", "scripts", "cron", "skill", "skills",
    "state", "test", "tests", "patch", "diff", "验证", "gateway/", ".py",
    "routing governance", "background wakeups.py", "docs/protocols",
)
_SELF_GOVERNANCE_FOREGROUND_ONLY_MARKERS = (
    "不用查文件", "不查文件", "不用读文件", "不读文件", "不用看代码", "不看代码",
    "不用改", "别改", "只判断", "轻量判断",
)
_OSS_BENCHMARK_RESEARCH_KEYWORDS = (
    "开源社区", "社区先进", "先进案例", "最佳实践", "best practice", "best practices",
    "benchmark", "benchmarks", "oss", "open source", "开源案例", "对照案例",
)


def _matches_self_governance_repo_work(normalized_prompt: str) -> bool:
    if _contains_any(normalized_prompt, _SELF_GOVERNANCE_FOREGROUND_ONLY_MARKERS):
        return False
    return (
        _contains_any(normalized_prompt, _SELF_GOVERNANCE_REPO_SUBJECT_KEYWORDS)
        and _contains_any(normalized_prompt, _SELF_GOVERNANCE_REPO_SIGNAL_KEYWORDS)
    )


def _matches_oss_benchmark_research(normalized_prompt: str) -> bool:
    return _contains_any(normalized_prompt, _OSS_BENCHMARK_RESEARCH_KEYWORDS)


_AUTOMATION_KEYWORDS = (
    "cron", "定时", "自动化", "schedule", "scheduled", "每天", "每周", "每月", "提醒",
)
_FEISHU_DOC_KEYWORDS = (
    "feishu", "lark", "飞书", "云文档", "wiki", "docx", "飞书文档",
)
_GOOGLE_DOC_KEYWORDS = (
    "google doc", "google docs", "gdoc", "gdocs", "谷歌文档", "google 文档",
)
_PDF_DOC_KEYWORDS = (
    "turn into pdf", "convert to pdf", "make pdf", "generate pdf", "render pdf", "html to pdf",
    "转pdf", "转成pdf", "生成pdf", "导出pdf", "输出pdf",
    "报告转pdf", "正式文档转pdf", "proposal pdf", "technical doc pdf", "whitepaper pdf",
    "doc-to-pdf", "markdown to pdf", "word to pdf", "local doc to pdf", "reportlab pdf",
)
_DOCUMENT_KEYWORDS = (
    "文档", "document", "写文档", "写个文档", "doc", "memo", "文稿", "整理成文",
)
_HTML_KEYWORDS = (
    "html slides", "html slide", "html deck", "html presentation", "html 展示稿", "html 演示稿", "html",
)
_PPT_KEYWORDS = (
    "ppt", ".pptx", "slides", "slide", "deck", "presentation", "演示文稿", "幻灯片",
)
_EDITABLE_DECK_KEYWORDS = (
    ".pptx", "editable", "可编辑", "powerpoint", "pptx",
)
_OWNER_ALIASES: dict[str, tuple[str, ...]] = {
    "bran": ("bran",),
    "claire": ("claire",),
    "frank": ("frank",),
    "sam": ("sam",),
    "seth": ("seth",),
}
_ROUTABLE_OWNERS = frozenset({"bran", "claire", "frank"})
_GOVERNANCE_ONLY_OWNERS = frozenset({"sam", "seth"})
_OWNER_DEFAULT_WORK_CLASS = {
    "bran": "research",
    "claire": "document",
    "frank": "repo",
}
_OWNER_DEFAULT_ROUTE = {
    "bran": "research",
    "claire": "doc_feishu",
    "frank": "repo",
}
_OWNER_ALLOWED_WORK_CLASSES: dict[str, frozenset[str]] = {
    "bran": frozenset({"research"}),
    "claire": frozenset({"document", "feishu_doc", "google_doc", "ppt", "html"}),
    "frank": frozenset({"repo"}),
}
_WORK_CLASS_TO_OWNER = {
    "research": "bran",
    "document": "claire",
    "feishu_doc": "claire",
    "google_doc": "claire",
    "ppt": "claire",
    "html": "claire",
    "repo": "frank",
}
_WORK_CLASS_TO_ROUTE = {
    "research": "research",
    "document": "doc_feishu",
    "feishu_doc": "doc_feishu",
    "google_doc": "doc_google",
    "ppt": "ppt",
    "html": "ppt",
    "repo": "repo",
}
_WORK_CLASS_ORDER = (
    "research",
    "document",
    "feishu_doc",
    "google_doc",
    "ppt",
    "html",
    "repo",
)

_ROUTE_COMMANDS: dict[str, tuple[str, ...]] = {
    "background": (),
    "bg": (),
    "research": ("research",),
    # /doc is an abstract document wrapper.  Keep it abstract here so
    # _apply_route can choose the concrete lane from prompt semantics
    # (Feishu by default, Google/PDF when explicitly requested).
    "doc": ("doc",),
    "ppt": ("ppt",),
    "repo": ("repo",),
}
_ROUTE_FAMILY_BY_ROUTE: dict[str, str] = {
    "default": "default",
    "work": "work",
    "research": "research",
    "scan": "research",
    "multi_agent": "multi_agent",
    "repo": "repo",
    "automation": "automation",
    "doc_feishu": "doc",
    "doc_google": "doc",
    "doc_pdf": "doc",
    "ppt": "ppt",
}

_ROUTE_RECEIPT_BINDINGS: dict[str, tuple[str, str]] = {
    "research": ("research-specialist", "bran"),
    "scan": ("research-specialist", "bran"),
    "doc_feishu": ("document-specialist", "claire"),
    "doc_google": ("document-specialist", "claire"),
    "doc_pdf": ("document-specialist", "claire"),
    "ppt": ("document-specialist", "claire"),
    "repo": ("execution-specialist", "frank"),
    "automation": ("execution-specialist", "frank"),
}

_BASE_ROUTE_CATALOG: dict[str, dict[str, object]] = {
    "research": {
        "command": "research",
        "display_command": "/research",
        "skills": (),
        "append_toolsets": ("web",),
    },
    "scan": {
        "command": "research",
        "display_command": "/research",
        "skills": (),
        "append_toolsets": ("web",),
    },
    "multi_agent": {
        "command": "background",
        "display_command": "/bg",
        "skills": (),
        "append_toolsets": ("delegation",),
    },
    "repo": {
        "command": "repo",
        "display_command": "/repo",
        "skills": (),
        # Repo/codebase lanes must be able to inspect the local checkout and run
        # read-only verification commands even when the Feishu parent session is
        # configured as a thin document/comment cockpit.
        "append_toolsets": ("terminal", "file", "skills", "session_search"),
        "route_examples": (
            "inspect a repository and patch failing tests",
            "review codebase implementation details and return verified diffs",
        ),
    },
    "automation": {
        "command": "background",
        "display_command": "/bg",
        "skills": (),
        "append_toolsets": ("cronjob",),
    },
    "doc_feishu": {
        "command": "doc",
        "display_command": "/doc",
        "skills": (),
    },
    "doc_google": {
        "command": "doc",
        "display_command": "/doc",
        "skills": (),
    },
    "doc_pdf": {
        "command": "doc",
        "display_command": "/doc",
        "skills": (),
    },
    "ppt": {
        "command": "ppt",
        "display_command": "/ppt",
        "skills": (),
        "conditional_skills": {
            "powerpoint": _EDITABLE_DECK_KEYWORDS,
        },
    },
}

BACKGROUND_ROUTE_COMMANDS = frozenset(
    name for name, forced in _ROUTE_COMMANDS.items() if forced
)


def get_base_background_route_catalog() -> dict[str, dict[str, object]]:
    """Return the seed route catalog before skill metadata binding."""

    return {
        route: {
            "command": str(spec.get("command", "")),
            "display_command": str(spec.get("display_command", "")),
            "skills": tuple(spec.get("skills", ()) or ()),
            "append_toolsets": tuple(spec.get("append_toolsets", ()) or ()),
            "prepend_toolsets": tuple(spec.get("prepend_toolsets", ()) or ()),
            "drop_toolsets": tuple(spec.get("drop_toolsets", ()) or ()),
            "conditional_skills": {
                str(skill_name): tuple(keywords or ())
                for skill_name, keywords in (spec.get("conditional_skills", {}) or {}).items()
            },
            "route_examples": tuple(spec.get("route_examples", ()) or ()),
        }
        for route, spec in _BASE_ROUTE_CATALOG.items()
    }


@lru_cache(maxsize=8)
def _get_background_wake_manifest_cached(
    platform: str,
    hermes_home: str,
    config_marker: tuple[str, int, int],
) -> dict[str, object]:
    del hermes_home, config_marker  # key-only cache inputs for test/runtime invalidation
    return build_wake_manifest(get_base_background_route_catalog(), platform=platform)


def _wake_manifest_config_marker() -> tuple[str, int, int]:
    config_path = get_config_path()
    try:
        stat = config_path.stat()
        return (str(config_path), int(stat.st_mtime_ns), int(stat.st_size))
    except FileNotFoundError:
        return (str(config_path), 0, 0)


def clear_background_wake_manifest_cache() -> None:
    """Clear the in-process wake manifest cache (mainly for tests)."""

    _get_background_wake_manifest_cached.cache_clear()


def get_background_wake_manifest(platform: str = "feishu") -> dict[str, object]:
    """Return the cached wake manifest for one platform."""

    platform_key = str(platform or "feishu").strip().lower() or "feishu"
    return _get_background_wake_manifest_cached(
        platform_key,
        str(get_hermes_home()),
        _wake_manifest_config_marker(),
    )


def forced_routes_for_command(command_name: str | None) -> tuple[str, ...]:
    """Return forced wake-up routes for a wrapper command."""

    normalized = str(command_name or "").strip().lower().lstrip("/")
    return _ROUTE_COMMANDS.get(normalized, ())


def get_background_route_catalog(platform: str = "feishu") -> dict[str, dict[str, object]]:
    """Return the runtime route catalog.

    This catalog is the live routing truth for wrapper hints, worker wake-ups,
    and foreground guidance. Keep user-visible routing language derived from
    this runtime view instead of duplicating stale route maps elsewhere.
    """

    routes = (get_background_wake_manifest(platform).get("routes") or {})
    return {
        route: {
            "command": str(spec.get("command", "")),
            "display_command": str(spec.get("display_command", "")),
            "skills": tuple(spec.get("skills", ()) or ()),
            "route_examples": tuple(spec.get("route_examples", ()) or ()),
        }
        for route, spec in routes.items()
    }


def _get_background_route_specs(platform: str = "feishu") -> dict[str, dict[str, object]]:
    """Return full runtime route specs including behavior fields."""

    routes = (get_background_wake_manifest(platform).get("routes") or {})
    return {
        route: dict(spec)
        for route, spec in routes.items()
    }


def _metadata_matched_routes(
    normalized_prompt: str,
    route_catalog: dict[str, dict[str, object]],
) -> tuple[tuple[str, str], ...]:
    """Return manifest-derived route matches from aggregated aliases/keywords."""

    matched: list[tuple[str, str]] = []
    for route_name, route_spec in route_catalog.items():
        aliases = tuple(_normalize_match_term(item) for item in (route_spec.get("aliases") or ()))
        keywords = tuple(_normalize_match_term(item) for item in (route_spec.get("keywords") or ()))
        route_examples = tuple(str(item or "") for item in (route_spec.get("route_examples") or ()))
        alias_hits = _find_matching_terms(normalized_prompt, aliases)
        if alias_hits:
            matched.append((str(route_name), f"metadata_alias:{alias_hits[0]}"))
            continue
        keyword_hits = _find_matching_terms(normalized_prompt, keywords)
        if keyword_hits:
            matched.append((str(route_name), f"metadata_keyword:{keyword_hits[0]}"))
            continue
        example_match = _best_route_example_match(normalized_prompt, route_examples)
        if example_match:
            example, score = example_match
            matched.append((str(route_name), f"route_example:{score}:{example[:64]}"))
    return tuple(matched)


def _apply_route(
    route: str,
    normalized_prompt: str,
    route_names: list[str],
    toolsets: list[str],
    skills: list[str],
    *,
    route_catalog: dict[str, dict[str, object]],
    match_details: list[str] | None = None,
    match_reason: str | None = None,
) -> None:
    route_key = str(route or "").strip().lower()
    if not route_key:
        return

    if route_key == "doc":
        if _contains_any(normalized_prompt, _GOOGLE_DOC_KEYWORDS):
            route_key = "doc_google"
        elif _contains_any(normalized_prompt, _FEISHU_DOC_KEYWORDS):
            route_key = "doc_feishu"
        elif _contains_any(normalized_prompt, _PDF_DOC_KEYWORDS):
            route_key = "doc_pdf"
        else:
            route_key = "doc_feishu"

    route_spec = route_catalog.get(route_key, {})
    if not route_spec:
        return

    route_skill_names = tuple(route_spec.get("skills", ()) or ())
    route_names.append(route_key)
    if match_details is not None and match_reason:
        _append_match_detail(match_details, route_key, match_reason)

    drop_toolsets = tuple(route_spec.get("drop_toolsets", ()) or ())
    if drop_toolsets:
        toolsets[:] = [ts for ts in toolsets if ts not in drop_toolsets]

    prepend_toolsets = tuple(route_spec.get("prepend_toolsets", ()) or ())
    if prepend_toolsets:
        toolsets[:0] = list(prepend_toolsets)

    append_toolsets = tuple(route_spec.get("append_toolsets", ()) or ())
    if append_toolsets:
        toolsets.extend(append_toolsets)

    conditional_skills = route_spec.get("conditional_skills", {}) or {}
    conditional_skill_names = set(conditional_skills)
    skills.extend(
        skill_name for skill_name in route_skill_names if skill_name not in conditional_skill_names
    )
    for skill_name, keywords in conditional_skills.items():
        if skill_name in route_skill_names and _contains_any(normalized_prompt, tuple(keywords or ())):
            skills.append(skill_name)


def _route_families_for_routes(route_names: Sequence[str]) -> tuple[str, ...]:
    families: list[str] = []
    for route_name in _dedupe(route_names):
        family = _ROUTE_FAMILY_BY_ROUTE.get(route_name)
        if family:
            families.append(family)
    return tuple(_dedupe(families))


def _concrete_runtime_routes(route_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        route_name
        for route_name in _dedupe(route_names)
        if route_name in _ROUTE_FAMILY_BY_ROUTE and route_name not in {"default", "work"}
    )


def _build_execution_plan(owner_work_plan: OwnerWorkDispatchPlan | None) -> ExecutionPlan | None:
    if owner_work_plan is None or not owner_work_plan.needs_delegation_planner:
        return None

    units = tuple(
        ExecutionPlanUnit(
            unit_index=index,
            owner=work_unit.owner,
            work_class=work_unit.work_class,
            route_name=work_unit.route_name,
            explicit_owner=work_unit.explicit_owner,
            depends_on=() if index == 1 else (index - 1,),
            parallelizable=False,
            input_contract="consume prior unit output" if index > 1 else "initial task prompt",
            output_contract=f"{work_unit.route_name} worker result",
            merge_strategy="serial_handoff",
            review_required=True,
        )
        for index, work_unit in enumerate(owner_work_plan.work_units, start=1)
    )
    return ExecutionPlan(units=units, dispatch_policy="serial")


def _build_route_level_execution_plan(route_names: Sequence[str]) -> ExecutionPlan | None:
    concrete_routes = tuple(route for route in _concrete_runtime_routes(route_names) if route != "multi_agent")
    if "repo" not in concrete_routes or not any(route in {"research", "scan"} for route in concrete_routes):
        return None

    route_specs = {
        "research": ("bran", "research", "external research findings"),
        "scan": ("bran", "scan", "source scan findings"),
        "repo": ("frank", "repo", "codebase inspection findings"),
    }
    units: list[ExecutionPlanUnit] = []
    for route in concrete_routes:
        if route not in route_specs:
            continue
        owner, work_class, output_contract = route_specs[route]
        units.append(
            ExecutionPlanUnit(
                unit_index=len(units) + 1,
                owner=owner,
                work_class=work_class,
                route_name=route,
                explicit_owner=False,
                depends_on=(),
                parallelizable=True,
                input_contract="initial task prompt",
                output_contract=output_contract,
                merge_strategy="director_synthesis",
                review_required=True,
            )
        )
    if len(units) < 2:
        return None
    return ExecutionPlan(units=tuple(units), dispatch_policy="parallel")


def _execution_plan_payload(execution_plan: ExecutionPlan | None) -> dict[str, object] | None:
    if execution_plan is None:
        return None

    return {
        "dispatch_policy": execution_plan.dispatch_policy,
        "units": [
            {
                "unit_index": unit.unit_index,
                "owner": unit.owner,
                "work_class": unit.work_class,
                "route_name": unit.route_name,
                "explicit_owner": unit.explicit_owner,
                "depends_on": list(unit.depends_on),
                "parallelizable": unit.parallelizable,
                "input_contract": unit.input_contract,
                "output_contract": unit.output_contract,
                "merge_strategy": unit.merge_strategy,
                "review_required": unit.review_required,
            }
            for unit in execution_plan.units
        ],
    }


def _infer_foreground_work_shape_reasons(
    prompt: str,
    plan: BackgroundWakeupPlan,
    *,
    meaningful_routes: Sequence[str],
) -> tuple[str, ...]:
    normalized = _normalize(prompt)
    reasons: list[str] = []

    if plan.execution_plan is not None or "multi_agent" in meaningful_routes:
        reasons.append("requires_orchestration")

    research_like = any(route in {"research", "scan"} for route in meaningful_routes)
    long_running_scope = _contains_any(normalized, _LONG_RUNNING_SCOPE_KEYWORDS)
    synthesis_requested = _contains_any(normalized, _SYNTHESIS_KEYWORDS)
    if research_like and (long_running_scope or synthesis_requested):
        reasons.append("likely_long_running")

    return tuple(_dedupe(reasons))


def suggested_commands_for_routes(
    route_names: Sequence[str],
    *,
    platform: str = "feishu",
    forced_routes: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return preferred wrapper commands for the resolved routes."""

    route_catalog = get_background_route_catalog(platform)
    history_summary = _route_history_summary()
    forced_route_set = set(_dedupe(tuple(forced_routes or ())))
    entries: list[tuple[str, str, int, bool, bool, float]] = []
    for index, route in enumerate(_dedupe(tuple(route_names))):
        route_key = str(route or "").strip()
        spec = route_catalog.get(route_key)
        if not spec:
            continue
        display = str(spec.get("display_command", "")).strip()
        if not display:
            continue
        has_history, sort_signal, _ = _route_history_sort_signal(history_summary, route_key)
        entries.append((display, route_key, index, route_key in forced_route_set, has_history, sort_signal))

    def _sort_key(item: tuple[str, str, int, bool, bool, float]) -> tuple[int, int, float, int]:
        _, _, original_index, forced, has_history, sort_signal = item
        # Forced routes always win. Otherwise preserve the P3 priority bands:
        # positive history > neutral/no history > negative history. The decayed
        # signal only orders routes inside the positive/negative bands.
        if not has_history or sort_signal == 0:
            history_status = 1
        elif sort_signal > 0:
            history_status = 2
        else:
            history_status = 0
        return (0 if forced else 1, -history_status, -sort_signal, original_index)

    entries.sort(key=_sort_key)
    return tuple(_dedupe([entry[0] for entry in entries]))


def route_history_match_details_for_routes(route_names: Sequence[str]) -> tuple[str, ...]:
    """Return explainability details for telemetry-fed wrapper ranking."""

    history_summary = _route_history_summary()
    details: list[str] = []
    for route in _dedupe(tuple(route_names)):
        if route in {"default", "work"}:
            continue
        has_history, sort_signal, explanation = _route_history_sort_signal(history_summary, route)
        if has_history:
            _append_match_detail(details, route, f"history:sort_signal={sort_signal:.3g};{explanation}")
    return tuple(details)


def resolve_specialist_receipt_binding(route_names: Sequence[str]) -> SpecialistReceiptBinding | None:
    """Resolve an unambiguous route→specialist binding for runtime receipts.

    Returns None when the resolved routes do not map cleanly onto one
    specialist alias / route_role pair. That keeps receipts truthful instead of
    pretending mixed or orchestration-only lanes are entity dispatches.
    """

    normalized_routes = _concrete_runtime_routes(route_names)
    bindings = [
        _ROUTE_RECEIPT_BINDINGS[route]
        for route in normalized_routes
        if route in _ROUTE_RECEIPT_BINDINGS
    ]
    if not bindings:
        return None

    route_roles = {route_role for route_role, _ in bindings}
    target_ids = {target_agent_id for _, target_agent_id in bindings}
    if len(route_roles) != 1 or len(target_ids) != 1:
        return None

    route_role = next(iter(route_roles))
    target_agent_id = next(iter(target_ids))
    bound_routes = tuple(route for route in normalized_routes if route in _ROUTE_RECEIPT_BINDINGS)
    return SpecialistReceiptBinding(
        route_role=route_role,
        target_agent_id=target_agent_id,
        route_names=bound_routes,
    )


def resolve_runtime_receipt_contract(route_names: Sequence[str]) -> RuntimeReceiptContract | None:
    """Resolve the truthful receipt contract for live runtime lanes.

    - entity binding: real specialist alias binding exists
    - route binding: route-scoped ledger entry only (single unbound route or
      multi-route mixed work)
    """

    normalized_routes = _concrete_runtime_routes(route_names)
    if not normalized_routes:
        return None

    binding = resolve_specialist_receipt_binding(normalized_routes)
    if binding is not None:
        return RuntimeReceiptContract(
            route_role=binding.route_role,
            target_agent_id=binding.target_agent_id,
            route_names=binding.route_names,
            binding_kind="entity",
        )

    if len(normalized_routes) == 1:
        route_name = normalized_routes[0]
        return RuntimeReceiptContract(
            route_role=f"{route_name}-route",
            target_agent_id=f"route:{route_name}",
            route_names=normalized_routes,
            binding_kind="route",
        )

    return RuntimeReceiptContract(
        route_role="mixed-route",
        target_agent_id="route:mixed",
        route_names=normalized_routes,
        binding_kind="route",
    )


def _summarize_route_catalog_for_foreground(platform: str = "feishu") -> str:
    """Return a compact wrapper→route summary derived from the live catalog."""

    catalog = get_background_route_catalog(platform)
    grouped: dict[str, list[str]] = {}
    for route_name, spec in catalog.items():
        display_command = str(spec.get("display_command", "")).strip()
        if not display_command:
            continue
        grouped.setdefault(display_command, []).append(str(route_name))

    parts: list[str] = []
    for command, routes in grouped.items():
        parts.append(f"{command} → {', '.join(routes)}")
    return "; ".join(parts)


def resolve_background_wakeup(
    prompt: str,
    *,
    platform: str | None = None,
    default_toolsets: Sequence[str] | None = None,
    forced_routes: Sequence[str] | None = None,
) -> BackgroundWakeupPlan:
    """Resolve the background worker wake-up plan.

    Keeps the behavior compact:
    - only Feishu gets special tool-lane routing
    - other platforms fall back to the existing toolsets
    - route behavior remains heuristic, but skill bundles now come from the
      validated wake manifest built from installed skills
    """

    normalized = _normalize(prompt)
    platform_key = str(platform or "").strip().lower()
    resolved_default_toolsets = _resolve_default_toolsets(platform_key, default_toolsets)

    if platform_key != "feishu":
        return BackgroundWakeupPlan(
            route_names=("default",),
            enabled_toolsets=tuple(resolved_default_toolsets),
            skill_names=(),
            route_families=("default",),
            wrapper_commands=(),
        )

    route_catalog = _get_background_route_specs(platform_key)
    route_names: list[str] = ["work"]
    toolsets: list[str] = list(resolved_default_toolsets)
    skills: list[str] = []
    match_details: list[str] = []

    owner_work_plan = resolve_owner_work_dispatch(prompt)
    execution_plan = _build_execution_plan(owner_work_plan)
    owner_work_conflict = owner_work_plan.conflict if owner_work_plan else None
    owner_work_unsupported = owner_work_plan.unsupported_owner if owner_work_plan else None
    owner_work_blocked = owner_work_conflict is not None or owner_work_unsupported is not None
    applied_routes = set(route_names)
    forced_route_names: list[str] = []

    if owner_work_plan and not owner_work_blocked:
        for work_unit in owner_work_plan.work_units:
            _apply_route(
                work_unit.route_name,
                normalized,
                route_names,
                toolsets,
                skills,
                route_catalog=route_catalog,
                match_details=match_details,
                match_reason=(
                    f"owner_work:{work_unit.owner}:{work_unit.work_class}"
                    if work_unit.explicit_owner else
                    f"work_only:{work_unit.work_class}:{work_unit.owner}"
                ),
            )
            applied_routes.add(work_unit.route_name)

        if execution_plan is not None and "multi_agent" not in applied_routes:
            _apply_route(
                "multi_agent",
                normalized,
                route_names,
                toolsets,
                skills,
                route_catalog=route_catalog,
                match_details=match_details,
                match_reason="owner_work:delegation_planner",
            )
            applied_routes.add("multi_agent")

    if not owner_work_blocked:
        matched_research = _contains_any(normalized, _RESEARCH_KEYWORDS)
        matched_info = _contains_any(normalized, _INFO_KEYWORDS)
        matched_oss_benchmark_research = _matches_oss_benchmark_research(normalized)
        matched_self_governance_repo = _matches_self_governance_repo_work(normalized)
        if (matched_research or matched_oss_benchmark_research) and "research" not in applied_routes and "scan" not in applied_routes:
            _apply_route(
                "research",
                normalized,
                route_names,
                toolsets,
                skills,
                route_catalog=route_catalog,
                match_details=match_details,
                match_reason=(
                    "heuristic:research_keywords"
                    if matched_research else "heuristic:oss_benchmark_research"
                ),
            )
            applied_routes.add("research")
        elif matched_info and "research" not in applied_routes and "scan" not in applied_routes:
            _apply_route(
                "scan",
                normalized,
                route_names,
                toolsets,
                skills,
                route_catalog=route_catalog,
                match_details=match_details,
                match_reason="heuristic:info_keywords",
            )
            applied_routes.add("scan")

        if _contains_any(normalized, _MULTI_AGENT_KEYWORDS):
            _apply_route(
                "multi_agent",
                normalized,
                route_names,
                toolsets,
                skills,
                route_catalog=route_catalog,
                match_details=match_details,
                match_reason="heuristic:multi_agent_keywords",
            )
            applied_routes.add("multi_agent")

        matched_repo = _contains_any(normalized, _REPO_KEYWORDS)
        if (matched_repo or matched_self_governance_repo) and "repo" not in applied_routes:
            _apply_route(
                "repo",
                normalized,
                route_names,
                toolsets,
                skills,
                route_catalog=route_catalog,
                match_details=match_details,
                match_reason=(
                    "heuristic:repo_keywords"
                    if matched_repo else "heuristic:self_governance_repo"
                ),
            )
            applied_routes.add("repo")

        if (
            "repo" in applied_routes
            and ("research" in applied_routes or "scan" in applied_routes)
            and "multi_agent" not in applied_routes
        ):
            _apply_route(
                "multi_agent",
                normalized,
                route_names,
                toolsets,
                skills,
                route_catalog=route_catalog,
                match_details=match_details,
                match_reason="heuristic:mixed_repo_research",
            )
            applied_routes.add("multi_agent")

        if _contains_any(normalized, _AUTOMATION_KEYWORDS) and "automation" not in applied_routes:
            _apply_route(
                "automation",
                normalized,
                route_names,
                toolsets,
                skills,
                route_catalog=route_catalog,
                match_details=match_details,
                match_reason="heuristic:automation_keywords",
            )
            applied_routes.add("automation")

        if _contains_any(normalized, _FEISHU_DOC_KEYWORDS) and "doc_feishu" not in applied_routes:
            _apply_route(
                "doc_feishu",
                normalized,
                route_names,
                toolsets,
                skills,
                route_catalog=route_catalog,
                match_details=match_details,
                match_reason="heuristic:feishu_doc_keywords",
            )
            applied_routes.add("doc_feishu")

        if _contains_any(normalized, _GOOGLE_DOC_KEYWORDS) and "doc_google" not in applied_routes:
            _apply_route(
                "doc_google",
                normalized,
                route_names,
                toolsets,
                skills,
                route_catalog=route_catalog,
                match_details=match_details,
                match_reason="heuristic:google_doc_keywords",
            )
            applied_routes.add("doc_google")

        if _contains_any(normalized, _PPT_KEYWORDS) and "ppt" not in applied_routes:
            _apply_route(
                "ppt",
                normalized,
                route_names,
                toolsets,
                skills,
                route_catalog=route_catalog,
                match_details=match_details,
                match_reason="heuristic:ppt_keywords",
            )
            applied_routes.add("ppt")

        for route, reason in _metadata_matched_routes(normalized, route_catalog):
            if route in applied_routes or route in {"default", "work"}:
                continue
            _apply_route(
                route,
                normalized,
                route_names,
                toolsets,
                skills,
                route_catalog=route_catalog,
                match_details=match_details,
                match_reason=reason,
            )
            applied_routes.add(route)

    for route in forced_routes or ():
        previous_route_count = len(route_names)
        _apply_route(
            str(route),
            normalized,
            route_names,
            toolsets,
            skills,
            route_catalog=route_catalog,
            match_details=match_details,
            match_reason=f"forced:{str(route).strip().lower()}",
        )
        forced_route_names.extend(route_names[previous_route_count:])

    resolved_route_names = tuple(_dedupe(route_names))
    match_details.extend(route_history_match_details_for_routes(resolved_route_names))
    final_execution_plan = execution_plan or _build_route_level_execution_plan(resolved_route_names)
    return BackgroundWakeupPlan(
        route_names=resolved_route_names,
        enabled_toolsets=tuple(_dedupe(toolsets)),
        skill_names=tuple(_dedupe(skills)),
        route_families=_route_families_for_routes(resolved_route_names),
        wrapper_commands=suggested_commands_for_routes(
            resolved_route_names,
            platform=platform_key,
            forced_routes=tuple(_dedupe(forced_route_names)),
        ),
        owner_work_plan=owner_work_plan,
        execution_plan=final_execution_plan,
        match_details=tuple(_dedupe(match_details)),
    )


def record_background_plan_usage(
    plan: BackgroundWakeupPlan,
    *,
    platform: str,
    task_id: str | None = None,
    source: str = "background_worker",
) -> None:
    """Persist route-hit and router-selected skill telemetry for a live worker run."""

    meaningful_routes = tuple(route for route in plan.route_names if route not in {"default", "work"})
    if not meaningful_routes and not plan.skill_names:
        return

    try:
        from agent.skill_usage import log_route_usage_event, log_skill_usage_event
    except Exception:
        return

    wrapper_command = plan.wrapper_commands[0] if plan.wrapper_commands else ""
    base_details = {
        "platform": str(platform or "").strip().lower(),
        "task_id": str(task_id or "").strip(),
        "source": str(source or "").strip().lower(),
        "selected_routes": list(meaningful_routes),
        "wrapper_commands": list(plan.wrapper_commands),
        "route_families": list(plan.route_families),
        "match_details": list(plan.match_details),
    }

    for route in meaningful_routes:
        route_details = dict(base_details)
        route_details["route"] = route
        route_details["wrapper_command"] = wrapper_command
        try:
            log_route_usage_event(
                route_name=route,
                event="route_selected_for_background",
                details=route_details,
            )
        except Exception:
            continue

    for skill_name in plan.skill_names:
        try:
            log_skill_usage_event(
                skill_name=str(skill_name),
                event="skill_selected_by_router",
                details=base_details,
            )
        except Exception:
            continue



def _build_owner_work_delegation_guidance(execution_plan: ExecutionPlan | None) -> str:
    if execution_plan is None:
        return ""

    payload = _execution_plan_payload(execution_plan)
    plan_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return (
        " Execution plan: "
        f"{plan_json}. "
        "Multi-work dispatch has already been resolved into ordered execution units. "
        "Feed this execution plan into your delegation planner instead of re-inferring it. "
        "If delegate_task is available, keep one child per execution unit, pass receipt_binding.owner with the matching route_names for each child, and preserve this order. "
        "Only run children in parallel when a unit is explicitly marked parallelizable and does not depend on earlier output."
    )


def build_background_ephemeral_prompt(plan: BackgroundWakeupPlan) -> str:
    """Build a tiny worker-side note for the resolved wake-up plan."""

    if not plan.route_names or plan.route_names == ("default",):
        return ""

    route_text = ", ".join(plan.route_names)
    skill_text = ""
    if plan.skill_names:
        skill_text = " Load these route-linked skills first if relevant: " + ", ".join(plan.skill_names[:4]) + "."
    delegation_text = _build_owner_work_delegation_guidance(plan.execution_plan)
    return (
        "[SYSTEM: This is a background worker session. "
        f"Active wake-up routes: {route_text}. "
        "Stay scoped to the assigned task, then return crisp results to the director agent for synthesis and final review."
        f"{delegation_text}"
        f"{skill_text}]"
    )


def build_feishu_director_hint() -> str:
    """Return a concise Feishu-only hint for the main session."""

    route_summary = _summarize_route_catalog_for_foreground("feishu")
    return (
        "[SYSTEM: On Feishu, keep the main chat lightweight. You are the director and final reviewer. "
        f"Treat the live background route catalog as the routing truth: {route_summary}. "
        "When a task fits one of those lanes, recommend the matching wrapper/worker instead of pretending the thin foreground already has that lane active. "
        "In user-visible language, prefer lane / wrapper / worker phrasing like '适合走研究链路，建议用 /research' or '这次走 /repo worker' and do not say Bran/Claire/Frank already picked it up unless a real receipt exists.]"
    )


def resolve_feishu_capability_gap(
    prompt: str,
    *,
    active_toolsets: Sequence[str] | None = None,
) -> ForegroundWakeSuggestion | None:
    """Return a structured foreground wake suggestion for thin Feishu parents."""

    owner_work_plan = resolve_owner_work_dispatch(prompt)
    if owner_work_plan and (owner_work_plan.conflict or owner_work_plan.unsupported_owner):
        return None

    resolved_active_toolsets = tuple(
        _resolve_default_toolsets(
            "feishu",
            active_toolsets if active_toolsets is not None else ("hermes-feishu-work",),
        )
    )
    plan = resolve_background_wakeup(
        prompt,
        platform="feishu",
        default_toolsets=resolved_active_toolsets,
    )
    meaningful_routes = tuple(
        route for route in plan.route_names if route not in {"default", "work"}
    )
    if not meaningful_routes:
        return None

    commands = suggested_commands_for_routes(meaningful_routes, platform="feishu") or ("/bg",)
    active_toolset_set = set(resolved_active_toolsets)
    missing_toolsets = tuple(
        toolset for toolset in plan.enabled_toolsets if toolset not in active_toolset_set
    )
    skill_names = tuple(plan.skill_names)
    work_shape_reasons = _infer_foreground_work_shape_reasons(
        prompt,
        plan,
        meaningful_routes=meaningful_routes,
    )
    if not missing_toolsets and not skill_names and not work_shape_reasons:
        return None

    return ForegroundWakeSuggestion(
        route_names=meaningful_routes,
        suggested_commands=commands,
        missing_toolsets=missing_toolsets,
        skill_names=skill_names,
        work_shape_reasons=work_shape_reasons,
        match_details=tuple(plan.match_details),
    )


def build_feishu_capability_gap_hint(
    prompt: str,
    *,
    active_toolsets: Sequence[str] | None = None,
) -> str:
    """Return a per-turn hint nudging the thin parent to suggest worker lanes."""

    owner_work_plan = resolve_owner_work_dispatch(prompt)
    if owner_work_plan and owner_work_plan.unsupported_owner:
        owner = owner_work_plan.unsupported_owner.owner
        return (
            "[SYSTEM: This Feishu request explicitly names "
            f"{owner}, which is a known governance alias but currently has no live wrapper or route in runtime. "
            "Do not pretend that worker-lane routing for that owner already exists. "
            "If replying, say this is a governance-only owner right now and ask Hank whether to switch to an available lane / wrapper instead. "
            "Do not say a worker already picked it up.]"
        )
    if owner_work_plan and owner_work_plan.conflict:
        conflict = owner_work_plan.conflict
        allowed_text = ", ".join(conflict.allowed_work_classes)
        return (
            "[SYSTEM: This Feishu request explicitly names "
            f"{conflict.owner} but also matches {conflict.work_class} work, which conflicts with the fixed Hank dispatch map "
            "(Claire=document/feishu_doc/ppt/html, Bran=research, Frank=repo). "
            f"That owner only supports: {allowed_text}. "
            "If replying, point out the mismatch and ask Hank whether to keep the named owner or switch to the matching lane. "
            "Do not silently reroute, and do not say a worker already picked it up.]"
        )

    suggestion = resolve_feishu_capability_gap(prompt, active_toolsets=active_toolsets)
    if not suggestion:
        return ""

    commands = suggestion.suggested_commands
    if len(commands) == 1:
        commands_text = commands[0]
    else:
        commands_text = ", ".join(commands[:2])

    capability_bits: list[str] = []
    if suggestion.missing_toolsets:
        capability_bits.append(
            "missing foreground toolsets: " + ", ".join(suggestion.missing_toolsets)
        )
    if suggestion.skill_names:
        capability_bits.append(
            "route-linked skills available: " + ", ".join(suggestion.skill_names[:3])
        )
    if suggestion.work_shape_reasons:
        capability_bits.append(
            "work-shape routing: " + ", ".join(suggestion.work_shape_reasons)
        )

    capability_summary = " ".join(bit + "." for bit in capability_bits if bit)
    explainability = suggestion.explainability_summary
    reason_summary = f" Match: {explainability}." if explainability else ""
    route_summary = ", ".join(suggestion.route_names)
    live_catalog_summary = _summarize_route_catalog_for_foreground("feishu")

    return (
        "[SYSTEM: This Feishu request appears to fit live worker lanes: "
        f"{route_summary}.{reason_summary} "
        f"{capability_summary} "
        f"Live wrapper map: {live_catalog_summary}. "
        f"If the thin foreground parent would benefit from worker routing, explicitly suggest {commands_text} "
        "(or /bg for mixed work). Use lane / wrapper / worker language in the reply, and do not say Bran/Claire/Frank already picked it up unless a real receipt exists; do not pretend dormant lanes or bundled skills are already active in the foreground.]"
    )
