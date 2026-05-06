"""Route-aware evaluator for background worker outcomes.

The evaluator is intentionally lightweight and deterministic. It does not judge
content quality like a model would; it verifies that a routed worker returned the
minimum evidence/artifact shape promised by the route contract, so telemetry can
separate useful dispatches from empty "I'll do it later" responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RouteContract:
    """Minimum verifiable output contract for a background route."""

    route_name: str
    description: str
    required_evidence: tuple[str, ...]
    pass_threshold: float = 0.7

    def to_dict(self) -> dict[str, object]:
        return {
            "route_name": self.route_name,
            "description": self.description,
            "required_evidence": list(self.required_evidence),
            "pass_threshold": self.pass_threshold,
        }


@dataclass(frozen=True)
class WorkerOutcomeEvaluation:
    """Deterministic route-contract evaluation result."""

    passed: bool
    score: float
    issues: tuple[str, ...]
    route_contracts: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "score": self.score,
            "issues": list(self.issues),
            "route_contracts": list(self.route_contracts),
        }


ROUTE_CONTRACTS: Mapping[str, RouteContract] = {
    "repo": RouteContract(
        "repo",
        "Codebase/repository worker must cite inspected files and verification commands.",
        ("inspected_files", "verification_command"),
        0.8,
    ),
    "research": RouteContract(
        "research",
        "Research worker must cite sources or URLs, not only provide unsupported synthesis.",
        ("sources",),
        0.75,
    ),
    "scan": RouteContract(
        "scan",
        "Scan worker must report scope and concrete findings.",
        ("scope", "findings"),
        0.7,
    ),
    "doc_feishu": RouteContract(
        "doc_feishu",
        "Feishu document worker must return a document artifact or verifiable document link.",
        ("document_artifact",),
        0.75,
    ),
    "doc_google": RouteContract(
        "doc_google",
        "Google document worker must return a document artifact or verifiable document link.",
        ("document_artifact",),
        0.75,
    ),
    "doc_pdf": RouteContract(
        "doc_pdf",
        "PDF worker must return a PDF or render/export evidence.",
        ("pdf_artifact",),
        0.75,
    ),
    "ppt": RouteContract(
        "ppt",
        "Presentation worker must return slides, PPTX, or an HTML deck artifact.",
        ("presentation_artifact",),
        0.75,
    ),
    "automation": RouteContract(
        "automation",
        "Automation worker must provide execution evidence such as commands, logs, or output paths.",
        ("automation_evidence",),
        0.75,
    ),
    "multi_agent": RouteContract(
        "multi_agent",
        "Multi-agent worker must summarize worker outputs and synthesis/merge decisions.",
        ("orchestration_summary",),
        0.75,
    ),
    "difficult_web_extract": RouteContract(
        "difficult_web_extract",
        "Difficult web extraction worker must return an auditable selector/fallback receipt.",
        ("difficult_web_extract_receipt",),
        0.75,
    ),
}


def _normalize_routes(route_names: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in route_names or ():
        route = str(raw or "").strip().lower().replace("-", "_")
        if not route or route in seen:
            continue
        seen.add(route)
        ordered.append(route)
    return tuple(ordered)


def _has_document_artifact(text: str) -> bool:
    return any(token in text for token in (
        "media:", ".docx", ".pdf", "docs.google.com", "feishu", "larksuite", "document link", "文档链接",
    ))


def _has_presentation_artifact(text: str) -> bool:
    if "no slides" in text or "no presentation" in text or "slides needed" in text:
        return False
    return any(token in text for token in (
        "media:", ".pptx", "slide deck", "html deck", "presentation artifact", "幻灯片", "演示稿",
    )) or ("slides" in text and any(token in text for token in ("created", "exported", "delivered", "attached", "生成", "已做")))


def _has_automation_evidence(text: str) -> bool:
    return any(token in text for token in (
        "ran ", "command", "pytest", "output", "log", "script", "receipt", "exit code", "执行", "日志",
    ))


def _has_orchestration_summary(text: str) -> bool:
    if "no synthesis" in text or "one worker result only" in text:
        return False
    return any(token in text for token in (
        "worker", "subagent", "parallel", "synthesis", "merge", "merged", "workers", "协同", "汇总",
    )) and any(token in text for token in ("synthesis", "merge", "summary", "汇总", "综合"))


def _has_difficult_web_extract_receipt(text: str) -> bool:
    has_backend = "backend=scrapling" in text or '"backend": "scrapling"' in text or "'backend': 'scrapling'" in text
    has_url = "url=" in text or '"url"' in text or "https://" in text or "http://" in text
    has_selector_or_reason = (
        "selector=" in text
        or '"selector"' in text
        or "fallback_reason=" in text
        or '"fallback_reason"' in text
        or "web_extract_empty" in text
    )
    has_errors = "errors=" in text or '"errors"' in text
    return has_backend and has_url and has_selector_or_reason and has_errors


def _issue_for_route(route: str, response_lc: str) -> str | None:
    if route == "research":
        if "http://" not in response_lc and "https://" not in response_lc and "source" not in response_lc and "来源" not in response_lc:
            return "missing_sources"
    elif route == "repo":
        has_file = ".py" in response_lc or "inspected" in response_lc or "file" in response_lc or "文件" in response_lc
        has_verification = "pytest" in response_lc or "test" in response_lc or "git diff" in response_lc or "验证" in response_lc
        if not (has_file and has_verification):
            return "missing_repo_verification"
    elif route in {"doc_feishu", "doc_google"}:
        if not _has_document_artifact(response_lc):
            return "missing_document_artifact"
    elif route == "doc_pdf":
        if ".pdf" not in response_lc and "media:" not in response_lc and "render" not in response_lc and "export" not in response_lc:
            return "missing_pdf_artifact"
    elif route == "ppt":
        if not _has_presentation_artifact(response_lc):
            return "missing_presentation_artifact"
    elif route == "automation":
        if not _has_automation_evidence(response_lc) or "would automate" in response_lc:
            return "missing_automation_evidence"
    elif route == "multi_agent":
        if not _has_orchestration_summary(response_lc):
            return "missing_orchestration_summary"
    elif route == "difficult_web_extract":
        if not _has_difficult_web_extract_receipt(response_lc):
            return "missing_difficult_web_extract_receipt"
    elif route == "scan":
        if not any(token in response_lc for token in ("scope", "scanned", "findings", "发现", "扫描")):
            return "missing_scan_findings"
    return None


def evaluate_background_worker_outcome(
    *,
    prompt: str,
    route_names: Sequence[str],
    response: str,
) -> WorkerOutcomeEvaluation:
    """Evaluate whether a background worker response satisfies route contracts."""

    del prompt  # Reserved for future prompt-sensitive checks.
    routes = _normalize_routes(route_names)
    response_lc = str(response or "").strip().lower()
    issues: list[str] = []

    if not response_lc:
        issues.append("empty_response")

    for route in routes:
        issue = _issue_for_route(route, response_lc)
        if issue:
            issues.append(issue)

    # Unknown routes do not fail by themselves; the caller may be rolling out a
    # new lane. Known-route failures are capped so multi-route successes can
    # still show partial utility, but pass/fail remains thresholded.
    score = max(0.0, 1.0 - 0.35 * len(issues))
    pass_threshold = max((ROUTE_CONTRACTS[route].pass_threshold for route in routes if route in ROUTE_CONTRACTS), default=0.7)
    passed = score >= pass_threshold and not issues
    return WorkerOutcomeEvaluation(
        passed=passed,
        score=round(score, 3),
        issues=tuple(dict.fromkeys(issues)),
        route_contracts=routes,
    )
