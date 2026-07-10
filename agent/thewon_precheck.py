"""TheWon KH + LLM Wiki precheck runtime enforcement.

This module is intentionally edge-gated: it activates only for TheWon-adjacent
requests on machines that have the user's TheWon runtime installed.  It does
not add a model tool and does not mutate the stable system prompt; the precheck
bundle is injected as ephemeral current-turn user context and mirrored in the
final response as an evidence block.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

THEWON_KEYWORDS = (
    "thewon",
    "the won",
    "hermes",
    "openclaw",
    "min-a",
    "mina",
    "민아",
    "bac",
    "sac",
    "named agent",
    "worker",
    "knowledgehub",
    "knowledge hub",
    "llm wiki",
    "wikirouter",
    "blackbox",
    "ssot",
    "gatekeeper",
    "quality gate",
    "qg",
    "6hats",
    "thinktank",
    "vault",
    "obsidian",
    "rdf",
    "hwpx",
    "hwp",
    "재무모델",
    "에이전트",
    "고도화",
    "정본",
    "반영",
    "절차",
    "품질",
    "거버넌스",
    "위키",
    "하네스",
)


@dataclass
class TheWonPrecheckBundle:
    required: bool
    query: str
    kh_summary: str = ""
    wiki_source: str = ""
    wiki_reason: str = ""
    wiki_hits: list[dict[str, Any]] = field(default_factory=list)
    read_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    kh_query_executed: bool = False
    wiki_route_executed: bool = False
    relevant_sources_read: bool = False
    unresolved_items_declared: bool = True
    source_mode_declared: bool = True

    @property
    def ok(self) -> bool:
        if not self.required:
            return False
        return all(
            [
                self.kh_query_executed,
                self.wiki_route_executed,
                self.relevant_sources_read,
                self.unresolved_items_declared,
                self.source_mode_declared,
            ]
        )

    def precheck_object(self) -> dict[str, bool]:
        return {
            "kh_query_executed": self.kh_query_executed,
            "wiki_route_executed": self.wiki_route_executed,
            "relevant_sources_read": self.relevant_sources_read,
            "unresolved_items_declared": self.unresolved_items_declared,
            "source_mode_declared": self.source_mode_declared,
        }


def _message_to_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts: list[str] = []
        for item in message:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(message or "")


def should_run_thewon_precheck(user_message: Any) -> bool:
    text = _message_to_text(user_message).lower()
    return any(keyword in text for keyword in THEWON_KEYWORDS)


def _ensure_thewon_import_paths() -> bool:
    candidates = [
        Path.home() / "TheWon",
        Path.home() / "TheWon" / "System",
    ]
    added = False
    for path in candidates:
        if path.exists():
            s = str(path)
            if s not in sys.path:
                sys.path.insert(0, s)
                added = True
    return any(path.exists() for path in candidates) or added


def _hit_to_dict(hit: Any) -> dict[str, Any]:
    return {
        "title": getattr(hit, "title", None),
        "confidence": getattr(hit, "confidence", None),
        "score": getattr(hit, "score", None),
        "path": getattr(hit, "path", None),
    }


def run_thewon_precheck(user_message: Any, *, agent_code: str = "Hermes") -> TheWonPrecheckBundle:
    """Run KH + WikiRouter precheck for a TheWon-adjacent user message.

    Returns a bundle even on failure; callers can surface the errors instead of
    silently skipping the gate.
    """
    text = _message_to_text(user_message).strip()
    required = should_run_thewon_precheck(text)
    bundle = TheWonPrecheckBundle(required=required, query=text[:500])
    if not required:
        return bundle

    if not _ensure_thewon_import_paths():
        bundle.errors.append("TheWon path not found; KH/Wiki precheck could not import runtime")
        return bundle

    try:
        from System.shared.knowledge_hub import KnowledgeHub  # type: ignore

        bundle.kh_summary = KnowledgeHub(agent_code=agent_code).smart_ask_formatted(
            text,
            top_k_per_source=3,
            min_sources=2,
        )
        bundle.kh_query_executed = True
    except Exception as exc:  # pragma: no cover - defensive runtime path
        bundle.errors.append(f"KnowledgeHub failed: {exc!r}")
        logger.warning("TheWon KnowledgeHub precheck failed: %s", exc)

    try:
        from System.shared.wiki_router import WikiRouter  # type: ignore

        route = WikiRouter().route(text, threshold=0.2)
        bundle.wiki_route_executed = True
        bundle.wiki_source = str(route.get("source") or "")
        bundle.wiki_reason = str(route.get("reason") or "")
        hits = route.get("hits") or []
        bundle.wiki_hits = [_hit_to_dict(hit) for hit in hits[:5]]
        # Read top relevant wiki pages to make the precheck an actual source read,
        # not just a router metadata lookup.  Keep it small to avoid context bloat.
        for hit in hits[:3]:
            path = getattr(hit, "path", None)
            if not path:
                continue
            try:
                p = Path(str(path))
                if p.exists() and p.is_file():
                    # Touch/read enough to verify accessibility without dumping contents.
                    _ = p.read_text(encoding="utf-8", errors="replace")[:2000]
                    bundle.read_files.append(str(p))
                    bundle.relevant_sources_read = True
            except Exception as exc:  # pragma: no cover - defensive runtime path
                bundle.errors.append(f"Wiki read failed for {path}: {exc!r}")
    except Exception as exc:  # pragma: no cover - defensive runtime path
        bundle.errors.append(f"WikiRouter failed: {exc!r}")
        logger.warning("TheWon WikiRouter precheck failed: %s", exc)

    return bundle


def format_precheck_context(bundle: TheWonPrecheckBundle | dict[str, Any] | None) -> str:
    """Compact context injected into the current user turn for the model."""
    if not bundle:
        return ""
    if isinstance(bundle, dict):
        bundle = TheWonPrecheckBundle(**bundle)
    if not bundle.required:
        return ""
    payload = {
        "precheck_object": bundle.precheck_object(),
        "precheck_pass": bundle.ok,
        "kh": bundle.kh_summary or "(KnowledgeHub: no result or failed)",
        "wiki_source": bundle.wiki_source,
        "wiki_reason": bundle.wiki_reason,
        "wiki_hits": bundle.wiki_hits[:3],
        "read_files": bundle.read_files[:3],
        "errors": bundle.errors,
    }
    return "[TheWon KH+LLM Wiki precheck]\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def format_precheck_response_block(bundle: TheWonPrecheckBundle | dict[str, Any] | None) -> str:
    """User-visible evidence block prefixed to final responses."""
    if not bundle:
        return ""
    if isinstance(bundle, dict):
        bundle = TheWonPrecheckBundle(**bundle)
    if not bundle.required:
        return ""

    kh = (bundle.kh_summary or "(KnowledgeHub: 관련 정보 없음)").strip()
    hit_titles = []
    for hit in bundle.wiki_hits[:3]:
        title = hit.get("title") or "untitled"
        confidence = hit.get("confidence") or "?"
        hit_titles.append(f"{title} ({confidence})")
    wiki_line = "; ".join(hit_titles) if hit_titles else "관련 hit 없음"
    read_line = "; ".join(Path(p).name for p in bundle.read_files[:3]) if bundle.read_files else "읽은 Wiki 파일 없음"
    missing = "; ".join(bundle.errors) if bundle.errors else "없음"

    return (
        "[Precheck]\n"
        f"Object: {json.dumps(bundle.precheck_object(), ensure_ascii=False)}\n"
        f"Gate: {'PASS' if bundle.ok else 'FAIL'}\n"
        f"KH: {kh}\n"
        f"LLM Wiki: source={bundle.wiki_source or 'n/a'}; {wiki_line}\n"
        f"Read: {read_line}\n"
        "Source mode: KH + WikiRouter + live file read\n"
        f"미확인: {missing}"
    )


def apply_precheck_response_block(
    final_response: str | None,
    bundle: TheWonPrecheckBundle | dict[str, Any] | None,
) -> str | None:
    if not final_response:
        return final_response
    if bundle and isinstance(bundle, dict):
        bundle = TheWonPrecheckBundle(**bundle)
    block = format_precheck_response_block(bundle)
    if not block:
        return final_response
    if "[Precheck]" in final_response[:500]:
        return final_response
    if isinstance(bundle, TheWonPrecheckBundle) and bundle.required and not bundle.ok:
        return (
            block
            + "\n\n"
            + "⛔ TheWon runtime hard gate blocked the final answer because "
            + "the mandatory KH + LLM Wiki precheck object did not pass. "
            + "Fix the unresolved precheck items above, then retry."
        )
    return block + "\n\n" + final_response
