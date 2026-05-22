"""Read-only retrieval introspection for LLM Wiki search results."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

import yaml

from hermes_wiki.config import WikiConfig
from hermes_wiki.search import WikiSearch


class Searcher(Protocol):
    """Minimal search protocol used by WikiSearch and deterministic test fakes."""

    def search(
        self,
        query: str,
        limit: int = 5,
        page_type: str | None = None,
        tags: list[str] | None = None,
        exclude_sources: bool = False,
        search_mode: str = "dense",
    ) -> list[Any]: ...


@dataclass(frozen=True)
class RetrievalHit:
    """One ranked retrieval hit/chunk."""

    rank: int
    page_path: str
    title: str
    page_type: str
    chunk_index: int
    score: float | None
    tags: list[str] = field(default_factory=list)
    text_preview: str = ""


@dataclass(frozen=True)
class PageCoverage:
    """Aggregate view of retrieval hits by page/source path."""

    page_path: str
    best_rank: int
    best_score: float | None
    hit_count: int


@dataclass(frozen=True)
class RetrievalIntrospectionReport:
    """Agent-readable diagnostics for a single retrieval query."""

    query: str
    search_mode: str
    top_k: int
    hits: list[RetrievalHit]
    pages: list[PageCoverage]
    expected_pages: set[str] = field(default_factory=set)
    missing_expected_pages: set[str] = field(default_factory=set)

    @property
    def passed_expected_pages(self) -> bool:
        return not self.missing_expected_pages


def _get_attr(result: Any, name: str, default: Any = "") -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _clean_preview(text: Any, *, max_chars: int = 200) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 1].rstrip()}…"


def _coerce_score(score: Any) -> float | None:
    if score is None or score == "":
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _validate_expected_pages(expected_pages: Iterable[str] | None) -> set[str]:
    cleaned: set[str] = set()
    for raw_page in expected_pages or []:
        page = str(raw_page).strip()
        if not page:
            continue
        candidate = Path(page)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("expected pages must be relative wiki page paths")
        cleaned.add(page)
    return cleaned


def _call_searcher(
    searcher: Searcher,
    query: str,
    *,
    top_k: int,
    page_type: str | None,
    exclude_sources: bool,
    search_mode: str,
) -> list[Any]:
    kwargs: dict[str, Any] = {}
    if page_type:
        kwargs["page_type"] = page_type
    if exclude_sources:
        kwargs["exclude_sources"] = True
    if search_mode != "dense":
        kwargs["search_mode"] = search_mode
    return searcher.search(query, limit=top_k, **kwargs)


def introspect_retrieval(
    searcher: Searcher,
    query: str,
    *,
    top_k: int = 5,
    expected_pages: Iterable[str] | None = None,
    page_type: str | None = None,
    exclude_sources: bool = False,
    search_mode: str = "dense",
) -> RetrievalIntrospectionReport:
    """Run one read-only retrieval query and expose ranked chunk/page diagnostics.

    This helper intentionally performs search only. It does not call an LLM,
    mutate wiki pages, append logs, ingest sources, reindex, or queue proposals.
    """

    cleaned_query = str(query or "").strip()
    if not cleaned_query:
        raise ValueError("query is required")
    search_mode = (search_mode or "dense").strip().lower()
    if search_mode not in {"dense", "sparse", "hybrid"}:
        raise ValueError("search_mode must be one of: dense, sparse, hybrid")
    top_k = max(1, int(top_k or 1))
    expected = _validate_expected_pages(expected_pages)

    raw_hits = _call_searcher(
        searcher,
        cleaned_query,
        top_k=top_k,
        page_type=page_type,
        exclude_sources=exclude_sources,
        search_mode=search_mode,
    )
    hits: list[RetrievalHit] = []
    pages_by_path: dict[str, PageCoverage] = {}

    for index, raw_hit in enumerate(raw_hits, start=1):
        page_path = str(_get_attr(raw_hit, "page_path", "") or "").strip()
        if not page_path:
            continue
        score = _coerce_score(_get_attr(raw_hit, "score", None))
        hit = RetrievalHit(
            rank=index,
            page_path=page_path,
            title=str(_get_attr(raw_hit, "title", "") or ""),
            page_type=str(_get_attr(raw_hit, "page_type", "") or ""),
            chunk_index=int(_get_attr(raw_hit, "chunk_index", 0) or 0),
            score=score,
            tags=list(_get_attr(raw_hit, "tags", []) or []),
            text_preview=_clean_preview(_get_attr(raw_hit, "text", "")),
        )
        hits.append(hit)

        existing = pages_by_path.get(page_path)
        if existing is None:
            pages_by_path[page_path] = PageCoverage(
                page_path=page_path,
                best_rank=index,
                best_score=score,
                hit_count=1,
            )
        else:
            pages_by_path[page_path] = PageCoverage(
                page_path=existing.page_path,
                best_rank=existing.best_rank,
                best_score=existing.best_score,
                hit_count=existing.hit_count + 1,
            )

    retrieved_pages = set(pages_by_path)
    return RetrievalIntrospectionReport(
        query=cleaned_query,
        search_mode=search_mode,
        top_k=top_k,
        hits=hits,
        pages=list(pages_by_path.values()),
        expected_pages=expected,
        missing_expected_pages=expected - retrieved_pages,
    )


def _score_to_json(score: float | None) -> float | None:
    if score is None:
        return None
    return round(score, 6)


def introspection_to_dict(report: RetrievalIntrospectionReport) -> dict[str, Any]:
    """Convert an introspection report into stable JSON-serializable data."""

    return {
        "query": report.query,
        "search_mode": report.search_mode,
        "top_k": report.top_k,
        "passed_expected_pages": report.passed_expected_pages,
        "expected_pages": sorted(report.expected_pages),
        "missing_expected_pages": sorted(report.missing_expected_pages),
        "pages": [
            {
                "page_path": page.page_path,
                "best_rank": page.best_rank,
                "best_score": _score_to_json(page.best_score),
                "hit_count": page.hit_count,
            }
            for page in report.pages
        ],
        "hits": [
            {
                "rank": hit.rank,
                "page_path": hit.page_path,
                "title": hit.title,
                "page_type": hit.page_type,
                "chunk_index": hit.chunk_index,
                "score": _score_to_json(hit.score),
                "tags": hit.tags,
                "text_preview": hit.text_preview,
            }
            for hit in report.hits
        ],
    }


def render_introspection_markdown(report: RetrievalIntrospectionReport) -> str:
    """Render retrieval diagnostics as compact Markdown for agent review."""

    lines = [
        "# LLM Wiki Retrieval Introspection",
        "",
        f"- Query: `{report.query}`",
        f"- Search mode: `{report.search_mode}`",
        f"- Top K: `{report.top_k}`",
    ]
    if report.expected_pages:
        status = "✅ Expected page coverage passed" if report.passed_expected_pages else "❌ Expected page coverage failed"
        lines.append(f"- {status}")
        lines.append(f"- Expected pages: {', '.join(f'`{p}`' for p in sorted(report.expected_pages))}")
        if report.missing_expected_pages:
            lines.append(f"- Missing expected pages: {', '.join(f'`{p}`' for p in sorted(report.missing_expected_pages))}")

    lines.extend([
        "",
        "## Page coverage",
        "",
        "| Page | Best rank | Best score | Hits |",
        "| --- | ---: | ---: | ---: |",
    ])
    if report.pages:
        for page in report.pages:
            score = "" if page.best_score is None else f"{_score_to_json(page.best_score):g}"
            lines.append(f"| `{page.page_path}` | {page.best_rank} | {score} | {page.hit_count} |")
    else:
        lines.append("| _No pages returned_ |  |  |  |")

    lines.extend([
        "",
        "## Ranked chunk hits",
        "",
        "| Rank | Page | Score | Type | Chunk | Preview |",
        "| ---: | --- | ---: | --- | ---: | --- |",
    ])
    if report.hits:
        for hit in report.hits:
            score = "" if hit.score is None else f"{_score_to_json(hit.score):g}"
            preview = hit.text_preview.replace("|", "\\|")
            lines.append(
                f"| {hit.rank} | `{hit.page_path}` | {score} | {hit.page_type} | {hit.chunk_index} | {preview} |"
            )
    else:
        lines.append("|  | _No chunks returned_ |  |  |  |  |")

    return "\n".join(lines)


def _load_explicit_wiki_config(config_path: str | Path) -> WikiConfig:
    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Hermes config not found: {config_path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Hermes config must be a mapping: {config_path}")
    if "wiki" not in data:
        raise ValueError(f"Hermes config has no wiki section: {config_path}")
    return WikiConfig.from_dict(data)


def _build_searcher(config_path: str | None) -> Searcher:
    config = _load_explicit_wiki_config(config_path) if config_path is not None else WikiConfig.from_hermes_config()
    return WikiSearch(config, ensure_collection=False)


def main(argv: list[str] | None = None) -> int:
    """Run read-only retrieval introspection from the command line."""

    parser = argparse.ArgumentParser(description="Inspect LLM Wiki retrieval hits for one query")
    parser.add_argument("query", help="Question/search query to inspect")
    parser.add_argument("--config", help="Hermes config.yaml path to load wiki settings from")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunk hits to retrieve")
    parser.add_argument("--expected-page", action="append", default=[], help="Expected wiki page path; repeatable")
    parser.add_argument("--page-type", help="Optional page_type filter")
    parser.add_argument("--exclude-sources", action="store_true", help="Exclude raw source chunks from search results")
    parser.add_argument(
        "--search-mode",
        choices=["dense", "sparse", "hybrid"],
        default="dense",
        help="Retrieval mode to inspect; default preserves dense semantic search",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    args = parser.parse_args(argv)

    try:
        report = introspect_retrieval(
            _build_searcher(args.config),
            args.query,
            top_k=args.top_k,
            expected_pages=args.expected_page,
            page_type=args.page_type,
            exclude_sources=args.exclude_sources,
            search_mode=args.search_mode,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(introspection_to_dict(report), sort_keys=True))
    else:
        print(render_introspection_markdown(report))
    return 0 if report.passed_expected_pages else 1


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI use
    raise SystemExit(main())
