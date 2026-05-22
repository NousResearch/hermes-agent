"""Retrieval evaluation helpers for LLM Wiki dogfood/regression checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

import yaml


class Searcher(Protocol):
    """Minimal protocol implemented by WikiSearch and test fakes."""

    def search(self, query: str, limit: int = 5) -> list[Any]: ...


@dataclass(frozen=True)
class RetrievalEvalCase:
    """A retrieval expectation for one query."""

    query: str
    expected_pages: set[str]
    top_k: int = 5


@dataclass(frozen=True)
class RetrievalEvalCaseResult:
    """Result for one retrieval expectation."""

    query: str
    expected_pages: set[str]
    retrieved_pages: list[str]
    missing_pages: set[str]

    @property
    def passed(self) -> bool:
        return not self.missing_pages


@dataclass(frozen=True)
class RetrievalEvalResult:
    """Aggregate retrieval evaluation result."""

    cases: list[RetrievalEvalCaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def failures(self) -> list[RetrievalEvalCaseResult]:
        return [case for case in self.cases if not case.passed]

    @property
    def passed(self) -> bool:
        return not self.failures


def _result_page_path(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("page_path") or "").strip()
    return str(getattr(result, "page_path", "") or "").strip()


def evaluate_retrieval(searcher: Searcher, cases: Iterable[RetrievalEvalCase]) -> RetrievalEvalResult:
    """Run retrieval expectations against a wiki searcher.

    The helper intentionally validates page presence only. It does not judge answer
    quality or generate text, so it can be used in deterministic tests and dogfood
    smoke checks without invoking an LLM.
    """

    results: list[RetrievalEvalCaseResult] = []
    for case in cases:
        top_k = max(1, int(case.top_k or 1))
        raw_results = searcher.search(case.query, limit=top_k)
        retrieved_pages: list[str] = []
        seen_pages: set[str] = set()
        for result in raw_results:
            page = _result_page_path(result)
            if page and page not in seen_pages:
                retrieved_pages.append(page)
                seen_pages.add(page)
        missing_pages = set(case.expected_pages) - set(retrieved_pages)
        results.append(
            RetrievalEvalCaseResult(
                query=case.query,
                expected_pages=set(case.expected_pages),
                retrieved_pages=retrieved_pages,
                missing_pages=missing_pages,
            )
        )
    return RetrievalEvalResult(cases=results)


def _coerce_expected_pages(value: Any, *, index: int) -> set[str]:
    if isinstance(value, str):
        pages = [value]
    elif isinstance(value, list | tuple | set):
        pages = list(value)
    else:
        raise ValueError(f"case {index}: expected_pages must be a string or list of strings")

    cleaned: set[str] = set()
    for raw_page in pages:
        page = str(raw_page).strip()
        if not page:
            continue
        candidate = Path(page)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"case {index}: expected_pages must be relative wiki page paths")
        cleaned.add(page)
    if not cleaned:
        raise ValueError(f"case {index}: expected_pages must include at least one page path")
    return cleaned


def _case_from_mapping(item: Any, *, index: int) -> RetrievalEvalCase:
    if not isinstance(item, dict):
        raise ValueError(f"case {index}: expected a mapping")

    query = str(item.get("query") or "").strip()
    if not query:
        raise ValueError(f"case {index}: query is required")

    expected_pages = _coerce_expected_pages(item.get("expected_pages"), index=index)
    top_k_raw = item.get("top_k", 5)
    try:
        top_k = max(1, int(top_k_raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"case {index}: top_k must be an integer") from exc

    return RetrievalEvalCase(query=query, expected_pages=expected_pages, top_k=top_k)


def load_retrieval_cases(path: str | Path) -> list[RetrievalEvalCase]:
    """Load retrieval eval cases from YAML or JSON.

    Supported shapes:
    - a top-level list of case mappings;
    - a mapping with a `cases` list.
    """

    case_path = Path(path)
    text = case_path.read_text(encoding="utf-8")
    if case_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)

    if isinstance(data, dict):
        data = data.get("cases")
    if not isinstance(data, list):
        raise ValueError("retrieval eval file must contain a list or a mapping with a cases list")
    if not data:
        raise ValueError("retrieval eval file must contain at least one case")

    return [_case_from_mapping(item, index=index) for index, item in enumerate(data, start=1)]


def result_to_dict(result: RetrievalEvalResult) -> dict[str, Any]:
    """Convert a retrieval eval result into stable JSON-serializable data."""

    return {
        "passed": result.passed,
        "total": result.total,
        "failures": len(result.failures),
        "cases": [
            {
                "query": case.query,
                "passed": case.passed,
                "expected_pages": sorted(case.expected_pages),
                "retrieved_pages": case.retrieved_pages,
                "missing_pages": sorted(case.missing_pages),
            }
            for case in result.cases
        ],
    }


def _load_explicit_wiki_config(config_path: str | Path):
    """Load exactly one explicit Hermes config file, with no fallback search."""

    from hermes_wiki.config import WikiConfig

    path = Path(config_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Hermes config not found: {config_path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Hermes config must be a mapping: {config_path}")
    if "wiki" not in data:
        raise ValueError(f"Hermes config has no wiki section: {config_path}")
    return WikiConfig.from_dict(data)


def _build_searcher(config_path: str | None = None) -> Searcher:
    from hermes_wiki.config import WikiConfig
    from hermes_wiki.search import WikiSearch

    config = _load_explicit_wiki_config(config_path) if config_path is not None else WikiConfig.from_hermes_config()
    return WikiSearch(config, ensure_collection=False)


def main(argv: list[str] | None = None) -> int:
    """Run retrieval eval cases from the command line."""

    parser = argparse.ArgumentParser(description="Run LLM Wiki retrieval regression checks")
    parser.add_argument("cases", help="YAML or JSON retrieval eval cases file")
    parser.add_argument("--config", help="Hermes config.yaml path to load wiki settings from")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args(argv)

    try:
        cases = load_retrieval_cases(args.cases)
        result = evaluate_retrieval(_build_searcher(args.config), cases)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(json.dumps(result_to_dict(result), indent=2 if args.pretty else None, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI use
    raise SystemExit(main())
