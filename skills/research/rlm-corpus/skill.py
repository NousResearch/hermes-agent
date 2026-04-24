#!/usr/bin/env python3
"""Top-level entry point for the rlm-corpus skill.

Usage:
    python skill.py query \\
        --corpus ~/.hermes/rlm-cache/svt-corpus/ \\
        --query "Compare how Volovik and Liberati handle effective Lorentz invariance."

    python skill.py query \\
        --corpus ~/physics/svt-corpus/ \\
        --query "..." \\
        --auto-ingest

The ``--corpus`` arg accepts either a cache dir (the output of ``ingestion.py``)
or a source dir of documents (with ``--auto-ingest``, we'll run ingestion first).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure sibling modules are importable when invoked as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RLMConfig  # noqa: E402
from corpus_loader import is_cache_dir, load_corpus, corpus_summary  # noqa: E402
from ingestion import (  # noqa: E402
    SUPPORTED_EXTENSIONS,
    _read_url_list,
    ingest_crawl,
    ingest_directory,
    ingest_urls,
)
from llm_clients import make_client  # noqa: E402
from rlm_engine import RLMEngine, format_answer_with_references  # noqa: E402

log = logging.getLogger("rlm_corpus.skill")


def _dir_looks_like_source(path: Path) -> bool:
    if not path.is_dir():
        return False
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            return True
    return False


def _resolve_corpus_path(
    corpus_arg: Path,
    config: RLMConfig,
    *,
    auto_ingest: bool,
    workers: int,
    backend: str,
) -> Path:
    """Return a cache dir, ingesting the source dir first if needed."""
    corpus_arg = corpus_arg.expanduser().resolve()
    if is_cache_dir(corpus_arg):
        return corpus_arg

    if not _dir_looks_like_source(corpus_arg):
        raise FileNotFoundError(
            f"{corpus_arg} is neither a cache dir (has no *.json) nor a source dir "
            "(has no supported files)."
        )

    if not auto_ingest:
        raise ValueError(
            f"{corpus_arg} looks like a source dir. Re-run with --auto-ingest "
            "or ingest it first via ingestion.py."
        )

    cache_dir = config.cache_dir / corpus_arg.name
    log.info("auto-ingesting %s -> %s", corpus_arg, cache_dir)
    ingest_directory(
        source_root=corpus_arg,
        cache_dir=cache_dir,
        backend=backend,
        workers=workers,
    )
    return cache_dir


def run_query(
    corpus_path: Path,
    query: str,
    *,
    auto_ingest: bool = False,
    workers: int = 1,
    backend: str = "auto",
    config: RLMConfig | None = None,
    output_format: str = "text",
) -> dict[str, object]:
    config = config or RLMConfig()
    cache_dir = _resolve_corpus_path(
        corpus_path, config, auto_ingest=auto_ingest, workers=workers, backend=backend
    )
    corpus = load_corpus(cache_dir)
    if not corpus:
        raise RuntimeError(f"no documents loaded from {cache_dir}")

    log.info("loaded %d docs; %s", len(corpus), config.describe())

    root = make_client(config.sub_llm_endpoint, config.root_model)
    sub_spec = {
        "endpoint": config.sub_llm_endpoint,
        "model": config.sub_model,
        "base_url": config.sub_llm_base_url,
    }

    with RLMEngine(corpus, root, sub_spec, config) as engine:
        result = engine.answer(query)

    answer = result.get("answer") or "[no answer produced]"
    formatted = format_answer_with_references(answer, corpus)
    result["formatted_answer"] = formatted
    result["corpus_summary"] = corpus_summary(corpus)
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="rlm-corpus: RLM over a local document corpus")
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", help="Answer a query over the corpus")
    q.add_argument("--corpus", required=True, type=Path,
                   help="Path to cache dir (or source dir, with --auto-ingest)")
    q.add_argument("--query", required=True, type=str)
    q.add_argument("--auto-ingest", action="store_true",
                   help="If --corpus is a source dir, ingest it first")
    q.add_argument("--workers", type=int, default=1)
    q.add_argument("--backend", default="auto",
                   choices=["auto", "pymupdf", "marker", "pypdf"])
    q.add_argument("--json", action="store_true",
                   help="Emit full result as JSON instead of just the answer text")
    q.add_argument("--save-trajectory", type=Path,
                   help="Write the full message trajectory to this path")

    summ = sub.add_parser("summary", help="Print corpus summary without running a query")
    summ.add_argument("--corpus", required=True, type=Path)

    urls = sub.add_parser("ingest-urls", help="Fetch URLs as markdown and add to cache")
    urls.add_argument("--cache", required=True, type=Path)
    urls.add_argument("--url", action="append", default=[])
    urls.add_argument("--urls-file", type=Path)
    urls.add_argument("--fetcher", default=None, choices=["cloudflare", "jina"])
    urls.add_argument("--force", action="store_true")

    crawl = sub.add_parser("ingest-crawl", help="Crawl a site and ingest every discovered page")
    crawl.add_argument("--cache", required=True, type=Path)
    crawl.add_argument("--start-url", required=True, type=str)
    crawl.add_argument("--max-depth", type=int, default=2)
    crawl.add_argument("--limit", type=int, default=50)
    crawl.add_argument("--include", action="append", default=[])
    crawl.add_argument("--exclude", action="append", default=[])
    crawl.add_argument("--fetcher", default=None, choices=["cloudflare", "jina"])
    crawl.add_argument("--force", action="store_true")

    return p


def _print_answer(result: dict[str, object]) -> None:
    print(result.get("formatted_answer") or "[no answer]")
    print()
    print(
        f"[stopped: {result.get('stopped_because')}; "
        f"steps: {len(result.get('trajectory') or [])}]",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "summary":
        corpus = load_corpus(args.corpus.expanduser().resolve())
        print(json.dumps(corpus_summary(corpus), indent=2))
        return 0

    if args.command == "ingest-urls":
        urls = list(args.url)
        if args.urls_file:
            if not args.urls_file.exists():
                print(f"error: --urls-file does not exist: {args.urls_file}", file=sys.stderr)
                return 2
            urls.extend(_read_url_list(args.urls_file))
        if not urls:
            print("error: no URLs provided (use --url and/or --urls-file)", file=sys.stderr)
            return 2
        result = ingest_urls(
            urls=urls,
            cache_dir=args.cache,
            fetcher_name=args.fetcher,
            force=args.force,
        )
        print(json.dumps(result["manifest"], indent=2))
        return 0

    if args.command == "ingest-crawl":
        result = ingest_crawl(
            start_url=args.start_url,
            cache_dir=args.cache,
            max_depth=args.max_depth,
            limit=args.limit,
            include_patterns=args.include or None,
            exclude_patterns=args.exclude or None,
            fetcher_name=args.fetcher,
            force=args.force,
        )
        print(json.dumps(result["manifest"], indent=2))
        return 0

    if args.command == "query":
        try:
            result = run_query(
                corpus_path=args.corpus,
                query=args.query,
                auto_ingest=args.auto_ingest,
                workers=args.workers,
                backend=args.backend,
            )
        except Exception as exc:  # noqa: BLE001 -- surface cleanly to user
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

        if args.save_trajectory:
            args.save_trajectory.write_text(
                json.dumps(result.get("messages") or [], indent=2),
                encoding="utf-8",
            )

        if args.json:
            print(json.dumps({
                "answer": result.get("formatted_answer"),
                "stopped_because": result.get("stopped_because"),
                "corpus_summary": result.get("corpus_summary"),
            }, indent=2))
        else:
            _print_answer(result)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
