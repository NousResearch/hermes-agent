"""LLM filing-section summarization for mined investment-assistant data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from .pydantic_runtime import create_pydantic_agent, pydantic_event_stream_handler, usage_metadata
from .storage import new_id, utc_now

DEFAULT_DATA_ROOT = Path("data/investment_assistant")
SUMMARY_LAYER = "filing_summary"
SUMMARY_FILENAME = "filing_summary.md"
SUMMARY_META_FILENAME = "filing_summary.meta.json"
DEFAULT_MAX_INPUT_CHARS = 120_000

SECTION_LABELS = {
    "part_i_item_1": "Item 1",
    "part_i_item_1a": "Item 1A",
    "part_i_item_1c": "Item 1C",
    "part_ii_item_7": "Item 7",
    "part_ii_item_7a": "Item 7A",
    "part_i_item_2": "Item 2",
    "part_ii_item_1a": "Item 1A",
    "item_202": "Item 2.02",
    "item_901": "Item 9.01",
    "item_3d": "Item 3.D",
    "item_4": "Item 4",
    "item_5": "Item 5",
    "item_11": "Item 11",
}

SECTION_PRIORITY = {
    ("latest_10q", "part_i_item_2"): 10,
    ("latest_8k", "item_202"): 20,
    ("latest_10k", "part_ii_item_7"): 30,
    ("latest_10k", "part_i_item_1"): 40,
    ("latest_10q", "part_ii_item_1a"): 50,
    ("latest_10k", "part_i_item_1a"): 60,
    ("latest_10k", "part_i_item_1c"): 70,
    ("latest_10k", "part_ii_item_7a"): 80,
    ("latest_8k", "item_901"): 90,
    ("latest_20f", "item_5"): 35,
    ("latest_20f", "item_4"): 45,
    ("latest_20f", "item_3d"): 65,
    ("latest_20f", "item_11"): 85,
}


class FilingSummaryOutput(BaseModel):
    """Minimal typed wrapper around the markdown summary."""

    markdown: str = Field(min_length=100)
    warnings: list[str] = Field(default_factory=list)


class FilingSummaryRunArtifact(BaseModel):
    artifact_type: str = "filing_summary_run"
    run_id: str = Field(default_factory=lambda: new_id("fsr"))
    generated_at: str = Field(default_factory=utc_now)
    root: str
    symbols: list[str]
    status_counts: dict[str, int] = Field(default_factory=dict)
    summary_paths: dict[str, str] = Field(default_factory=dict)
    meta_paths: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def summarize_filings_for_symbols(
    *,
    root: str | Path = DEFAULT_DATA_ROOT,
    symbols: Sequence[str] | None = None,
    all_symbols: bool = False,
    max_symbols: int | None = None,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    force: bool = False,
    skip_existing: bool = False,
) -> FilingSummaryRunArtifact:
    """Generate filing summaries for selected symbols under a file-backed data root."""

    data_root = Path(root)
    selected = _select_symbols(data_root, symbols=symbols, all_symbols=all_symbols, max_symbols=max_symbols)
    run = FilingSummaryRunArtifact(root=str(data_root), symbols=selected)
    for symbol in selected:
        symbol_dir = data_root / "symbols" / symbol
        try:
            result = summarize_symbol_filings(
                symbol_dir,
                root=data_root,
                max_input_chars=max_input_chars,
                force=force,
                skip_existing=skip_existing,
                run_id=run.run_id,
            )
        except Exception as exc:
            status = "error"
            run.warnings.append(f"{symbol}: {type(exc).__name__}: {exc}")
            _write_error_meta(symbol_dir, symbol, exc, run_id=run.run_id)
        else:
            status = str(result.get("status") or "unknown")
            if result.get("summary_path"):
                run.summary_paths[symbol] = str(result["summary_path"])
            if result.get("meta_path"):
                run.meta_paths[symbol] = str(result["meta_path"])
            run.warnings.extend(f"{symbol}: {warning}" for warning in result.get("warnings", []))
        run.status_counts[status] = run.status_counts.get(status, 0) + 1
    _write_run_artifact(data_root, run)
    return run


def summarize_symbol_filings(
    symbol_dir: str | Path,
    *,
    root: str | Path = DEFAULT_DATA_ROOT,
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS,
    force: bool = False,
    skip_existing: bool = False,
    run_id: str = "",
) -> dict[str, Any]:
    """Summarize filing section markdown files for one symbol directory."""

    symbol_path = Path(symbol_dir)
    data_root = Path(root)
    symbol = symbol_path.name
    summary_path = symbol_path / SUMMARY_FILENAME
    meta_path = symbol_path / SUMMARY_META_FILENAME
    if summary_path.exists() and meta_path.exists() and skip_existing and not force:
        meta = _read_json(meta_path)
        return {
            "status": meta.get("status", "fresh"),
            "summary_path": summary_path,
            "meta_path": meta_path,
            "warnings": ["existing summary skipped"],
        }

    manifest = _read_json(symbol_path / "manifest.json") if (symbol_path / "manifest.json").exists() else {}
    filing_metadata = (
        _read_json(symbol_path / "filing_metadata.json") if (symbol_path / "filing_metadata.json").exists() else {}
    )
    filing_sections = (
        _read_json(symbol_path / "filing_sections.json") if (symbol_path / "filing_sections.json").exists() else {}
    )
    source_files = _collect_source_files(symbol_path, filing_metadata, filing_sections)
    if not source_files:
        warnings = ["No filing section markdown files found for summarization."]
        meta = _summary_meta(
            symbol=symbol,
            status="skipped",
            summary_path=summary_path,
            source_files=[],
            manifest=manifest,
            runtime={},
            usage={},
            warnings=warnings,
            run_id=run_id,
        )
        _write_text(summary_path, _skipped_summary(symbol, warnings))
        _write_json(meta_path, meta)
        _update_manifest(symbol_path, data_root, meta, status="skipped", warnings=warnings, run_id=run_id)
        return {"status": "skipped", "summary_path": summary_path, "meta_path": meta_path, "warnings": warnings}

    prompt_payload, source_files = _summary_prompt_payload(
        symbol=symbol,
        manifest=manifest,
        filing_metadata=filing_metadata,
        source_files=source_files,
        max_input_chars=max_input_chars,
    )
    output, runtime, usage = _run_summary_agent(prompt_payload)
    markdown = _normalize_markdown(symbol, output.markdown)
    warnings = _dedupe([*prompt_payload.get("input_warnings", []), *output.warnings])
    status = "partial" if warnings else "fresh"
    _write_text(summary_path, markdown)
    meta = _summary_meta(
        symbol=symbol,
        status=status,
        summary_path=summary_path,
        source_files=source_files,
        manifest=manifest,
        runtime=runtime,
        usage=usage,
        warnings=warnings,
        run_id=run_id,
    )
    _write_json(meta_path, meta)
    _update_manifest(symbol_path, data_root, meta, status=status, warnings=warnings, run_id=run_id)
    return {"status": status, "summary_path": summary_path, "meta_path": meta_path, "warnings": warnings}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "summarize":
        run = summarize_filings_for_symbols(
            root=args.root,
            symbols=args.symbols or None,
            all_symbols=args.all or not args.symbols,
            max_symbols=args.max_symbols,
            max_input_chars=args.max_input_chars,
            force=args.force,
            skip_existing=args.skip_existing,
        )
        if args.json:
            print(json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"run_id: {run.run_id}")
            print(f"root: {run.root}")
            print(f"symbols: {len(run.symbols)}")
            print(f"status_counts: {run.status_counts}")
            if run.warnings:
                print("warnings:")
                for warning in run.warnings:
                    print(f"  - {warning}")
        return 0
    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia-filing-summary",
        description="Summarize mined SEC filing sections with an LLM.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    summarize = subparsers.add_parser("summarize", help="Generate filing summaries for symbols.")
    summarize.add_argument("--root", default=str(DEFAULT_DATA_ROOT), help="Data root containing symbols/.")
    summarize.add_argument("--symbols", nargs="*", default=[], help="Symbols to summarize, e.g. US.MRVL.")
    summarize.add_argument("--all", action="store_true", help="Summarize all symbols under root/symbols.")
    summarize.add_argument("--max-symbols", type=int, help="Maximum symbols to process.")
    summarize.add_argument(
        "--max-input-chars",
        type=int,
        default=int(os.getenv("IA_FILING_SUMMARY_MAX_INPUT_CHARS", DEFAULT_MAX_INPUT_CHARS)),
        help="Approximate max source text characters sent per symbol.",
    )
    summarize.add_argument("--skip-existing", action="store_true", help="Skip symbols with existing summaries.")
    summarize.add_argument("--force", action="store_true", help="Overwrite existing summaries.")
    summarize.add_argument("--json", action="store_true", help="Print run artifact as JSON.")
    return parser


def _run_summary_agent(payload: dict[str, Any]) -> tuple[FilingSummaryOutput, dict[str, Any], dict[str, Any]]:
    agent, _model_config, runtime = create_pydantic_agent(
        output_type=FilingSummaryOutput,
        instructions=_SUMMARY_INSTRUCTIONS,
        agent_kind="filing_summary_agent",
        output_retries=1,
        agent_skill_names=["filing-narrative-summary"],
    )
    result = agent.run_sync(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        event_stream_handler=pydantic_event_stream_handler("filing_summary_agent"),
    )
    return result.output, runtime, usage_metadata(result)


def _summary_prompt_payload(
    *,
    symbol: str,
    manifest: dict[str, Any],
    filing_metadata: dict[str, Any],
    source_files: list[dict[str, Any]],
    max_input_chars: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    warnings: list[str] = []
    budget = max(2_000, max_input_chars)
    selected: list[dict[str, Any]] = []
    remaining = budget
    for item in sorted(source_files, key=_source_priority):
        text = item["text"]
        include_chars = min(len(text), _section_char_budget(item), remaining)
        if include_chars <= 0:
            item = {**item, "included_char_count": 0, "truncated": True}
            warnings.append(f"Skipped {item['source_label']} due to input budget.")
        else:
            truncated = include_chars < len(text)
            item = {
                **item,
                "text": text[:include_chars],
                "included_char_count": include_chars,
                "truncated": truncated,
            }
            if truncated:
                warnings.append(
                    f"Truncated {item['source_label']} from {len(text)} to {include_chars} characters."
                )
            remaining -= include_chars
        selected.append(item)

    payload = {
        "task": "Produce a source-grounded filing summary markdown for one company.",
        "symbol": symbol,
        "manifest_status": manifest.get("source_status", ""),
        "manifest_warnings": manifest.get("warnings", []),
        "filing_metadata": _compact_filing_metadata(filing_metadata),
        "source_sections": [
            {
                "filing_key": item["filing_key"],
                "form": item.get("form", ""),
                "filing_date": item.get("filing_date", ""),
                "section_key": item["section_key"],
                "source_label": item["source_label"],
                "title": item.get("title", ""),
                "path": item["path"],
                "char_count": item["char_count"],
                "included_char_count": item.get("included_char_count", 0),
                "truncated": item.get("truncated", False),
                "text": item.get("text", ""),
            }
            for item in selected
            if item.get("included_char_count", 0) > 0
        ],
        "input_warnings": warnings,
        "output_contract": (
            "Return FilingSummaryOutput. markdown must use the requested headings and source labels. "
            "Do not include investment recommendations, target prices, portfolio weights, or trade plans."
        ),
    }
    return payload, [_source_file_meta(item) for item in selected]


def _collect_source_files(
    symbol_path: Path,
    filing_metadata: dict[str, Any],
    filing_sections: dict[str, Any],
) -> list[dict[str, Any]]:
    filings_meta = filing_metadata.get("filings") if isinstance(filing_metadata.get("filings"), dict) else {}
    filings_sections = filing_sections.get("filings") if isinstance(filing_sections.get("filings"), dict) else {}
    source_files: list[dict[str, Any]] = []
    sections_root = symbol_path / "filing_sections"
    for path in sorted(sections_root.glob("*/*.md")):
        filing_key = path.parent.name
        section_key = path.stem
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        filing_info = filings_meta.get(filing_key, {}) if isinstance(filings_meta.get(filing_key), dict) else {}
        filing_section = filings_sections.get(filing_key, {}) if isinstance(filings_sections.get(filing_key), dict) else {}
        section_info = {}
        if isinstance(filing_section.get("sections"), dict):
            section_info = filing_section["sections"].get(section_key, {}) or {}
        item_label = section_info.get("item") or SECTION_LABELS.get(section_key) or section_key
        source_files.append(
            {
                "filing_key": filing_key,
                "form": filing_info.get("form") or filing_section.get("form") or "",
                "filing_date": filing_info.get("filing_date") or filing_section.get("filing_date") or "",
                "period_of_report": filing_info.get("period_of_report") or "",
                "accession_number": filing_info.get("accession_number") or "",
                "section_key": section_key,
                "source_label": f"{filing_key} / {item_label}",
                "title": section_info.get("title") or "",
                "path": path.relative_to(symbol_path).as_posix(),
                "char_count": len(text),
                "checksum": _sha256_text(text),
                "text": text,
            }
        )
    return source_files


def _source_file_meta(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "filing_key",
            "form",
            "filing_date",
            "period_of_report",
            "accession_number",
            "section_key",
            "source_label",
            "title",
            "path",
            "char_count",
            "included_char_count",
            "truncated",
            "checksum",
        )
    }


def _summary_meta(
    *,
    symbol: str,
    status: Literal["fresh", "partial", "skipped", "error"],
    summary_path: Path,
    source_files: list[dict[str, Any]],
    manifest: dict[str, Any],
    runtime: dict[str, Any],
    usage: dict[str, Any],
    warnings: list[str],
    run_id: str,
) -> dict[str, Any]:
    generated_at = utc_now()
    return {
        "artifact_type": "filing_summary",
        "symbol": symbol,
        "status": status,
        "summary_path": summary_path.name,
        "generated_at": generated_at,
        "run_id": run_id,
        "source_manifest_status": manifest.get("source_status", ""),
        "source_files": source_files,
        "model": runtime.get("model", ""),
        "api_mode": runtime.get("api_mode", ""),
        "pydantic_ai": {
            key: runtime.get(key)
            for key in ("available", "mode", "package_version", "model", "configured_model", "api_mode")
            if key in runtime
        },
        "usage": usage,
        "warnings": _dedupe(warnings),
    }


def _update_manifest(
    symbol_path: Path,
    root: Path,
    meta: dict[str, Any],
    *,
    status: str,
    warnings: list[str],
    run_id: str,
) -> None:
    manifest_path = symbol_path / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {
        "artifact_type": "symbol_data_manifest",
        "symbol": symbol_path.name,
        "market": symbol_path.name.split(".", 1)[0],
        "layers": {},
        "warnings": [],
    }
    summary_path = symbol_path / SUMMARY_FILENAME
    rel_summary = summary_path.relative_to(root).as_posix()
    rel_meta = (symbol_path / SUMMARY_META_FILENAME).relative_to(root).as_posix()
    now = utc_now()
    manifest.setdefault("layers", {})[SUMMARY_LAYER] = {
        "layer": SUMMARY_LAYER,
        "status": status,
        "source": "pydantic_ai",
        "provider": "pydantic_ai",
        "path": rel_summary,
        "meta_path": rel_meta,
        "asof": meta.get("generated_at", now),
        "updated_at": now,
        "run_id": run_id,
        "checksum": _sha256_file(summary_path) if summary_path.exists() else "",
        "warnings": _dedupe(warnings),
        "error": "",
    }
    manifest["updated_at"] = now
    manifest["source_status"] = _combined_manifest_status(manifest.get("layers", {}))
    manifest["warnings"] = _dedupe(
        str(item)
        for entry in manifest.get("layers", {}).values()
        for item in entry.get("warnings", [])
        if item
    )
    _write_json(manifest_path, manifest)


def _write_error_meta(symbol_path: Path, symbol: str, exc: Exception, *, run_id: str = "") -> None:
    symbol_path.mkdir(parents=True, exist_ok=True)
    meta = {
        "artifact_type": "filing_summary",
        "symbol": symbol,
        "status": "error",
        "summary_path": SUMMARY_FILENAME,
        "generated_at": utc_now(),
        "run_id": run_id,
        "source_files": [],
        "warnings": [f"{type(exc).__name__}: {exc}"],
        "error": str(exc),
    }
    _write_json(symbol_path / SUMMARY_META_FILENAME, meta)


def _write_run_artifact(root: Path, run: FilingSummaryRunArtifact) -> None:
    run_dir = root / "runs" / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "filing_summary_run.json", run.model_dump(mode="json"))
    _write_text(run_dir / "symbols.txt", "\n".join(run.symbols) + ("\n" if run.symbols else ""))


def _select_symbols(
    root: Path,
    *,
    symbols: Sequence[str] | None,
    all_symbols: bool,
    max_symbols: int | None,
) -> list[str]:
    if symbols:
        selected = [_normalize_symbol(symbol) for symbol in symbols]
    elif all_symbols:
        selected = sorted(path.name for path in (root / "symbols").iterdir() if path.is_dir())
    else:
        selected = []
    if max_symbols is not None:
        selected = selected[: max(0, max_symbols)]
    return selected


def _compact_filing_metadata(filing_metadata: dict[str, Any]) -> dict[str, Any]:
    filings = filing_metadata.get("filings") if isinstance(filing_metadata.get("filings"), dict) else {}
    return {
        "source_status": filing_metadata.get("source_status", ""),
        "risk_flags": filing_metadata.get("risk_flags", []),
        "event_context": filing_metadata.get("event_context", {}),
        "filings": {
            key: {
                field: item.get(field)
                for field in ("form", "filing_date", "period_of_report", "accession_number", "url", "filing_url")
            }
            for key, item in filings.items()
            if isinstance(item, dict)
        },
    }


def _source_priority(item: dict[str, Any]) -> tuple[int, str, str]:
    return (
        SECTION_PRIORITY.get((item.get("filing_key"), item.get("section_key")), 500),
        str(item.get("filing_key", "")),
        str(item.get("section_key", "")),
    )


def _section_char_budget(item: dict[str, Any]) -> int:
    key = (item.get("filing_key"), item.get("section_key"))
    if key == ("latest_10q", "part_i_item_2"):
        return 40_000
    if key == ("latest_10k", "part_ii_item_7"):
        return 35_000
    if key == ("latest_10k", "part_i_item_1"):
        return 30_000
    if item.get("section_key") in {"part_i_item_1a", "part_ii_item_1a"}:
        return 35_000
    if item.get("filing_key") == "latest_8k":
        return 12_000
    if key == ("latest_20f", "item_5"):
        return 35_000
    if key == ("latest_20f", "item_4"):
        return 30_000
    if item.get("filing_key") == "latest_20f":
        return 25_000
    return 12_000


def _normalize_markdown(symbol: str, markdown: str) -> str:
    text = str(markdown or "").strip()
    if not text.startswith("# "):
        text = f"# {symbol} Filing Summary\n\n{text}"
    return text + "\n"


def _skipped_summary(symbol: str, warnings: list[str]) -> str:
    lines = [
        f"# {symbol} Filing Summary",
        "",
        "## Source Files",
        "- No filing section markdown files were available.",
        "",
        "## Data Quality Notes",
        *[f"- {warning}" for warning in warnings],
        "",
    ]
    return "\n".join(lines)


def _combined_manifest_status(layers: dict[str, Any]) -> str:
    statuses = [str(entry.get("status") or "") for entry in layers.values() if isinstance(entry, dict)]
    if not statuses:
        return "missing"
    if all(status == "fresh" for status in statuses):
        return "fresh"
    if any(status in {"fresh", "partial"} for status in statuses):
        return "partial"
    if any(status == "not_implemented" for status in statuses):
        return "partial"
    return "unavailable"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_symbol(symbol: str, default_market: str = "US") -> str:
    value = str(symbol or "").strip().upper()
    if "." in value:
        return value
    return f"{default_market}.{value}"


def _dedupe(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in values or []:
        text = str(item)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


_SUMMARY_INSTRUCTIONS = """
You are a filing narrative summarization agent for an investment research
assistant. Read only the supplied JSON payload and filing section text. Produce
a concise but substantive markdown summary for one company.

Use the required markdown headings from the filing-narrative-summary skill.
Every important factual claim must include a source label that appears in the
payload, such as [latest_10q / Item 2]. Do not cite source labels that were not
provided.

Do not provide buy/sell/hold recommendations, target prices, portfolio weights,
trade plans, option strategies, or unsupported facts from memory. Do not treat
risk-factor boilerplate as an event that already happened unless MD&A or 8-K
text supports it.

Exact financial numbers should come from structured numeric artifacts, not from
this prose summary. If the filing text mentions demand, margin, inventory,
capex, customer, backlog, or AI/data-center exposure, summarize it
qualitatively with source labels.
"""


if __name__ == "__main__":
    raise SystemExit(main())
