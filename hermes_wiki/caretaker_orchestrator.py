"""Convert LLM Wiki caretaker findings into queued proposal drafts.

The orchestrator is intentionally proposal-mediated: it translates caretaker
actions into review artifacts and only writes under <wiki>/proposals when
explicitly queued with an explicit config.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hermes_wiki.caretaker import CaretakerAction, run_caretaker
from hermes_wiki.config import WikiConfig
from hermes_wiki.eval import _build_searcher
from hermes_wiki.proposal_lifecycle import proposal_record_to_dict, read_proposal
from hermes_wiki.proposals import MemoryProposal, _load_explicit_wiki_config, proposal_to_dict, queue_proposal, render_proposal_markdown


@dataclass(frozen=True)
class CaretakerProposalRun:
    proposals: list[MemoryProposal] = field(default_factory=list)
    queued_paths: list[Path] = field(default_factory=list)


def caretaker_action_to_proposal(action: CaretakerAction | Any) -> MemoryProposal | None:
    """Map a caretaker action to a safe memory proposal draft."""

    kind = str(getattr(action, "kind", "") or "")
    message = str(getattr(action, "message", "") or "")
    file_path = getattr(action, "file_path", None)
    target_pages = [str(file_path)] if file_path and str(file_path).startswith(("concepts/", "entities/", "comparisons/", "queries/")) else []
    source_refs = [str(file_path)] if file_path else ["caretaker report"]

    if kind == "repair_broken_link":
        return MemoryProposal(
            title=f"Repair broken link in {file_path or 'wiki'}",
            rationale="Caretaker found a broken wikilink that weakens Hermes memory navigation.",
            proposed_changes=[message, "Create the missing page or replace/remove the broken wikilink with a source-backed target."],
            source_refs=source_refs,
            target_pages=target_pages,
            tags=["llm-wiki", "caretaker", "broken-link"],
        )
    if kind == "strengthen_wiki_graph":
        return MemoryProposal(
            title=f"Strengthen wiki graph links for {file_path or 'orphan page'}",
            rationale="Caretaker found an orphan page with no inbound links, reducing agent navigability.",
            proposed_changes=[message, "Add source-backed wikilinks from related canonical pages to improve retrieval and graph traversal."],
            source_refs=source_refs,
            target_pages=target_pages,
            tags=["llm-wiki", "caretaker", "wiki-graph"],
        )
    if kind == "find_source_evidence":
        return MemoryProposal(
            title=f"Add source evidence for {file_path or 'wiki page'}",
            rationale="Caretaker found a canonical page without explicit source coverage.",
            proposed_changes=[message, "Find or ingest source-backed evidence before treating this page as durable memory."],
            source_refs=source_refs,
            target_pages=target_pages,
            tags=["llm-wiki", "caretaker", "source-coverage"],
        )
    if kind == "fix_retrieval_regression":
        return MemoryProposal(
            title="Fix LLM Wiki retrieval regression",
            rationale="Caretaker found a retrieval eval failure that blocks reliable Hermes memory recall.",
            proposed_changes=[message, "Inspect page titles, aliases, source coverage, and eval expectations; update source-backed wiki content or evals deliberately."],
            source_refs=["caretaker retrieval eval"],
            target_pages=[],
            tags=["llm-wiki", "caretaker", "retrieval-eval"],
        )
    return None


def run_caretaker_orchestrator(
    config: WikiConfig,
    *,
    queue: bool = False,
    eval_cases_path: str | Path | None = None,
) -> CaretakerProposalRun:
    searcher = None
    if eval_cases_path is not None and Path(eval_cases_path).exists():
        searcher = _build_searcher(None)
    report = run_caretaker(config, eval_cases_path=eval_cases_path, searcher=searcher)
    proposals = [proposal for action in report.actions if (proposal := caretaker_action_to_proposal(action)) is not None]
    queued_paths: list[Path] = []
    if queue:
        for proposal in proposals:
            queued_paths.append(queue_proposal(config, proposal, write=True))
    return CaretakerProposalRun(proposals=proposals, queued_paths=queued_paths)


def _run_to_dict(result: CaretakerProposalRun, config: WikiConfig, *, include_markdown: bool = False) -> dict[str, Any]:
    queued_records = []
    for path in result.queued_paths:
        queued_records.append(proposal_record_to_dict(read_proposal(config, path.stem)))
    proposals = []
    for proposal in result.proposals:
        payload = proposal_to_dict(proposal)
        if include_markdown:
            payload["markdown"] = render_proposal_markdown(proposal)
        proposals.append(payload)
    return {
        "queued": bool(result.queued_paths),
        "queued_paths": [str(path) for path in result.queued_paths],
        "queued_records": queued_records,
        "proposals": proposals,
    }


def _render_run(result: CaretakerProposalRun) -> str:
    if not result.proposals:
        return "No caretaker proposals needed.\n"
    return "\n---\n\n".join(render_proposal_markdown(proposal).strip() for proposal in result.proposals) + "\n"


def _build_config(config_path: str | None) -> WikiConfig:
    return _load_explicit_wiki_config(config_path) if config_path else WikiConfig.from_hermes_config()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draft or queue proposals from LLM Wiki caretaker findings")
    parser.add_argument("--config", help="Hermes config.yaml path to load wiki settings from")
    parser.add_argument("--evals", help="Optional retrieval eval YAML/JSON path")
    parser.add_argument("--queue", action="store_true", help="Queue generated proposals under <wiki>/proposals")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args(argv)

    try:
        if args.queue and not args.config:
            raise ValueError("--queue requires explicit --config to avoid writing to the wrong wiki")
        config = _build_config(args.config)
        result = run_caretaker_orchestrator(config, queue=args.queue, eval_cases_path=args.evals)
        if args.json:
            print(json.dumps(_run_to_dict(result, config, include_markdown=not args.queue), sort_keys=True))
        else:
            print(_render_run(result), end="")
        return 0
    except (FileNotFoundError, ValueError, yaml.YAMLError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
