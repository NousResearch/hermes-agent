"""Agent-native caretaker loop for LLM Wiki.

The caretaker coordinates read-only memory health signals for Hermes: wiki
maintenance issues, retrieval regressions, and pending memory proposals. It is
not a human UI and does not mutate canonical memory by default.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hermes_wiki.config import WikiConfig
from hermes_wiki.eval import Searcher, _build_searcher, evaluate_retrieval, load_retrieval_cases, result_to_dict
from hermes_wiki.frontmatter import write_page
from hermes_wiki.maintenance import (
    MaintenanceReport,
    _load_explicit_wiki_config,
    _validate_report_path,
    generate_maintenance_report,
    maintenance_report_to_dict,
)


@dataclass(frozen=True)
class CaretakerAction:
    """A next memory operation Hermes can reason about or perform later."""

    kind: str
    severity: str
    message: str
    autonomous: bool
    file_path: str | None = None


@dataclass(frozen=True)
class CaretakerReport:
    """Aggregated agent-native wiki health report."""

    maintenance: MaintenanceReport
    retrieval_eval: Any | None = None
    actions: list[CaretakerAction] = field(default_factory=list)

    @property
    def has_blockers(self) -> bool:
        if self.maintenance.error_count > 0:
            return True
        if self.retrieval_eval is not None and not self.retrieval_eval.passed:
            return True
        return any(action.severity == "error" for action in self.actions)


def _classify_maintenance_actions(report: MaintenanceReport) -> list[CaretakerAction]:
    actions: list[CaretakerAction] = []
    for issue in report.issues:
        severity = issue.severity.value
        if issue.category == "pending_proposal":
            actions.append(
                CaretakerAction(
                    kind="review_pending_proposal",
                    severity="info",
                    message=issue.message,
                    autonomous=True,
                    file_path=issue.file_path,
                )
            )
        elif issue.category == "broken_link":
            actions.append(
                CaretakerAction(
                    kind="repair_broken_link",
                    severity=severity,
                    message=issue.message,
                    autonomous=True,
                    file_path=issue.file_path,
                )
            )
        elif issue.category == "orphan_page":
            actions.append(
                CaretakerAction(
                    kind="strengthen_wiki_graph",
                    severity=severity,
                    message=issue.message,
                    autonomous=True,
                    file_path=issue.file_path,
                )
            )
        elif issue.category == "missing_source_coverage":
            actions.append(
                CaretakerAction(
                    kind="find_source_evidence",
                    severity=severity,
                    message=issue.message,
                    autonomous=False,
                    file_path=issue.file_path,
                )
            )
        else:
            actions.append(
                CaretakerAction(
                    kind=f"inspect_{issue.category}",
                    severity=severity,
                    message=issue.message,
                    autonomous=False,
                    file_path=issue.file_path,
                )
            )
    return actions


def _classify_retrieval_actions(retrieval_eval: Any | None) -> list[CaretakerAction]:
    if retrieval_eval is None or retrieval_eval.passed:
        return []
    actions: list[CaretakerAction] = []
    for failure in retrieval_eval.failures:
        missing = ", ".join(sorted(failure.missing_pages))
        actions.append(
            CaretakerAction(
                kind="fix_retrieval_regression",
                severity="error",
                message=f"Retrieval eval failed for query {failure.query!r}; missing pages: {missing}",
                autonomous=True,
            )
        )
    return actions


def _default_eval_cases_path(config: WikiConfig) -> Path:
    return config.wiki_path / "evals" / "retrieval.yaml"


def run_caretaker(
    config: WikiConfig,
    *,
    eval_cases_path: str | Path | None = None,
    searcher: Searcher | None = None,
) -> CaretakerReport:
    """Run a read-first caretaker pass for Hermes-owned wiki memory.

    This scans local wiki markdown and, when retrieval cases are present,
    runs deterministic retrieval evals. It does not ingest, reindex, call chat
    models, mutate canonical pages, append logs, or create directories.
    """

    maintenance = generate_maintenance_report(config)
    retrieval_eval = None

    cases_path = Path(eval_cases_path) if eval_cases_path is not None else _default_eval_cases_path(config)
    if cases_path.exists():
        active_searcher = searcher if searcher is not None else _build_searcher(None)
        retrieval_eval = evaluate_retrieval(active_searcher, load_retrieval_cases(cases_path))

    actions = _classify_maintenance_actions(maintenance)
    actions.extend(_classify_retrieval_actions(retrieval_eval))
    return CaretakerReport(maintenance=maintenance, retrieval_eval=retrieval_eval, actions=actions)


def caretaker_report_to_dict(report: CaretakerReport) -> dict[str, Any]:
    return {
        "has_blockers": report.has_blockers,
        "maintenance": maintenance_report_to_dict(report.maintenance),
        "retrieval_eval": result_to_dict(report.retrieval_eval) if report.retrieval_eval is not None else None,
        "actions": [
            {
                "kind": action.kind,
                "severity": action.severity,
                "message": action.message,
                "autonomous": action.autonomous,
                "file_path": action.file_path,
            }
            for action in report.actions
        ],
    }


def render_caretaker_report(report: CaretakerReport) -> str:
    payload = caretaker_report_to_dict(report)
    maintenance = payload["maintenance"]
    retrieval = payload["retrieval_eval"]
    lines = [
        "# LLM Wiki Caretaker Report",
        "",
        "This is an agent-native memory health report for Hermes-owned durable memory.",
        "",
        "## Summary",
        f"- Blockers: {'yes' if report.has_blockers else 'no'}",
        f"- Pages: {maintenance['total_pages']}",
        f"- Sources: {maintenance['total_sources']}",
        f"- Maintenance issues: {len(maintenance['issues'])} ({maintenance['errors']} errors, {maintenance['warnings']} warnings, {maintenance['infos']} info)",
        f"- Retrieval eval: {('not run' if retrieval is None else ('passed' if retrieval['passed'] else 'failed'))}",
        "",
        "## Hermes actions",
    ]
    if not report.actions:
        lines.append("- No memory actions needed.")
    else:
        for action in report.actions:
            agent_mode = "autonomous" if action.autonomous else "needs evidence"
            file_path = f" `{action.file_path}`" if action.file_path else ""
            lines.append(f"- **{action.severity} / {action.kind} / {agent_mode}**{file_path}: {action.message}")
    return "\n".join(lines) + "\n"


def _build_config(config_path: str | None) -> WikiConfig:
    return _load_explicit_wiki_config(config_path) if config_path else WikiConfig.from_hermes_config()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an agent-native LLM Wiki caretaker pass")
    parser.add_argument("--config", help="Hermes config.yaml path to load wiki settings from")
    parser.add_argument("--evals", help="Optional retrieval eval YAML/JSON path; defaults to <wiki>/evals/retrieval.yaml when present")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")
    parser.add_argument("--quiet", action="store_true", help="Print only when blockers exist")
    parser.add_argument("--write-report", help="Explicit relative markdown path under reports/ for a caretaker report")
    args = parser.parse_args(argv)

    try:
        config = _build_config(args.config)
        searcher = None
        eval_path = Path(args.evals) if args.evals else _default_eval_cases_path(config)
        if eval_path.exists():
            searcher = _build_searcher(args.config)
        report = run_caretaker(config, eval_cases_path=eval_path, searcher=searcher)
        if args.write_report:
            if not args.config:
                raise ValueError("--write-report requires explicit --config to avoid writing to the wrong wiki")
            path = _validate_report_path(config, args.write_report)
            write_page(path, {"title": "LLM Wiki Caretaker Report", "type": "caretaker_report", "status": "generated"}, render_caretaker_report(report))
            print(json.dumps({"written": True, "path": str(path)}, sort_keys=True))
            return 1 if report.has_blockers else 0
        if args.quiet and not report.has_blockers:
            return 0
        output = caretaker_report_to_dict(report) if args.json else render_caretaker_report(report)
        if args.json:
            print(json.dumps(output, sort_keys=True))
        else:
            print(output, end="")
        return 1 if report.has_blockers else 0
    except (FileNotFoundError, ValueError, json.JSONDecodeError, yaml.YAMLError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
