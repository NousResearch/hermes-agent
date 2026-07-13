"""CLI entrypoint for architecture-first project reviews."""

import argparse
import json
import sys
from dataclasses import asdict

from .architecture_first import ArchitectureReviewRequest, render_review_report, review_architecture
from .doc_generation import generate_missing_docs, write_review_artifact
from .persistence import SQLiteRepository, persist_review_report
from .scanners import scan_project
from .tasks import generate_tasks_from_review, next_task_number_from_files, write_task_artifacts


PASS = 0
WARNING = 2
BLOCKED = 3
INVALID_REQUEST = 64


def build_parser():
    parser = argparse.ArgumentParser(prog="hermes architect")
    subparsers = parser.add_subparsers(dest="command")
    review = subparsers.add_parser("review")
    review.add_argument("project")
    review.add_argument("--projects-root")
    review.add_argument("--json", action="store_true")
    review.add_argument("--scope", default="")
    review.add_argument("--block-on-critical", action="store_true")
    review.add_argument("--write-report", action="store_true")
    review.add_argument("--generate-docs", action="store_true")
    review.add_argument("--generate-tasks", action="store_true")
    review.add_argument("--overwrite", action="store_true")
    review.add_argument("--persist", action="store_true")
    review.add_argument("--db", default="")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "review":
        parser.print_help()
        return INVALID_REQUEST

    try:
        scan = scan_project(args.project, args.projects_root)
    except Exception as exc:
        return _emit_error(args, "invalid_project", str(exc))
    request = ArchitectureReviewRequest(
        project_id=scan.project_id,
        project_path=scan.project_path,
        present_documents=scan.present_documents,
        completed_stages=scan.completed_stages,
    )
    report = review_architecture(request)
    if args.scope:
        report = _apply_scope(report, args.scope)
    artifacts = []
    if args.generate_docs:
        generation = generate_missing_docs(scan.project_path, report, overwrite=args.overwrite)
        artifacts.extend(write.path for write in generation.writes if write.status == "written")
    if args.write_report:
        artifact = write_review_artifact(scan.project_path, report, overwrite=args.overwrite)
        if artifact.status == "written":
            artifacts.append(artifact.path)
    if args.persist:
        try:
            repository = SQLiteRepository(args.db or scan.project_path)
            artifacts.append(persist_review_report(repository, report))
        except Exception as exc:
            return _emit_error(args, "persistence_failed", str(exc))
    if args.generate_tasks:
        start = next_task_number_from_files([
            __import__("os").path.join(scan.project_path, "TASKS.md"),
            __import__("os").path.join(scan.project_path, ".hermes", "tasks.json"),
        ])
        tasks = generate_tasks_from_review(report, start_at=start)
        result = write_task_artifacts(scan.project_path, tasks, overwrite=args.overwrite)
        artifacts.extend(result["paths"])

    if args.json:
        payload = asdict(report)
        payload["project_path"] = scan.project_path
        payload["artifacts"] = artifacts
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(render_review_report(report))
        if artifacts:
            sys.stdout.write("\nArtifacts:\n")
            for artifact in artifacts:
                sys.stdout.write("- " + artifact + "\n")

    if report.blocked and args.block_on_critical:
        return BLOCKED
    if report.critical_gaps:
        return WARNING
    return PASS


def _emit_error(args, code: str, message: str):
    payload = {"error": {"code": code, "message": message}}
    if getattr(args, "json", False):
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        sys.stderr.write("%s: %s\n" % (code, message))
    return INVALID_REQUEST


def _apply_scope(report, scope: str):
    """Filter display-heavy lists for focused command output.

    Scoping is intentionally report-only: the underlying readiness score and
    blocked decision still reflect the whole project, so a narrow view cannot
    accidentally bless implementation work.
    """
    from dataclasses import replace

    requested = {item.strip().lower() for item in scope.split(",") if item.strip()}
    if not requested:
        return report
    keep_docs = not requested.isdisjoint({"documents", "docs"})
    keep_schemas = "schemas" in requested
    keep_dashboards = "dashboards" in requested
    keep_approvals = "approvals" in requested
    return replace(
        report,
        missing_documents=report.missing_documents if keep_docs else [],
        missing_schemas=report.missing_schemas if keep_schemas else [],
        missing_dashboards=report.missing_dashboards if keep_dashboards else [],
        missing_approvals=report.missing_approvals if keep_approvals else [],
    )


if __name__ == "__main__":
    raise SystemExit(main())
