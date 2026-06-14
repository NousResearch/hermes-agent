"""CLI entrypoint for architecture-first project reviews."""

import argparse
import json
import sys
from dataclasses import asdict

from .architecture_first import ArchitectureReviewRequest, render_review_report, review_architecture
from .doc_generation import generate_missing_docs, write_review_artifact
from .scanners import scan_project


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
    review.add_argument("--block-on-critical", action="store_true")
    review.add_argument("--write-report", action="store_true")
    review.add_argument("--generate-docs", action="store_true")
    review.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "review":
        parser.print_help()
        return INVALID_REQUEST

    scan = scan_project(args.project, args.projects_root)
    request = ArchitectureReviewRequest(
        project_id=scan.project_id,
        project_path=scan.project_path,
        present_documents=scan.present_documents,
        completed_stages=scan.completed_stages,
    )
    report = review_architecture(request)
    artifacts = []
    if args.generate_docs:
        generation = generate_missing_docs(scan.project_path, report, overwrite=args.overwrite)
        artifacts.extend(write.path for write in generation.writes if write.status == "written")
    if args.write_report:
        artifact = write_review_artifact(scan.project_path, report, overwrite=args.overwrite)
        if artifact.status == "written":
            artifacts.append(artifact.path)

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


if __name__ == "__main__":
    raise SystemExit(main())
