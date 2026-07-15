"""Parser for tier-0 validation and the local candidate evaluator."""

from __future__ import annotations

from typing import Callable


def build_providers_parser(subparsers, *, cmd_providers: Callable) -> None:
    """Attach ``providers`` and its verbs to the shared CLI parser."""

    parser = subparsers.add_parser(
        "providers",
        help="Validate providers or run a local candidate evaluation",
        description=(
            "Provider validation is a tier-0 compatibility smoke. Candidate "
            "evaluation is a local, paired, screening-grade CLI lane; it never "
            "changes routing or user configuration."
        ),
    )
    verbs = parser.add_subparsers(dest="providers_command")

    validate = verbs.add_parser(
        "validate",
        help="Run the six-case tier-0 compatibility smoke",
        description=(
            "Run the explicitly labeled tier-0 compatibility smoke through "
            "real `hermes chat -Q` turns and persisted SessionDB receipts. "
            "This is not qualification or replacement evidence."
        ),
    )
    validate.add_argument("--provider", help="Inference provider")
    validate.add_argument("--model", help="Model identifier")
    validate.add_argument(
        "--toolsets",
        default="file",
        choices=["file"],
        help="Tier-0 toolset (only file is supported)",
    )
    validate.add_argument(
        "--suite",
        default="agent-readiness",
        choices=["agent-readiness"],
        help="Tier-0 suite (default: agent-readiness)",
    )
    validate.add_argument("--out", help="Directory for smoke receipts")
    validate.add_argument(
        "--timeout", type=float, default=120.0, help="Per-case timeout in seconds"
    )
    validate.add_argument(
        "--hermes-executable",
        help="Hermes executable override for local/fake integration tests",
    )
    validate.set_defaults(func=cmd_providers)

    evaluate = verbs.add_parser(
        "evaluate",
        help="Run or dry-run paired cli-full-v1 candidate evaluation",
        description=(
            "Evaluate a candidate against an incumbent in the frozen 27-case "
            "cli-full-v1 lane. The default is an offline dry-run; live-shaped "
            "execution requires --execute and an operator-supplied keyless-local "
            "manifest. Network tools are excluded; compression is deferred."
        ),
    )
    evaluate.add_argument(
        "--candidate-manifest", required=True, help="Pinned candidate stack manifest"
    )
    evaluate.add_argument(
        "--incumbent-manifest", required=True, help="Pinned incumbent stack manifest"
    )
    evaluate.add_argument(
        "--evaluation-config", required=True, help="Read-only evaluation specification"
    )
    evaluate.add_argument("--lane", default="cli-full-v1", choices=["cli-full-v1"])
    evaluate.add_argument(
        "--suite", default="full-hermes-cli-v1", choices=["full-hermes-cli-v1"]
    )
    evaluate.add_argument("--out", required=True, help="Self-contained run directory")
    evaluate.add_argument("--repetitions", type=int, default=3)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--timeout", type=float, default=120.0)
    evaluate.add_argument(
        "--archive-index", help="Optional immutable local archive index"
    )
    execute_group = evaluate.add_mutually_exclusive_group()
    execute_group.add_argument(
        "--dry-run",
        dest="execute",
        action="store_false",
        help="Print prerequisites without invoking Hermes (default)",
    )
    execute_group.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        help="Execute the local paired schedule",
    )
    evaluate.set_defaults(func=cmd_providers, execute=False)
    evaluate.add_argument(
        "--hermes-home", help="Read-only Hermes-home snapshot for local tests"
    )
    evaluate.add_argument("--fixture-dir", help="Read-only suite fixture snapshot")
    evaluate.add_argument(
        "--hermes-executable", help="Hermes executable override for fake-provider E2E"
    )

    score = verbs.add_parser(
        "score",
        help="Offline-score saved candidate-evaluation receipts",
        description=(
            "Recalculate evidence-derived checks, including A/A parity, from "
            "receipts without contacting a provider."
        ),
    )
    score.add_argument(
        "--run-dir", required=True, help="Completed evaluation run directory"
    )
    score.add_argument("--archive-index", help="Optional immutable local archive index")
    score.set_defaults(func=cmd_providers)

    suites = verbs.add_parser("suites", help="List frozen evaluator suites")
    suite_commands = suites.add_subparsers(dest="suites_command")
    suite_commands.add_parser("list", help="List available local suites").set_defaults(
        func=cmd_providers
    )
    parser.set_defaults(func=cmd_providers)
