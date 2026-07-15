"""Parser for the provider validation and candidate-evaluation commands."""

from __future__ import annotations

from typing import Callable


def build_providers_parser(subparsers, *, cmd_providers: Callable) -> None:
    """Attach ``providers`` and its verbs to the shared CLI parser."""

    parser = subparsers.add_parser(
        "providers",
        help="Run the tier-0 provider compatibility smoke",
        description=(
            "Run the explicitly labeled tier-0 provider compatibility smoke. "
            "It is not Hermes qualification, replacement evidence, a benchmark, "
            "or a routing decision."
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
        help="Run or plan a paired candidate-vs-incumbent screen",
        description=(
            "Execute the frozen cli-full-v1 screening catalog with isolated, "
            "interleaved candidate and incumbent arms, or write its schedule "
            "without invoking an executable."
        ),
    )
    evaluate.add_argument("--candidate-manifest", required=True)
    evaluate.add_argument("--incumbent-manifest", required=True)
    evaluate.add_argument("--evaluation-config", required=True)
    evaluate.add_argument("--lane", default="cli-full-v1", choices=["cli-full-v1"])
    evaluate.add_argument("--suite", default="full-hermes-cli-v1", choices=["full-hermes-cli-v1"])
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--repetitions", type=int, default=3)
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--timeout", type=float, default=120.0)
    evaluate.add_argument("--fixture-dir", help="Read-only suite fixture snapshot")
    evaluate.add_argument("--hermes-home", help="Base temp HERMES_HOME snapshot")
    evaluate.add_argument("--hermes-executable", help="Local Hermes/fake executable")
    evaluate.add_argument("--archive-index", help="Immutable local archive index")
    evaluate.add_argument("--execute", action="store_true", help="Run local arms")
    evaluate.add_argument(
        "--dry-run", action="store_true", help="Write and validate the schedule only"
    )
    evaluate.set_defaults(func=cmd_providers)

    score = verbs.add_parser(
        "score",
        help="Offline-rescore a completed evaluation run",
        description="Verify receipt hashes and deterministically rescore a local run.",
    )
    score.add_argument("--run-dir", required=True)
    score.add_argument("--archive-index")
    score.set_defaults(func=cmd_providers)

    suites = verbs.add_parser("suites", help="List frozen evaluation suites")
    suite_verbs = suites.add_subparsers(dest="suites_command")
    suites_list = suite_verbs.add_parser("list", help="List suite ids and case counts")
    suites_list.set_defaults(func=cmd_providers)
    parser.set_defaults(func=cmd_providers)
