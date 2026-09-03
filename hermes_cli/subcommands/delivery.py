"""Parser and handler for the fixed Feature Delivery V1 runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.feature_delivery_runner import FeatureDeliveryRunner


def build_delivery_parser(subparsers, *, cmd_delivery) -> None:
    parser = subparsers.add_parser(
        "delivery",
        help="Run a durable Feature Delivery V1 workflow",
    )
    commands = parser.add_subparsers(dest="delivery_command", required=True)

    create = commands.add_parser("create", help="Create a delivery root from a JSON contract")
    create.add_argument("--contract", required=True, help="Path to a TaskContract JSON file")

    unblock = commands.add_parser(
        "unblock",
        help="Human-resume a recoverable delivery block",
        description=(
            "Only recoverable feature-delivery blocks can be resumed. "
            "Terminal integrity blocks remain blocked."
        ),
    )
    unblock.add_argument("task_id", help="Feature Delivery root task id")
    unblock.add_argument(
        "--resume-stage",
        choices=("previous", "developer"),
        default="previous",
        help="Resume the blocked stage, or safely return to development",
    )
    unblock.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm this human-approved state recovery",
    )

    for name, help_text in (
        ("run", "Run until delivered, blocked, or waiting for an executor"),
        ("resume", "Resume a durable delivery root"),
        ("status", "Show durable delivery state"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("task_id", help="Feature Delivery root task id")
        if name in {"run", "resume"}:
            command.add_argument(
                "--executor",
                choices=("profiles",),
                help="Use the fixed developer/tester/acceptance Hermes profiles",
            )
    parser.set_defaults(func=cmd_delivery)


def delivery_command(args, *, runner: "FeatureDeliveryRunner | None" = None) -> int:
    if runner is None:
        from hermes_cli.feature_delivery_runner import FeatureDeliveryRunner

        executor = None
        if getattr(args, "executor", None) == "profiles":
            from hermes_cli.profile_stage_executor import ProfileStageExecutor

            executor = ProfileStageExecutor()
        runner = FeatureDeliveryRunner(executor=executor)
    if args.delivery_command == "create":
        print(runner.create(args.contract))
        return 0
    if args.delivery_command == "run":
        print(runner.run(args.task_id).render())
        return 0
    if args.delivery_command == "resume":
        print(runner.resume(args.task_id).render())
        return 0
    if args.delivery_command == "unblock":
        print(
            runner.unblock(
                args.task_id,
                resume_stage=args.resume_stage,
                confirmed=args.confirm,
            ).render()
        )
        return 0
    if args.delivery_command == "status":
        print(runner.status(args.task_id).render())
        return 0
    raise ValueError(f"unknown delivery command: {args.delivery_command}")
