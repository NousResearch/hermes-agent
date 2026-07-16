"""`hermes training-data` — export agent trajectories for model training."""

from pathlib import Path


def build_training_data_parser(subparsers, cmd_status=None, cmd_export=None):
    """Build the training-data subcommand parser.

    Args:
        subparsers: argparse subparsers object
        cmd_status: callable for the status subcommand
        cmd_export: callable for the export subcommand
    """
    training_data_parser = subparsers.add_parser(
        "training-data",
        help="Export agent trajectories for model training",
        description="Validate, deduplicate, and export agent conversation data for training frameworks",
    )
    td_subparsers = training_data_parser.add_subparsers(dest="training_data_action")

    td_status = td_subparsers.add_parser("status", help="Show available training data statistics")
    td_status.add_argument("--days", type=int, default=30)
    td_status.set_defaults(func=cmd_status or _cmd_status)

    td_export = td_subparsers.add_parser("export", help="Export training data")
    td_export.add_argument("--format", choices=["sharegpt", "alpaca", "parquet"], default="sharegpt")
    td_export.add_argument("--min-score", type=float, help="Minimum evaluation score")
    td_export.add_argument("--domain", help="Filter by task domain")
    td_export.add_argument("--since-last", action="store_true", help="Only new records since last export")
    td_export.add_argument("--output", "-o", help="Output file path")
    td_export.set_defaults(func=cmd_export or _cmd_export)


def _cmd_status(args) -> int:
    """Show training data statistics."""
    from agent.training_data import TrainingDataBridge
    bridge = TrainingDataBridge()
    stats = bridge.get_stats(days=args.days or 30)

    print(f"Training Data (last {stats['period_days']} days):")
    print(f"  Total records:     {stats['total_records']}")
    print(f"  Unique records:    {stats['unique_records']}")
    print(f"  Scored records:    {stats['scored_records']}")
    print(f"  New since export:  {stats['new_since_last_export']}")
    if stats["last_export_at"]:
        print(f"  Last exported:     {stats['last_export_at'][:19]}")
    if stats["by_domain"]:
        print(f"\n  By domain:")
        for domain, count in sorted(stats["by_domain"].items()):
            print(f"    {domain}: {count}")
    return 0


def _cmd_export(args) -> int:
    """Export training data."""
    from agent.training_data import TrainingDataBridge
    bridge = TrainingDataBridge()

    result = bridge.export(
        output_path=args.output,
        fmt=args.format or "sharegpt",
        min_score=args.min_score or 0.0,
        domain=args.domain,
        since_last=args.since_last,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        return 1

    print(f"Exported {result['records_exported']} records")
    print(f"Format: {result['format']}")
    print(f"Path:   {result['output_path']}")
    print(f"\nReady for training. Use your framework of choice:")
    print(f"  slime:   slime train --data {result['output_path']}")
    print(f"  unsloth: unsloth train --dataset {result['output_path']}")
    print(f"  axolotl: configure dataset: {result['output_path']}")
    return 0
