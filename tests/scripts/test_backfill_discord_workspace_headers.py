"""CLI safety contract for the Discord workspace-header backfill."""

from scripts.backfill_discord_workspace_headers import build_parser


def test_backfill_defaults_to_read_only_dry_run():
    args = build_parser().parse_args([])

    assert args.apply is False


def test_backfill_requires_explicit_apply_flag_for_mutation():
    args = build_parser().parse_args(["--apply"])

    assert args.apply is True
