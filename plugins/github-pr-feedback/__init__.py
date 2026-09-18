"""Hermes directory-plugin entry point."""

try:
    from .github_pr_feedback.cli import cli_bindings
except ImportError:  # Direct module loading in lightweight test hosts.
    from github_pr_feedback.cli import cli_bindings


def register(ctx) -> None:
    """Register the standalone, host-owned CLI command tree."""

    setup_fn, handler_fn = cli_bindings(ctx)
    ctx.register_cli_command(
        name="github-pr-feedback",
        help="Scan governed GitHub pull-request feedback",
        setup_fn=setup_fn,
        handler_fn=handler_fn,
        description="Read-only GitHub feedback intake with blocked Kanban cards.",
    )
