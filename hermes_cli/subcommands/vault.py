"""``hermes vault`` subcommand parser."""

from __future__ import annotations


def build_vault_parser(subparsers) -> None:
    """Attach the local encrypted autofill vault subcommand."""
    vault_parser = subparsers.add_parser(
        "vault",
        help="Manage the local encrypted autofill vault (add/list/rm credentials)",
        description=(
            "Store login credentials in a locally encrypted vault. The agent "
            "sees handles and login identifiers (metadata); passwords are "
            "injected server-side by browser_vault_fill on the exact origin "
            "they were saved for and never enter the conversation."
        ),
    )
    from hermes_cli.vault import register_cli, vault_command

    register_cli(vault_parser)
    vault_parser.set_defaults(func=vault_command)
