"""``hermes vault`` subcommand parser."""

from __future__ import annotations


def build_vault_parser(subparsers) -> None:
    """Attach the credential broker and model-blind autofill subcommand."""
    vault_parser = subparsers.add_parser(
        "vault",
        help="Manage model-blind login storage and autofill",
        description=(
            "Store login credentials locally or in a configured password manager. "
            "The agent sees opaque handles; identifiers and passwords are filled "
            "server-side only on the exact origin they were saved for."
        ),
    )
    from hermes_cli.vault import register_cli, vault_command

    register_cli(vault_parser)
    vault_parser.set_defaults(func=vault_command)
