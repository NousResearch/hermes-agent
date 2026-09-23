"""``hermes codes`` subcommand parser."""

from __future__ import annotations


def build_codes_parser(subparsers) -> None:
    """Attach the model-blind verification-code handle subcommand (#119683)."""
    codes_parser = subparsers.add_parser(
        "codes",
        help="Park a verification code server-side (prints an otp_… handle, never the code)",
        description=(
            "Hand an SMS/email verification code to Hermes without putting it in the "
            "conversation: hermes codes put parks it for ~5 minutes and returns an "
            "opaque handle that browser_vault_enter_code can consume once."
        ),
    )
    from hermes_cli.codes import codes_command, register_cli

    register_cli(codes_parser)
    codes_parser.set_defaults(func=codes_command)
