"""``hermes secrets`` subcommand parser."""

from __future__ import annotations


def build_secrets_parser(subparsers) -> None:
    """Attach the ``secrets`` subcommand to ``subparsers``."""
    secrets_parser = subparsers.add_parser(
        "secrets", help="Manage external secret sources (Bitwarden, 1Password)",
        description="Pull API keys from an external secret manager at process startup "
            "instead of storing them in ~/.hermes/.env.  Supports Bitwarden "
            "Secrets Manager and 1Password.  See: "
            "https://hermes-agent.nousresearch.com/docs/user-guide/secrets/")
    secrets_subparsers = secrets_parser.add_subparsers(dest="secrets_command")

    secrets_bw = secrets_subparsers.add_parser(
        "bitwarden", aliases=["bw"], help="Bitwarden Secrets Manager integration")

    secrets_op = secrets_subparsers.add_parser(
        "onepassword", aliases=["op", "1password"], help="1Password (op:// references) integration")

    # Lazy import: secrets_cli pulls cryptography's native extension, which on Windows
    # maps into the updater process and defers its self-lock preflight. secrets_cli
    # defers its backend import to first use, so register_cli here costs no crypto load.
    # Lazy-import secrets_cli: the module imports agent.secret_sources.bitwarden which loads
    # cryptography._rust.pyd. See #86781.
    from hermes_cli import secrets_cli as _secrets_cli
    from hermes_cli import onepassword_secrets_cli as _op_secrets_cli

    _secrets_cli.register_cli(secrets_bw)
    _op_secrets_cli.register_cli(secrets_op)

    def _dispatch_secrets(args):  # noqa: ANN001
        sub = getattr(args, "secrets_command", None)
        if sub is None:
            secrets_parser.print_help()
            return 0
        return args.func(args)

    secrets_parser.set_defaults(func=_dispatch_secrets)
