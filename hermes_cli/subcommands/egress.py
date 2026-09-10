"""``hermes egress`` subcommand parser."""

from __future__ import annotations


def build_egress_parser(subparsers) -> None:
    """Attach the ``egress`` subcommand to ``subparsers``."""
    # OUTBOUND egress firewall (iron-proxy); `hermes proxy` is the separate INBOUND
    # OAuth-aggregator reverse proxy.
    egress_parser = subparsers.add_parser(
        "egress", help="Manage the iron-proxy egress credential-injection firewall",
        description="Manage iron-proxy, the optional TLS-intercepting egress firewall "
            "that swaps proxy tokens for real API credentials before outbound "
            "requests leave a sandbox.  Disabled by default.  See: "
            "https://hermes-agent.nousresearch.com/docs/user-guide/egress/iron-proxy")

    from hermes_cli import proxy_cli as _proxy_cli
    _proxy_cli.register_cli(egress_parser)

    def _dispatch_egress(args):  # noqa: ANN001
        # dest='egress_command' stays disjoint from ``hermes proxy`` (dest='proxy_command').
        sub = getattr(args, "egress_command", None)
        if sub is not None and hasattr(args, "func") and args.func is not _dispatch_egress:
            return args.func(args)
        egress_parser.print_help()
        return 0

    egress_parser.set_defaults(func=_dispatch_egress)
