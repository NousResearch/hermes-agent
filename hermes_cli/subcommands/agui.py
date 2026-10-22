"""``hermes agui``: thin registration for the standalone adapter launcher."""


def build_agui_parser(subparsers):
    from agui_adapter.cli import add_arguments, run

    parser = subparsers.add_parser(
        "agui",
        help="Run Hermes as an AG-UI HTTP/SSE server",
        description="Expose Hermes to AG-UI clients. Settings live under agui in config.yaml; flags override them per invocation.",
    )
    add_arguments(parser)
    parser.set_defaults(func=run)
