"""Classic interactive CLI process-identity setup."""


def register_interactive_cli() -> None:
    """Register the CLI and bind its Windows descendants to its lifetime."""
    from hermes_cli.process_identity import attach_self_to_kill_on_close_job, register_self

    register_self("cli")
    attach_self_to_kill_on_close_job()
