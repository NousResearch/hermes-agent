def register(ctx) -> None:
    """Register Telegram without importing its adapter during package discovery."""
    from .adapter import register as _register
    from .mini_app.cli import command as mini_app_command
    from .mini_app.cli import register_cli as register_mini_app_cli

    _register(ctx)
    ctx.register_cli_command(
        name="telegram-mini-app",
        help="Manage the Telegram Mini App",
        description=(
            "Configure and manage the profile-scoped Telegram Mini App. "
            "Public ingress must be an existing stable HTTPS reverse proxy; "
            "Hermes never creates or manages a tunnel."
        ),
        setup_fn=register_mini_app_cli,
        handler_fn=mini_app_command,
    )


__all__ = ["register"]
