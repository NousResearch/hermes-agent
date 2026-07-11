def register(ctx) -> None:
    """Register Telegram without importing its adapter during package discovery."""
    from .adapter import register as _register

    _register(ctx)


__all__ = ["register"]
