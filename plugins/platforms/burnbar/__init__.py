def register(ctx):
    """Register the BurnBar platform plugin."""
    from .adapter import register as _register

    return _register(ctx)

__all__ = ["register"]
