"""Linear Agent platform plugin."""


def register(ctx) -> None:
    """Register the Linear Agent platform adapter lazily.

    Lazy import keeps `python -m plugins.platforms.linear_agent.oauth` from
    importing the adapter and registering tool side effects before the OAuth CLI
    module executes.
    """
    from .adapter import register as _register

    return _register(ctx)


__all__ = ["register"]
