"""Native directory-plugin entry point (also usable without a pip install)."""


def register(ctx):
    from .plugin import register as register_plugin

    return register_plugin(ctx)
