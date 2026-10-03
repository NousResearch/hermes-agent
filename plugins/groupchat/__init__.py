"""One Groupchat addon: inbound relevance and outbound pingpong prevention."""
def register(ctx):
    from .policy import GroupchatAddon
    ctx.register_middleware("gateway_conversation", lambda adapter: GroupchatAddon(adapter))
