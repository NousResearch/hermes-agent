"""One Groupchat addon: inbound relevance and outbound pingpong prevention."""
def register(ctx):
    from .policy import GroupchatAddon
    ctx.register_middleware("gateway_conversation", lambda adapter: GroupchatAddon(adapter))
    from . import coordination_runtime as work
    ctx.register_hook("pre_llm_call", work.pre_llm_call)
    ctx.register_hook("transform_tool_result", work.transform_tool_result)
    ctx.register_hook("pre_tool_call", work.pre_tool_call)
    ctx.register_hook("transform_llm_output", work.transform_llm_output)
    ctx.register_tool(name="groupchat_work", toolset="groupchat", schema=work.TOOL_SCHEMA,
                      handler=work.work_tool, check_fn=work.enabled)
