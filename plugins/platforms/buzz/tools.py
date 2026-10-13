"""Discovery-time client tools for the deferred Buzz platform plugin."""

from __future__ import annotations


async def _handle_buzz_read_message_link(args: dict, **kwargs) -> str:
    from .adapter import _handle_buzz_read_message_link as handle

    return await handle(args, **kwargs)


def _check_buzz_link_reader() -> bool:
    from .adapter import _check_buzz_link_reader as check

    return check()


def register_tools(ctx) -> None:
    """Register Buzz client tools without materializing the platform adapter."""
    ctx.register_tool(
        name="buzz_read_message_link",
        toolset="buzz",
        schema={
            "name": "buzz_read_message_link",
            "description": "Read the exact Buzz message referenced by a canonical buzz://message link.",
            "parameters": {
                "type": "object",
                "properties": {"link": {"type": "string"}},
                "required": ["link"],
                "additionalProperties": False,
            },
        },
        handler=_handle_buzz_read_message_link,
        check_fn=_check_buzz_link_reader,
        requires_env=["BUZZ_RELAY_URL", "BUZZ_PRIVATE_KEY"],
        is_async=True,
        description="Read the exact Buzz message referenced by a canonical buzz://message link.",
        emoji="🐝",
    )
