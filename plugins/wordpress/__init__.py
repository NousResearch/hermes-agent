"""WordPress plugin for Hermes."""

from __future__ import annotations

from .schemas import (
    WP_POST_CREATE_SCHEMA,
    WP_POST_GET_SCHEMA,
    WP_POST_LIST_SCHEMA,
    WP_POST_UPDATE_SCHEMA,
    WP_SITE_INFO_SCHEMA,
)
from .tools import (
    handle_wp_post_create,
    handle_wp_post_get,
    handle_wp_post_list,
    handle_wp_post_update,
    handle_wp_site_info,
    wordpress_requirements_met,
)


def register(ctx) -> None:
    ctx.register_tool(
        name="wp_site_info",
        toolset="wordpress",
        schema=WP_SITE_INFO_SCHEMA,
        handler=handle_wp_site_info,
        check_fn=wordpress_requirements_met,
        emoji="W",
    )
    ctx.register_tool(
        name="wp_post_list",
        toolset="wordpress",
        schema=WP_POST_LIST_SCHEMA,
        handler=handle_wp_post_list,
        check_fn=wordpress_requirements_met,
        emoji="W",
    )
    ctx.register_tool(
        name="wp_post_get",
        toolset="wordpress",
        schema=WP_POST_GET_SCHEMA,
        handler=handle_wp_post_get,
        check_fn=wordpress_requirements_met,
        emoji="W",
    )
    ctx.register_tool(
        name="wp_post_create",
        toolset="wordpress",
        schema=WP_POST_CREATE_SCHEMA,
        handler=handle_wp_post_create,
        check_fn=wordpress_requirements_met,
        emoji="W",
    )
    ctx.register_tool(
        name="wp_post_update",
        toolset="wordpress",
        schema=WP_POST_UPDATE_SCHEMA,
        handler=handle_wp_post_update,
        check_fn=wordpress_requirements_met,
        emoji="W",
    )
