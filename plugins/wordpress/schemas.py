"""JSON schemas for WordPress plugin tools."""

from __future__ import annotations


WP_SITE_INFO_SCHEMA = {
    "name": "wp_site_info",
    "description": "Inspect the configured WordPress site and report API/auth status.",
    "parameters": {
        "type": "object",
        "properties": {
            "base_url": {
                "type": "string",
                "description": "Optional WordPress site URL override for this call. Required when WORDPRESS_BASE_URL env var is unset.",
            },
        },
        "additionalProperties": False,
    },
}


_POST_MUTATION_PROPERTIES = {
    "title": {"type": "string"},
    "content": {"type": "string"},
    "excerpt": {"type": "string"},
    "slug": {"type": "string"},
    "status": {
        "type": "string",
        "enum": ["publish", "future", "draft", "pending", "private"],
    },
    "featured_media": {"type": "integer"},
    "categories": {"type": "array", "items": {"type": "integer"}},
    "tags": {"type": "array", "items": {"type": "integer"}},
    "date": {"type": "string"},
    "date_gmt": {"type": "string"},
}


WP_POST_LIST_SCHEMA = {
    "name": "wp_post_list",
    "description": "List WordPress posts from the configured site.",
    "parameters": {
        "type": "object",
        "properties": {
            "base_url": {
                "type": "string",
                "description": "Optional WordPress site URL override. Required when WORDPRESS_BASE_URL env var is unset.",
            },
            "status": {"type": "string"},
            "search": {"type": "string"},
            "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
            "page": {"type": "integer", "minimum": 1},
        },
        "additionalProperties": False,
    },
}


WP_POST_GET_SCHEMA = {
    "name": "wp_post_get",
    "description": "Fetch a WordPress post by id.",
    "parameters": {
        "type": "object",
        "properties": {
            "base_url": {
                "type": "string",
                "description": "Optional WordPress site URL override. Required when WORDPRESS_BASE_URL env var is unset.",
            },
            "post_id": {"type": "integer", "minimum": 1},
            "context": {"type": "string", "enum": ["view", "embed", "edit"]},
        },
        "required": ["post_id"],
        "additionalProperties": False,
    },
}


WP_POST_CREATE_SCHEMA = {
    "name": "wp_post_create",
    "description": "Create a WordPress post via the REST API.",
    "parameters": {
        "type": "object",
        "properties": {
            "base_url": {
                "type": "string",
                "description": "Optional WordPress site URL override. Required when WORDPRESS_BASE_URL env var is unset.",
            },
            **_POST_MUTATION_PROPERTIES,
        },
        "additionalProperties": False,
    },
}


WP_POST_UPDATE_SCHEMA = {
    "name": "wp_post_update",
    "description": "Update an existing WordPress post via the REST API.",
    "parameters": {
        "type": "object",
        "properties": {
            "base_url": {
                "type": "string",
                "description": "Optional WordPress site URL override. Required when WORDPRESS_BASE_URL env var is unset.",
            },
            "post_id": {"type": "integer", "minimum": 1},
            **_POST_MUTATION_PROPERTIES,
        },
        "required": ["post_id"],
        "additionalProperties": False,
    },
}
