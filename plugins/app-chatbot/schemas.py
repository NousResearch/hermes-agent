"""Tool schemas for CRWD support tools."""

GET_ACTIVE_GIGS = {
    "name": "get_active_gigs",
    "description": (
        "Fetch paginated active/open gigs (status Active, not deleted, end date still open). "
        "Excludes gigs the user is already enrolled in. Returns JSON with items[] containing "
        "name, description, payout, gig_type, dates, proof_type, image, and image_url."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "MongoDB users._id (24-char hex). Defaults to APP_CHATBOT_DEFAULT_USER_ID from .env.",
            },
            "page": {
                "type": "integer",
                "description": "Page number (1-based). Default 1.",
            },
            "limit": {
                "type": "integer",
                "description": "Results per page (max 50). Default 10.",
            },
        },
        "required": [],
    },
}

GET_USER_PROFILE_BY_ID = {
    "name": "get_user_profile_by_id",
    "description": (
        "Look up a CRWD user by MongoDB users._id (24-char hex). "
        "Returns profile fields (name, email, phone, status, role) or a not-found message."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "MongoDB users._id (24-char hex).",
            },
        },
        "required": ["user_id"],
    },
}

GET_GIG_DETAILS = {
    "name": "get_gig_details",
    "description": (
        "Fetch one gig/campaign by Mongo _id, exact name, or fuzzy name match on active open gigs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "gig_id": {
                "type": "string",
                "description": "MongoDB crwds._id (24-char hex).",
            },
            "name": {
                "type": "string",
                "description": "Gig name or substring to search for.",
            },
        },
        "required": [],
    },
}

GET_USER_GIG_HISTORY = {
    "name": "get_user_gig_history",
    "description": (
        "Past gig participation for a user. Returns membership rows with status, dates, "
        "and payment-related fields."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "MongoDB users._id (24-char hex). Defaults to APP_CHATBOT_DEFAULT_USER_ID from .env.",
            },
            "limit": {
                "type": "integer",
                "description": "Max rows to return (default 50, max 100).",
            },
        },
        "required": [],
    },
}

GET_USER_JOINED_GIGS = {
    "name": "get_user_joined_gigs",
    "description": (
        "Gigs the user is currently linked to (active, non-deleted membership). "
        "Joins membership rows to full gig documents."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "MongoDB users._id (24-char hex). Defaults to APP_CHATBOT_DEFAULT_USER_ID from .env.",
            },
            "limit": {
                "type": "integer",
                "description": "Max rows to return (default 50, max 100).",
            },
        },
        "required": [],
    },
}

ALL_SCHEMAS = [
    GET_ACTIVE_GIGS,
    GET_USER_PROFILE_BY_ID,
    GET_GIG_DETAILS,
    GET_USER_GIG_HISTORY,
    GET_USER_JOINED_GIGS,
]
