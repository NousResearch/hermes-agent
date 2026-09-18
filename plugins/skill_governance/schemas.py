"""Tool schemas for the skill-governance plugin."""

SKILLS_PLUGIN_ROADMAP_SUMMARY_SCHEMA = {
    "name": "skills_plugin_roadmap_summary",
    "description": (
        "Summarize the static approved skill-to-plugin roadmap catalog: total candidates, "
        "priority distribution, wave distribution, and candidates that require guarded "
        "live/credentialed operations. This does not inspect installed skills or curator state. "
        "Use before deciding which Hermes plugin to build next."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

SKILLS_FIND_PLUGIN_CANDIDATES_SCHEMA = {
    "name": "skills_find_plugin_candidates",
    "description": (
        "List approved skill-to-plugin candidates, optionally filtered by area, wave, "
        "priority, or search text. Returns deterministic, business-prioritized items."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "area": {
                "type": "string",
                "description": "Optional area tag, e.g. operations, saturn-business, agent-infra.",
            },
            "wave": {
                "type": "integer",
                "description": "Optional roadmap wave number: 1, 2, or 3.",
            },
            "priority": {
                "type": "string",
                "description": "Optional priority: very_high, high, medium_high, or medium.",
            },
            "query": {
                "type": "string",
                "description": "Optional case-insensitive text search over id, title, sources, and tools.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum candidates to return; clamped to 1..12.",
            },
            "include_plan": {
                "type": "boolean",
                "description": "When true, include a compact implementation plan for each candidate.",
            },
        },
        "additionalProperties": False,
    },
}

SKILLS_TO_PLUGIN_PLAN_SCHEMA = {
    "name": "skills_to_plugin_plan",
    "description": (
        "Return an implementation plan for one approved skill-to-plugin candidate. "
        "The plan includes proposed tools, source skills, safety guardrails, phases, "
        "and verification gates."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "candidate_id": {
                "type": "string",
                "description": "Approved candidate id, e.g. bitrix_ops or telegram_thread_router.",
            },
        },
        "required": ["candidate_id"],
        "additionalProperties": False,
    },
}
