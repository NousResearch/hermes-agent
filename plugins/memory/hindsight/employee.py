from typing import Any

_CLIENT_POLICY: dict[str, Any] = {
    "mode": "local_external",
    "memory_mode": "hybrid",
    "recall_prefetch_method": "recall",
    "recall_sync": False,
    "recall_types": ["world", "experience", "observation"],
    "prefer_observations": True,
    "recall_budget": "mid",
    "recall_max_tokens": 4096,
    "auto_retain": True,
    "retain_async": True,
    "retain_every_n_turns": 4,
    "observation_scopes": "shared",
    "retain_context": (
        "Workplace chat between this workspace's human collaborators and "
        "their AI employee (the assistant turns). User turns may contain "
        "messages from multiple different people; author names identify "
        "speakers, not verified identities."
    ),
}

_RECALL_SESSION_SEARCH_SENTENCE = (
    " For the verbatim record of past conversations, use session_search."
)

RECALL_TOOL_SCHEMA: dict[str, Any] = {
    "name": "recall",
    "description": (
        "Ask your background memory a question about this organization — its "
        "people, customers, decisions and their status, and how things are "
        "done here. It reasons over everything learned across all "
        "conversations and channels and returns a synthesized answer, not a "
        "list of matches.\n\n"
        "Relevant memory already surfaces automatically each turn as recalled "
        "context; this is how you deliberately go deeper. Use it when framing "
        "work rather than executing it: taking on a new area, picking up a "
        "topic you haven't touched recently, an ambiguous request — check "
        "memory before asking the person to clarify — or a conversation's "
        "first turn, where nothing has surfaced yet. Once a task is underway "
        "and clear, the surfaced context and live sources usually carry it.\n\n"
        "The answer is supplementary context, not a source of truth — it may "
        "be stale or incomplete; ground claims and actions in live sources."
        + _RECALL_SESSION_SEARCH_SENTENCE
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A specific, self-contained question — 'What do we know "
                    "about ACME's renewal terms?' — not keywords."
                ),
            },
        },
        "required": ["query"],
    },
}


def config():
    import hashlib
    from hermes_constants import get_hermes_home
    from hermes_cli.config import load_config_readonly
    from agent.secret_scope import get_secret
    settings = load_config_readonly().get('hindsight', {})
    bank = settings.get('bank_id') or ('hermes-' + hashlib.sha256(str(get_hermes_home().resolve()).encode()).hexdigest()[:20])
    return {**_CLIENT_POLICY, 'api_url': settings.get('url', ''),
            'api_key': get_secret('HINDSIGHT_API_KEY', ''), 'bank_id': bank}
