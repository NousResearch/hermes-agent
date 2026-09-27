"""Honcho tool schemas sent to the model — byte-stable (prompt-cache parity).

Profile / search / reasoning / context / conclude, exposed via HonchoMemoryProvider.get_tool_schemas().
"""

_PEER = {"type": "string",
         "description": "Peer to query. Built-in aliases: 'user' (default), 'ai'. Or pass any peer ID from this workspace."}


def _tool(name: str, description: str, properties: dict, required: list) -> dict:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}


PROFILE_SCHEMA = _tool(
    "honcho_profile",
    "Read or write a peer's CARD — a short, curated list of standing facts "
    "about that peer (name, role, preferences, communication style, recurring "
    "patterns). This is the cheapest, fastest Honcho call: no query, no LLM, "
    "just the current card. Pass `card` to overwrite it; omit `card` to read. "
    "An empty read returns a `hint` explaining why (observation disabled, fresh "
    "peer, representation still warming up) — that is NOT an error; the card "
    "accumulates over time from observed conversation. "
    "Related tools: honcho_context for the fuller standing snapshot (card + "
    "representation + summary + recent messages); honcho_search to find "
    "specific things that were actually said; honcho_reasoning for a "
    "synthesized answer to a question.",
    {"peer": _PEER,
     "card": {"type": "array", "items": {"type": "string"},
              "description": "New peer card as a list of fact strings. Omit to read the current card."}},
    [],
)

SEARCH_SCHEMA = _tool(
    "honcho_search",
    "Hybrid (semantic + keyword) search over a peer's actual message "
    "history across ALL past sessions they took part in — not just the "
    "current one. Returns RRF-ranked raw message excerpts (what was "
    "literally said, including the assistant's own messages about the "
    "peer), no LLM synthesis. Cheaper and faster than honcho_reasoning. "
    "Use this to recall specific past facts — 'what did I say about X', "
    "'what was the regimen/decision/config we settled on' — and reason "
    "over the excerpts yourself. For nuanced questions needing synthesis, "
    "use honcho_reasoning instead.",
    {"query": {"type": "string",
               "description": "What to look for — a topic, keyword, name, or natural-language description of the fact you're trying to recall."},
     "max_tokens": {"type": "integer",
                    "description": "Approximate budget for returned excerpts (default 800, max 2000). Larger budgets return more/longer ranked snippets."},
     "peer": {"type": "string",
              "description": "Whose history to search. Built-in aliases: 'user' (default), 'ai'. Or pass any peer ID from this workspace. Spans every session that peer took part in."}},
    ["query"],
)

REASONING_SCHEMA = _tool(
    "honcho_reasoning",
    "Ask Honcho's dialectic agent a natural-language question about a peer and "
    "get back a SYNTHESIZED answer. This is the only Honcho tool that runs an "
    "LLM: it agentically searches both raw messages and derived conclusions, "
    "reasons over them, and writes a prose answer — so it is the slowest and "
    "most expensive call (seconds + tokens). Reach for it for nuanced or "
    "open-ended questions ('how does this person prefer to receive feedback?', "
    "'what's their relationship to project X?') where you want Honcho to do the "
    "synthesis. For a specific fact that was stated, prefer honcho_search "
    "(cheap, raw excerpts, you synthesize). For standing profile facts, prefer "
    "honcho_profile / honcho_context (no LLM). "
    "Pass reasoning_level to control depth: minimal (fast/cheap), low (default), "
    "medium, high, max (deep/expensive). Omit for the configured default.",
    {"query": {"type": "string", "description": "A natural language question."},
     "reasoning_level": {
            "type": "string",
            "description": (
                "Override the default reasoning depth. "
                "Omit to use the configured default (typically low).\n"
                "reasoning_level parameter guide:\n"
                "- minimal: use ONLY for a single quick factual lookup (e.g. "
                "'what is the user's name'). Honcho hard-caps this tier's output "
                "at 250 tokens combined with the model's own hidden reasoning "
                "tokens — a multi-part answer can get cut off mid-thought before "
                "it even reaches the final-answer phase, especially on models "
                "with reasoning/thinking enabled.\n"
                "- low/medium/high/max: use for anything requiring a synthesized, "
                "multi-fact, or summary-style answer (e.g. 'summarize known facts "
                "about this peer', 'what are their communication preferences'). "
                "These tiers have no output-token cap of their own (fall back to "
                "Honcho's 8192-token global default), so they don't have "
                "minimal's cutoff failure mode.\n"
                "  - low: straightforward questions with clear answers\n"
                "  - medium: multi-aspect questions requiring synthesis across observations\n"
                "  - high: complex behavioral patterns, contradictions, deep analysis\n"
                "  - max: thorough audit-level analysis, leave no stone unturned\n"
                "Default to at least 'low' unless the query is genuinely a single "
                "fact lookup."
            ),
            "enum": ["minimal", "low", "medium", "high", "max"]},
     "peer": _PEER},
    ["query"],
)

CONTEXT_SCHEMA = _tool(
    "honcho_context",
    "Retrieve the standing SNAPSHOT Honcho holds for the current session — "
    "session summary, the peer's representation, the peer card, and the most "
    "recent messages — in one call. No query, no LLM synthesis (cheaper than "
    "honcho_reasoning). Use it to orient yourself on what Honcho currently "
    "knows about this conversation and peer. This is a fixed snapshot, not a "
    "search: to look up a specific past fact use honcho_search; to ask a "
    "question and get a synthesized answer use honcho_reasoning; for just the "
    "compact card use honcho_profile.",
    {"peer": _PEER},
    [],
)

CONCLUDE_SCHEMA = _tool(
    "honcho_conclude",
    "Write, delete, or list CONCLUSIONS — persistent, derived facts about a peer that "
    "feeds their long-term profile (card + representation). Use this to record "
    "something durable you've learned about the peer (a stable preference, a "
    "correction, a standing constraint) so future sessions carry it forward. "
    "You MUST pass exactly one of `conclusion` (to create), `delete_id` (to "
    "delete), or `list` (to list/search); any other combination is an error. "
    "A deletion ID is an opaque server-generated string: first call with `list=true` "
    "and optionally `query`, then pass the returned ID as `delete_id`. "
    "Deletion exists only for "
    "PII removal — for merely wrong facts, write a corrected conclusion instead; "
    "Honcho self-heals contradictions over time. This is a WRITE tool: to read "
    "the profile use honcho_profile / honcho_context, and to search what was "
    "said use honcho_search.",
    {"conclusion": {"type": "string",
                    "description": "A factual statement to persist. Provide this when creating a conclusion. Do not send it together with delete_id or list."},
     "delete_id": {"type": "string",
                   "description": "Conclusion ID to delete for PII removal. Provide this when deleting a conclusion. Do not send it together with conclusion or list. Get this id from a prior `list` call — never guess it."},
     "list": {"type": "boolean",
              "description": "Set to true to list or search stored conclusions (with their ids) instead of creating or deleting one. Do not send together with conclusion or delete_id."},
     "query": {"type": "string",
               "description": "Optional semantic search query, used only when `list` is true. Omit to list the most recent conclusions instead of searching."},
     "peer": {"type": "string",
              "description": "The peer the conclusion is ABOUT. Built-in aliases: 'user' (default), 'ai'. Or pass any peer ID from this workspace."}},
    [],
)


ALL_TOOL_SCHEMAS = [PROFILE_SCHEMA, SEARCH_SCHEMA, REASONING_SCHEMA, CONTEXT_SCHEMA, CONCLUDE_SCHEMA]
