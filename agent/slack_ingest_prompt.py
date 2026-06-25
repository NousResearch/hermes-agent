"""Prompt builder for /slack-ingest.

The command is intentionally thin: it accepts a natural-language request from a
messaging platform, wraps it in a deterministic execution scaffold, then lets
the live agent perform the actual Slack history reads + wiki writes as a normal
turn. This keeps the core narrow while still giving users a reliable, terse
entrypoint from Slack threads.
"""

from __future__ import annotations

_SHARED_WIKI_PATH = (
    "/Users/dukho/Library/CloudStorage/GoogleDrive-duk921@gmail.com/내 드라이브/"
    "Feelma_Obsidian/LLM-Wiki/wiki"
)

_EXECUTION_RULES = f"""
Treat the text below as the user's Slack-ingest request.

Execution rules:
- Use Slack channel history as the source of truth, not Hermes session recall.
- Use the shared wiki workflow for this user when saving pages.
- Target the canonical shared wiki path: `{_SHARED_WIKI_PATH}`.
- If the request implies wiki writing, read the shared wiki's AGENTS/index/log as needed before writing.
- Prefer the `slack-channel-history-to-wiki-ingest` skill workflow.
- When summarizing channel activity, exclude side chat unless the user explicitly asks for full conversational context.
- Update `wiki/index.md` and `wiki/log.md` when you create or update wiki pages.
- In the final reply, state the exact files written and that the basis was Slack channel history.
""".strip()


def build_slack_ingest_prompt(user_request: str) -> str:
    """Build the agent-facing prompt for /slack-ingest.

    The user-facing command stays short (``!slack-ingest ...`` in Slack), while
    the live agent receives explicit workflow constraints so execution is
    repeatable and grounded.
    """

    request = (user_request or "").strip()
    if not request:
        raise ValueError("slack_ingest_requires_request")

    return (
        f"{_EXECUTION_RULES}\n\n"
        "User request:\n"
        f"{request}\n\n"
        "Carry out the request now. If the user asked for a multi-channel daily ingest, "
        "create the per-channel summaries first and then any requested integrated briefing."
    )
