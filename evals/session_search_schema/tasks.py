"""Task prompts + programmatic oracles for the session_search schema A/B eval.

Every oracle is a pure function of the model's final answer string — no LLM
judging. Tasks map 1:1 to the sessions seeded by fixtures.py.
"""
from __future__ import annotations

import re

# task_id -> (prompt, oracle_fn, note)
TASKS = {
    "t1_discover": (
        "What approach did we settle on for the postgres migration of the "
        "analytics cluster? Check our past conversations.",
        lambda a: "pglogical" in a.lower(),
        "discovery basics",
    ),
    "t2_scroll": (
        "A while back we had that checkout 504 incident. What exactly was the "
        "FINAL fix we shipped (the precise setting and value)? Look it up in "
        "our session history.",
        lambda a: (
            ("statement_timeout" in a.lower() or "statement timeout" in a.lower())
            and "45" in a
        ),
        "requires forward scroll past the ±5 window (and past bookends)",
    ),
    "t3_broaden": (
        "Where do I access the grafana graphs for the beehive sensors? We set "
        "this up together — search our history.",
        lambda a: "3000" in a,
        "AND-query misses; needs broadening (OR / fewer terms)",
    ),
    "t4_link": (
        "Find the session where we planned the reef aquarium build and point "
        "me to it so I can reopen it.",
        lambda a: (
            bool(re.search(r"(?<!`)@session:[\w./-]*20260822_183000_a4b4c4(?!`)", a))
            and "`@session" not in a
            and not re.search(r"\]\(@session", a)
        ),
        "must emit link value verbatim, not backticked/markdown",
    ),
    "t5_profile": (
        "@session:work/20260815_110000_beef01 — what did we decide in there?",
        lambda a: "vault" in a.lower() and "90" in a,
        "resolve profile-qualified link (read shape + profile)",
    ),
    "t6_browse": (
        "What have I been working on in my recent sessions? Just give me a "
        "quick rundown.",
        lambda a: sum(
            k in a.lower()
            for k in ("fan", "tax", "aquarium", "beehive", "apiary", "504", "checkout")
        ) >= 3,
        "browse shape",
    ),
}

SYSTEM = (
    "You are Hermes, a personal AI agent with persistent memory across "
    "sessions. You have a session_search tool over the user's past "
    "conversation history. Answer the user's question accurately and "
    "concisely. Today is 2026-08-26."
)
