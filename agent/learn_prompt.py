#!/usr/bin/env python3
"""``/learn`` — build the standards-guided prompt that turns whatever the user
described into a reusable skill.

``/learn`` is open-ended. The user can point it at anything they can describe:
a directory of code, an API doc URL, a workflow they just walked the agent
through in this conversation, or pasted notes. This module builds ONE prompt
that instructs the live agent to:

  1. Gather the sources the user named, using the tools it already has
     (``read_file`` / ``search_files`` for dirs, ``web_extract`` for URLs, the
     current conversation for "what I just did", the user's text for pasted
     material).
  2. Author a single ``SKILL.md`` via ``skill_manage`` that follows the Hermes
     skill-authoring standards (description <=60 chars, the modern section
     order, Hermes-tool framing, no invented commands).

There is no separate distillation engine and no model-tool footprint: the
agent does the work with its existing toolset, so this works identically on
local, Docker, and remote terminal backends. Every surface (CLI ``/learn``,
gateway ``/learn``, the dashboard "Learn a skill" panel) calls
:func:`build_learn_prompt` and feeds the result to the agent as a normal turn.
"""

from __future__ import annotations

# The house-style rules, distilled from AGENTS.md "Skill authoring standards
# (HARDLINE)" and the hermes-agent-dev new-skill salvage reference. Embedded in
# the prompt so the agent authors skills the way a maintainer would by hand.
_AUTHORING_STANDARDS = """\
Follow the Hermes skill-authoring standards exactly:

Frontmatter:
- name: lowercase-hyphenated, <=64 chars, no spaces.
- description: ONE sentence, <=60 characters, ends with a period. State the
  capability, not the implementation. No marketing words (powerful,
  comprehensive, seamless, advanced). Do NOT repeat the skill name. If the
  description contains a colon, wrap the whole value in double quotes.
- version: 0.1.0
- metadata.hermes.tags: a few Capitalized, Relevant, Tags.

Body section order (omit a section only if it genuinely has no content):
1. "# <Human Title>" then a 2-3 sentence intro: what it does, what it does NOT
   do, and the key dependency stance (e.g. "stdlib only").
2. "## When to Use" — bullet list of concrete trigger phrases.
3. "## Prerequisites" — exact env vars, install steps, credentials.
4. "## How to Run" — the canonical invocation, framed through Hermes tools.
5. "## Quick Reference" — a flat command/endpoint list, no narration.
6. "## Procedure" — numbered steps with copy-paste-exact commands.
7. "## Pitfalls" — known limits, rate limits, things that look broken but aren't.
8. "## Verification" — a single command/check that proves the skill worked.

Hermes-tool framing (this is what makes it a skill, not shell docs):
- Frame running scripts as "invoke through the `terminal` tool".
- Use `read_file` (not cat/head/tail), `search_files` (not grep/find/ls),
  `patch` (not sed/awk), `web_extract` (not curl-to-scrape),
  `vision_analyze` for images. Reference these tools by name in backticks.
- Do NOT name shell utilities the agent already has wrapped.

Quality bar:
- Prefer exact commands, endpoint URLs, function signatures, and config keys
  that appear VERBATIM in the source. NEVER invent flags, paths, or APIs — if
  you didn't see it in the source, don't write it.
- Keep it tight and scannable: ~100 lines for a simple skill, ~200 for a
  complex one. Don't re-paste the source docs.
- Don't write a router/index/hub skill that only points at other skills.
- Larger scripts/parsers belong in a `scripts/` file (add via
  `skill_manage` write_file), referenced from SKILL.md by relative path — not
  inlined for the agent to re-type every run."""


def build_learn_prompt(user_request: str) -> str:
    """Build the agent prompt for an open-ended ``/learn`` request.

    Args:
        user_request: the free-text the user gave after ``/learn`` — a
            description of the workflow, paths, URLs, or "what I just did".

    Returns:
        A complete instruction the agent runs as a normal turn. The agent
        gathers the described sources with its existing tools and authors the
        skill via ``skill_manage``.
    """
    req = (user_request or "").strip()
    if not req:
        req = (
            "the workflow we just went through in this conversation — review "
            "the steps taken and distill them into a reusable skill"
        )

    return (
        "[/learn] The user wants you to learn a reusable skill from the "
        "source(s) they described below, and save it.\n\n"
        f"WHAT TO LEARN FROM:\n{req}\n\n"
        "Do this:\n"
        "1. Gather the material. Resolve whatever the user named using the "
        "tools you already have — `read_file`/`search_files` for local files "
        "or directories, `web_extract` for URLs, the current conversation "
        "history if they referred to something you just did, and the text "
        "they pasted as-is. If the request is ambiguous about scope, make a "
        "reasonable choice and note it; do not stall.\n"
        "2. Author ONE SKILL.md and save it with the `skill_manage` tool "
        "(action=\"create\"). Pick a sensible category. If the procedure needs "
        "a non-trivial script, add it under the skill's `scripts/` with "
        "`skill_manage` write_file and reference it by relative path.\n\n"
        f"{_AUTHORING_STANDARDS}\n\n"
        "When done, tell the user the skill name, its category, and a "
        "one-line summary of what it captured."
    )
