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
  2. Author a skill via ``skill_manage`` that follows the Hermes
     skill-authoring standards (description <=60 chars, the modern section
     order, Hermes-tool framing, no invented commands). Small sources get one
     tight SKILL.md; large prose sources (books, paper stacks, specs, doc
     corpora) get the knowledge-base layout — a lean SKILL.md index plus
     per-chapter ``references/`` files loaded on demand via ``skill_view``
     (the shape popularized by virgiliojr94/book-to-skill).

There is no separate distillation engine and no model-tool footprint: the
agent does the work with its existing toolset, so this works identically on
local, Docker, and remote terminal backends. Every surface (CLI ``/learn``,
gateway ``/learn``, the dashboard "Learn a skill" panel) calls
:func:`build_learn_prompt` and feeds the result to the agent as a normal turn.
"""

from __future__ import annotations

# The house-style rules (authoring + source hygiene) live in
# agent.skill_standards, SHARED with /upskill. Importing from there (rather
# than defining them inline here) means a refactor of either prompt module
# can't silently break the other. We re-export under the legacy underscore
# names so existing tests and callers that import them keep working.
from agent.skill_standards import AUTHORING_STANDARDS, SOURCE_HYGIENE

# Back-compat aliases.
_AUTHORING_STANDARDS = AUTHORING_STANDARDS
_SOURCE_HYGIENE = SOURCE_HYGIENE


# Rules for the expansive shape: a book, a paper stack, a large docs folder, a
# spec — anything too big to distill into one ~200-line file without lossy
# summarization. Modeled on the layout that makes book-to-skill
# (virgiliojr94/book-to-skill, MIT) work: a lean always-loaded index plus
# per-chapter files loaded on demand, so query cost stays proportional to the
# answer instead of the source.
_KNOWLEDGE_SKILL_STANDARDS = """\
Knowledge-base skills (books, paper stacks, large doc corpora, specs):

When the source is a large body of prose rather than a workflow, do NOT cram
it into one SKILL.md and do NOT reduce it to a lossy summary. Author an
expansive skill:

- SKILL.md is a lean core, always loaded in full: the source's central mental
  models and the decision rules worth having in every session, followed by an
  index of every reference file with a one-line "load this when ..."
  description. Keep SKILL.md itself within the normal size bar; the bulk
  lives in `references/`.
- One file per chapter or major topic under `references/` (e.g.
  `references/ch04-replication.md`), each added with `skill_manage`
  write_file. Distill STRUCTURE, not summary: frameworks, definitions,
  decision rules, anti-patterns, key numbers and tables, with
  chapter/section refs back to the source. Bullet-dense, roughly 100-150
  lines per file.
- Process large sources incrementally: inventory the chapters/topics first,
  then read, distill, and persist ONE chapter or topic at a time before moving
  to the next. Never load an entire large corpus into conversation context at
  once. After all units are written, reconcile the SKILL.md index against the
  actual reference files so none are missing or stale.
- Add cross-cutting files when the source earns them: a `references/`
  glossary (terms with chapter refs), patterns/techniques, and a cheatsheet
  of decision tables. Skip any that would be padding.
- SKILL.md must tell the reader to load a chapter on demand with
  `skill_view` (file_path="references/<file>") — reference files cost
  nothing until a question actually needs them.
- Synthesize, never reproduce: the output is structured notes ABOUT the
  source, not a copy of it. No verbatim passages beyond a short quoted
  phrase. This is both the quality bar and the copyright line.
- Fold-in, don't duplicate: if a skill for this source or topic already
  exists, extend it (`skill_manage` patch / write_file) with the new
  material instead of creating a near-duplicate skill."""


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
        "request below, and save it.\n\n"
        f"THE REQUEST:\n{req}\n\n"
        "The request is open-ended and may mix two kinds of content, in any "
        "order: SOURCES to gather (directories, file paths, URLs, \"what we "
        "just did\", pasted notes) AND REQUIREMENTS that shape the skill "
        "(what to focus on, what to leave out, scope, naming, the angle to "
        "take). Treat EVERY part of the request as load-bearing. In "
        "particular, prose that comes after a path or link is NOT incidental "
        "— it is the user telling you what they want from that source. A "
        "request like `<url> focus on the auth flow, skip the deprecated "
        "endpoints` means: gather the URL AND honor \"focus on auth, skip "
        "deprecated\" as authoring requirements. Never fetch the first source "
        "and ignore the rest.\n\n"
        "Do this:\n"
        "1. Inventory every source the user named, using the tools you already "
        "have — `read_file`/`search_files` for local files or directories, "
        "`web_extract` for URLs, the current conversation history if they "
        "referred to something you just did, and the text they pasted as-is. "
        "Gather a small source now. For a large source, inspect enough to map "
        "its chapters or major topics, but do not load the whole corpus into "
        "conversation context; process it incrementally in step 2b. "
        "If the request is ambiguous about scope, make a reasonable choice "
        "and note it; do not stall.\n"
        "1b. Apply every requirement, focus, and constraint in the request to "
        "the skill you author — these govern what the SKILL.md covers and "
        "emphasizes, not just which sources you read.\n"
        "2. Save the skill with `skill_manage`. First check the available "
        "skills for one covering this source or topic. If one exists, load it "
        "with `skill_view`, then extend its SKILL.md with `skill_manage` patch "
        "(or edit for a necessary full rewrite) and add or update supporting "
        "files with `skill_manage` write_file. Only when no matching skill "
        "exists, create one with `skill_manage` action=\"create\" and pick a "
        "sensible category. If the procedure needs a non-trivial script, add "
        "it under the skill's `scripts/` with `skill_manage` write_file and "
        "reference it by relative path.\n"
        "2b. Pick the shape by the source, not by habit: a workflow or small "
        "source gets ONE tight SKILL.md; a book, paper stack, spec, or large "
        "docs corpus gets the knowledge-base layout below — a lean SKILL.md "
        "index plus per-chapter `references/` files added with `skill_manage` "
        "write_file. If a single SKILL.md would force you to summarize away "
        "most of the material, that is the signal to go expansive. For this "
        "layout, create or load the skill after inventorying the source, then "
        "read, distill, and persist one chapter/topic at a time before reading "
        "the next; finish by reconciling the SKILL.md index with every "
        "reference file you wrote.\n\n"
        f"{_SOURCE_HYGIENE}\n\n"
        f"{_AUTHORING_STANDARDS}\n\n"
        f"{_KNOWLEDGE_SKILL_STANDARDS}\n\n"
        "When done, tell the user the skill name, its category, a one-line "
        "summary of what it captured, and — for a knowledge-base skill — the "
        "list of reference files it can load on demand."
    )