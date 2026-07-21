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
  2. Load the bundled Hermes authoring skill and contract without executing
     inline shell.
  3. Inspect existing skills, choose update/create/consolidate/skip, then apply
     and validate only the selected decision.

There is no separate distillation engine. Prompt construction uses the shared
raw bundled-guidance loader, then the agent does the work with its existing
toolset, so this works identically on local, Docker, and remote terminal
backends. Every surface (CLI ``/learn``, gateway ``/learn``, the dashboard
"Learn a skill" panel) calls :func:`build_learn_prompt` and feeds the result to
the agent as a normal turn.
"""

from __future__ import annotations

from agent.skill_authoring_guidance import (
    load_bundled_skill_authoring_guidance,
)

# This is deliberately compact. The bundled authoring skill and its linked
# contract are the primary source of truth; this text only keeps /learn safe
# and useful when bundled guidance is missing, disabled, or invalid.
_FALLBACK_AUTHORING_GUIDANCE = """\
The bundled Hermes authoring guidance is unavailable. Use this conservative
fallback:

- Inspect existing skills with `skills_list`, then read plausible owners with
  `skill_view` before writing anything.
- Choose exactly one outcome: update an existing owner, create a distinct
  reusable skill, consolidate overlapping skills, or skip when the behavior is
  already covered or not reusable.
- Use a lowercase hyphenated name that matches its directory. Keep the
  description to one capability sentence of at most 60 characters ending in a
  period; avoid marketing language.
- Preserve human authorship. Credit the human contributor first and Hermes
  second when both contributed. Never derive an author from host identity.
- Use the modern body order: title and scope, `When to Use`, `Prerequisites`,
  `How to Run`, `Quick Reference`, `Procedure`, `Pitfalls`, and `Verification`.
- Name Hermes native tools in prose. Put repeated deterministic logic in
  `scripts/`, detailed material in `references/`, and reusable skeletons in
  `templates/`.
- Validate frontmatter, links, platform claims, helpers, and behavior. Ask
  before an unrequested deletion or other destructive consolidation step.
"""


def _load_authoring_guidance() -> str:
    """Format raw bundled authoring files with compact fallbacks.

    The shared loader reads the exact bundled path without ``skill_view``, so
    neither inline-shell preprocessing nor a user-local same-name skill can
    affect ``/learn``.
    """
    try:
        guidance = load_bundled_skill_authoring_guidance()
    except Exception:
        guidance = None
    if not guidance:
        return _FALLBACK_AUTHORING_GUIDANCE

    parts = [
        "AUTHORITATIVE HERMES AUTHORING SKILL:\n\n"
        + guidance.skill_content.strip(),
    ]
    if guidance.contract_content:
        parts.append(
            "AUTHORITATIVE HERMES AUTHORING CONTRACT:\n\n"
            + guidance.contract_content.strip()
        )
    else:
        parts.append(
            "The linked authoring contract could not be loaded. Apply this "
            "fallback in addition to the skill:\n\n"
            + _FALLBACK_AUTHORING_GUIDANCE
        )
    return "\n\n".join(parts)


def build_learn_prompt(user_request: str) -> str:
    """Build the agent prompt for an open-ended ``/learn`` request.

    Args:
        user_request: the free-text the user gave after ``/learn`` — a
            description of the workflow, paths, URLs, or "what I just did".

    Returns:
        A complete instruction the agent runs as a normal turn. The agent
        gathers the described sources, selects an authoring decision, and uses
        the appropriate skill-management operation when a write is warranted.
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
        "1. Gather every source the user named, using the tools you already "
        "have — `read_file`/`search_files` for local files or directories, "
        "`web_extract` for URLs, the current conversation history if they "
        "referred to something you just did, and the text they pasted as-is. "
        "If the request is ambiguous about scope, make a reasonable choice "
        "and note it; do not stall.\n"
        "1b. Apply every requirement, focus, and constraint in the request to "
        "the skill you author — these govern what the SKILL.md covers and "
        "emphasizes, not just which sources you read.\n"
        "2. Before writing, inspect the installed library with `skills_list` "
        "and read every plausible owner with `skill_view`. Search repository "
        "and Hub skills too when those sources are in scope.\n"
        "3. Choose exactly one decision from the evidence: UPDATE an existing "
        "owner, CREATE a distinct reusable skill, CONSOLIDATE overlapping "
        "skills under one owner, or SKIP when the behavior is already covered "
        "or is not reusable. Do not default to creation.\n"
        "4. Execute only that decision. Use `skill_manage` patch/edit for an "
        "installed update, create for a genuinely new user-local skill, and "
        "write_file for needed references/scripts/templates. Update the "
        "umbrella before deleting a duplicate and obtain confirmation before "
        "an unrequested deletion. A SKIP decision performs no write.\n"
        "5. Validate the result as required by the authoring guidance below. "
        "Do not invent commands, paths, APIs, or verification results.\n\n"
        f"{_load_authoring_guidance()}\n\n"
        "When done, report the UPDATE/CREATE/CONSOLIDATE/SKIP decision, the "
        "skill name and category when applicable, what changed, and the "
        "validation evidence."
    )
