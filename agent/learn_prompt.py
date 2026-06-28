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
Follow the Hermes skill-authoring standards exactly. These are the same
HARDLINE rules a maintainer enforces in review:

Frontmatter:
- name: lowercase-hyphenated, <=64 chars, no spaces.
- description: ONE sentence, **<=60 characters**, ends with a period. State the
  capability, not the implementation. No marketing words (powerful,
  comprehensive, seamless, advanced, robust). Do NOT repeat the skill name. If
  the description contains a colon, wrap the whole value in double quotes.
  This is the most-violated rule and it is NOT cosmetic: the system-prompt
  skill index truncates the description to 60 chars and loads it every
  session, so anything past char 60 is silently cut and never routes. After
  you write the description, COUNT the characters; if it is over 60, cut it
  down before saving — do not ship a sentence and hope.
    Good (<=60): `Search arXiv papers by keyword, author, or ID.`
    Bad (123):   `A comprehensive skill that lets the agent search arXiv for
                  academic papers using keywords, authors, and categories.`
- version: 0.1.0
- author: always the literal value `Hermes`. NEVER fill it from the host
  environment — the OS/login username (e.g. the `user=` line in your
  environment hints), git config, or any identity you can probe must not be
  written. Skills get shared and published, so an environment-derived name is
  a privacy leak the user never opted into; the skill names itself as Hermes.
- platforms: declare `[macos]`, `[linux]`, and/or `[windows]` IF the skill
  uses OS-bound primitives (osascript/apt/systemctl => the matching OS; /proc,
  os.setsid, signal.SIGKILL => linux; fcntl/termios => POSIX). Prefer fixing it
  cross-platform first (tempfile.gettempdir(), pathlib.Path, psutil); gate only
  when the dependency is genuinely platform-bound. Omit the field for portable
  skills.
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
- Reference Hermes tools by name in backticks: `terminal`, `read_file`,
  `write_file`, `search_files`, `patch`, `web_extract`, `web_search`,
  `vision_analyze`, `browser_navigate`, `delegate_task`, `image_generate`,
  `text_to_speech`, `cronjob`, `memory`, `skill_view`, `execute_code`.
- Do NOT name shell utilities the agent already has wrapped: say `read_file`
  not cat/head/tail, `search_files` not grep/rg/find/ls, `patch` not sed/awk,
  `web_extract` not curl-to-scrape, `write_file` not echo>file or heredocs.
- Third-party CLIs (ffmpeg, gh, an SDK) are fine inside a script file, but the
  prose still frames them as "invoke through the `terminal` tool". If the
  skill needs an MCP server, name it and document its setup in Prerequisites.

Quality bar:
- Prefer exact commands, endpoint URLs, function signatures, and config keys
  that appear VERBATIM in the source. NEVER invent flags, paths, or APIs — if
  you didn't see it in the source, don't write it.
- Keep it tight and scannable: ~100 lines for a simple skill, ~200 for a
  complex one. Don't re-paste the source docs.
- Don't write a router/index/hub skill that only points at other skills.
- Larger scripts/parsers belong in a `scripts/` file (add via
  `skill_manage` write_file), referenced from SKILL.md by relative path — not
  inlined for the agent to re-type every run. References go in `references/`,
  templates in `templates/`."""


def parse_learn_request(user_request: str) -> tuple[str | None, str]:
    """Split a ``/learn`` argument into ``(update_target, remaining_text)``.

    ``/learn`` accepts exactly one flag — ``--update <skill>`` — and only when
    it leads the argument. This is the single source of truth for that parsing
    so the CLI, gateway, and TUI surfaces never re-implement it.

    Returns:
        ``(None, stripped_request)`` for an ordinary create-mode request, so
        the create path stays byte-identical to the pre-``--update`` behavior.
        ``(skill_name, notes)`` when ``--update`` leads the argument;
        ``skill_name`` is ``""`` when the user typed ``--update`` without
        naming a skill (still update mode — never create).
    """
    req = (user_request or "").strip()
    if not req:
        return None, ""

    tokens = req.split(None, 1)
    first = tokens[0]
    rest = tokens[1].strip() if len(tokens) > 1 else ""

    if first == "--update":
        if not rest:
            return "", ""
        name_parts = rest.split(None, 1)
        target = name_parts[0]
        notes = name_parts[1].strip() if len(name_parts) > 1 else ""
        return target, notes

    return None, req


def _build_update_prompt(skill_name: str, notes: str) -> str:
    """Build the agent prompt for ``/learn --update <skill> [notes]``.

    Update mode edits an EXISTING skill in place: read it with ``skill_view``,
    then apply ``skill_manage`` ``patch`` (small fixes) or ``edit`` (major
    rewrite). It never authors a new skill. Carries the same authoring
    standards so an edited skill still matches house style.
    """
    name = (skill_name or "").strip()
    notes = (notes or "").strip()

    if name:
        target_line = f"SKILL TO UPDATE: {name}\n\n"
        read_step = (
            f"1. Read the current skill first with `skill_view(\"{name}\")` so "
            "you edit the real content rather than a guess. If no skill by "
            "that name exists, say so and use `skills_list` to help the user "
            "pick the right one — do not author a brand-new skill.\n"
        )
    else:
        target_line = "SKILL TO UPDATE: (the user did not name one)\n\n"
        read_step = (
            "1. The user ran `/learn --update` without naming a skill. Use "
            "`skills_list` to find the one they mean (ask if it is ambiguous), "
            "then read it with `skill_view` — do not author a brand-new "
            "skill.\n"
        )

    if notes:
        change_line = f"WHAT TO CHANGE:\n{notes}\n\n"
    else:
        change_line = (
            "WHAT TO CHANGE:\nThe user did not spell out the change — infer the "
            "stale, missing, or incorrect parts from the current skill and the "
            "conversation so far (errors hit, steps that were wrong or "
            "missing) and improve those.\n\n"
        )

    return (
        "[/learn --update] The user wants you to update an EXISTING skill in "
        "place, not author a new one.\n\n"
        f"{target_line}"
        f"{change_line}"
        "Do this:\n"
        f"{read_step}"
        "2. Apply the change with the `skill_manage` tool. Prefer "
        "action=\"patch\" (old_string/new_string) for small, targeted fixes — "
        "a typo, an added pitfall, a corrected command, a tightened trigger. "
        "Use action=\"edit\" (full SKILL.md rewrite) ONLY for a major "
        "overhaul. Never author a brand-new skill in update mode, and do not "
        "change the skill's name or directory.\n\n"
        f"{_AUTHORING_STANDARDS}\n\n"
        "When done, tell the user the skill name and a one-line summary of "
        "what you changed."
    )


def build_learn_prompt(user_request: str) -> str:
    """Build the agent prompt for an open-ended ``/learn`` request.

    Args:
        user_request: the free-text the user gave after ``/learn`` — a
            description of the workflow, paths, URLs, or "what I just did".
            A leading ``--update <skill>`` switches to in-place update mode
            (edit an existing skill via ``skill_manage`` patch/edit) instead
            of authoring a new one.

    Returns:
        A complete instruction the agent runs as a normal turn. The agent
        gathers the described sources with its existing tools and authors the
        skill via ``skill_manage``.
    """
    update_target, notes = parse_learn_request(user_request)
    if update_target is not None:
        return _build_update_prompt(update_target, notes)

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
