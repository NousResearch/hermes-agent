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

from dataclasses import dataclass
from enum import Enum

# Per-skill content ceiling enforced by ``skill_manage``
# (``MAX_SKILL_CONTENT_CHARS`` in ``tools/skill_manager_tool.py``). Named here so
# the decomposition threshold can refer to it explicitly in the prompt.
MAX_SKILL_CONTENT_CHARS = 100_000

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


def _baseline_single_skill_prompt(user_request: str) -> str:
    """Build the open-ended single-skill ``/learn`` prompt (upstream baseline).

    This is the original ``/learn`` behavior: gather the described sources and
    author ONE ``SKILL.md`` via ``skill_manage``. It is reused as the base for
    AUTO and SINGLE modes; DECOMPOSE and UPDATE assemble their own prompts.

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


# ---------------------------------------------------------------------------
# Decomposition & delta-update enhancement
#
# `/learn` accepts optional flags that layer two capabilities on top of the
# open-ended baseline above, without adding any new model-facing tool:
#   --decompose / --no-decompose : force or forbid a parent+children hierarchy
#   --update <name>              : delta-update an existing skill (also the
#                                  natural-language "update <name> with ...")
#   --dry-run                    : preview the plan, write nothing
# Flags are recognized as whitespace-delimited tokens anywhere in the request;
# unrecognized --flags pass through untouched as part of the source text.
# ---------------------------------------------------------------------------


class LearnMode(str, Enum):
    """Resolved authoring mode for a ``/learn`` request."""

    AUTO = "auto"            # no flag: single by default, decompose if large
    SINGLE = "single"        # --no-decompose: force one skill
    DECOMPOSE = "decompose"  # --decompose: force a hierarchy
    UPDATE = "update"        # --update <name> / "update <name> with ..."


@dataclass(frozen=True)
class LearnDirective:
    """Parsed intent extracted from a ``/learn`` request. Pure data, no I/O."""

    mode: LearnMode
    source_text: str                 # request text with recognized flags removed
    update_target: str | None = None
    dry_run: bool = False
    conflicting_flags: bool = False   # --decompose AND --no-decompose both given


_FLAG_DECOMPOSE = "--decompose"
_FLAG_NO_DECOMPOSE = "--no-decompose"
_FLAG_DRY_RUN = "--dry-run"
_FLAG_UPDATE = "--update"


def parse_learn_request(user_request: str) -> LearnDirective:
    """Extract control flags and free text from a ``/learn`` request.

    Deterministic and side-effect free. Recognizes ``--decompose``,
    ``--no-decompose``, ``--update <name>`` and ``--dry-run`` as
    whitespace-delimited tokens anywhere in the text, plus the natural-language
    ``update <name> with <sources>`` phrasing. Recognized flags are stripped;
    the remainder becomes ``source_text``. Unrecognized ``--flags`` are left
    untouched in ``source_text``.

    Mode precedence: ``--update`` > ``--no-decompose`` (sets
    ``conflicting_flags`` when ``--decompose`` is also present) > ``--decompose``
    > AUTO. ``dry_run`` is orthogonal. ``--update`` with no following name
    yields no target (mode falls through to AUTO).
    """
    text = user_request if isinstance(user_request, str) else ""
    tokens = text.split()

    decompose = False
    no_decompose = False
    dry_run = False
    update_target: str | None = None

    remaining: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        token = tokens[i]
        if token == _FLAG_DECOMPOSE:
            decompose = True
        elif token == _FLAG_NO_DECOMPOSE:
            no_decompose = True
        elif token == _FLAG_DRY_RUN:
            dry_run = True
        elif token == _FLAG_UPDATE:
            if i + 1 < n and not tokens[i + 1].startswith("--"):
                update_target = tokens[i + 1]
                i += 1
        else:
            remaining.append(token)
        i += 1

    # Natural-language form: "update <name> with <sources>".
    if (
        update_target is None
        and len(remaining) >= 3
        and remaining[0].lower() == "update"
        and remaining[2].lower() == "with"
    ):
        update_target = remaining[1]
        source_text = " ".join(remaining[3:])
    else:
        source_text = " ".join(remaining)

    conflicting_flags = decompose and no_decompose

    if update_target is not None:
        mode = LearnMode.UPDATE
    elif no_decompose:
        mode = LearnMode.SINGLE
    elif decompose:
        mode = LearnMode.DECOMPOSE
    else:
        mode = LearnMode.AUTO

    return LearnDirective(
        mode=mode,
        source_text=source_text,
        update_target=update_target,
        dry_run=dry_run,
        conflicting_flags=conflicting_flags,
    )


# Fallback source description for an empty request in decompose/update modes
# (the baseline single-skill prompt has its own conversation fallback).
_CONVERSATION_FALLBACK = (
    "the workflow we just went through in this conversation - review the steps "
    "taken and distill them into reusable skills"
)

# Decomposition decision rule (AUTO mode). Defines the Decomposition Threshold
# relative to MAX_SKILL_CONTENT_CHARS and the distinct-topic count, and states
# the single-skill default below the threshold.
_THRESHOLD_RULE = """\
DECOMPOSITION CHECK (automatic): before authoring, estimate the scope. Apply the
Decomposition Threshold, measured by two signals: the estimated authored content
size relative to MAX_SKILL_CONTENT_CHARS (100,000 characters), and the count of
distinct single-responsibility topics the material covers. Decompose the material
into a skill hierarchy WHEN the estimated authored content would exceed
MAX_SKILL_CONTENT_CHARS (100,000 characters) OR would span more than one distinct
single-responsibility topic. Otherwise author a single SKILL.md via the
`skill_manage` action `create` (the default). Aim for approximately 200 lines per
skill; relocate overflow into `references/`/`templates/`/`scripts/`/`assets/`."""

# Parent/child structure + hierarchy frontmatter, shared by AUTO (conditional)
# and DECOMPOSE (forced).
_DECOMP_GUIDANCE = """\
HIERARCHY RULES (when you decompose):
- Author ONE Parent Skill with an overview, a `## When to Use` section, and links
  to each Child Skill. The Parent Skill orchestrates the hierarchy.
- Author each Child Skill as a focused, single-responsibility skill covering
  exactly one topic.
- Direct the Parent Skill to load Child Skills on demand through Progressive
  Disclosure, so only the relevant sub-skill loads.
- Follow the modern SKILL.md section order for every skill, parent and children
  alike.
- Record hierarchy/dependency links strictly under `metadata.hermes.*`: child
  names under `metadata.hermes.children` on the parent, the parent name under
  `metadata.hermes.parent` on each child, and any cross-skill dependencies under
  `metadata.hermes.depends_on`. Do not introduce any new top-level frontmatter
  keys for these relationships."""

# Child-reference partitioning guidance. Emitted alongside `_DECOMP_GUIDANCE`
# for both AUTO (conditional decomposition path) and DECOMPOSE (forced) modes,
# so any prompt that instructs decomposition also instructs partitioning the
# gathered source material across the parent and children. Distilled from
# Requirement 14 and the `skill_manage` write_file constraints in
# `tools/skill_manager_tool.py` (ALLOWED_SUBDIRS, MAX_SKILL_FILE_BYTES).
_CHILD_REFERENCE_GUIDANCE = """\
CHILD-REFERENCE PARTITIONING (when you decompose):
- Partition the gathered source material across the Parent Skill and the Child
  Skills by single responsibility, so each piece of source content lives in
  exactly ONE skill. Do NOT duplicate the same source content across skills.
- Place in each Child Skill's OWN `references/` folder only the partitioned
  source material relevant to that child's single responsibility, and write each
  child's reference files with the `skill_manage` action `write_file`.
- Keep the Parent Skill limited to a high-level overview plus pointers to the
  child skills and their references. EXCLUDE the full body of any child reference
  from the Parent Skill — the parent links to a child's references, it does not
  copy them.
- Respect BOTH size caps: the 100,000-character `SKILL.md` content cap AND the
  separate 1 MiB (1,048,576-byte) per-file cap (`MAX_SKILL_FILE_BYTES`) that
  applies to each supporting file. If a child's partitioned references would
  exceed 1,048,576 bytes in a single file, split that material across multiple
  files under the child's `references/` folder.
- Supporting files are confined to ALLOWED_SUBDIRS = {references, templates,
  scripts, assets}: `skill_manage` `write_file`/`remove_file` reject any path
  outside those folders (and reject `..` traversal), so write every child
  reference under that child's own `references/` folder.
- Preserve three-level Progressive Disclosure for child references: Level 0
  `skills_list()` (the catalog), Level 1 `skill_view(name)` (a skill's SKILL.md
  body), and Level 2 `skill_view(name, "references/specific-file.md")` (one
  reference file within a skill).
- Author each Child Skill together with its references so the child is
  independently loadable and usable on its own, without first loading the
  Parent Skill."""

# Delta-update guidance (UPDATE mode).
_UPDATE_GUIDANCE = """\
DELTA UPDATE: change only what the new sources affect; preserve everything else.
- Read the existing skill before modifying it. If the named skill does not exist,
  report that the skill was not found rather than create a new skill.
- Identify the changed sections or references implied by the new sources.
- Apply changes with the `skill_manage` action `patch` or `write_file` for
  targeted edits. Reserve the `edit` action for a full rewrite.
- Leave unchanged sections and references untouched.
- Increment the `version` frontmatter field of each modified skill; if a modified
  skill has no `version`, add one set to an initial value. Record a summary of the
  change in each modified skill.
- If the named skill is a Parent Skill, read it and its children listed under
  `metadata.hermes.children`, apply delta updates only to the children affected by
  the new sources, and keep `metadata.hermes.children`/`metadata.hermes.parent`
  consistent when a child is added or removed.
- When the new sources change a Child Skill's partitioned content (its Child
  References), add, update, or remove the affected files within THAT child's own
  `references/` folder using the `skill_manage` actions `write_file`, `patch`, or
  `remove_file`. Leave any reference files the new sources do not touch unchanged.
- When a child's Child References change, update the Parent Pointers in the Parent
  Skill so the parent's links to the affected child references stay consistent —
  add links for new reference files, repoint links whose target moved or was
  renamed, and drop links to references that were removed. The Parent Skill keeps
  pointing at the children's references; it does not copy their content."""

# Dry-run wrapper (any mode).
_DRYRUN_GUIDANCE = """\
DRY RUN (preview only): do NOT modify any skill on disk. List the planned skills
to create, the planned patches to apply, and the planned supporting files. Invoke
no write action of `skill_manage` (do not call `create`, `edit`, `patch`,
`delete`, `write_file`, or `remove_file`). When the plan involves decomposition,
include the planned Parent Skill and Child Skill names in the preview."""


def _gather_and_standards_block(req: str) -> str:
    """Shared gather-sources + authoring-standards + save block.

    Used by the decompose/update prompts so they carry the same invariants as
    the baseline single-skill prompt (gather tools named, `skill_manage`-only
    writes, the full HARDLINE standards, and the source text embedded verbatim).
    """
    return (
        f"WHAT TO LEARN FROM:\n{req}\n\n"
        "Gather the material with the tools you already have - `read_file` and "
        "`search_files` for local files or directories, `web_extract` for URLs, "
        "the current conversation history if they referred to something you just "
        "did, and the text they pasted as-is. Save every skill with the "
        "`skill_manage` tool; do not write skill files with any other tool and do "
        "not introduce a new tool.\n\n"
        f"{_AUTHORING_STANDARDS}"
    )


def build_learn_prompt(user_request: str) -> str:
    """Construct the standards-guided ``/learn`` prompt.

    Pure, total, and deterministic: a single ``str`` in, a non-empty ``str`` out;
    the same input always yields byte-identical output and the function never
    raises. With no flags it reproduces the open-ended single-skill baseline
    (plus an automatic decomposition-threshold check); flags switch it to a
    forced hierarchy, a forced single skill, or a delta update, and ``--dry-run``
    wraps any mode as a write-free preview.
    """
    directive = parse_learn_request(user_request)

    if directive.mode is LearnMode.UPDATE and directive.update_target is not None:
        req = directive.source_text.strip() or _CONVERSATION_FALLBACK
        prompt = (
            f"[/learn] Delta-update the existing skill `{directive.update_target}` "
            "from the source(s) described below.\n\n"
            f"{_gather_and_standards_block(req)}\n\n"
            f"{_UPDATE_GUIDANCE}"
        )
    elif directive.mode is LearnMode.DECOMPOSE:
        req = directive.source_text.strip() or _CONVERSATION_FALLBACK
        prompt = (
            "[/learn] Decompose the source(s) described below into a modular skill "
            "hierarchy (a parent skill plus focused child skills) rather than one "
            "monolithic skill.\n\n"
            f"{_gather_and_standards_block(req)}\n\n"
            f"{_DECOMP_GUIDANCE}\n\n"
            f"{_CHILD_REFERENCE_GUIDANCE}"
        )
    elif directive.mode is LearnMode.SINGLE:
        prompt = _baseline_single_skill_prompt(directive.source_text)
        note = (
            "\n\nSINGLE SKILL (--no-decompose): author exactly one skill; do not "
            "split the material into a hierarchy."
        )
        if directive.conflicting_flags:
            note += (
                " Note: both `--decompose` and `--no-decompose` were supplied; "
                "these flags conflict, so `--no-decompose` takes precedence and a "
                "single skill is authored."
            )
        prompt += note
    else:  # AUTO
        prompt = (
            f"{_baseline_single_skill_prompt(directive.source_text)}\n\n"
            f"{_THRESHOLD_RULE}\n\n"
            f"{_DECOMP_GUIDANCE}\n\n"
            f"{_CHILD_REFERENCE_GUIDANCE}"
        )

    if directive.dry_run:
        prompt += f"\n\n{_DRYRUN_GUIDANCE}"

    return prompt
