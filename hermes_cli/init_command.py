#!/usr/bin/env python3
"""``/init`` — build the prompt that generates or updates a project AGENTS.md.

Port of Codex ``/init`` (Claude Code has the same for CLAUDE.md). Hermes
already *loads* AGENTS.md / CLAUDE.md / .cursorrules as project context, but
had no command to bootstrap one. ``/init`` hands the live agent ONE
guidance-laden prompt instructing it to:

  1. Inspect the project with its own read-only tools (``read_file`` /
     ``search_files`` on manifests, CI configs, lockfiles, existing docs) to
     learn the layout, toolchain, and the exact build/test/lint commands.
  2. Write a CONCISE ``AGENTS.md`` (target under 100 lines) with the sections
     an agent actually needs — overview, setup, commands, conventions,
     pitfalls — not an essay.
  3. If an AGENTS.md already exists, UPDATE it: preserve the user's existing
     content and merge in what's missing, never blow it away.

There is no engine and no model-tool footprint: the agent does the work with
its existing toolset, so this works identically on local, Docker, and remote
terminal backends. Every surface (CLI ``/init``, gateway ``/init``, TUI
``/init``) calls :func:`build_init_prompt` and feeds the result to the agent
as a normal user turn — the same prompt-injection pattern as ``/learn`` and
``/blueprint``, which preserves prompt-cache invariants (no system-prompt or
history mutation).
"""

from __future__ import annotations

# The quality bar, embedded in every prompt so the generated file reads like a
# maintainer wrote it — concrete and command-exact, not generic advice.
_QUALITY_BAR = """\
Quality bar for the file you write (this is what separates a useful AGENTS.md
from noise):
- CONCISE: target under 100 lines. Agents load this file every session — every
  line costs context. No essays, no marketing prose, no filler.
- Commands must be EXACT invocations you verified from the repo (package.json
  scripts, Makefile targets, pyproject/tox/CI config, existing docs). Write
  `npm run test:unit` or `scripts/run_tests.sh tests/foo`, never "run the
  tests". NEVER invent a command you didn't see evidence for.
- No generic advice. "Write tests for new code" and "follow best practices"
  are banned — if a line would be true of any repo, cut it.
- Conventions must be OBSERVED, not assumed: naming patterns, module layout,
  error-handling style, commit-message format — only what the code actually
  shows.
- Include pitfalls that would genuinely trip up a newcomer or an agent
  (required env vars, generated files not to hand-edit, slow test suites,
  ports already in use), if you found any. Skip the section if you found none.
- Markdown structure: a short title + one-paragraph overview, then focused
  sections (e.g. "Dev environment", "Build & test", "Conventions",
  "Pitfalls"). Flat and scannable — no deep nesting."""


def build_init_prompt(
    cwd: str,
    existing_file: str | None = None,
    extra: str = "",
) -> str:
    """Build the agent prompt for a ``/init`` request.

    Args:
        cwd: the project directory the agent should scan and write
            ``AGENTS.md`` into (usually the session working directory).
        existing_file: the current content of ``AGENTS.md`` if one already
            exists, else ``None``. When present the prompt switches to
            update-and-merge discipline instead of fresh generation.
        extra: free-text the user gave after ``/init`` — emphasis or notes to
            honor while authoring (e.g. "focus on the test setup").

    Returns:
        A complete instruction the agent runs as a normal turn.
    """
    extra = (extra or "").strip()

    parts: list[str] = [
        "[/init] The user wants you to "
        + (
            "UPDATE the existing AGENTS.md project-instructions file"
            if existing_file is not None
            else "generate an AGENTS.md project-instructions file"
        )
        + f" for the project at: {cwd}\n",
        "AGENTS.md is the instruction file coding agents (Hermes included) "
        "load as project context every session. It should teach an agent how "
        "to work in THIS repo: what the project is, how to set up, the exact "
        "build/test/lint commands, the conventions the code actually follows, "
        "and the pitfalls that waste time.\n",
        "Do this:\n"
        "1. Inspect the project with your read-only tools (`read_file`, "
        "`search_files`) — start with manifests and toolchain files "
        "(package.json, pyproject.toml, Cargo.toml, go.mod, Makefile, "
        "CI workflow configs, lockfiles), then the directory layout, existing "
        "README/docs, and test/lint configuration. Learn the real commands, "
        "don't guess them.\n"
        "2. Write the file to "
        f"{cwd.rstrip('/')}/AGENTS.md with `write_file`"
        + (
            " — but this is an UPDATE, so follow the merge discipline below."
            if existing_file is not None
            else "."
        )
        + "\n"
        "3. Confirm to the user the exact path you wrote and summarize in one "
        "or two lines what the file covers.\n",
    ]

    if existing_file is not None:
        parts.append(
            "MERGE DISCIPLINE — an AGENTS.md already exists (its current "
            "content is below). Do NOT overwrite or regenerate it from "
            "scratch. Preserve the user's existing content — their wording, "
            "their sections, their rules — and merge in only what is missing "
            "or verifiably stale (e.g. a command that no longer exists in the "
            "repo). When existing content conflicts with what you observed, "
            "prefer minimal surgical edits over rewrites, and keep the "
            "user's intent. The result must still meet the quality bar.\n\n"
            "CURRENT AGENTS.md CONTENT:\n"
            "<<<EXISTING_AGENTS_MD\n"
            f"{existing_file}\n"
            "EXISTING_AGENTS_MD\n"
        )

    parts.append(_QUALITY_BAR)

    if extra:
        parts.append(
            "\nUSER NOTES — honor these while authoring (they override the "
            f"defaults above where they conflict):\n{extra}"
        )

    return "\n".join(parts)


def build_init_prompt_for_cwd(cwd: str | None = None, extra: str = "") -> str:
    """Convenience wrapper used by the dispatch surfaces.

    Resolves ``cwd`` (defaults to the process working directory), reads an
    existing ``AGENTS.md`` there if present, and returns the full prompt.
    """
    import os

    resolved = os.path.abspath(cwd or os.getcwd())
    existing: str | None = None
    agents_path = os.path.join(resolved, "AGENTS.md")
    try:
        if os.path.isfile(agents_path):
            with open(agents_path, encoding="utf-8", errors="replace") as fh:
                existing = fh.read()
    except OSError:
        existing = None
    return build_init_prompt(resolved, existing_file=existing, extra=extra)
