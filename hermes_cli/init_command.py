"""``/init`` — build the prompt that asks the agent to generate or update a project AGENTS.md."""

from __future__ import annotations

import os

# Embedded in every prompt so the generated file reads like a maintainer wrote it.
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


def build_init_prompt(cwd: str, existing_file: str | None = None, extra: str = "") -> str:
    """Build the ``/init`` prompt; ``existing_file`` (current AGENTS.md) switches to merge discipline,
    ``extra`` is the user's free text after ``/init``."""
    extra = (extra or "").strip()
    update = existing_file is not None
    parts: list[str] = [
        "[/init] The user wants you to "
        + ("UPDATE the existing" if update else "generate an")
        + f" AGENTS.md project-instructions file for the project at: {cwd}\n",
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
        + (" — but this is an UPDATE, so follow the merge discipline below." if update else ".")
        + "\n"
        "3. Confirm to the user the exact path you wrote and summarize in one "
        "or two lines what the file covers.\n",
    ]
    if update:
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


def _resolve_session_cwd(session_key: str | None) -> str:
    """The session's ACTIVE directory — where /init should scan and write.

    Ladder: the terminal tool's per-session cwd record (seeded when a surface attaches a
    workspace to the session — desktop project picker, ``project_create``/``project_switch``,
    gateway ``terminal.cwd`` — and updated after every completed command, so it tracks
    ``cd``), then ``agent.runtime_cwd.resolve_agent_cwd()`` (session-pinned cwd →
    ``TERMINAL_CWD`` → process cwd). A bare ``os.getcwd()`` is only right for a CLI launched
    inside a project: on the desktop app it is the home directory, so /init scanned and
    updated the HOME's AGENTS.md instead of the workspace attached to the session.

    ``session_key`` is the surface's own key (multi-session hosts must pass it); an empty
    string reads the single-session CLI's ``"default"`` record, and ``None`` falls back to
    the ambient ``HERMES_SESSION_KEY``.
    """
    try:
        from gateway.session_context import get_session_env
        from tools.terminal_tool import get_session_cwd

        key = session_key if session_key is not None else get_session_env("HERMES_SESSION_KEY", "")
        recorded = get_session_cwd(key)
        if recorded and os.path.isdir(recorded):
            return recorded
    except Exception:
        pass
    try:
        from agent.runtime_cwd import resolve_agent_cwd

        return str(resolve_agent_cwd())
    except Exception:
        return os.getcwd()


def build_init_prompt_for_cwd(cwd: str | None = None, extra: str = "",
                              session_key: str | None = None) -> str:
    """Convenience wrapper used by the dispatch surfaces.

    An explicit ``cwd`` wins while it still exists; a stale one (deleted project, removed
    worktree) falls through to :func:`_resolve_session_cwd` — the guard lives here so every
    dispatch surface shares it rather than each one validating its own record.
    """
    resolved = os.path.abspath(cwd if cwd and os.path.isdir(cwd) else _resolve_session_cwd(session_key))
    existing: str | None = None
    agents_path = os.path.join(resolved, "AGENTS.md")
    try:
        if os.path.isfile(agents_path):
            with open(agents_path, encoding="utf-8", errors="replace") as fh:
                existing = fh.read()
    except OSError:
        existing = None
    return build_init_prompt(resolved, existing_file=existing, extra=extra)
