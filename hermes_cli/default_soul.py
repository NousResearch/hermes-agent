"""Default SOUL.md template seeded into HERMES_HOME on first run."""

# Kept identical to agent/prompt_builder.py's DEFAULT_AGENT_IDENTITY (#95681,
# maintainer-directed rewrite) -- this is the text virtually every real user
# actually gets, since _ensure_default_soul_md() seeds it into SOUL.md on
# first run. DEFAULT_AGENT_IDENTITY only serves sessions with no SOUL.md at
# all (e.g. skip_context_files), which is not the common case. The old
# "targeted and efficient exploration" line is deliberately absent -- see the
# comment on DEFAULT_AGENT_IDENTITY for why -- never re-add it here either.
DEFAULT_SOUL_MD = (
    "You are Hermes Agent, built by Nous Research. Be direct: match the "
    "length of your reply to the weight of the ask — a one-line question "
    "gets a one-line answer, and finished work gets a short report of what "
    "changed, what's verified, and what's left, never a replay of the "
    "process. No filler (\"Great question,\" \"I'd be happy to\"), no "
    "restating the request back, no re-summarizing what you already said, "
    "no narrating tool calls the user can see. Plain claims over "
    "adjectives; when unsure, say so plainly. Agree because it's right, "
    "not because the user said it. Depth is earned — give it when the "
    "user asks for detail, teaches, or the stakes demand it, not by "
    "default."
)

# Legacy SOUL.md boilerplate that older installers (install.sh / install.ps1 /
# docker/SOUL.md) seeded before they were switched to write DEFAULT_SOUL_MD.
# These templates contain no persona text -- they are pure comment scaffolding,
# so a SOUL.md whose content matches one of these was demonstrably never
# customized by the user and is safe to upgrade to DEFAULT_SOUL_MD in place.
#
# Match on normalized content (stripped, line-endings unified) so trailing
# newlines or CRLF from Windows installers don't defeat the comparison. NEVER
# add anything here that a user might have intentionally written -- the whole
# safety guarantee is that these strings carry zero user intent.
_LEGACY_TEMPLATE_SOULS = (
    (
        "# Hermes Agent Persona\n"
        "\n"
        "<!--\n"
        "This file defines the agent's personality and tone.\n"
        "The agent will embody whatever you write here.\n"
        "Edit this to customize how Hermes communicates with you.\n"
        "\n"
        "Examples:\n"
        '  - "You are a warm, playful assistant who uses kaomoji occasionally."\n'
        '  - "You are a concise technical expert. No fluff, just facts."\n'
        '  - "You speak like a friendly coworker who happens to know everything."\n'
        "\n"
        "This file is loaded fresh each message -- no restart needed.\n"
        "Delete the contents (or this file) to use the default personality.\n"
        "-->"
    ),
    # docker/SOUL.md and the install.sh heredoc differ only by an "Examples"
    # block / trailing newline in some historical revisions; the bare scaffold
    # (no Examples block) was also shipped briefly.
    (
        "# Hermes Agent Persona\n"
        "\n"
        "<!--\n"
        "This file defines the agent's personality and tone.\n"
        "The agent will embody whatever you write here.\n"
        "Edit this to customize how Hermes communicates with you.\n"
        "\n"
        "This file is loaded fresh each message -- no restart needed.\n"
        "Delete the contents (or this file) to use the default personality.\n"
        "-->"
    ),
    # The pre-#95681 DEFAULT_SOUL_MD text: every install between that text's
    # introduction and this fix got it auto-seeded on first run, so it also
    # carries zero user intent (it's the same auto-seed mechanism, just an
    # older generation of the same non-customized string) and is safe to
    # upgrade in place, same as the comment-only scaffolds above.
    (
        "You are Hermes Agent, an intelligent AI assistant created by Nous "
        "Research. You are helpful, knowledgeable, and direct. You assist "
        "users with a wide range of tasks including answering questions, "
        "writing and editing code, analyzing information, creative work, "
        "and executing actions via your tools. You communicate clearly, "
        "admit uncertainty when appropriate, and prioritize being "
        "genuinely useful over being verbose unless otherwise directed "
        "below. Be targeted and efficient in your exploration and "
        "investigations."
    ),
)


def _normalize_soul(text: str) -> str:
    """Normalize SOUL.md content for legacy-template comparison."""
    # Unify line endings (Windows installer writes CRLF-free but be defensive),
    # strip a leading UTF-8 BOM, and trim surrounding whitespace.
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff").strip()


def is_legacy_template_soul(text: str) -> bool:
    """True if ``text`` is a non-customized, auto-seeded SOUL.md.

    Covers two generations of non-user-authored content: older installers'
    comment-only scaffold (which shadowed the runtime default and left users
    with no persona), and the pre-#95681 generation of DEFAULT_SOUL_MD itself
    (auto-seeded, never edited). A file matching one of those known strings
    carries zero user intent and is safe to upgrade in place. Any deviation
    (the user typed a persona, even one character outside the comment) makes
    this return False.
    """
    normalized = _normalize_soul(text)
    return any(normalized == _normalize_soul(t) for t in _LEGACY_TEMPLATE_SOULS)
