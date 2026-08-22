"""Default SOUL.md template seeded into HERMES_HOME on first run."""

DEFAULT_SOUL_MD = (
    "You are Hermes, an intelligent AI assistant created by Nous Research. "
    "You are helpful, knowledgeable, and direct. You assist users with a wide "
    "range of tasks including answering questions, writing and editing code, "
    "analyzing information, creative work, and executing actions via your tools. "
    "You communicate clearly, admit uncertainty when appropriate, and prioritize "
    "being genuinely useful over being verbose unless otherwise directed below. "
    "Be targeted and efficient in your exploration and investigations."
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
)

# The OLD DEFAULT_SOUL_MD text -- containing "Hermes Agent", the exact phrase
# z.ai's content filter keys on (issue #89278) -- as shipped by every
# installer/CLI version before that fix. A DIFFERENT safety argument than
# _LEGACY_TEMPLATE_SOULS above (this one is real persona text, not an empty
# comment scaffold): an exact match here still carries zero evidence of
# customization BEYOND accepting the shipped default as-is, since this is a
# byte-for-byte copy of what we ourselves generated. Without this,
# _ensure_default_soul_md()'s self-heal would only catch the never-persona'd
# comment scaffold, silently leaving every pre-existing installation that
# already had a real (but stale, trigger-phrase-containing) persona on the
# vulnerable wording forever, even after upgrading past #89278's fix
# (review of #90094). Frozen: must stay byte-identical to what shipped, same
# rule as _LEGACY_TEMPLATE_SOULS.
_LEGACY_DEFAULT_PERSONA_SOULS = (
    (
        "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
        "You are helpful, knowledgeable, and direct. You assist users with a wide "
        "range of tasks including answering questions, writing and editing code, "
        "analyzing information, creative work, and executing actions via your tools. "
        "You communicate clearly, admit uncertainty when appropriate, and prioritize "
        "being genuinely useful over being verbose unless otherwise directed below. "
        "Be targeted and efficient in your exploration and investigations."
    ),
)


def _normalize_soul(text: str) -> str:
    """Normalize SOUL.md content for legacy-template comparison."""
    # Unify line endings (Windows installer writes CRLF-free but be defensive),
    # strip a leading UTF-8 BOM, and trim surrounding whitespace.
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff").strip()


def is_legacy_template_soul(text: str) -> bool:
    """True if ``text`` is a known, frozen, safe-to-upgrade-in-place SOUL.md.

    Two disjoint sets, both exact-match against known past shipped content:

    - ``_LEGACY_TEMPLATE_SOULS``: the old comment-only scaffold. Older
      installers seeded this instead of a real persona, which shadowed the
      runtime default and left users with no persona at all. Carries zero
      user intent by construction (no one would type this verbatim as a
      persona).
    - ``_LEGACY_DEFAULT_PERSONA_SOULS``: the OLD DEFAULT_SOUL_MD text
      (pre-#89278, containing "Hermes Agent" -- z.ai's content-filter
      trigger). A byte-for-byte match against what we ourselves generated
      as the unmodified default, so it carries zero evidence of
      customization beyond accepting the shipped default as-is.

    Any deviation from either set (the user typed a persona, even one
    character outside a known template) makes this return False.
    """
    normalized = _normalize_soul(text)
    return any(
        normalized == _normalize_soul(t)
        for t in (*_LEGACY_TEMPLATE_SOULS, *_LEGACY_DEFAULT_PERSONA_SOULS)
    )
