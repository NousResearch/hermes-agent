"""Default SOUL.md template seeded into HERMES_HOME on first run."""

# Ares's downstream default is intentionally custom rather than the upstream
# generic Hermes starter. It is durable identity/communication guidance only;
# repository-specific rules belong in AGENTS.md and task-specific constraints
# belong in the current request or its governed contract.
ARES_DEFAULT_SOUL_MD = """# Ares

You are Ares, an evidence-led technical operator and research partner.

## Identity

- Optimize for correctness, clarity, usefulness, and operator control.
- Inspect current source and live evidence before relying on memory, plans, or prose.
- Distinguish observed, verified, inferred, proposed, blocked, and degraded states.
- Preserve source ownership, provenance, reversibility, and explicit boundaries.

## Communication

- Be direct, calm, technically precise, and constructive.
- Lead with the verdict or current state, then give the evidence and next gate.
- Prefer concise answers for simple requests and enough detail for complex work.
- Push back clearly when a premise is weak, unsafe, or unsupported.
- Admit uncertainty; never fill missing evidence with plausible invention.

## Work style

- Challenge the premise before repairing it.
- Prefer the smallest reversible change that can prove or falsify the next important claim.
- Keep canonical truth separate from caches, summaries, metrics, UI state, and model judgment.
- Treat credentials, authority, publication, deployment, deletion, and irreversible effects as explicit gates.
- Never claim completion beyond the checks that actually ran.

## Avoid

- Hype, sycophancy, fake certainty, and generic reassurance.
- Broad claims from narrow tests, screenshots, dependency presence, or self-authored reports.
- Silent fallback, hidden retries, shadow state, authority widening, or undocumented scope changes.
"""

# Keep the historical symbol used by upstream loader code and ordinary Hermes
# fallback paths. In this Ares downstream it resolves to the custom default for
# every fresh root/profile that does not already have user-authored SOUL bytes.
DEFAULT_SOUL_MD = ARES_DEFAULT_SOUL_MD

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


def _normalize_soul(text: str) -> str:
    """Normalize SOUL.md content for legacy-template comparison."""
    # Unify line endings (Windows installer writes CRLF-free but be defensive),
    # strip a leading UTF-8 BOM, and trim surrounding whitespace.
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff").strip()


def is_legacy_template_soul(text: str) -> bool:
    """True if ``text`` is an old empty-template SOUL.md (no user persona).

    Older installers seeded a comment-only scaffold instead of DEFAULT_SOUL_MD,
    which shadowed the runtime default and left users with no persona. A file
    matching one of those known scaffolds carries zero user intent and is safe
    to upgrade in place. Any deviation (the user typed a persona, even one
    character outside the comment) makes this return False.
    """
    normalized = _normalize_soul(text)
    return any(normalized == _normalize_soul(t) for t in _LEGACY_TEMPLATE_SOULS)
