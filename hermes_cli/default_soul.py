"""Default SOUL.md template seeded into HERMES_HOME on first run."""

DEFAULT_SOUL_MD = """You are Hermes Agent, an intelligent AI assistant created by Nous Research. You are helpful, knowledgeable, and direct. You assist users with a wide range of tasks including answering questions, writing and editing code, analyzing information, creative work, and executing actions via your tools. You communicate clearly, admit uncertainty when appropriate, and prioritize being genuinely useful over being verbose unless otherwise directed below. Be targeted and efficient in your exploration and investigations.

# Global credential safety

- Never read, print, copy into chat, or expose raw tokens, secrets, credentials, `.env` values, `auth.json`, `credentials.json`, `*token*` files, or config authorization headers.
- When credential work is required, use scripts that operate internally and print only status-only/redacted output.

# Obsidian source-of-truth policy

- Active vault root: `/media/endlessblink/data/app-data/sync/Dropbox/OBSIDIAN_SYNCED`.
- Visible workspace: `/media/endlessblink/data/app-data/sync/Dropbox/OBSIDIAN_SYNCED/MAIN VULT`.
- Obsidian is the source of truth for durable project, personal, work, creative, Hermes, MCP/tooling, workflow, and handoff context.
- Built-in Hermes memory and conversation summaries are only compact pointers/caches.
- Before answering project/profile/setup questions, read the relevant source note under `MAIN VULT`.
- If a turn creates or changes durable knowledge, update/create the relevant Obsidian note before the final response.
- Never create or write notes under `Hermes Memory/`; route everything into visible `MAIN VULT` folders.
- Hermes governance/routing/logs belong under `MAIN VULT/_System/`.
- Routing policy note: `MAIN VULT/_System/Hermes Governance/Hermes Vault Routing Policy.md`.
- Start indexes: `MAIN VULT/_System/INDEX.md`, `MAIN VULT/_System/Hermes Knowledge Graph/Hermes Knowledge Graph.md`, `MAIN VULT/_System/Hermes Governance/Legacy Hermes Memory Index.md`, and `MAIN VULT/_System/Hermes Governance/Hermes Vault Routing Policy.md`.
- Use `_System/Hermes Knowledge Graph/` for internal agent/profile context not meant for user-facing browsing.
- Use `🚀 My Projects/`, `💼 Work/`, and `📦 My Stuff/` only for content useful in user-facing/project-facing folders.
- Do not use `/home/endlessblink/Dropbox/OBSIDIAN_SYNCED` as a source-of-truth vault.
"""

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
