"""Which staged profile-export paths are secret-scrubbed, and the scrub pass itself.

Split out of ``hermes_cli/profiles.py`` so that facade stays inside its line cap; the profile
export staging walk (``_default_export_ignore`` / ``export_profile``) is the only caller.

The pass runs on the STAGED copy only, never on the live profile, and only over text/config
suffixes: redacting a byte range inside a binary DB or an image corrupts it, while markdown,
config and script files are where a pasted token actually lives. Credential stores themselves
are excluded from the export before this runs (``_EXPORT_CREDENTIAL_FILES``), so the pass is
the second line for secrets that leaked into ordinary text.
"""

from __future__ import annotations

from pathlib import Path

# Text/config suffixes secret-scrubbed on export; binary DBs, images etc. are left alone.
_EXPORT_REDACT_SUFFIXES = frozenset({
    ".md", ".txt", ".yaml", ".yml", ".json", ".jsonl", ".toml", ".ini", ".cfg", ".conf", ".py", ".sh",
    ".bash", ".zsh", ".js", ".ts", ".tsx", ".jsx", ".css", ".html", ".xml", ".csv",
})
# ``Path(".cursorrules").suffix`` is "" — name-match; ``*.env.example`` uses endswith; a hand-named
# config copy (``config.yaml.bak-my-note``) holds config.yaml's secrets under a suffix of its own.
_EXPORT_REDACT_NAMES = frozenset({".cursorrules"})


def _should_redact_export_file(path: Path) -> bool:
    name = path.name
    return (
        name in _EXPORT_REDACT_NAMES
        or name.lower().endswith(".env.example")
        or name.lower().startswith("config.yaml.")
        or path.suffix.lower() in _EXPORT_REDACT_SUFFIXES
    )


def _scrub_export_secrets(staged: Path) -> None:
    """Force-redact secret-shaped strings in a staged export tree (same pass as ``hermes
    sessions export --redact``). Runs on the staged copy only; symlinks to text files are
    materialized when content changes so redaction never follows a link back into the source."""
    from agent.redact import redact_sensitive_text
    for path in staged.rglob("*"):
        try:
            is_link = path.is_symlink()
            if not path.is_file():  # broken links, symlinked dirs, non-files
                continue
        except OSError:
            continue
        if not _should_redact_export_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError):
            continue
        redacted = redact_sensitive_text(text, force=True)
        if redacted == text:
            continue
        if is_link:
            path.unlink()
        path.write_text(redacted, encoding="utf-8")
