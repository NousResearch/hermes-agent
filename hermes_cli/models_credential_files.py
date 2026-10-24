"""Credential-file mtime parts for ``hermes_cli.models._credential_fingerprint``.

A provider's cached catalog row is keyed by a fingerprint of the credentials its catalog fetch
can read. Credential STORES are part of that: a rewrite of ``~/.hermes/auth.json`` or of an
external tool's token file must re-key the row so a changed account never serves the previous
account's catalog. The external stores are scoped PER PROVIDER: they rotate on their owner's
schedule (Claude Code refreshes ``~/.claude/.credentials.json`` every few hours; ``copilot login``
rewrites ``~/.copilot/config.json``), and folding every store into every provider's fingerprint
evicted unrelated rows — a Claude Code refresh dropped the Copilot row, so the non-blocking
``/model`` picker fell back to the curated Copilot list (no Opus tier) until the SWR refresh
landed (#139795, #129094).
"""

import logging
import os

logger = logging.getLogger(__name__)

# External (non-Hermes) credential stores, keyed by the ONLY provider whose catalog fetch reads
# them. Adding a provider here means its ``_fetch_*_models`` borrows a token from that file.
_PROVIDER_EXTERNAL_CREDENTIAL_FILES: dict[str, tuple[str, ...]] = {
    "anthropic": ("~/.claude/.credentials.json",),
    "copilot": ("~/.config/github-copilot/hosts.json", "~/.config/github-copilot/apps.json",
                "~/.copilot/config.json"),
    "copilot-acp": ("~/.config/github-copilot/hosts.json", "~/.config/github-copilot/apps.json",
                    "~/.copilot/config.json"),
    "minimax-oauth": ("~/.minimax/credentials.json",),
}


def append_mtime_part(parts: list[str], label: str, path) -> None:
    """Append ``label@<mtime_ns>`` (``@missing`` / ``@unreadable``) so a rewrite of ``path`` re-keys the row."""
    try:
        parts.append(f"{label}@{os.stat(path).st_mtime_ns}")
    except FileNotFoundError:
        parts.append(f"{label}@missing")
    except OSError as exc:
        # Permission / stale-mount errors: still a stable part (so the row stays servable) but
        # never silently dropped — the next "wrong account's catalog" report starts here.
        parts.append(f"{label}@unreadable")
        logger.debug("credential store %s unreadable while fingerprinting: %s", path, exc, exc_info=True)


def append_credential_file_parts(parts: list[str], provider: str) -> None:
    """Append the Hermes auth-store mtimes plus the external stores ``provider`` can read."""
    from hermes_constants import get_hermes_home

    for rel in ("auth.json", "credentials.json"):
        append_mtime_part(parts, rel, get_hermes_home() / rel)
    for rel in _PROVIDER_EXTERNAL_CREDENTIAL_FILES.get(provider, ()):
        path = os.path.expanduser(rel)
        append_mtime_part(parts, path, path)
