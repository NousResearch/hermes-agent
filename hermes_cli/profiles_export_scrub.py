"""What a profile export scrubs from its staged copy, and the saved-gateways stores profiles never carry.

Split out of :mod:`hermes_cli.profiles` (its facade over the clone/export paths).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# The machine's saved-gateways list and its OAuth sign-ins (``hermes_cli.gateway_connections``), with
# their ``.lock`` dir and ``.corrupt-<ms>`` sidecars: machine-wide root state every profile reads at the
# root, holding plaintext session tokens. A copy in a clone is dead weight, and an export must not ship it.
GATEWAY_CONNECTION_STORES = ("gateway-connections.json", "gateway-connection-tokens.json")
GATEWAY_CONNECTION_STORE_RE = re.compile(r"gateway-connections?(?:-tokens)?\.json(?:\..+)?")


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


def _masked_envelopes(node):
    """*node* with the value of every ``{"encoding", "value"}`` secret envelope masked."""
    if isinstance(node, list):
        return [_masked_envelopes(item) for item in node]
    if not isinstance(node, dict):
        return node
    if set(node) == {"encoding", "value"} and isinstance(node["value"], str) and node["value"]:
        return {"encoding": node["encoding"], "value": "***"}
    return {key: _masked_envelopes(value) for key, value in node.items()}


def _mask_secret_envelopes(text: str) -> str:
    """Hermes stores secrets as ``{"encoding": "plain", "value": ...}`` envelopes, whose key names
    no secret: the text redactor cannot see them, so JSON documents are masked structurally."""
    try:
        doc = json.loads(text)
    except ValueError:
        return text
    masked = _masked_envelopes(doc)
    return text if masked == doc else json.dumps(masked, indent=2)


def scrub_export_secrets(staged: Path) -> None:
    """Force-redact secret-shaped strings in a staged export tree (same pass as ``hermes
    sessions export --redact``). Runs on the staged copy only; symlinks to text files are
    materialized when content changes so redaction never follows a link back into the source."""
    from hermes_agent.agent.redact import redact_sensitive_text
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
        redacted = redact_sensitive_text(_mask_secret_envelopes(text), force=True)
        if redacted == text:
            continue
        if is_link:
            path.unlink()
        path.write_text(redacted, encoding="utf-8")
