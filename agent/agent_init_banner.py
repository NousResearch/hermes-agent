"""Init-time credential banner (the masked key line printed when a client is built)."""

from __future__ import annotations


def _print_key_banner(key, label: str, warn_missing: bool = False) -> None:
    """Masked credential line. A callable ``key`` (Entra ID, Vertex OAuth, key_cmd, ...) is labelled
    by its tagged source and never invoked. Keys ≤ 12 chars (incl. "dummy-key") are not shown."""
    from agent.credential_source import credential_label
    if source_label := credential_label(key):
        print(f"🔑 Using credentials: {source_label}")
    elif isinstance(key, str) and len(key) > 12:
        print(f"🔑 Using {label}: {key[:8]}...{key[-4:]}")
    elif warn_missing:
        print("⚠️  Warning: API key appears invalid or missing")
