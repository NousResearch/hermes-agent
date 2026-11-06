"""Codex auxiliary endpoint policy."""


def codex_base_url_override() -> str:
    """Profile-scoped ``HERMES_CODEX_BASE_URL`` (same read as the API-key env vars: under a
    multiplexer the routed profile's .env decides the endpoint, never a sibling's process env)."""
    from agent.auxiliary_client import _scoped_key_env
    return _scoped_key_env("HERMES_CODEX_BASE_URL").rstrip("/")
