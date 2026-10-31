"""Gateway session-content search defaults and env registration.

Empty embedding_model keeps search fuzzy-only (FTS/trigram). A model plus
embedding_base_url (OpenAI-compatible /embeddings) adds a semantic index beside
state.db and fuses it with fuzzy hits via reciprocal rank fusion. The API key is
SESSION_SEARCH_EMBEDDING_API_KEY, else OPENAI_API_KEY — never stored in config.
"""

from hermes_cli.config_defaults_env import _env

SESSION_SEARCH_DEFAULT = {
    "embedding_model": "",
    "embedding_base_url": "",
}

SESSION_SEARCH_ENV = {
    "SESSION_SEARCH_EMBEDDING_API_KEY": _env(
        "API key for session-search embeddings when session_search.embedding_model is set. "
        "Falls back to OPENAI_API_KEY. Leave empty for fuzzy-only search.",
        "Session search embedding API key",
        password=True,
        category="provider",
        advanced=True,
    ),
}
