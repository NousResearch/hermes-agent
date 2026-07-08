"""Domain-specific exceptions."""


class LlmWikiError(Exception):
    """Base exception for the project."""


class VaultNotInitializedError(LlmWikiError):
    """Raised when a vault is missing required structure."""


class InvalidFrontmatterError(LlmWikiError):
    """Raised when a note's frontmatter cannot be parsed or validated."""


class IndexCorruptionError(LlmWikiError):
    """Raised when the SQLite sidecar index is missing or invalid."""


class IngestInputError(LlmWikiError):
    """Raised when ingest input is missing or malformed."""


class ConfigurationError(LlmWikiError):
    """Raised when required configuration is unavailable."""
