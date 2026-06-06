"""Configuration for the Hermes document knowledge integration."""

from __future__ import annotations

import logging
import os
from pathlib import Path


VERBOSE_MODE = os.getenv("HERMES_COCOINDEX_VERBOSE", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

LOGGER = logging.getLogger("hermes_cocoindex_docs")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO if VERBOSE_MODE else logging.WARNING)

DOCUMENTS_DIR = Path(
    os.getenv("HERMES_DOCS_SOURCE_DIR", str(Path.home() / "Dokumente"))
).expanduser()

STATE_DIR = Path(
    os.getenv(
        "HERMES_COCOINDEX_STATE_DIR",
        str(Path.home() / ".local" / "state" / "hermes" / "cocoindex"),
    )
).expanduser()

SHARE_DIR = Path(
    os.getenv(
        "HERMES_COCOINDEX_SHARE_DIR",
        str(Path.home() / ".local" / "share" / "hermes" / "cocoindex"),
    )
).expanduser()

COCOINDEX_DB_PATH = Path(
    os.getenv("COCOINDEX_DB", str(STATE_DIR / "cocoindex.db"))
).expanduser()

SQLITE_DB_PATH = Path(
    os.getenv("HERMES_DOCS_SQLITE_DB", str(SHARE_DIR / "documents.sqlite3"))
).expanduser()

TABLE_NAME = os.getenv("HERMES_DOCS_TABLE", "hermes_documents")
EMBED_MODEL = os.getenv(
    "HERMES_DOCS_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
MAX_FILE_BYTES = int(os.getenv("HERMES_DOCS_MAX_FILE_BYTES", str(80 * 1024 * 1024)))
DEFAULT_LIMIT = int(os.getenv("HERMES_DOCS_DEFAULT_LIMIT", "5"))
MAX_LIMIT = int(os.getenv("HERMES_DOCS_MAX_LIMIT", "20"))

INCLUDED_PATTERNS = [
    "**/*.md",
    "**/*.MD",
    "**/*.mdx",
    "**/*.MDX",
    "**/*.txt",
    "**/*.TXT",
    "**/*.pdf",
    "**/*.PDF",
    "**/*.py",
    "**/*.js",
    "**/*.jsx",
    "**/*.ts",
    "**/*.tsx",
    "**/*.json",
    "**/*.JSON",
    "**/*.eml",
    "**/*.EML",
]

EXCLUDED_PATTERNS = [
    "**/.git",
    "**/.git/**",
    "**/.svn",
    "**/.svn/**",
    "**/__pycache__",
    "**/__pycache__/**",
    "**/node_modules",
    "**/node_modules/**",
    "**/site-packages",
    "**/site-packages/**",
    "**/venv",
    "**/venv/**",
    "**/venv-*",
    "**/venv-*/**",
    "**/.venv",
    "**/.venv/**",
    "**/env",
    "**/env/**",
    "**/env-*",
    "**/env-*/**",
    "**/env_*",
    "**/env_*/**",
    "**/build",
    "**/build/**",
    "**/dist",
    "**/dist/**",
    "**/*.pyc",
    "**/*.so",
    "**/*.mov",
    "**/*.MOV",
    "**/*.mp4",
    "**/*.MP4",
    "**/*.psd",
    "**/*.PSD",
    "**/*.xcf",
    "**/*.XCF",
    "**/*.bmp",
    "**/*.BMP",
]


def ensure_directories() -> None:
    """Create runtime directories used by the indexer and search tool."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SHARE_DIR.mkdir(parents=True, exist_ok=True)
    COCOINDEX_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Runtime directories are ready")


def configure_cocoindex_environment() -> None:
    """Set CocoIndex defaults before the app is initialized."""
    ensure_directories()
    os.environ.setdefault("COCOINDEX_DB", str(COCOINDEX_DB_PATH))
    LOGGER.info("COCOINDEX_DB=%s", os.environ["COCOINDEX_DB"])
