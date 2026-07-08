"""Project-wide constants."""

from pathlib import Path

APP_NAME = "llmwiki-hermes"
PROVIDER_NAME = "wiki"
DEFAULT_TOP_K_SEMANTIC = 5
DEFAULT_TOP_K_EPISODIC = 4
DEFAULT_RECALL_TOP_K = 8
DEFAULT_AUTO_WRITEBACK = False
CURRENT_SCHEMA_VERSION = 1
VAULT_ROOT_NAME = "LLM-Wiki"
VAULT_DIRS = (
    "00_inbox",
    "10_sources",
    "20_semantic",
    "30_episodic",
    "40_indexes",
    "90_system",
    ".wiki",
)
INDEX_DB_NAME = "index.sqlite"
INGEST_LOG_NAME = "ingest.log"
STATE_FILE_NAME = "state.json"
SESSION_LOG_DIR = "sessions"
PRECOMPRESS_DIR = "precompress"
CONFIG_ENV_VAR = "LLMWIKI_VAULT_PATH"
DEFAULT_CONFIG_PATH = Path("~/.config/llmwiki-hermes/config.yaml").expanduser()
