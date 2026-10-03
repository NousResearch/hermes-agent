"""skill-rag plugin configuration.

Paths and embedding parameters are constants.
Behavioral settings are read from ctx.get_config() at runtime
(plugins.entries.skill-rag.settings.* in config.yaml).
API_KEY is a secret and stays in environment variables.
"""
from pathlib import Path

from hermes_constants import get_hermes_home, get_skills_dir

# --- Paths ---
HERMES_HOME = get_hermes_home()
SKILLS_ROOT = get_skills_dir()
FALLBACK_INDEX_DIR = HERMES_HOME / "skills_index"
DB_FILENAME = ".skill_rag.db"  # SQLite index filename
FTS_TABLE = "skills_fts"  # FTS5 virtual table name (hardcoded — do not change)
SKILL_FILE = "SKILL.md"  # Skill file name for scanning

# --- Embedding provider constants ---
# "openai_compatible" — API (LM Studio, Ollama, vLLM, llama.cpp)
# "local" — offline via sentence-transformers (~2GB)
LOCAL_MODEL = "intfloat/multilingual-e5-small"
LOCAL_REVISION = None  # Model revision (None = latest)
API_BATCH_SIZE = 16
API_TIMEOUT_CONNECT = 2.0
API_TIMEOUT_READ = 10.0

# --- Embedding parameters ---
EMBEDDING_DIM = 1024  # Vector dimension (384 for nomic/e5, 1024 for bge-m3)
PREFIX_QUERY = "query: "  # Prefix for queries (required by e5 models)
PREFIX_PASSAGE = "passage: "  # Prefix for passages (required by e5 models)

# --- Frontmatter fields for embedding ---
EMBED_FIELDS = ("name", "description", "when_to_use", "triggers", "tags", "category")
FIELD_MAX_LEN = 300

# --- Skill tools ---
SKILL_TOOLS = {"skill_view", "skill_manage", "skills_list"}

# --- Logging ---
LOG_PREFIX = "[skill-rag]"

# --- Behavioral settings defaults ---
# Read from ctx.get_config() at runtime; these are fallbacks.
DEFAULTS = {
    "provider": "openai_compatible",
    "api_base": "http://localhost:1234/v1",
    "api_model": "text-embedding-bge-m3",
    "top_k": 5,
    "threshold": 0.3,
    "history_window": 4,
    "assistant_truncate": 500,
}
