"""QMD client configuration.

Configuration for QMD client, resolved from config.yaml or environment variables.
When qmd.enabled=true in config.yaml, this takes precedence over honcho settings.

Supported embedding models:
  - BAAI/bge-m3 (default, 1024 dimensions)
  - Qwen/Qwen3-Embedding-4B-GGUF via QMD server (4096 dimensions, highest quality)
  - Qwen/Qwen3-Embedding-0.6B-GGUF via QMD server (1024 dimensions, faster)
  - sentence-transformers/all-MiniLM-L6-v2 (fallback, 384 dimensions)

Lite mode models (for resource-constrained environments):
  - BAAI/bge-micro-v2 (256 dimensions)
  - jinaai/jina-reranker-tiny (reranker)

Note: QMD server manages its own embedding models independently. The models listed
above are for sentence-transformers compatibility. QMD server uses GGUF-format
models (Qwen3-Embedding-4B, Qwen3-Reranker-4B) via llama.cpp on Metal/CPU.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default QMD server settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8181
DEFAULT_INDEX_NAME = "qmd_memory"

# Environment variable for QMD data directory
QMD_DIR_ENV = "HERMES_HOME"


def _resolve_hermes_home() -> Path:
    """Resolve HERMES_HOME or default to ~/.hermes."""
    return Path(os.getenv(QMD_DIR_ENV, Path.home() / ".hermes"))


QMD_DATA_DIR = _resolve_hermes_home() / "qmd_data"
QMD_DATA_DIR.mkdir(parents=True, exist_ok=True)


# Valid embedding models
VALID_EMBED_MODELS = {
    "default": "BAAI/bge-m3",  # 1024 dimensions
    "lite": "BAAI/bge-micro-v2",  # 256 dimensions
    # Integration point: Add NousResearch embedding models here when available
    # "nousresearch/...": "NousResearch/...",
}

# Embedding dimensions for known models
EMBEDDING_DIMS = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-micro-v2": 256,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    # Ollama models
    "qwen3.5b:0.8b": 896,
    "qwen3:0.8b": 896,
    "nomic-embed-text": 768,
    # MLX models (Apple Silicon)
    "mlx-community/bge-micro-v2": 256,
    "mlx-community/Nomic-embed-text": 768,
    # Aliases
    "mxbai-embed-large": 1024,
}

# Model format (pytorch vs mlx)
MODEL_FORMATS = {
    # Ollama
    "qwen3.5b:0.8b": "ollama",
    "qwen3:0.8b": "ollama",
    "nomic-embed-text": "ollama",
    # MLX (Apple Silicon)
    "mlx-community/bge-micro-v2": "mlx",
    "mlx-community/Nomic-embed-text": "mlx",
    # HuggingFace sentence-transformers
    "BAAI/bge-m3": "huggingface",
    "BAAI/bge-micro-v2": "huggingface",
    "sentence-transformers/all-MiniLM-L6-v2": "huggingface",
}


@dataclass
class QMDClientConfig:
    """Configuration for QMD client.

    When enabled=true, QMD takes precedence over Honcho settings.
    This is a strict OR relationship — both cannot be active simultaneously.
    """

    # Server connection
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    base_url: str | None = None  # Optional: override for remote QMD server

    # Identity
    peer_name: str = "hermes"

    # Index settings
    index_name: str = DEFAULT_INDEX_NAME
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # Memory settings
    memory_char_limit: int = 2200
    top_k: int = 5

    # Toggles
    enabled: bool = False
    lite_mode: bool = False  # Use lightweight embedding model

    # Write frequency: "async" (background), "turn" (sync per turn),
    # "session" (flush on session end), or int (every N turns)
    write_frequency: str | int = "async"

    # Anticipatory context settings
    anticipatory_enabled: bool = True
    anticipatory_max_results: int = 3

    # Session resolution
    session_strategy: str = "per-session"  # per-session | per-repo | per-directory | global

    # Raw config for anything else consumers need
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def server_url(self) -> str:
        """Get the full server URL."""
        if self.base_url:
            return f"{self.base_url}:{self.port}"
        return f"http://{self.host}:{self.port}"

    @property
    def data_dir(self) -> Path:
        """Get the QMD data directory."""
        return QMD_DATA_DIR

    @property
    def index_path(self) -> Path:
        """Get the FAISS index path."""
        return self.data_dir / f"{self.index_name}.faiss"

    @property
    def metadata_path(self) -> Path:
        """Get the metadata JSON path."""
        return self.data_dir / f"{self.index_name}.meta.json"

    def resolve_embedding(self) -> tuple[str, int]:
        """Resolve the embedding model and dimensions.

        Returns:
            Tuple of (model_name, embedding_dim).

        When lite_mode=True, returns the lite embedding model.
        Integration point for NousResearch models — check here first when
        NousResearch releases official embedding models.
        """
        if self.lite_mode:
            return ("BAAI/bge-micro-v2", 256)

        # Integration point: Check for NousResearch embedding model
        # Example when available:
        # if os.getenv("QMD_USE_NOUS_EMBEDDING"):
        #     return ("NousResearch/...", 1024)

        return (self.embedding_model, self.embedding_dim)

    @classmethod
    def from_config_dict(
        cls,
        config: dict[str, Any],
    ) -> QMDClientConfig:
        """Create config from a config.yaml qmd section.

        Args:
            config: The 'qmd' section from config.yaml.

        Returns:
            QMDClientConfig instance with resolved values.
        """
        if not config:
            return cls()

        enabled = config.get("enabled", False)

        # Lite mode auto-selects lightweight embedding
        lite_mode = config.get("lite_mode", False)

        # Resolve embedding model
        embed_model = config.get("embedding_model", "BAAI/bge-m3")
        embed_dim = EMBEDDING_DIMS.get(embed_model, 1024)

        if lite_mode:
            embed_model = "BAAI/bge-micro-v2"
            embed_dim = 256

        # Write frequency: accept int or string
        raw_wf = config.get("write_frequency", "async")
        try:
            write_frequency: str | int = int(raw_wf)
        except (TypeError, ValueError):
            write_frequency = str(raw_wf)

        return cls(
            host=config.get("host", DEFAULT_HOST),
            port=config.get("port", DEFAULT_PORT),
            base_url=config.get("base_url"),
            peer_name=config.get("peer_name", "hermes"),
            index_name=config.get("index_name", DEFAULT_INDEX_NAME),
            embedding_model=embed_model,
            embedding_dim=embed_dim,
            memory_char_limit=config.get("memory_char_limit", 2200),
            top_k=config.get("top_k", 5),
            enabled=enabled,
            lite_mode=lite_mode,
            write_frequency=write_frequency,
            anticipatory_enabled=config.get("anticipatory_enabled", True),
            anticipatory_max_results=config.get("anticipatory_max_results", 3),
            session_strategy=config.get("session_strategy", "per-session"),
            raw=config,
        )

    @classmethod
    def from_env(cls) -> QMDClientConfig:
        """Create config from environment variables (fallback).

        Environment variables:
            QMD_ENABLED: Set to 'true' to enable
            QMD_HOST: Server host
            QMD_PORT: Server port
            QMD_BASE_URL: Full base URL (overrides host/port)
            QMD_LITE_MODE: Use lightweight embedding
            QMD_INDEX_NAME: Index name
        """
        enabled = os.getenv("QMD_ENABLED", "").lower() in ("true", "1", "yes")
        lite_mode = os.getenv("QMD_LITE_MODE", "").lower() in ("true", "1", "yes")

        embed_model = "BAAI/bge-m3"
        embed_dim = 1024
        if lite_mode:
            embed_model = "BAAI/bge-micro-v2"
            embed_dim = 256

        return cls(
            host=os.getenv("QMD_HOST", DEFAULT_HOST),
            port=int(os.getenv("QMD_PORT", str(DEFAULT_PORT))),
            base_url=os.getenv("QMD_BASE_URL") or None,
            enabled=enabled,
            lite_mode=lite_mode,
            embedding_model=embed_model,
            embedding_dim=embed_dim,
            index_name=os.getenv("QMD_INDEX_NAME", DEFAULT_INDEX_NAME),
        )

    @staticmethod
    def _git_repo_name(cwd: str) -> str | None:
        """Return the git repo root directory name, or None if not in a repo."""
        try:
            import subprocess

            root = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=5,
            )
            if root.returncode == 0:
                return Path(root.stdout.strip()).name
        except (OSError, subprocess.TimeoutExpired):
            pass
        return None

    def resolve_session_name(
        self,
        cwd: str | None = None,
        session_title: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Resolve QMD session name based on session strategy.

        Resolution order:
          1. per-session: Hermes session_id ({timestamp}_{hex})
          2. per-repo: git repo root directory name
          3. per-directory: directory basename
          4. global: "qmd_global"
        """
        import re

        if not cwd:
            cwd = os.getcwd()

        # /title mid-session remap
        if session_title:
            sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", session_title).strip("-")
            if sanitized:
                return f"{self.peer_name}-{sanitized}"

        # per-session: inherit Hermes session_id
        if self.session_strategy == "per-session" and session_id:
            return f"{self.peer_name}-{session_id}"

        # per-repo: one session per git repository
        if self.session_strategy == "per-repo":
            base = self._git_repo_name(cwd) or Path(cwd).name
            return f"{self.peer_name}-{base}"

        # per-directory: one session per working directory
        if self.session_strategy == "per-directory":
            base = Path(cwd).name
            return f"{self.peer_name}-{base}"

        # global: single session
        return "qmd_global"
