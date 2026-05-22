from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WikiConfig:
    wiki_path: Path = field(default_factory=lambda: Path(
        os.environ.get("WIKI_PATH", os.path.expanduser("~/.hermes-dev/wiki"))
    ))
    wiki_name: str = "personal"

    embedding_url: str = field(default_factory=lambda: os.environ.get(
        "EMBEDDING_BASE_URL", "http://localhost:22222"
    ))
    embedding_model: str = "Qwen3-Embedding-8B"
    embedding_dim: int = 4096
    embedding_cache_path: Path | None = None
    embedding_cache_max_entries: int = 100000

    qdrant_url: str = field(default_factory=lambda: os.environ.get(
        "QDRANT_URL", "http://localhost:6333"
    ))
    collection_prefix: str = "hermes_wiki"

    llm_base_url: str = field(default_factory=lambda: os.environ.get(
        "WIKI_LLM_URL", "http://localhost:8011/v1"
    ))
    llm_model: str = "gpt-5.5"
    llm_api_key: str = "not-needed"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.3

    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50

    log_rotation_threshold: int = 500
    page_split_threshold: int = 200

    @property
    def collection_name(self) -> str:
        return f"{self.collection_prefix}_{self.wiki_name}"

    @property
    def raw_dir(self) -> Path:
        return self.wiki_path / "raw"

    @property
    def entities_dir(self) -> Path:
        return self.wiki_path / "entities"

    @property
    def concepts_dir(self) -> Path:
        return self.wiki_path / "concepts"

    @property
    def comparisons_dir(self) -> Path:
        return self.wiki_path / "comparisons"

    @property
    def queries_dir(self) -> Path:
        return self.wiki_path / "queries"

    @property
    def archive_dir(self) -> Path:
        return self.wiki_path / "_archive"

    @property
    def index_path(self) -> Path:
        return self.wiki_path / "index.md"

    @property
    def log_path(self) -> Path:
        return self.wiki_path / "log.md"

    @property
    def schema_path(self) -> Path:
        return self.wiki_path / "SCHEMA.md"

    def ensure_dirs(self) -> None:
        for d in [
            self.wiki_path,
            self.raw_dir / "articles",
            self.raw_dir / "papers",
            self.raw_dir / "transcripts",
            self.raw_dir / "assets",
            self.entities_dir,
            self.concepts_dir,
            self.comparisons_dir,
            self.queries_dir,
            self.archive_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, d: dict) -> WikiConfig:
        wiki_cfg = d.get("wiki", {})
        if not isinstance(wiki_cfg, dict):
            raise ValueError("wiki config section must be a mapping")

        embedding = wiki_cfg.get("embedding", {})
        if not isinstance(embedding, dict):
            raise ValueError("wiki.embedding config section must be a mapping")
        vector = wiki_cfg.get("vector_store", {})
        if not isinstance(vector, dict):
            raise ValueError("wiki.vector_store config section must be a mapping")
        llm = wiki_cfg.get("llm", {})
        if not isinstance(llm, dict):
            raise ValueError("wiki.llm config section must be a mapping")

        kwargs = {}
        if "path" in wiki_cfg:
            kwargs["wiki_path"] = Path(os.path.expanduser(wiki_cfg["path"]))
        if "name" in wiki_cfg:
            kwargs["wiki_name"] = wiki_cfg["name"]
        if "url" in embedding:
            kwargs["embedding_url"] = embedding["url"]
        if "model" in embedding:
            kwargs["embedding_model"] = embedding["model"]
        if "dim" in embedding:
            kwargs["embedding_dim"] = embedding["dim"]
        if "cache_path" in embedding:
            kwargs["embedding_cache_path"] = Path(os.path.expanduser(str(embedding["cache_path"])))
        if "cache_max_entries" in embedding:
            kwargs["embedding_cache_max_entries"] = int(embedding["cache_max_entries"])
        if "url" in vector:
            kwargs["qdrant_url"] = vector["url"]
        if "collection_prefix" in vector:
            kwargs["collection_prefix"] = vector["collection_prefix"]
        if "url" in llm:
            kwargs["llm_base_url"] = llm["url"]
        if "model" in llm:
            kwargs["llm_model"] = llm["model"]
        if "api_key" in llm:
            kwargs["llm_api_key"] = llm["api_key"]

        return cls(**kwargs)

    @classmethod
    def from_hermes_config(cls, config_path: str | Path | None = None) -> WikiConfig:
        """Load wiki config from a Hermes config.yaml file.

        Tries in order:
        1. Explicit config_path argument
        2. HERMES_HOME env var + /config.yaml
        3. ~/.hermes-dev/config.yaml (dev install)
        4. ~/.hermes/config.yaml (production)
        5. Falls back to defaults
        """
        import yaml

        candidates = []
        if config_path:
            candidates.append(Path(config_path))

        hermes_home = os.environ.get("HERMES_HOME")
        if hermes_home:
            candidates.append(Path(hermes_home) / "config.yaml")

        candidates.append(Path.home() / ".hermes-dev" / "config.yaml")
        candidates.append(Path.home() / ".hermes" / "config.yaml")

        for path in candidates:
            if path.exists():
                try:
                    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                    if "wiki" in data:
                        return cls.from_dict(data)
                except Exception:
                    continue

        return cls()
