"""Shared constants, paths, and utilities for the finetune pipeline."""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("hermes.finetune")

# ── Paths ──


def _default_hermes_home() -> Path:
    """Platform-native default, mirroring hermes_constants.get_hermes_home().

    These scripts run standalone (dispatched via subprocess from /finetune),
    so they can't import hermes_constants. The dispatcher always exports
    HERMES_HOME for profile isolation; this fallback only covers direct
    invocation from a shell.
    """
    import sys as _sys
    if _sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


HERMES_HOME = Path(os.environ.get("HERMES_HOME") or _default_hermes_home())
FINETUNE_DIR = HERMES_HOME / "finetune"
DATA_DIR = FINETUNE_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
SCORED_DIR = DATA_DIR / "scored"
CLUSTERS_DIR = DATA_DIR / "clusters"
IMPORTED_DIR = DATA_DIR / "imported"
ADAPTERS_DIR = FINETUNE_DIR / "adapters"
MODELS_DIR = FINETUNE_DIR / "models" / "merged"
LOGS_DIR = FINETUNE_DIR / "logs"
BENCH_DIR = FINETUNE_DIR / "bench"
FEEDBACK_PATH = FINETUNE_DIR / "feedback.jsonl"
REGISTRY_PATH = ADAPTERS_DIR / "registry.json"
CLUSTER_STATE_PATH = ADAPTERS_DIR / "cluster_state.json"
STATE_DB_PATH = HERMES_HOME / "state.db"

# Extraction state
EXTRACT_STATE_PATH = FINETUNE_DIR / "extract_state.json"


def ensure_dirs():
    """Create the full finetune directory tree."""
    for d in [
        EXTRACTED_DIR, SCORED_DIR, CLUSTERS_DIR, IMPORTED_DIR,
        ADAPTERS_DIR, MODELS_DIR, LOGS_DIR,
        BENCH_DIR / "results", BENCH_DIR / "custom",
    ]:
        d.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default=None):
    """Load JSON file, returning default if missing or invalid."""
    if not path.exists():
        return default if default is not None else {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load %s: %s", path, e)
        return default if default is not None else {}


def save_json(path: Path, data, indent=2):
    """Atomically write JSON to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=indent, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.rename(path)


def append_jsonl(path: Path, records: list):
    """Append records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list:
    """Read all records from a JSONL file. Malformed lines are skipped with a warning."""
    if not path.exists():
        return []
    records = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1
    if skipped:
        logger.warning("Skipped %d malformed line(s) while reading %s", skipped, path)
    return records


def load_records_dedup(directory: Path, pattern: str, key: str = "session_id") -> list:
    """Load records from every JSONL file in `directory` matching `pattern`,
    deduplicated by `key`.

    Files are read in sorted (i.e. timestamp) order and later records win, so
    repeated snapshot runs (scored_*.jsonl, extract_*.jsonl) yield exactly one
    record per session — the one from the newest snapshot. Records missing the
    key are kept as-is.
    """
    by_key: dict = {}
    extras: list = []
    for path in sorted(directory.glob(pattern)):
        for record in read_jsonl(path):
            k = record.get(key) if isinstance(record, dict) else None
            if k:
                by_key[k] = record
            else:
                extras.append(record)
    return list(by_key.values()) + extras


def content_to_text(content) -> str:
    """Normalize a message `content` field to plain text.

    Hermes messages may carry multipart list content (e.g.
    [{"type": "text", "text": ...}, {"type": "image_url", ...}]) or plain
    strings. Non-text parts (images, etc.) are dropped rather than serialized.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(p for p in parts if p)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content)


def load_config() -> dict:
    """Load finetune config from ~/.hermes/config.yaml, falling back to defaults."""
    defaults = {
        "enabled": True,
        "extract": {
            "min_turns": 2,
            "exclude_sources": [],
        },
        "scoring": {
            "weights": {
                "conversation_signal": 0.3,
                "turn_signal": 0.4,
                "sentiment_modifier": 0.1,
                "judge_score": 0.2,
            },
            "thresholds": {
                "good": 0.7,
                "neutral": 0.4,
            },
        },
        "clustering": {
            "embedding_model": "all-MiniLM-L6-v2",
            "min_cluster_size": 30,
            "confidence_threshold": 0.6,
        },
        "training": {
            # HF repo id (matches templates/base_qlora.yaml). Must be a
            # transformers-loadable model — a GGUF path cannot be trained on.
            "base_model": "kai-os/Carnice-9b",
            "chat_template": "chatml",
            "quantization": "Q5_K_M",
            "terminal_backend": "local",
            # Per-turn extraction (see format.py::extract_training_turns).
            # context_window_turns: how many preceding turns to include as
            # context for each training example. min_turn_score: only emit
            # assistant turns whose effective score (automated or retro
            # override) meets this threshold.
            "context_window_turns": 8,
            "min_turn_score": 0.7,
        },
        "routing": {
            "enabled": True,
            "providers": ["local", "llama-cpp", "custom"],
        },
        "retraining": {
            "data_growth_trigger": 0.2,
            "schedule": "weekly",
        },
        "feedback": {
            "cli_keybindings": True,
            "gateway_reactions": True,
        },
        # Auto-redeploy: convert the active adapter to GGUF and restart
        # llama-server with it loaded after each promote. Off by default
        # because it requires user-specific config (server command,
        # converter path, etc.). See SKILL.md for the full setup.
        "serving": {
            "auto_redeploy": False,
            "converter": "",                 # empty = resolved at use-site
            "base_model_snapshot": "auto",  # "auto" = detect from HF cache
            "server_command": "",            # empty = no auto-restart
            # Empty = resolved at use-site to FINETUNE_DIR/llama-server.pid
            # and FINETUNE_DIR/llama-server.log (see manage.py).
            "server_pid_file": "",
            "server_log_path": "",
            "health_check_url": "http://localhost:8008/v1/models",
            "health_check_timeout": 30,
        },
    }

    config_path = HERMES_HOME / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                full_config = yaml.safe_load(f) or {}
            ft_config = full_config.get("finetune", {})
            if ft_config:
                _deep_merge(defaults, ft_config)
        except Exception as e:
            logger.warning("Failed to load finetune config: %s", e)

    # Env override (declared in SKILL.md frontmatter): takes precedence over
    # both defaults and config.yaml.
    env_base_model = os.environ.get("FINETUNE_BASE_MODEL", "").strip()
    if env_base_model:
        defaults.setdefault("training", {})["base_model"] = env_base_model

    return defaults


def _deep_merge(base: dict, override: dict):
    """Merge override into base in-place, recursing into nested dicts."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
