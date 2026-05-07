"""Optional local embedding helpers for research-search hybrid retrieval."""

from __future__ import annotations

import importlib
from typing import Any


_MODEL_CACHE: dict[str, Any] = {}


def _numpy():
    return importlib.import_module("numpy")


def vector_status(config: dict[str, Any]) -> dict[str, Any]:
    """Return vector-search availability without loading a model eagerly."""
    vector_cfg = config.get("vector") or {}
    if not vector_cfg.get("enabled", True):
        return {
            "enabled": False,
            "available": False,
            "provider": str(vector_cfg.get("provider") or "sentence_transformers"),
            "model": str(vector_cfg.get("model") or ""),
            "error": "disabled",
        }
    provider = str(vector_cfg.get("provider") or "sentence_transformers")
    model = str(vector_cfg.get("model") or "BAAI/bge-small-en-v1.5")
    try:
        importlib.import_module("numpy")
        if provider == "sentence_transformers":
            importlib.import_module("sentence_transformers")
        else:
            return {
                "enabled": True,
                "available": False,
                "provider": provider,
                "model": model,
                "error": f"unsupported provider: {provider}",
            }
    except Exception as exc:
        return {
            "enabled": True,
            "available": False,
            "provider": provider,
            "model": model,
            "error": str(exc),
        }
    return {
        "enabled": True,
        "available": True,
        "provider": provider,
        "model": model,
        "error": "",
    }


def _load_sentence_transformer(model_name: str):
    if model_name not in _MODEL_CACHE:
        module = importlib.import_module("sentence_transformers")
        _MODEL_CACHE[model_name] = module.SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_texts(texts: list[str], config: dict[str, Any]) -> list[Any]:
    """Embed text strings using the configured optional provider."""
    vector_cfg = config.get("vector") or {}
    provider = str(vector_cfg.get("provider") or "sentence_transformers")
    if provider != "sentence_transformers":
        raise RuntimeError(f"unsupported embedding provider: {provider}")
    model_name = str(vector_cfg.get("model") or "BAAI/bge-small-en-v1.5")
    model = _load_sentence_transformer(model_name)
    batch_size = int(vector_cfg.get("batch_size") or 32)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    np = _numpy()
    return [np.asarray(vec, dtype=np.float32) for vec in vectors]


def vector_to_blob(vector: Any) -> tuple[bytes, int]:
    np = _numpy()
    arr = np.asarray(vector, dtype=np.float32)
    return arr.tobytes(), int(arr.shape[0])


def blob_to_vector(blob: bytes):
    np = _numpy()
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(left: Any, right: Any) -> float:
    np = _numpy()
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)
