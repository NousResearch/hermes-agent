"""Generic embedding client for the skill-graph plugin.

Refactored from ``hermes_mem/tei_client.py`` (MIT).  Keeps the three proven
mechanisms — protocol auto-detection, health-check fallback, CPU lazy-load —
and decouples the config source so any plugin can reuse it.

Backends (config ``embedding_backend``):
  tei     — Text Embeddings Inference, POST /embed
  ollama  — POST /api/embeddings
  openai  — OpenAI-compatible, POST /v1/embeddings
  cpu     — local sentence-transformers (BAAI/bge-m3, 1024-dim)
  auto    — probe endpoint, GPU first, CPU fallback on any failure

Config keys (read from ``skills.config.skill-graph.*`` via injected reader):
  embedding_backend: auto
  embedding_api_url: http://localhost:8081
  embedding_model:   bge-m3
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import requests

_logger = logging.getLogger("skill-graph.embedding")

_DEFAULT_BACKEND = "auto"
_DEFAULT_URL = "http://localhost:8081"
_DEFAULT_MODEL = "bge-m3"

# Lazy CPU model singleton
_CPU_MODEL = None


def _read_cfg(config_reader: Callable[[], dict]) -> dict:
    """Read plugin config dict from the injected reader (never raises)."""
    try:
        cfg = config_reader()
        if not isinstance(cfg, dict):
            return {}
        return cfg
    except Exception:
        return {}


def _get_backend(cfg: dict) -> str:
    return str(cfg.get("embedding_backend", _DEFAULT_BACKEND)).strip().lower()


def _get_endpoint(cfg: dict) -> str:
    return str(cfg.get("embedding_api_url", _DEFAULT_URL)).rstrip("/")


def _get_model(cfg: dict) -> str:
    return str(cfg.get("embedding_model", _DEFAULT_MODEL))


def _detect_protocol(endpoint: str) -> str:
    """Detect API protocol from endpoint.

    Detection probes (fast, 2s timeout each):
      ``/health``    → TEI
      ``/api/tags``  → Ollama
      ``/v1/models`` → OpenAI-compatible

    Returns ``'unknown'`` if none respond.
    """
    ep = endpoint.rstrip("/")

    try:
        r = requests.get(f"{ep}/health", timeout=2)
        if r.status_code == 200:
            return "tei"
    except requests.RequestException:
        pass

    try:
        r = requests.get(f"{ep}/api/tags", timeout=2)
        if r.status_code == 200:
            return "ollama"
    except requests.RequestException:
        pass

    try:
        r = requests.get(f"{ep}/v1/models", timeout=2)
        if r.status_code == 200:
            return "openai"
    except requests.RequestException:
        pass

    return "unknown"


def gpu_health_check(endpoint: str, timeout_s: float = 2.0) -> bool:
    """Fast health check against the GPU embedding endpoint.

    Probes ``/health`` (TEI), root (Ollama), then ``/v1/models`` (OpenAI).
    Returns ``True`` on first 200 response.
    """
    ep = endpoint.rstrip("/")
    health_paths = ["/health", "", "/v1/models"]
    for path in health_paths:
        try:
            url = f"{ep}{path}" if path else ep
            r = requests.get(url, timeout=min(timeout_s, 2.0))
            if r.status_code == 200:
                return True
        except requests.RequestException:
            continue
    return False


class EmbeddingClient:
    """Thread-safe-ish embedding client with GPU→CPU fallback.

    Config is read lazily on first use via ``config_reader`` so a plugin
    can point it at any config section (e.g. ``skills.config.skill-graph``).
    """

    def __init__(self, config_reader: Callable[[], dict]) -> None:
        self._config_reader = config_reader
        self._cfg: dict | None = None
        self._availability_cache: dict = {"available": None, "checked_at": 0.0}

    # ── config ──────────────────────────────────────────────────

    def _config(self) -> dict:
        if self._cfg is None:
            self._cfg = _read_cfg(self._config_reader)
        return self._cfg

    def backend(self) -> str:
        return _get_backend(self._config())

    def endpoint(self) -> str:
        return _get_endpoint(self._config())

    def model_name(self) -> str:
        return _get_model(self._config())

    # ── availability ────────────────────────────────────────────

    def is_available(self, cache_seconds: float = 60.0) -> bool:
        """Check if any embedding backend (GPU or CPU) is reachable.

        Cached for *cache_seconds* to avoid 2s timeout probes on every call.
        Returns True if the GPU endpoint is reachable OR the CPU fallback
        module is importable.
        """
        now = time.time()
        cache = self._availability_cache
        if (
            cache["available"] is not None
            and (now - cache["checked_at"]) < cache_seconds
        ):
            return bool(cache["available"])

        gpu_ok = gpu_health_check(self.endpoint(), timeout_s=1.0)
        cpu_ok = False
        try:
            import sentence_transformers  # noqa: F401
            cpu_ok = True
        except ImportError:
            cpu_ok = False

        available = gpu_ok or cpu_ok
        cache["available"] = available
        cache["checked_at"] = now
        if not available:
            _logger.warning(
                "No embedding backend available (GPU down, CPU model not "
                "installed) — vector retrieval will be skipped"
            )
        return available

    # ── embed ──────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings via configured backend, CPU fallback."""
        if not texts:
            return []

        t0 = time.time()
        preview = texts[0][:80] + ("..." if len(texts[0]) > 80 else "")
        n = len(texts)
        backend = self.backend()

        # Fast path: cpu-only
        if backend == "cpu":
            return _embed_cpu(texts)

        ep = self.endpoint()

        # GPU path — fast pre-check to avoid long timeouts when service is down
        if not gpu_health_check(ep, timeout_s=2.0):
            _logger.warning(
                "Embed [GPU] health check failed after %.2fs — falling back to CPU",
                time.time() - t0,
            )
            return _embed_cpu(texts)

        protocol = _detect_protocol(ep)

        try:
            if protocol == "ollama":
                result = _embed_ollama(texts, ep, self.model_name())
                label = "Ollama"
            elif protocol == "openai":
                result = _embed_openai(texts, ep, self.model_name())
                label = "OpenAI"
            elif protocol == "tei":
                result = _embed_tei(texts, ep)
                label = "TEI"
            else:
                _logger.warning("Unknown protocol at %s — falling back to CPU", ep)
                return _embed_cpu(texts)

            dim = len(result[0]) if result else 0
            _logger.info(
                "Embed [%s] %d texts, %d-dim, %.2fs — preview=%r",
                label, n, dim, time.time() - t0, preview,
            )
            return result

        except Exception as e:
            _logger.warning(
                "Embed [GPU] %d texts failed after %.2fs: %s — falling back to CPU",
                n, time.time() - t0, e,
            )
            return _embed_cpu(texts)


# ── GPU backends ────────────────────────────────────────────────


def _embed_tei(texts: list[str], endpoint: str) -> list[list[float]]:
    url = endpoint + "/embed"
    payload = {"inputs": texts if len(texts) > 1 else texts[0]}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data and isinstance(data[0], float):
        return [data]
    if isinstance(data, list) and data and isinstance(data[0], list):
        return data
    if isinstance(data, list):
        return data
    raise RuntimeError(f"Unexpected TEI response format: {type(data)}")


def _embed_ollama(
    texts: list[str], endpoint: str, model: str
) -> list[list[float]]:
    url = endpoint + "/api/embeddings"
    results = []
    for text in texts:
        payload = {"model": model, "prompt": text}
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        emb = data.get("embedding")
        if emb:
            results.append(emb)
        else:
            raise RuntimeError(f"Unexpected Ollama response: {data}")
    return results


def _embed_openai(
    texts: list[str], endpoint: str, model: str
) -> list[list[float]]:
    url = endpoint + "/v1/embeddings"
    payload = {"model": model, "input": texts}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    entries = data.get("data", [])
    entries.sort(key=lambda x: x.get("index", 0))
    return [e["embedding"] for e in entries]


# ── CPU fallback ────────────────────────────────────────────────


def _embed_cpu(texts: list[str]) -> list[list[float]]:
    """Fallback embedding using local sentence-transformers on CPU."""
    global _CPU_MODEL
    if _CPU_MODEL is None:
        _logger.info("Loading CPU embedding model (BAAI/bge-m3, 1024-dim)...")
        from sentence_transformers import SentenceTransformer
        _CPU_MODEL = SentenceTransformer("BAAI/bge-m3", device="cpu")
    vecs = _CPU_MODEL.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


# ── vector helpers ─────────────────────────────────────────────


def to_blob(vec: list[float]) -> bytes:
    """Serialize a float list to a float32 BLOB (1024-dim → 4096 bytes)."""
    import numpy as np
    return np.array(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> "list[float]":
    """Deserialize a float32 BLOB back to a float list."""
    import numpy as np
    return np.frombuffer(blob, dtype=np.float32).tolist()


def cosine(a_blob: bytes, b_blob: bytes) -> float:
    """Cosine similarity between two normalized embedding BLOBs.

    Since encoders normalize, this is equivalent to dot product.
    """
    import numpy as np
    va = np.frombuffer(a_blob, dtype=np.float32)
    vb = np.frombuffer(b_blob, dtype=np.float32)
    return float(np.dot(va, vb))


def cosine_batch(query_blob: bytes, blobs: list[bytes]) -> list[float]:
    """Cosine similarity of a query against a batch of BLOBs (vectorized)."""
    import numpy as np
    q = np.frombuffer(query_blob, dtype=np.float32)
    m = np.array([np.frombuffer(b, dtype=np.float32) for b in blobs])
    return (m @ q).tolist()
