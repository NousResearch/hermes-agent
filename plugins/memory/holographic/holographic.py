"""Holographic Reduced Representations (HRR) with phase encoding.

HRRs are a vector symbolic architecture for encoding compositional structure
into fixed-width distributed representations. This module uses *phase vectors*:
each concept is a vector of angles in [0, 2π). The algebraic operations are:

  bind   — circular convolution (phase addition)  — associates two concepts
  unbind — circular correlation (phase subtraction) — retrieves a bound value
  bundle — superposition (circular mean)           — merges multiple concepts

Phase encoding is numerically stable, avoids the magnitude collapse of
traditional complex-number HRRs, and maps cleanly to cosine similarity.

Atoms are generated deterministically from SHA-256 so representations are
identical across processes, machines, and language versions.

References:
  Plate (1995) — Holographic Reduced Representations
  Gayler (2004) — Vector Symbolic Architectures answer Jackendoff's challenges
"""

import hashlib
import logging
import struct
import math

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

logger = logging.getLogger(__name__)

_TWO_PI = 2.0 * math.pi
_FLOAT32_BLOB_PREFIX = b"HRR1"


def _require_numpy() -> None:
    if not _HAS_NUMPY:
        raise RuntimeError("numpy is required for holographic operations")


def _np():
    """Return the numpy module after the runtime availability guard."""
    _require_numpy()
    return np  # type: ignore[name-defined]


def encode_atom(word: str, dim: int = 1024) -> "np.ndarray":
    """Deterministic phase vector via SHA-256 counter blocks.

    Uses hashlib (not numpy RNG) for cross-platform reproducibility.

    Algorithm:
    - Generate enough SHA-256 blocks by hashing f"{word}:{i}" for i=0,1,2,...
    - Concatenate digests, interpret as uint16 values via struct.unpack
    - Scale to [0, 2π): phases = values * (2π / 65536)
    - Truncate to dim elements
    - Returns np.float64 array of shape (dim,)
    """
    _require_numpy()

    # Each SHA-256 digest is 32 bytes = 16 uint16 values.
    values_per_block = 16
    blocks_needed = math.ceil(dim / values_per_block)

    uint16_values: list[int] = []
    for i in range(blocks_needed):
        digest = hashlib.sha256(f"{word}:{i}".encode()).digest()
        uint16_values.extend(struct.unpack("<16H", digest))

    phases = np.array(uint16_values[:dim], dtype=np.float64) * (_TWO_PI / 65536.0)
    return phases


def bind(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    """Circular convolution = element-wise phase addition.

    Binding associates two concepts into a single composite vector.
    The result is dissimilar to both inputs (quasi-orthogonal).
    """
    _require_numpy()
    return (a + b) % _TWO_PI


def unbind(memory: "np.ndarray", key: "np.ndarray") -> "np.ndarray":
    """Circular correlation = element-wise phase subtraction.

    Unbinding retrieves the value associated with a key from a memory vector.
    unbind(bind(a, b), a) ≈ b  (up to superposition noise)
    """
    _require_numpy()
    return (memory - key) % _TWO_PI


def bundle(*vectors: "np.ndarray") -> "np.ndarray":
    """Superposition via circular mean of complex exponentials.

    Bundling merges multiple vectors into one that is similar to each input.
    The result can hold O(sqrt(dim)) items before similarity degrades.
    """
    _require_numpy()
    complex_sum = np.sum([np.exp(1j * v) for v in vectors], axis=0)
    return np.angle(complex_sum) % _TWO_PI


def similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    """Phase cosine similarity. Range [-1, 1].

    Returns 1.0 for identical vectors, near 0.0 for random (unrelated) vectors,
    and -1.0 for perfectly anti-correlated vectors.
    """
    _require_numpy()
    return float(np.mean(np.cos(a - b)))


def encode_text(text: str, dim: int = 1024) -> "np.ndarray":
    """Bag-of-words: bundle of atom vectors for each token.

    Tokenizes by lowercasing, splitting on whitespace, and stripping
    leading/trailing punctuation from each token.

    Returns bundle of all token atom vectors.
    If text is empty or produces no tokens, returns encode_atom("__hrr_empty__", dim).
    """
    _require_numpy()

    tokens = [
        token.strip(".,!?;:\"'()[]{}")
        for token in text.lower().split()
    ]
    tokens = [t for t in tokens if t]

    if not tokens:
        return encode_atom("__hrr_empty__", dim)

    atom_vectors = [encode_atom(token, dim) for token in tokens]
    return bundle(*atom_vectors)


def encode_fact(content: str, entities: list[str], dim: int = 1024) -> "np.ndarray":
    """Structured encoding: content bound to ROLE_CONTENT, each entity bound to ROLE_ENTITY, all bundled.

    Role vectors are reserved atoms: "__hrr_role_content__", "__hrr_role_entity__"

    Components:
    1. bind(encode_text(content, dim), encode_atom("__hrr_role_content__", dim))
    2. For each entity: bind(encode_atom(entity.lower(), dim), encode_atom("__hrr_role_entity__", dim))
    3. bundle all components together

    This enables algebraic extraction:
        unbind(fact, bind(entity, ROLE_ENTITY)) ≈ content_vector
    """
    _require_numpy()

    role_content = encode_atom("__hrr_role_content__", dim)
    role_entity = encode_atom("__hrr_role_entity__", dim)

    components: list[np.ndarray] = [
        bind(encode_text(content, dim), role_content)
    ]

    for entity in entities:
        components.append(bind(encode_atom(entity.lower(), dim), role_entity))

    return bundle(*components)


def phases_to_bytes(phases: "np.ndarray", dim: int | None = None) -> bytes:
    """Serialize phase vectors as float32 blobs.

    float32 halves SQLite BLOB storage versus the legacy float64 format
    (4 KB + a 4-byte format prefix instead of 8 KB at dim=1024) while
    preserving enough precision for phase-similarity retrieval.
    ``bytes_to_phases`` keeps reading legacy float64 blobs for backward
    compatibility.

    When ``dim`` is 1 the prefixed float32 blob (8 bytes) collides in size
    with a raw float64 blob (8 bytes), making the format ambiguous.  In
    that case we fall back to writing raw float64 so that ``bytes_to_phases``
    can never misinterpret the blob.
    """
    numpy = _np()
    if dim is None:
        dim = int(phases.shape[0])
    float32_blob_bytes = len(_FLOAT32_BLOB_PREFIX) + dim * numpy.dtype(numpy.float32).itemsize
    float64_bytes = dim * numpy.dtype(numpy.float64).itemsize
    if float32_blob_bytes == float64_bytes:
        # dim=1: sizes collide, write legacy float64 to stay unambiguous
        return numpy.asarray(phases, dtype=numpy.float64).tobytes()
    payload = numpy.asarray(phases, dtype=numpy.float32).tobytes()
    return _FLOAT32_BLOB_PREFIX + payload


def bytes_to_phases(data: bytes, dim: int | None = None) -> "np.ndarray":
    """Deserialize a phase vector from new float32 or legacy float64 storage.

    New float32 blobs carry a small prefix so callers can round-trip without
    knowing ``dim``. Legacy float64 blobs are raw NumPy bytes and remain
    readable for backward compatibility. The returned array is copied and
    promoted to float64 so downstream HRR math keeps the existing numerical
    behavior.

    When ``dim`` is 1 the prefixed float32 blob and the raw float64 blob are
    both 8 bytes, so size alone cannot disambiguate.  ``phases_to_bytes``
    avoids writing prefixed blobs in that case; here we guard the remaining
    collision window (a legacy float64 blob that happens to start with the
    ``HRR1`` prefix) by preferring the legacy interpretation when sizes
    match and the caller supplied ``dim``.
    """
    numpy = _np()

    if dim is not None:
        float32_payload_bytes = dim * numpy.dtype(numpy.float32).itemsize
        float32_blob_bytes = len(_FLOAT32_BLOB_PREFIX) + float32_payload_bytes
        float64_bytes = dim * numpy.dtype(numpy.float64).itemsize

        # When sizes collide (dim=1), prefer legacy float64 for a blob that
        # starts with the prefix, because phases_to_bytes never writes a
        # prefixed float32 blob at dim=1 — any such blob must be legacy.
        if float32_blob_bytes == float64_bytes:
            if len(data) == float64_bytes:
                return numpy.frombuffer(data, dtype=numpy.float64).copy()
            if data.startswith(_FLOAT32_BLOB_PREFIX):
                payload_len = len(data) - len(_FLOAT32_BLOB_PREFIX)
                raise ValueError(
                    f"HRR vector blob has {len(data)} bytes ({payload_len} payload bytes after "
                    f"the float32 prefix); expected {float64_bytes} (legacy float64) for dim={dim}"
                )
            raise ValueError(
                f"HRR legacy vector blob has {len(data)} bytes; expected "
                f"{float64_bytes} (float64) for dim={dim}"
            )

        if data.startswith(_FLOAT32_BLOB_PREFIX) and len(data) == float32_blob_bytes:
            payload = data[len(_FLOAT32_BLOB_PREFIX):]
            return numpy.frombuffer(payload, dtype=numpy.float32).astype(numpy.float64)
        if len(data) == float64_bytes:
            return numpy.frombuffer(data, dtype=numpy.float64).copy()
        if data.startswith(_FLOAT32_BLOB_PREFIX):
            payload_len = len(data) - len(_FLOAT32_BLOB_PREFIX)
            raise ValueError(
                f"HRR vector blob has {len(data)} bytes ({payload_len} payload bytes after "
                f"the float32 prefix); expected {float32_blob_bytes} (prefixed float32) "
                f"or {float64_bytes} (legacy float64) for dim={dim}"
            )
        raise ValueError(
            f"HRR legacy vector blob has {len(data)} bytes; expected "
            f"{float64_bytes} (float64) for dim={dim}"
        )

    if data.startswith(_FLOAT32_BLOB_PREFIX):
        payload = data[len(_FLOAT32_BLOB_PREFIX):]
        if len(payload) % numpy.dtype(numpy.float32).itemsize != 0:
            raise ValueError(
                f"HRR float32 vector blob has invalid payload byte length: {len(payload)}"
            )
        return numpy.frombuffer(payload, dtype=numpy.float32).astype(numpy.float64)

    if len(data) % numpy.dtype(numpy.float64).itemsize != 0:
        raise ValueError(f"HRR legacy vector blob has invalid byte length: {len(data)}")
    return numpy.frombuffer(data, dtype=numpy.float64).copy()


def snr_estimate(dim: int, n_items: int) -> float:
    """Signal-to-noise ratio estimate for holographic storage.

    SNR = sqrt(dim / n_items) when n_items > 0, else inf.

    The SNR falls below 2.0 when n_items > dim / 4, meaning retrieval
    errors become likely. Logs a warning when this threshold is crossed.
    """
    _require_numpy()

    if n_items <= 0:
        return float("inf")

    snr = math.sqrt(dim / n_items)

    if snr < 2.0:
        logger.warning(
            "HRR storage near capacity: SNR=%.2f (dim=%d, n_items=%d). "
            "Retrieval accuracy may degrade. Consider increasing dim or reducing stored items.",
            snr,
            dim,
            n_items,
        )

    return snr
