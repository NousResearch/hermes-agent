# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
fast_ops.pyx — Cython-accelerated hot paths for Neural Memory.

Targets:
  - cosine_similarity (385K calls in 100 recalls)
  - batch_cosine_similarity (score N memories against query)
  - temporal_decay (math.exp loop)
  - hash_embed (deterministic hash-based embedding)
  - content_differs (string comparison for conflict detection)
"""

import cython
from libc.math cimport sqrt, exp, fabs
from libc.string cimport memcpy
cimport cython as cy

# ---------------------------------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def cosine_similarity(double[:] a, double[:] b) -> double:
    """Cosine similarity between two vectors. ~50x faster than Python sum()."""
    cdef int n = a.shape[0]
    cdef int i
    cdef double dot = 0.0, na = 0.0, nb = 0.0

    for i in range(n):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]

    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


@cython.boundscheck(False)
@cython.wraparound(False)
def batch_cosine_similarity(double[:] query, double[:, :] matrix):
    """Score query against N vectors. Returns list of similarities.

    query: shape (D,)
    matrix: shape (N, D)

    Returns: list of N floats
    """
    cdef int n = matrix.shape[0]
    cdef int d = matrix.shape[1]
    cdef int i, j
    cdef double dot, nb, nq = 0.0

    # Precompute query norm
    for j in range(d):
        nq += query[j] * query[j]
    nq = sqrt(nq)

    if nq == 0.0:
        return [0.0] * n

    cdef list results = []
    for i in range(n):
        dot = 0.0
        nb = 0.0
        for j in range(d):
            dot += query[j] * matrix[i, j]
            nb += matrix[i, j] * matrix[i, j]
        if nb == 0.0:
            results.append(0.0)
        else:
            results.append(dot / (nq * sqrt(nb)))

    return results


# ---------------------------------------------------------------------------
# Temporal Decay
# ---------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def temporal_decay(double[:] timestamps, double now, double half_life=24.0):
    """Exponential decay scores for timestamps.

    timestamps: array of epoch seconds
    now: current epoch
    half_life: hours for 50% decay

    Returns: list of decay scores (0-1)
    """
    cdef int n = timestamps.shape[0]
    cdef int i
    cdef double decay_factor = -0.693 / (half_life * 3600.0)
    cdef double age

    cdef list results = []
    for i in range(n):
        if timestamps[i] <= 0:
            results.append(0.5)
        else:
            age = now - timestamps[i]
            results.append(exp(decay_factor * age))
    return results


# ---------------------------------------------------------------------------
# Hash Embedding (deterministic, no dependencies)
# ---------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def hash_embed(str text, int dim=384):
    """Deterministic hash-based embedding. ~3x faster than Python version.

    Returns: list of floats length dim, normalized.
    """
    import hashlib

    cdef bytes h = hashlib.sha256(text.encode('utf-8')).digest()
    cdef int hlen = len(h)
    cdef list vec = []
    cdef int i
    cdef double v, norm = 0.0

    for i in range(dim):
        v = <double>(<unsigned char>h[i % hlen] - 128) / 128.0
        vec.append(v)
        norm += v * v

    norm = sqrt(norm)
    if norm > 0:
        vec = [v / norm for v in vec]

    return vec


# ---------------------------------------------------------------------------
# Batch Hash Embed
# ---------------------------------------------------------------------------

def batch_hash_embed(list texts, int dim=384):
    """Hash embed a batch of texts. Returns list of vectors."""
    return [hash_embed(t, dim) for t in texts]


# ---------------------------------------------------------------------------
# Content Differs (for conflict detection)
# ---------------------------------------------------------------------------

def content_differs(str old_text, str new_text) -> bool:
    """Check if two texts contain different information.

    Heuristics: different numbers, significant word differences.
    ~2x faster than Python regex version.
    """
    import re

    old_clean = old_text.replace("[SUPERSEDED]", "").replace("[UPDATED TO]", "").strip()

    # Extract numbers
    old_nums = set(re.findall(r'\d+\.?\d*', old_clean))
    new_nums = set(re.findall(r'\d+\.?\d*', new_text))

    if old_nums and new_nums and old_nums != new_nums:
        return True

    # Significant word difference
    old_words = set(old_clean.lower().split())
    new_words = set(new_text.lower().split())
    if len(old_words) == 0 or len(new_words) == 0:
        return False

    overlap = len(old_words & new_words)
    total = len(old_words | new_words)
    return (1.0 - <double>overlap / total) > 0.5


# ---------------------------------------------------------------------------
# Top-K selection (avoid full sort for large arrays)
# ---------------------------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def top_k_indices(double[:] scores, int k):
    """Find indices of top-k scores without full sort.

    Returns: list of (index, score) tuples sorted descending.
    """
    cdef int n = scores.shape[0]
    cdef int i, j
    cdef list heap = []  # min-heap of (score, index)

    if k <= 0 or n == 0:
        return []

    if k >= n:
        return sorted(enumerate(scores), key=lambda x: -x[1])[:k]

    # Simple linear scan for top-k (faster than heapq for small k)
    cdef list top_indices = list(range(min(k, n)))
    top_indices.sort(key=lambda i: -scores[i])

    cdef double min_score = scores[top_indices[-1]]

    for i in range(k, n):
        if scores[i] > min_score:
            # Insert in sorted position
            for j in range(k):
                if scores[i] > scores[top_indices[j]]:
                    top_indices.insert(j, i)
                    top_indices.pop()
                    min_score = scores[top_indices[-1]]
                    break

    return [(idx, scores[idx]) for idx in top_indices]
