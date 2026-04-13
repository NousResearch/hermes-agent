import math
import re
from typing import Callable, List, Dict, Any, Awaitable

def split_sentences(text: str) -> List[str]:
    """Split text into sentences, handling basic punctuation."""
    # Split on sentence-ending punctuation followed by whitespace or newline
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a)
    norm_b_sq = sum(y * y for y in b)
    denom = math.sqrt(norm_a) * math.sqrt(norm_b_sq)
    return dot / denom if denom != 0 else 0.0

def _transpose(m: List[List[float]]) -> List[List[float]]:
    rows = len(m)
    cols = len(m[0])
    res = [[0.0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            res[j][i] = m[i][j]
    return res

def _mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    res = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            for k in range(inner):
                res[i][j] += a[i][k] * b[k][j]
    return res

def _invert_matrix(m: List[List[float]]) -> List[List[float]]:
    n = len(m)
    aug = [row + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(m)]
    
    for col in range(n):
        # find pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular")
            
        for j in range(2 * n):
            aug[col][j] /= pivot
            
        for row in range(n):
            if row == col: continue
            factor = aug[row][col]
            for j in range(2 * n):
                aug[row][j] -= factor * aug[col][j]
                
    return [row[n:] for row in aug]

def _factorial(n: int) -> int:
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res

def savitzky_golay(data: List[float], window_size: int, poly_order: int, deriv_order: int) -> List[float]:
    half = window_size // 2
    n = len(data)
    if n < window_size:
        return data[:]
        
    # Build Vandermonde matrix
    J = []
    for i in range(-half, half + 1):
        row = [math.pow(i, j) for j in range(poly_order + 1)]
        J.append(row)
        
    JT = _transpose(J)
    JTJ = _mat_mul(JT, J)
    JTJinv = _invert_matrix(JTJ)
    coeffs = _mat_mul(JTJinv, JT)
    
    filter_row = coeffs[deriv_order]
    fact = _factorial(deriv_order)
    
    res = [0.0] * n
    for i in range(n):
        val = 0.0
        for j in range(-half, half + 1):
            idx = min(max(i + j, 0), n - 1)
            val += filter_row[j + half] * data[idx]
        res[i] = val * fact
    return res

def _percentile(arr: List[float], p: float) -> float:
    if not arr: return 0.0
    sorted_arr = sorted(arr)
    idx = int(math.floor(p * len(sorted_arr)))
    return sorted_arr[min(idx, len(sorted_arr) - 1)]

def _enforce_min_distance(boundaries: List[int], min_dist: int) -> List[int]:
    if len(boundaries) <= 1:
        return boundaries
    res = [boundaries[0]]
    for i in range(1, len(boundaries)):
        if boundaries[i] - res[-1] >= min_dist:
            res.append(boundaries[i])
    return res

def find_boundaries_savgol(similarities: List[float]) -> List[int]:
    deriv = savitzky_golay(similarities, 5, 3, 1)
    minima = []
    for i in range(1, len(deriv)):
        if deriv[i - 1] < 0 and deriv[i] >= 0:
            minima.append(i)
            
    if not similarities: return []
    threshold = _percentile(similarities, 0.2)
    filtered = []
    for m in minima:
        sim_idx = min(m, len(similarities) - 1)
        if similarities[sim_idx] <= threshold:
            filtered.append(m)
            
    return _enforce_min_distance(filtered, 2)

def find_boundaries_percentile(similarities: List[float]) -> List[int]:
    if not similarities: return []
    threshold = _percentile(similarities, 0.2)
    boundaries = []
    for i, sim in enumerate(similarities):
        if sim <= threshold:
            boundaries.append(i + 1)
    return _enforce_min_distance(boundaries, 2)

def find_boundaries(similarities: List[float]) -> List[int]:
    if len(similarities) < 5:
        return find_boundaries_percentile(similarities)
    try:
        return find_boundaries_savgol(similarities)
    except Exception:
        return find_boundaries_percentile(similarities)

def group_at_boundaries(sentences: List[str], boundaries: List[int]) -> List[List[str]]:
    groups = []
    start = 0
    for b in boundaries:
        if 0 < b < len(sentences):
            groups.append(sentences[start:b])
            start = b
    if start < len(sentences):
        groups.append(sentences[start:])
    return groups if groups else [sentences]

async def semantic_chunk_text(text: str, embed_fn: Callable[[List[str]], Awaitable[List[List[float]]]], chunk_size: int = 300) -> List[str]:
    """Mathematically detects topic shifts and chunks text while preserving meaning.
    Requires an async embed_fn that takes List[str] and returns List[vector].
    """
    sentences = split_sentences(text)
    if len(sentences) <= 3:
        return [text]
        
    try:
        embeddings = await embed_fn(sentences)
        if len(embeddings) != len(sentences):
            raise ValueError("Embedding count mismatch")
            
        similarities = []
        for i in range(len(embeddings) - 1):
            similarities.append(cosine_similarity(embeddings[i], embeddings[i+1]))
            
        boundaries = find_boundaries(similarities)
        groups = group_at_boundaries(sentences, boundaries)
        
        chunks = []
        for g in groups:
            chunk_text = " ".join(g)
            word_count = len(chunk_text.split())
            if word_count > chunk_size * 1.5:
                # Naive split if still too large
                half = len(g) // 2
                if half > 0:
                    chunks.append(" ".join(g[:half]))
                    chunks.append(" ".join(g[half:]))
                else:
                    chunks.append(chunk_text)
            else:
                chunks.append(chunk_text)
        return chunks
    except Exception:
        # Fallback to naive
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i:i+chunk_size]))
        return chunks
