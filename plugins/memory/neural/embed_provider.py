#!/usr/bin/env python3
"""
embed_provider.py - Text Embedding for Neural Memory Adapter
Tries sentence-transformers first, falls back to TF-IDF+SVD.
"""

import os
import sys
import pickle
import hashlib
import re
from pathlib import Path

CACHE_DIR = Path.home() / ".neural_memory"
CACHE_FILE = CACHE_DIR / "embed_cache.pkl"
MODEL_DIR = CACHE_DIR / "models"
DIMENSION = 384  # all-MiniLM-L6-v2 output dim

# ============================================================================
# Embedding backends
# ============================================================================

class SentenceTransformerBackend:
    """Uses sentence-transformers (all-MiniLM-L6-v2, 384d, ~80MB)
    
    Singleton: model loaded once and shared across all instances.
    Cached locally at ~/.neural_memory/models/.
    """
    MODEL_NAME = 'all-MiniLM-L6-v2'
    _shared_model = None
    _shared_dim = 384
    
    def __init__(self):
        if SentenceTransformerBackend._shared_model is not None:
            self.model = SentenceTransformerBackend._shared_model
            self.dim = SentenceTransformerBackend._shared_dim
            return
        
        from sentence_transformers import SentenceTransformer
        import torch
        
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Detect best device (with GPU memory check)
        device = 'cpu'
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                gpu_mem = getattr(props, 'total_mem', None) or getattr(props, 'total_global_memory', 0) / 1024**3
                free_mem = (torch.cuda.mem_get_info(0)[0]) / 1024**2  # MB free
                if free_mem > 500:  # Need at least 500MB for sentence-transformers
                    device = 'cuda'
                    print(f"[embed] CUDA: {gpu_name} ({gpu_mem:.1f} GB, {free_mem:.0f} MB free)")
                else:
                    print(f"[embed] CUDA: {gpu_name} but only {free_mem:.0f} MB free — using CPU")
            except Exception:
                print(f"[embed] CUDA detected but memory check failed — using CPU")
        if device == 'cpu':
            print(f"[embed] CPU only")
        
        cached_model_dir = MODEL_DIR / f"models--sentence-transformers--{self.MODEL_NAME}"
        is_cached = cached_model_dir.exists()
        
        if is_cached:
            print(f"[embed] Loading {self.MODEL_NAME} from local cache...")
        else:
            print(f"[embed] Downloading {self.MODEL_NAME} (~80MB) to {MODEL_DIR}...")
            print("This only happens once. Please wait...", flush=True)
        
        try:
            self.model = SentenceTransformer(
                self.MODEL_NAME,
                cache_folder=str(MODEL_DIR),
                device=device,
                local_files_only=is_cached
            )
            self.dim = 384
            SentenceTransformerBackend._shared_model = self.model
            SentenceTransformerBackend._shared_dim = self.dim
            
            if not is_cached:
                print(f"[embed] Model cached successfully!", flush=True)
            
            print(f"[embed] {self.MODEL_NAME} ready on {device}")
        except Exception as e:
            print(f"[embed] Failed to load model: {e}", file=sys.stderr)
            raise
    
    def embed(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


class TfidfSvdBackend:
    """Pure numpy TF-IDF + SVD embedding (no ML dependencies)"""
    def __init__(self, dim: int = DIMENSION):
        import numpy as np
        self.np = np
        self.dim = dim
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.svd_components: np.ndarray | None = None
        self._trained = False
        self._corpus: list[str] = []
    
    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()
    
    def _hash_embed(self, text: str) -> list[float]:
        """Hash-based fallback for pre-training"""
        import math
        vec = [0.0] * self.dim
        tokens = self._tokenize(text)
        for i, token in enumerate(tokens):
            h = hash(token)
            for j in range(4):
                idx = (h ^ (j * 2654435761)) % self.dim
                vec[idx] += 1.0 / (1.0 + i * 0.1)
                h = (h >> 8) | ((h & 0xFF) << 24)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
    
    def fit(self, texts: list[str]):
        """Fit on a corpus of texts"""
        np = self.np
        # Build vocabulary
        doc_freq = {}
        all_tokens = []
        for text in texts:
            tokens = set(self._tokenize(text))
            all_tokens.append(list(tokens))
            for t in tokens:
                doc_freq[t] = doc_freq.get(t, 0) + 1
        
        # Keep top 10000 tokens
        sorted_vocab = sorted(doc_freq.items(), key=lambda x: -x[1])[:10000]
        self.vocab = {word: i for i, (word, _) in enumerate(sorted_vocab)}
        vocab_size = len(self.vocab)
        
        # IDF
        n_docs = len(texts)
        self.idf = np.zeros(vocab_size)
        for word, idx in self.vocab.items():
            self.idf[idx] = np.log((n_docs + 1) / (doc_freq.get(word, 1) + 1)) + 1
        
        # Build TF-IDF matrix
        tfidf = np.zeros((n_docs, vocab_size))
        for i, tokens in enumerate(all_tokens):
            for t in tokens:
                if t in self.vocab:
                    tfidf[i, self.vocab[t]] += 1
            # Apply IDF
            tfidf[i] *= self.idf
            # Normalize
            norm = np.linalg.norm(tfidf[i])
            if norm > 0:
                tfidf[i] /= norm
        
        # SVD for dimensionality reduction
        if vocab_size > self.dim:
            # Randomized SVD for speed
            U, S, Vt = np.linalg.svd(tfidf[:, :min(vocab_size, 5000)], full_matrices=False)
            self.svd_components = Vt[:self.dim].T  # (vocab, dim)
        else:
            self.svd_components = np.eye(vocab_size, self.dim)
        
        self._trained = True
    
    def embed(self, text: str) -> list[float]:
        self._corpus.append(text)
        if not self._trained:
            if len(self._corpus) >= 5:
                self.fit(self._corpus)
            return self._hash_embed(text)
        
        np = self.np
        tokens = self._tokenize(text)
        
        vec = np.zeros(len(self.vocab))
        for t in tokens:
            if t in self.vocab:
                vec[self.vocab[t]] += 1
        
        if self.idf is not None:
            vec *= self.idf
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        
        if self.svd_components is not None:
            vec = vec @ self.svd_components
        
        # Pad or truncate to dim
        result = np.zeros(self.dim)
        result[:min(len(vec), self.dim)] = vec[:self.dim]
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
        
        return result.tolist()
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Optimized batch embedding using vectorized operations."""
        np = self.np
        
        # If not trained yet, check if we have enough to train
        if not self._trained:
            self._corpus.extend(texts)
            if len(self._corpus) >= 5:
                self.fit(self._corpus)
            # Return hash embeddings for each text
            return [self._hash_embed(t) for t in texts]
        
        # Tokenize all texts at once
        all_tokens = [self._tokenize(t) for t in texts]
        n_texts = len(texts)
        vocab_size = len(self.vocab)
        
        # Build TF matrix for all texts at once
        tfidf = np.zeros((n_texts, vocab_size))
        for i, tokens in enumerate(all_tokens):
            for t in tokens:
                if t in self.vocab:
                    tfidf[i, self.vocab[t]] += 1
        
        # Apply IDF
        if self.idf is not None:
            tfidf *= self.idf
        
        # Normalize rows
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        tfidf /= norms
        
        # Apply SVD projection
        if self.svd_components is not None:
            result = tfidf @ self.svd_components
        else:
            result = tfidf
        
        # Pad or truncate to dim and normalize
        batch_result = np.zeros((n_texts, self.dim))
        copy_len = min(result.shape[1], self.dim) if result.ndim > 1 else min(len(result), self.dim)
        if result.ndim > 1:
            batch_result[:, :copy_len] = result[:, :copy_len]
        else:
            batch_result[0, :copy_len] = result[:copy_len]
        
        # Final normalization
        norms = np.linalg.norm(batch_result, axis=1, keepdims=True)
        norms[norms == 0] = 1
        batch_result /= norms
        
        return [row.tolist() for row in batch_result]


class HashBackend:
    """Simple hash-based embedding (zero dependencies)"""
    def __init__(self, dim: int = DIMENSION):
        self.dim = dim
    
    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()
    
    def embed(self, text: str) -> list[float]:
        import math
        vec = [0.0] * self.dim
        tokens = self._tokenize(text)
        
        for i, token in enumerate(tokens):
            h = hash(token)
            # Distribute across dimensions
            for j in range(3):  # 3 positions per token
                idx = (h ^ (j * 2654435761)) % self.dim
                vec[idx] += 1.0 / (1.0 + i * 0.1)  # Position decay
                h = (h >> 8) | ((h & 0xFF) << 24)
        
        # Normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        
        return vec
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# ============================================================================
# Provider with caching
# ============================================================================

class EmbeddingProvider:
    def __init__(self, backend: str = "auto"):
        self.cache: dict[str, list[float]] = {}
        self._load_cache()
        
        if backend == "auto":
            self.backend = self._auto_detect()
        elif backend == "sentence-transformers":
            self.backend = SentenceTransformerBackend()
        elif backend == "tfidf":
            self.backend = TfidfSvdBackend()
        else:
            self.backend = HashBackend()
        
        print(f"Embedding backend: {self.backend.__class__.__name__} ({self.backend.dim}d)")
    
    def _auto_detect(self):
        """Auto-detect best available backend.
        Priority: CUDA sentence-transformers > CPU sentence-transformers > TF-IDF > hash
        """
        # Try sentence-transformers first (CUDA or CPU)
        try:
            import sentence_transformers
            import torch
            device = "CUDA" if torch.cuda.is_available() else "CPU"
            backend = SentenceTransformerBackend()
            print(f"[embed] Auto-selected: sentence-transformers ({device})")
            return backend
        except (ImportError, Exception) as e:
            if not isinstance(e, ImportError):
                print(f"[embed] sentence-transformers failed: {e}", file=sys.stderr)
        
        # TF-IDF+SVD
        try:
            import numpy
            print("[embed] Auto-selected: TF-IDF+SVD")
            return TfidfSvdBackend()
        except ImportError:
            pass
        
        # Hash fallback
        print("[embed] Auto-selected: hash (zero dependencies)")
        return HashBackend()
    
    def _load_cache(self):
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'rb') as f:
                    self.cache = pickle.load(f)
            except Exception:
                self.cache = {}
    
    def _save_cache(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def embed(self, text: str) -> list[float]:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        
        vec = self.backend.embed(text)
        self.cache[key] = vec
        
        # Save periodically
        if len(self.cache) % 100 == 0:
            self._save_cache()
        
        return vec
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = []
        to_compute = []
        indices = []
        
        for i, text in enumerate(texts):
            key = hashlib.md5(text.encode()).hexdigest()
            if key in self.cache:
                results.append(self.cache[key])
            else:
                results.append(None)
                to_compute.append(text)
                indices.append(i)
        
        if to_compute:
            computed = self.backend.embed_batch(to_compute)
            for idx, vec in zip(indices, computed):
                results[idx] = vec
                key = hashlib.md5(texts[idx].encode()).hexdigest()
                self.cache[key] = vec
            self._save_cache()
        
        return results
    
    @property
    def dim(self) -> int:
        return self.backend.dim


# ============================================================================
# Standalone test
# ============================================================================

if __name__ == "__main__":
    provider = EmbeddingProvider()
    
    texts = [
        "The user has a dog named Lou",
        "Working on BTQuant trading platform",
        "Neural memory adapter with Hopfield networks",
        "The dog is a Chihuahua",
    ]
    
    for text in texts:
        vec = provider.embed(text)
        print(f"'{text[:50]}...' -> {len(vec)}d vector")
    
    # Similarity test
    import math
    def cosine(a, b):
        dot = sum(x*y for x,y in zip(a,b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(x*x for x in b))
        return dot / (na * nb) if na * nb > 0 else 0
    
    v1 = provider.embed("The user has a dog named Lou")
    v2 = provider.embed("The dog is a Chihuahua")
    v3 = provider.embed("Neural memory adapter with Hopfield networks")
    
    print(f"\nSimilarity 'dog/Lou' vs 'Chihuahua': {cosine(v1, v2):.3f}")
    print(f"Similarity 'dog/Lou' vs 'neural memory': {cosine(v1, v3):.3f}")
    print(f"(Dog-dog should be higher than dog-tech)")
