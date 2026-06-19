"""ONNX embedding module for semantic search.

Wraps nomic-embed-text-v1.5 ONNX model for text embedding generation.
Provides graceful degradation when model is unavailable.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_PATH = Path("~/.aingram/models/nomic-embed-text-v1.5/onnx/model.onnx").expanduser()
_EMBED_DIM = 768


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.
    
    Returns:
        Float in [-1, 1]. 1.0 for identical, 0.0 for orthogonal.
    """
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))


class OnnxEmbedder:
    """ONNX-based text embedder using nomic-embed-text-v1.5."""
    
    def __init__(self, model_path: Optional[Path] = None):
        self._model_path = model_path or _MODEL_PATH
        self._session = None
        self._tokenizer = None
        self._available = False
        
        if self._model_path.exists():
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(str(self._model_path))
                self._available = True
                logger.info("ONNX embedder loaded: %s", self._model_path)
            except Exception as e:
                logger.warning("Failed to load ONNX model: %s", e)
        else:
            logger.warning("ONNX model not found: %s", self._model_path)
    
    @property
    def available(self) -> bool:
        return self._available
    
    @property
    def dim(self) -> int:
        return _EMBED_DIM
    
    def _load_tokenizer(self) -> None:
        """Load tokenizer from tokenizer.json using tokenizers library."""
        if self._tokenizer is not None:
            return
        # Model is at .../nomic-embed-text-v1.5/onnx/model.onnx;
        # tokenizer.json is at .../nomic-embed-text-v1.5/tokenizer.json
        tokenizer_path = self._model_path.parent.parent / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        
        try:
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except ImportError:
            raise ImportError("tokenizers library required: pip install tokenizers")
    
    def _tokenize(self, text: str) -> list[int]:
        """Tokenize text using loaded tokenizer."""
        self._load_tokenizer()
        
        encoding = self._tokenizer.encode(text)
        return encoding.ids
    
    def embed(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text.
        
        Args:
            text: Input text to embed.
            
        Returns:
            np.ndarray of shape (768,) with float32 values, or None if unavailable.
        """
        if not self._available or not text.strip():
            return None
        
        try:
            tokens = self._tokenize(text)
            
            input_ids = np.array([tokens], dtype=np.int64)
            attention_mask = np.ones_like(input_ids)
            token_type_ids = np.zeros_like(input_ids)
            
            outputs = self._session.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }
            )
            
            embedding = outputs[0].mean(axis=1).squeeze()
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            return None


_embedder: Optional[OnnxEmbedder] = None


def get_embedder() -> Optional[OnnxEmbedder]:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = OnnxEmbedder()
    return _embedder if _embedder.available else None
