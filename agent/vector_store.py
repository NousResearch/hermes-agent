"""
VectorStore — semantic memory via TF-IDF on a fixed vocabulary.

WARNING: This is a PLACEHOLDER implementation for development only.
Semantic similarity computed via TF-IDF on a fixed word list is approximate
and does NOT capture true semantic meaning. For production, inject an
implementation using OpenAI embeddings, sentence-transformers, or FAISS/ChromaDB.

This implementation:
  - Uses a fixed vocabulary of ~5000 common English words
  - Computes TF-IDF vectors via sklearn
  - Projects to embedding_dim=384 via random projection ( Johnson-Lindenstrauss )
  - Cosine similarity for search
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# Fixed vocabulary — common English words used for TF-IDF vectorization
# Covers everyday language; not comprehensive but sufficient for demos.
_FIXED_VOCAB: List[str] = [
    # Articles, pronouns, common verbs
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "need", "dare", "ought", "used",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "and", "but", "or", "if", "because", "until", "while", "although", "though",
    "also", "about", "over", "out", "up", "down", "off", "away", "back", "still",
    "now", "always", "never", "ever", "today", "yesterday", "tomorrow",
    "this", "that", "these", "those", "i", "me", "my", "myself", "we", "our",
    "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "whose",
    # Common nouns
    "time", "year", "people", "way", "day", "man", "woman", "child", "children",
    "world", "life", "hand", "part", "place", "case", "week", "company", "system",
    "program", "question", "work", "government", "number", "night", "point",
    "home", "water", "room", "mother", "area", "money", "story", "fact", "month",
    "lot", "right", "study", "book", "eye", "job", "word", "business", "issue",
    "side", "kind", "head", "house", "service", "friend", "father", "power",
    "hour", "game", "line", "end", "member", "law", "car", "city", "name",
    "president", "team", "minute", "idea", "kid", "body", "information", "back",
    "parent", "face", "others", "level", "office", "door", "health", "person",
    "art", "war", "history", "party", "result", "change", "morning", "reason",
    "research", "girl", "guy", "moment", "air", "teacher", "force", "education",
    "foot", "boy", "age", "policy", "process", "music", "market", "sense",
    "nation", "plan", "college", "interest", "death", "experience", "effect",
    "use", "class", "field", "development", "role", "effort", "rate", "heart",
    "drug", "show", "leader", "light", "voice", "wife", "police", "mind",
    "difference", "period", "value", "building", "action", "authority", "model",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "hundred", "thousand", "million", "billion",
    # Common adjectives
    "good", "new", "first", "last", "long", "great", "little", "own", "old",
    "right", "big", "high", "different", "small", "large", "next", "early",
    "young", "important", "few", "public", "bad", "same", "able", "human",
    "local", "sure", "free", "better", "true", "whole", "real", "best",
    "special", "easy", "hard", "clear", "recent", "certain", "personal",
    "open", "red", "difficult", "available", "likely", "short", "single",
    "medical", "current", "wrong", "private", "past", "foreign", "fine", "common",
    "poor", "natural", "significant", "similar", "hot", "dead", "central",
    "happy", "serious", "ready", "simple", "left", "physical", "general", "environmental",
    "financial", "blue", "democratic", "dark", "various", "entire", "close",
    "legal", "religious", "cold", "final", "main", "green", "nice", "huge",
    "popular", "traditional", "cultural", "beautiful", "wonderful", "amazing",
    "perfect", "quick", "fast", "slow", "strong", "weak", "powerful",
    # Common verbs
    "get", "make", "go", "know", "take", "see", "come", "think", "look",
    "want", "give", "use", "find", "tell", "ask", "work", "seem", "feel",
    "try", "leave", "call", "keep", "let", "begin", "show", "hear", "play",
    "run", "move", "live", "believe", "hold", "bring", "happen", "write",
    "provide", "sit", "stand", "lose", "pay", "meet", "include", "continue",
    "set", "learn", "change", "lead", "understand", "watch", "follow", "stop",
    "create", "speak", "read", "spend", "grow", "open", "walk", "win", "offer",
    "remember", "love", "consider", "appear", "buy", "wait", "serve", "die",
    "send", "expect", "build", "stay", "fall", "cut", "reach", "kill",
    "remain", "suggest", "raise", "pass", "sell", "require", "report",
    "decide", "pull", "develop", "return", "explain", "hope", "carry",
    "break", "receive", "agree", "support", "hit", "produce", "eat", "cover",
    "catch", "draw", "choose", "study", "state", "pay", "cost", "push",
    "describe", "prefer", "supply", "discover", "operate", "detect", "identify",
    # Tech/computing terms
    "code", "file", "data", "program", "software", "system", "computer",
    "network", "server", "database", "application", "api", "web", "internet",
    "email", "message", "user", "account", "password", "login", "error",
    "bug", "feature", "function", "class", "method", "variable", "object",
    "string", "number", "boolean", "array", "list", "dictionary", "hash",
    "map", "reduce", "filter", "sort", "search", "index", "query", "table",
    "row", "column", "key", "value", "cache", "memory", "storage", "disk",
    "request", "response", "header", "body", "status", "code", "format",
    "json", "xml", "html", "css", "javascript", "python", "java", "script",
    "run", "execute", "test", "debug", "deploy", "build", "compile", "install",
    "update", "delete", "create", "read", "write", "open", "close", "save",
]

# Ensure vocab is sorted and deduplicated
_FIXED_VOCAB = sorted(set(_FIXED_VOCAB))
_VOCAB_SIZE = len(_FIXED_VOCAB)
_WORD_TO_IDX = {w: i for i, w in enumerate(_FIXED_VOCAB)}


def _build_tfidf_vectorizer():
    """Build and return a TfidfVectorizer with the fixed vocabulary."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        vocabulary=_FIXED_VOCAB,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )
    # Fit on a dummy corpus covering all vocabulary words so IDF is stable
    vectorizer.fit([" ".join(_FIXED_VOCAB)])
    return vectorizer


def _compute_tfidf(text: str, vectorizer) -> np.ndarray:
    """Compute TF-IDF vector for text using the fitted vectorizer."""
    try:
        vec = vectorizer.transform([text]).toarray().flatten()
    except Exception:
        # Fallback: bag of words if TF-IDF fails
        words = text.lower().split()
        vec = np.zeros(_VOCAB_SIZE, dtype=np.float32)
        for w in words:
            if w in _WORD_TO_IDX:
                vec[_WORD_TO_IDX[w]] += 1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
    return vec.astype(np.float32)


class VectorStore:
    """
    Semantic memory store using TF-IDF on a fixed vocabulary.

    WARNING: This is a PLACEHOLDER — semantic similarity is approximate and
    does NOT capture true semantic meaning. For production, inject an
    implementation using OpenAI embeddings, sentence-transformers, or
    FAISS/ChromaDB with proper embeddings.
    """

    def __init__(self, embedding_dim: int = 384):
        self._embedding_dim = embedding_dim
        self._vectors: List[np.ndarray] = []
        self._metadata: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._event_bus = None

        # Project TF-IDF vectors (~5000d) down to embedding_dim via random projection
        # Johnson-Lindenstrauss: random Gaussian matrix preserves distances approximately
        self._projection_matrix = np.random.randn(_VOCAB_SIZE, embedding_dim).astype(np.float32)
        # Normalize columns for stable projection
        col_norms = np.linalg.norm(self._projection_matrix, axis=0, keepdims=True)
        self._projection_matrix /= (col_norms + 1e-8)

        # Lazy-init TF-IDF vectorizer
        self._vectorizer = None

    def _get_vectorizer(self):
        if self._vectorizer is None:
            self._vectorizer = _build_tfidf_vectorizer()
        return self._vectorizer

    def set_event_bus(self, event_bus) -> None:
        """Set the EventBus for emitting events."""
        self._event_bus = event_bus

    def _embed(self, text: str) -> np.ndarray:
        """Convert text to a dense embedding vector via TF-IDF + projection."""
        vectorizer = self._get_vectorizer()
        tfidf = _compute_tfidf(text, vectorizer)
        # Project from vocab_size down to embedding_dim
        embedded = tfidf @ self._projection_matrix
        norm = np.linalg.norm(embedded)
        if norm > 0:
            embedded /= norm
        return embedded.astype(np.float32)

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Emit an event via EventBus with graceful fallback."""
        if self._event_bus is None:
            return
        try:
            from hermes.analytics import Event
            self._event_bus.emit(Event(event_type, payload))
        except Exception:
            # EventBus may not be initialized (Phase 1); fail silently
            pass

    def add(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add a text entry to the store.

        Returns the assigned ID.
        """
        embedding = self._embed(text)
        with self._lock:
            id = f"vec_{len(self._vectors)}"
            self._vectors.append(embedding)
            self._metadata.append({"id": id, "text": text, **(metadata or {})})
            self._emit("semantic.add", {"id": id, "text": text[:100] if text else ""})
            return id

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for the top-k most similar entries to the query.

        Returns list of (id, score, metadata) tuples sorted by descending score.
        """
        query_emb = self._embed(query)
        with self._lock:
            if not self._vectors:
                self._emit("semantic.search", {"query": query[:50] if query else "", "top_k": top_k, "results": 0})
                return []
            scores = []
            for v in self._vectors:
                score = float(np.dot(query_emb, v))
                scores.append(score)
            # Get top-k indices sorted descending by score
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            results = [(self._metadata[i]["id"], scores[i], dict(self._metadata[i])) for i in sorted_indices]
            self._emit("semantic.search", {
                "query": query[:50] if query else "",
                "top_k": top_k,
                "results": len(results),
            })
            return results

    def delete(self, id: str) -> bool:
        """
        Delete the entry with the given ID.

        Returns True if deleted, False if not found.
        """
        with self._lock:
            for i, m in enumerate(self._metadata):
                if m.get("id") == id:
                    self._vectors.pop(i)
                    self._metadata.pop(i)
                    self._emit("semantic.delete", {"id": id})
                    return True
            return False

    def __len__(self) -> int:
        with self._lock:
            return len(self._vectors)
