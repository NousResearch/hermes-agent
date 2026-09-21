"""LLM-based entity extraction and consolidation memory plugin for Hermes Agent."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = __import__('logging').getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "ollama_host": "http://localhost:11434",
    "embed_model": "phi3",
    "chat_model": "phi3",
    "similarity_threshold": 0.85,
    "max_facts_per_turn": 10,
    "enable_consolidation": True,
    "fact_ttl_days": 365,  # Optional: time-to-live for facts
}

# SQL for the fact embeddings table
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS fact_embeddings (
    triple_hash TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at REAL NOT NULL
);
"""

# SQL for cleaning up old facts (optional)
CLEANUP_SQL = """
DELETE FROM fact_embeddings WHERE created_at < ?;
"""


def _default_config() -> dict:
    return DEFAULT_CONFIG.copy()


def _sanitize_config(raw: dict) -> dict:
    config = _default_config()
    if not isinstance(raw, dict):
        return config
    for key, value in raw.items():
        if key in config:
            config[key] = value
    return config


def _load_llmextract_config(hermes_home: str) -> dict:
    config_path = Path(hermes_home) / "llmextract.json"
    config = _default_config()
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                config.update({k: v for k, v in raw.items() if v is not None})
        except Exception as e:
            logger.debug("Failed to parse %s: %s", config_path, e)
    return _sanitize_config(config)


def _save_llmextract_config(values: dict, hermes_home: str) -> None:
    config_path = Path(hermes_home) / "llmextract.json"
    existing = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(values)
    from utils import atomic_json_write
    atomic_json_write(config_path, existing, mode=0o600, sort_keys=True)


def _hash_triple(subject: str, predicate: str, object_: str) -> str:
    """Generate a SHA256 hash of the triple string."""
    triple_string = f"{subject}|{predicate}|{object_}"
    return hashlib.sha256(triple_string.encode()).hexdigest()


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class LLMExtractProvider(MemoryProvider):
    """Memory provider that extracts and consolidates facts using local LLMs."""

    def __init__(self):
        self.config: dict = {}
        self.hermes_home: str = ""
        self.conn: Optional[sqlite3.Connection] = None
        self.ollama_host: str = ""
        self.embed_model: str = ""
        self.chat_model: str = ""
        self.similarity_threshold: float = 0.0
        self.max_facts_per_turn: int = 0
        self.enable_consolidation: bool = False
        self.fact_ttl_days: int = 0

    @property
    def name(self) -> str:
        return "llmextract"

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            import ollama
        except ImportError:
            logger.warning("Ollama package not installed. Install with: pip install ollama")
            return False

        # Load config if not already loaded (is_available can be called before initialize)
        if not self.ollama_host:
            hermes_home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
            config = _load_llmextract_config(hermes_home)
            self.ollama_host = config.get("ollama_host", "http://localhost:11434")
            self.embed_model = config.get("embed_model", "phi3")
            self.chat_model = config.get("chat_model", "phi3")

        try:
            client = ollama.Client(host=self.ollama_host)
            # Check if the model exists by trying to list models
            models = client.list()
            model_names = [m.get("model") for m in models.get("models", [])]
            # Normalize model names: strip :latest suffix and any other tag
            normalized_names = set()
            for name in model_names:
                normalized_names.add(name.split(":")[0])
            embed_model_base = self.embed_model.split(":")[0]
            chat_model_base = self.chat_model.split(":")[0]
            if embed_model_base not in normalized_names or chat_model_base not in normalized_names:
                logger.warning(
                    "Model(s) not found in Ollama. Expected embed_model='%s' (base: '%s'), chat_model='%s' (base: '%s'). Available: %s",
                    self.embed_model,
                    embed_model_base,
                    self.chat_model,
                    chat_model_base,
                    model_names,
                )
                return False
            return True
        except Exception as e:
            logger.warning("Failed to connect to Ollama at %s: %s", self.ollama_host, e)
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize the plugin for a session."""
        self.hermes_home = kwargs.get("hermes_home", "")
        if not self.hermes_home:
            self.hermes_home = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))

        self.config = _load_llmextract_config(self.hermes_home)
        self.ollama_host = self.config.get("ollama_host", "http://localhost:11434")
        self.embed_model = self.config.get("embed_model", "phi3")
        self.chat_model = self.config.get("chat_model", "phi3")
        self.similarity_threshold = float(self.config.get("similarity_threshold", 0.85))
        self.max_facts_per_turn = int(self.config.get("max_facts_per_turn", 10))
        self.enable_consolidation = bool(self.config.get("enable_consolidation", True))
        self.fact_ttl_days = int(self.config.get("fact_ttl_days", 365))

        # Initialize the database connection for fact embeddings
        mnemosyne_data_dir = os.environ.get(
            "MNEMOSYNE_DATA_DIR",
            str(Path(self.hermes_home) / "mnemosyne" / "data"),
        )
        db_path = Path(mnemosyne_data_dir) / "triples.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute(CREATE_TABLE_SQL)
        self.conn.commit()

        # Clean up old facts if TTL is set
        if self.fact_ttl_days > 0:
            cutoff = time.time() - (self.fact_ttl_days * 86400)
            self.conn.execute(CLEANUP_SQL, (cutoff,))
            self.conn.commit()

        logger.info("LLMExtract plugin initialized for hermes_home=%s", self.hermes_home)

    def system_prompt_block(self) -> str:
        """Return static text to include in the system prompt."""
        return (
            "You have access to a long-term memory system that extracts and consolidates facts from conversations. "
            "When relevant, recall memories to inform your responses."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant context for the upcoming turn."""
        if not self.enable_consolidation or not self.conn:
            return ""

        try:
            # Get embedding for the query
            query_embedding = self._get_embedding(query)
            if not query_embedding:
                return ""

            # Fetch all embeddings and compute similarities
            cursor = self.conn.execute(
                "SELECT triple_hash, subject, predicate, object, embedding FROM fact_embeddings"
            )
            rows = cursor.fetchall()
            if not rows:
                return ""

            similarities = []
            for row in rows:
                triple_hash, subject, predicate, object_, blob = row
                stored_embedding = self._deserialize_embedding(blob)
                if stored_embedding is None:
                    continue
                sim = _cosine_similarity(query_embedding, stored_embedding)
                similarities.append((sim, subject, predicate, object_))

            # Sort by similarity descending and take top-k
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_k = similarities[: self.max_facts_per_turn]

            # Format as recall context
            lines = []
            for sim, subject, predicate, object_ in top_k:
                if sim >= self.similarity_threshold:
                    lines.append(f"- {subject} {predicate} {object_} (similarity: {sim:.2f})")
            if not lines:
                return ""

            return "Relevant memories:\n" + "\n".join(lines)
        except Exception as e:
            logger.exception("Error in prefetch: %s", e)
            return ""

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist a completed turn by extracting and consolidating facts."""
        if not self.enable_consolidation or not self.conn:
            return

        try:
            # Combine user and assistant content for fact extraction
            conversation = f"User: {user_content}\nAssistant: {assistant_content}"
            facts = self._extract_facts(conversation)
            if not facts:
                return

            # Limit the number of facts per turn
            if len(facts) > self.max_facts_per_turn:
                facts = facts[: self.max_facts_per_turn]

            # Process each fact
            for subject, predicate, object_ in facts:
                self._process_fact(subject, predicate, object_)
        except Exception as e:
            logger.exception("Error in sync_turn: %s", e)

    def _process_fact(self, subject: str, predicate: str, object_: str) -> None:
        """Process a single fact: extract embedding, consolidate, and store."""
        if not self.conn:
            return

        triple_hash = _hash_triple(subject, predicate, object_)
        fact_string = f"{subject} {predicate} {object_}"

        # Get embedding for the fact
        embedding = self._get_embedding(fact_string)
        if embedding is None:
            return

        # Check for similar existing facts
        similar_fact = self._find_similar_fact(embedding, exclude_hash=triple_hash)
        if similar_fact and self.enable_consolidation:
            existing_subj, existing_pred, existing_obj, existing_hash = similar_fact
            existing_fact_string = f"{existing_subj} {existing_pred} {existing_obj}"

            # Use LLM judge to decide action
            action = self._judge(existing_fact_string, fact_string)
            if action == "ADD":
                # Add as new fact (even though similar, we treat as new)
                self._store_fact(subject, predicate, object_, triple_hash, embedding)
            elif action == "UPDATE":
                # Remove the old fact and add the new one
                self._delete_fact(existing_subj, existing_pred, existing_obj)
                self._store_fact(subject, predicate, object_, triple_hash, embedding)
            elif action == "DELETE":
                # Delete the existing fact, do not add the new one
                self._delete_fact(existing_subj, existing_pred, existing_obj)
            # NOOP: do nothing
        else:
            # No similar fact found, or consolidation disabled: store as new
            self._store_fact(subject, predicate, object_, triple_hash, embedding)

    def _store_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        triple_hash: str,
        embedding: List[float],
    ) -> None:
        """Store a fact in the triple store and its embedding in the fact_embeddings table."""
        from mnemosyne.core.triples import TripleStore

        triple_store = TripleStore()
        # Use current date as valid_from
        valid_from = time.strftime("%Y-%m-%d")
        triple_id = triple_store.add(
            subject=subject,
            predicate=predicate,
            object=object_,
            valid_from=valid_from,
            source="llmextract",
            confidence=0.9,
        )
        # Store embedding
        self.conn.execute(
            """
            INSERT OR REPLACE INTO fact_embeddings (triple_hash, subject, predicate, object, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                triple_hash,
                subject,
                predicate,
                object_,
                self._serialize_embedding(embedding),
                time.time(),
            ),
        )
        self.conn.commit()
        logger.debug("Stored fact: %s %s %s (triple_id=%s)", subject, predicate, object_, triple_id)

    def _delete_fact(self, subject: str, predicate: str, object_: str) -> None:
        """Delete a fact from the triple store and its embedding."""
        from mnemosyne.core.triples import TripleStore

        triple_store = TripleStore()
        # Close any open triple matching (subject, predicate, object)
        triple_store.end(subject=subject, predicate=predicate, object=object_)
        # Remove from fact_embeddings
        triple_hash = _hash_triple(subject, predicate, object_)
        self.conn.execute(
            "DELETE FROM fact_embeddings WHERE triple_hash = ?", (triple_hash,)
        )
        self.conn.commit()
        logger.debug("Deleted fact: %s %s %s", subject, predicate, object_)

    def _find_similar_fact(
        self, embedding: List[float], exclude_hash: Optional[str] = None
    ) -> Optional[Tuple[str, str, str, str]]:
        """Find the most similar existing fact to the given embedding."""
        if not self.conn:
            return None

        cursor = self.conn.execute(
            "SELECT triple_hash, subject, predicate, object, embedding FROM fact_embeddings"
        )
        rows = cursor.fetchall()
        best_sim = -1.0
        best_fact = None
        for row in rows:
            triple_hash, subject, predicate, object_, blob = row
            if exclude_hash and triple_hash == exclude_hash:
                continue
            stored_embedding = self._deserialize_embedding(blob)
            if stored_embedding is None:
                continue
            sim = _cosine_similarity(embedding, stored_embedding)
            if sim > best_sim and sim >= self.similarity_threshold:
                best_sim = sim
                best_fact = (subject, predicate, object_, triple_hash)
        return best_fact

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding vector for text using Ollama."""
        try:
            import ollama

            client = ollama.Client(host=self.ollama_host)
            response = client.embeddings(model=self.embed_model, prompt=text)
            return response.get("embedding")
        except Exception as e:
            logger.warning("Failed to get embedding for text: %s", e)
            return None

    def _extract_facts(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract facts as triples from text using the chat model."""
        try:
            import ollama

            client = ollama.Client(host=self.ollama_host)
            prompt = f"""Extract facts from the following conversation as a list of triples (subject, predicate, object).
Only extract facts that are useful for future conversation (e.g., user preferences, facts about the user, etc.).
Ignore trivial chit-chat.

Conversation:
{text}

Output format: a JSON list of objects, each with keys "subject", "predicate", "object".
Example: [{{"subject": "user", "predicate": "likes", "object": "coffee"}}]

If no facts are found, return an empty list.
"""
            response = client.chat(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            content = response["message"]["content"]
            facts = json.loads(content)
            if not isinstance(facts, list):
                return []
            result = []
            for f in facts:
                if isinstance(f, dict) and all(k in f for k in ("subject", "predicate", "object")):
                    result.append((str(f["subject"]), str(f["predicate"]), str(f["object"])))
            return result
        except Exception as e:
            logger.warning("Failed to extract facts: %s", e)
            return []

    def _judge(self, existing_fact: str, new_fact: str) -> str:
        """Use LLM judge to decide action between existing and new fact."""
        try:
            import ollama

            client = ollama.Client(host=self.ollama_host)
            prompt = f"""You are a judge deciding whether a new fact should be added, update an existing fact, delete the existing fact, or be ignored.

Existing fact: {existing_fact}
New fact: {new_fact}

Choose one of: ADD, UPDATE, DELETE, NOOP.

Output only the word.
"""
            response = client.chat(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response["message"]["content"].strip().upper()
            if content in ("ADD", "UPDATE", "DELETE", "NOOP"):
                return content
            return "NOOP"
        except Exception as e:
            logger.warning("Failed to judge facts: %s", e)
            return "NOOP"

    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Serialize a list of floats to bytes using JSON."""
        return json.dumps(embedding).encode("utf-8")

    def _deserialize_embedding(self, blob: bytes) -> Optional[List[float]]:
        """Deserialize bytes to a list of floats."""
        try:
            return json.loads(blob.decode("utf-8"))
        except Exception:
            return None

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas for manual fact operations (optional)."""
        return [
            {
                "name": "llmextract_extract_facts",
                "description": "Extract facts from text as triples (subject, predicate, object).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to extract facts from.",
                        }
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "llmextract_store_fact",
                "description": "Store a fact as a triple in long-term memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "Subject of the fact.",
                        },
                        "predicate": {
                            "type": "string",
                            "description": "Predicate of the fact.",
                        },
                        "object": {
                            "type": "string",
                            "description": "Object of the fact.",
                        },
                    },
                    "required": ["subject", "predicate", "object"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle tool calls for manual fact operations."""
        if tool_name == "llmextract_extract_facts":
            text = args.get("text", "")
            facts = self._extract_facts(text)
            return json.dumps({"facts": [{"s": s, "p": p, "o": o} for s, p, o in facts]})
        elif tool_name == "llmextract_store_fact":
            subject = args.get("subject", "")
            predicate = args.get("predicate", "")
            object_ = args.get("object", "")
            if subject and predicate and object_:
                triple_hash = _hash_triple(subject, predicate, object_)
                embedding = self._get_embedding(f"{subject} {predicate} {object_}")
                if embedding:
                    self._store_fact(subject, predicate, object_, triple_hash, embedding)
                    return json.dumps({"status": "stored"})
            return json.dumps({"status": "failed", "error": "Invalid fact or embedding failed"})
        else:
            return tool_error(f"Tool {tool_name} not found")

    def shutdown(self) -> None:
        """Clean shutdown — flush queues, close connections."""
        if self.conn:
            self.conn.close()
            self.conn = None
        logger.info("LLMExtract plugin shut down")


# Factory function for plugin loader
def create() -> MemoryProvider:
    return LLMExtractProvider()