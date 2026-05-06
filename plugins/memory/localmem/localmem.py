"""
localmem.py — Thin local memory wrapper using qwen2.5:7b + nomic-embed-text + ChromaDB.
Replaces Mem0's broken middleware with direct component access.
API-compatible with Mem0's Memory class for easy Hermes plugin integration.
"""

import json
import time
import uuid
import hashlib
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from ollama import Client as OllamaClient


# ── Tight extraction prompt (no hallucination-prone categories) ────────────

EXTRACTION_PROMPT = """Extract factual information about the user from this conversation. 
Return ONLY a JSON object with a "facts" key containing a list of strings.

Rules:
- Extract technical preferences, professional details, project decisions, and tool choices.
- Do NOT invent facts. Only extract what is explicitly stated.
- Do NOT extract greetings, small talk, or meta-commentary about the conversation.
- If nothing factual is stated, return {{"facts": []}}.
- Combine related details into single, concise facts.
- Facts should be in third person. Examples:
  - "User prefers Python and FastAPI for backend development"
  - "User is CTO of CodeWalnut with £9,500/month retainer"
  - "Oneness platform uses React with TypeScript, ThemeProvider wraps all routes"

Conversation:
{conversation}

Return ONLY the JSON:"""


class LocalMemory:
    """Local-first memory store with LLM extraction + vector search. Zero cloud."""

    def __init__(
        self,
        llm_model: str = "qwen2.5:7b",
        embed_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        chroma_path: str = "/tmp/localmem_chroma",
        collection_name: str = "localmem",
    ):
        self.ollama = OllamaClient(host=ollama_url)
        self.llm_model = llm_model
        self.embed_model = embed_model

        # ChromaDB — embedded, no server needed
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Stats
        self.extraction_count = 0
        self.total_extraction_time = 0.0

    # ── Public API (Mem0-compatible) ────────────────────────────────────

    def add(self, messages: list, user_id: str = "default", agent_id: str = "agent",
            infer: bool = True) -> dict:
        """Extract facts from conversation messages and store them.

        When infer=False, stores the user message verbatim (no LLM extraction).
        """
        t0 = time.time()
        facts = []

        if not infer:
            # Verb store — use the last user message directly
            content = messages[-1]["content"] if messages else ""
            if content.strip():
                facts = [content]
        else:
            # Build conversation text
            convo = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            )

            # Extract facts via LLM
            prompt = EXTRACTION_PROMPT.format(conversation=convo)
            response = self.ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "top_p": 0.1, "num_predict": 500},
            )
            raw = response["message"]["content"].strip()

            # Parse JSON — handle markdown wrapping
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            facts = json.loads(raw).get("facts", [])

        if not facts:
            return {"results": []}

        # Store each fact
        results = []
        for fact in facts:
            fact_id = hashlib.md5(f"{user_id}:{fact}".encode()).hexdigest()[:16]
            emb = self._embed(fact)
            self.collection.upsert(
                ids=[fact_id],
                embeddings=[emb],
                metadatas=[{
                    "memory": fact,
                    "user_id": user_id,
                    "agent_id": agent_id,
                }],
            )
            results.append({"id": fact_id, "memory": fact, "event": "ADD"})

        elapsed = time.time() - t0
        self.extraction_count += 1
        self.total_extraction_time += elapsed

        return {"results": results}

    def search(
        self, query: str, user_id: str = "default", top_k: int = 5, threshold: float = 0.0,
        rerank: bool = True,
    ) -> dict:
        """Semantic search over stored memories with optional LLM re-ranking."""
        # Fetch more candidates than needed for re-ranking
        fetch_k = top_k * 3 if rerank else top_k
        emb = self._embed(query)
        raw = self.collection.query(
            query_embeddings=[emb],
            n_results=fetch_k,
            where={"user_id": user_id},
        )

        candidates = []
        if raw["ids"] and raw["ids"][0]:
            for i, mem_id in enumerate(raw["ids"][0]):
                score = raw["distances"][0][i] if raw.get("distances") else 1.0
                similarity = 1.0 - score
                if similarity >= threshold:
                    candidates.append({
                        "id": mem_id,
                        "memory": raw["metadatas"][0][i].get("memory", ""),
                        "score": round(similarity, 4),
                    })

        if not candidates:
            return {"results": []}

        if rerank and len(candidates) > top_k:
            candidates = self._rerank(query, candidates, top_k)

        return {"results": candidates[:top_k]}

    def _rerank(self, query: str, candidates: list, top_k: int) -> list:
        """Use LLM to re-rank candidates by relevance to the query."""
        if len(candidates) <= 1:
            return candidates

        options = "\n".join(
            f"{i}. {c['memory']}" for i, c in enumerate(candidates)
        )
        prompt = f"""Query: {query}

Which of these facts answers the query best? Return ONLY a JSON array of indices in order of relevance, like [3, 0, 1].

Facts:
{options}

Ranking:"""

        response = self.ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 100},
        )
        raw = response["message"]["content"].strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            ranking = json.loads(raw)
            if isinstance(ranking, list) and all(isinstance(i, int) for i in ranking):
                seen = set()
                ordered = []
                for idx in ranking:
                    if 0 <= idx < len(candidates) and idx not in seen:
                        ordered.append(candidates[idx])
                        seen.add(idx)
                # Append any remaining candidates not in the ranking
                for i, c in enumerate(candidates):
                    if i not in seen:
                        ordered.append(c)
                return ordered[:top_k]
        except (json.JSONDecodeError, ValueError):
            pass

        return candidates[:top_k]

    def get_all(self, user_id: str = "default") -> dict:
        """Retrieve all stored memories for a user."""
        raw = self.collection.get(where={"user_id": user_id})
        results = []
        if raw["metadatas"]:
            for i, meta in enumerate(raw["metadatas"]):
                results.append({
                    "id": raw["ids"][i] if raw["ids"] else str(i),
                    "memory": meta.get("memory", ""),
                })
        return {"results": results}

    def stats(self) -> dict:
        """Return performance statistics."""
        avg_time = (
            self.total_extraction_time / self.extraction_count
            if self.extraction_count > 0
            else 0
        )
        return {
            "extractions": self.extraction_count,
            "total_extraction_time": round(self.total_extraction_time, 2),
            "avg_extraction_time": round(avg_time, 2),
            "llm_model": self.llm_model,
            "embed_model": self.embed_model,
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _embed(self, text: str) -> list:
        """Generate embedding vector via Ollama."""
        resp = self.ollama.embeddings(model=self.embed_model, prompt=text)
        return resp["embedding"]
