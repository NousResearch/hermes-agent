#!/usr/bin/env python3
"""
Kensei Super-Memory MCP Server — Enhanced memory tools for external agents.

Exposes four enhanced memory tools via the Model Context Protocol (MCP):
  - mem0_extract:       LLM-based entity/fact extraction from text
  - cognee_cognify:     Multi-step ETL pipeline (text → entities → enriched triples)
  - zep_temporal_query: Temporal knowledge graph query with validity intervals
  - simplemem_compress: Semantic compression of memory entries

All tools wrap Mnemosyne's existing capabilities (triple store, canonical store,
beam memory) and are safe to run alongside the standard Mnemosyne MCP server.

Usage:
    python3 kensei-super-memory-mcp.py          # stdio (default)
    python3 kensei-super-memory-mcp.py --sse    # SSE on 127.0.0.1:8081

MCP client config (claude_desktop_config.json):
    {
        "mcpServers": {
            "kensei-super-memory": {
                "command": "python3",
                "args": ["/path/to/kensei-super-memory-mcp.py"]
            }
        }
    }
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kensei.super_memory.mcp")

# ---------------------------------------------------------------------------
# Lazy MCP SDK import
# ---------------------------------------------------------------------------

_MCP_AVAILABLE = False
try:
    from mcp.server.fastmcp import FastMCP
    _MCP_AVAILABLE = True
except ImportError:
    FastMCP = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Mnemosyne imports (guarded)
# ---------------------------------------------------------------------------

_MNEMOSYNE_AVAILABLE = False
try:
    from mnemosyne.core.memory import Mnemosyne
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.triples import TripleStore
    from mnemosyne.core.canonical import CanonicalStore
    _MNEMOSYNE_AVAILABLE = True
except ImportError:
    Mnemosyne = None  # type: ignore[assignment,misc]
    BeamMemory = None
    TripleStore = None
    CanonicalStore = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
_MNEMOSYNE_HOME = Path(os.environ.get("MNEMOSYNE_HOME", str(_HERMES_HOME / "mnemosyne")))
_DEFAULT_BANK = os.environ.get("KENSEI_MEMORY_BANK", "default")

# ---------------------------------------------------------------------------
# Mnemosyne instance helpers
# ---------------------------------------------------------------------------


def _create_instance(bank: str = _DEFAULT_BANK) -> Mnemosyne:
    """Create a fresh Mnemosyne instance for each tool call."""
    return Mnemosyne(
        session_id=f"kensei_super_memory_{bank}",
        bank=bank,
    )


def _get_triple_store(bank: str = _DEFAULT_BANK) -> TripleStore:
    """Get a TripleStore instance pointing at the given bank."""
    mem = _create_instance(bank)
    db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path
    return TripleStore(db_path=db_path)


def _get_canonical_store(bank: str = _DEFAULT_BANK) -> CanonicalStore:
    """Get a CanonicalStore instance pointing at the given bank."""
    mem = _create_instance(bank)
    store = getattr(mem.beam, "canonical", None)
    if store is None:
        db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path
        store = CanonicalStore(db_path=db_path, conn=mem.beam.conn)
    return store


def _serialize(obj: Any) -> Any:
    """Recursively convert non-serializable objects (datetime, etc.) to strings."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, tuple):
        return [_serialize(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# LLM helper — calls the configured auxiliary LLM for extraction/compression
# ---------------------------------------------------------------------------


def _call_llm(prompt: str, max_tokens: int = 1000, timeout: int = 60) -> Optional[str]:
    """Call the configured auxiliary LLM and return the response text.

    Uses the same auxiliary client as Mnemosyne's MemRefine auto-compaction.
    Falls back to a simple HTTP call to Ollama if the client is unavailable.
    """
    # Try the Hermes auxiliary client first
    try:
        from agent.auxiliary_client import call_llm

        response = call_llm(
            task="compression",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return response.choices[0].message.content.strip()
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("Auxiliary LLM call failed: %s", exc)

    # Fallback: direct Ollama call
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("KENSEI_EXTRACTION_MODEL", "tinyllama:1.1b-chat")
    try:
        import urllib.request

        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{ollama_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.debug("Ollama fallback LLM call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Tool: mem0_extract — LLM-based entity/fact extraction
# ---------------------------------------------------------------------------

_MEM0_EXTRACT_PROMPT = """\
You are a fact extraction system. Given a text passage, extract all factual
statements as structured triples (subject, predicate, object). Also extract
named entities (people, places, organizations, concepts).

Rules:
- Each triple must be a verifiable fact stated in the text, not inferred.
- Subject and object should be specific entities or concepts.
- Predicate should be a concise relationship verb (e.g. "works_at", "lives_in",
  "prefers", "created", "is_a", "has_property").
- Return ONLY valid JSON with no extra text.

Output format:
{
    "entities": ["entity1", "entity2", ...],
    "triples": [
        {"subject": "...", "predicate": "...", "object": "..."}
    ]
}

Text:
"""


def _handle_mem0_extract(
    text: str,
    bank: str = _DEFAULT_BANK,
    source: str = "mem0_extract",
    confidence: float = 0.8,
) -> Dict[str, Any]:
    """Extract entities and fact triples from text using LLM, store in triple store."""
    if not text or not text.strip():
        return {"status": "error", "message": "text is required", "entities": [], "triples_stored": 0}

    prompt = _MEM0_EXTRACT_PROMPT + text
    llm_response = _call_llm(prompt, max_tokens=2000, timeout=120)

    if not llm_response:
        return {
            "status": "error",
            "message": "LLM extraction failed — no response from model",
            "entities": [],
            "triples_stored": 0,
        }

    # Parse JSON from LLM response
    try:
        # Find JSON block in response (handle markdown-wrapped output)
        json_str = llm_response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        result = json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return {
            "status": "error",
            "message": f"LLM returned unparseable JSON: {llm_response[:200]}",
            "entities": [],
            "triples_stored": 0,
        }

    entities = result.get("entities", [])
    triples = result.get("triples", [])

    # Store triples in Mnemosyne triple store
    kg = _get_triple_store(bank)
    stored_count = 0
    errors = []
    for triple in triples:
        subj = triple.get("subject", "").strip()
        pred = triple.get("predicate", "").strip()
        obj = triple.get("object", "").strip()
        if not all([subj, pred, obj]):
            errors.append(f"Invalid triple: {triple}")
            continue
        try:
            kg.add(
                subject=subj,
                predicate=pred,
                object=obj,
                source=source,
                confidence=confidence,
            )
            stored_count += 1
        except Exception as exc:
            errors.append(f"Failed to store triple ({subj}, {pred}, {obj}): {exc}")

    return {
        "status": "ok",
        "entities": entities,
        "triples_extracted": len(triples),
        "triples_stored": stored_count,
        "errors": errors if errors else None,
        "bank": bank,
    }


# ---------------------------------------------------------------------------
# Tool: cognee_cognify — multi-step ETL pipeline
# ---------------------------------------------------------------------------

_COGNEE_COGNIFY_PROMPT = """\
You are a knowledge graph enrichment system. Given a text passage, perform a
multi-step analysis:

Step 1 — Entity Extraction: Identify all named entities (people, places,
organizations, concepts, events, dates).

Step 2 — Relation Extraction: For each pair of entities that have a meaningful
relationship, extract the relationship as a triple.

Step 3 — Enrichment: For each entity, add a brief description/context.

Return ONLY valid JSON with no extra text.

Output format:
{
    "entities": [
        {"name": "...", "type": "person|place|org|concept|event|date", "description": "..."}
    ],
    "relations": [
        {"subject": "...", "predicate": "...", "object": "..."}
    ]
}

Text:
"""


def _handle_cognee_cognify(
    text: str,
    bank: str = _DEFAULT_BANK,
    source: str = "cognee_cognify",
    confidence: float = 0.7,
) -> Dict[str, Any]:
    """Run a multi-step ETL pipeline: extract entities → extract relations → enrich → store."""
    if not text or not text.strip():
        return {"status": "error", "message": "text is required", "entities": [], "relations_stored": 0}

    prompt = _COGNEE_COGNIFY_PROMPT + text
    llm_response = _call_llm(prompt, max_tokens=3000, timeout=180)

    if not llm_response:
        return {
            "status": "error",
            "message": "LLM cognify failed — no response from model",
            "entities": [],
            "relations_stored": 0,
        }

    # Parse JSON
    try:
        json_str = llm_response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        result = json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return {
            "status": "error",
            "message": f"LLM returned unparseable JSON: {llm_response[:200]}",
            "entities": [],
            "relations_stored": 0,
        }

    entities = result.get("entities", [])
    relations = result.get("relations", [])

    # Store entity descriptions as canonical facts
    canonical = _get_canonical_store(bank)
    entity_count = 0
    for ent in entities:
        name = ent.get("name", "").strip()
        desc = ent.get("description", "").strip()
        etype = ent.get("type", "concept").strip()
        if name and desc:
            try:
                canonical.remember(
                    owner_id="kensei_super_memory",
                    category=f"entity/{etype}",
                    name=name,
                    body=desc,
                    source=source,
                    confidence=confidence,
                )
                entity_count += 1
            except Exception:
                pass

    # Store relations as triples
    kg = _get_triple_store(bank)
    relation_count = 0
    errors = []
    for rel in relations:
        subj = rel.get("subject", "").strip()
        pred = rel.get("predicate", "").strip()
        obj = rel.get("object", "").strip()
        if not all([subj, pred, obj]):
            errors.append(f"Invalid relation: {rel}")
            continue
        try:
            kg.add(
                subject=subj,
                predicate=pred,
                object=obj,
                source=source,
                confidence=confidence,
            )
            relation_count += 1
        except Exception as exc:
            errors.append(f"Failed to store relation ({subj}, {pred}, {obj}): {exc}")

    return {
        "status": "ok",
        "entities_extracted": len(entities),
        "entities_stored": entity_count,
        "relations_extracted": len(relations),
        "relations_stored": relation_count,
        "errors": errors if errors else None,
        "bank": bank,
    }


# ---------------------------------------------------------------------------
# Tool: zep_temporal_query — temporal knowledge graph query
# ---------------------------------------------------------------------------


def _handle_zep_temporal_query(
    subject: str = "",
    predicate: str = "",
    object: str = "",
    as_of: str = "",
    bank: str = _DEFAULT_BANK,
    include_invalidated: bool = False,
) -> Dict[str, Any]:
    """Query the temporal knowledge graph with validity interval awareness.

    Supports:
    - as_of: ISO date string — only return facts valid at that point in time
    - include_invalidated: if True, also return facts that have been superseded
    """
    kg = _get_triple_store(bank)
    results = kg.query(
        subject=subject or None,
        predicate=predicate or None,
        object=object or None,
        as_of=as_of or None,
    )

    # Filter by temporal validity
    filtered = []
    for r in results:
        r = dict(r) if hasattr(r, "keys") else r
        valid_from = r.get("valid_from")
        valid_until = r.get("valid_until")

        # If include_invalidated is False, skip facts with a valid_until set
        if not include_invalidated and valid_until:
            continue

        filtered.append(_serialize(r))

    return {
        "status": "ok",
        "count": len(filtered),
        "results": filtered,
        "bank": bank,
        "as_of": as_of or "now",
    }


# ---------------------------------------------------------------------------
# Tool: simplemem_compress — semantic compression of memory entries
# ---------------------------------------------------------------------------


def _handle_simplemem_compress(
    bank: str = _DEFAULT_BANK,
    target: str = "working",
    max_entries: int = 50,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Compress memory entries by merging related facts and removing redundancy.

    Reads recent working/episodic memories, uses LLM to merge related entries,
    and replaces them with compressed versions. Similar to Mnemosyne's sleep
    cycle but with explicit LLM-guided semantic merging.

    Args:
        bank: Memory bank to operate on
        target: 'working' (default) or 'episodic'
        max_entries: Max entries to consider for compression
        dry_run: If True, report what would be compressed without writing
    """
    mem = _create_instance(bank)
    beam = mem.beam

    # Read entries from the target store
    if target == "episodic":
        try:
            entries = beam.get_episodic_stats()
            raw_entries = beam.get_episodic_memories(limit=max_entries) if hasattr(beam, "get_episodic_memories") else []
        except Exception:
            return {"status": "error", "message": f"Cannot read episodic store for bank '{bank}'"}
    else:
        try:
            raw_entries = beam.get_working_memories(limit=max_entries) if hasattr(beam, "get_working_memories") else []
            entries = beam.get_working_stats()
        except Exception:
            return {"status": "error", "message": f"Cannot read working store for bank '{bank}'"}

    if not raw_entries:
        return {
            "status": "ok",
            "message": f"No {target} entries to compress in bank '{bank}'",
            "entries_before": 0,
            "entries_after": 0,
            "compressed": 0,
        }

    # Build a text representation of entries for LLM
    entry_texts = []
    for entry in raw_entries:
        entry = dict(entry) if hasattr(entry, "keys") else entry
        eid = entry.get("id", entry.get("memory_id", "?"))
        content = entry.get("content", str(entry))[:300]
        entry_texts.append(f"[{eid}] {content}")

    entries_before = len(entry_texts)
    if entries_before < 2:
        return {
            "status": "ok",
            "message": f"Only {entries_before} entry — nothing to merge",
            "entries_before": entries_before,
            "entries_after": entries_before,
            "compressed": 0,
        }

    prompt = f"""\
You are a memory compression system. You have {entries_before} memory entries.
Merge related entries into fewer, shorter entries that preserve ALL unique
factual information. Remove redundancy. Group related facts.

Current entries:
{chr(10).join(entry_texts)}

Output ONLY a JSON array of strings, each string being a compressed entry.
Example: ["compressed entry 1", "compressed entry 2"]
"""

    llm_response = _call_llm(prompt, max_tokens=2000, timeout=120)

    if not llm_response:
        return {
            "status": "error",
            "message": "LLM compression failed — no response from model",
            "entries_before": entries_before,
            "entries_after": entries_before,
            "compressed": 0,
        }

    # Parse JSON array from response
    try:
        json_str = llm_response
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        compressed = json.loads(json_str)
        if not isinstance(compressed, list):
            compressed = [compressed]
    except (json.JSONDecodeError, IndexError):
        return {
            "status": "error",
            "message": f"LLM returned unparseable JSON: {llm_response[:200]}",
            "entries_before": entries_before,
            "entries_after": entries_before,
            "compressed": 0,
        }

    entries_after = len(compressed)
    compressed_count = entries_before - entries_after

    if dry_run:
        return {
            "status": "ok",
            "message": f"DRY RUN: Would compress {entries_before} entries into {entries_after}",
            "entries_before": entries_before,
            "entries_after": entries_after,
            "compressed": compressed_count,
            "compressed_entries": compressed,
            "bank": bank,
            "target": target,
        }

    # Apply compression: clear and re-write compressed entries
    try:
        if target == "episodic":
            if hasattr(beam, "clear_episodic"):
                beam.clear_episodic()
        else:
            if hasattr(beam, "clear_working"):
                beam.clear_working()

        for entry_text in compressed:
            mem.remember(
                content=entry_text,
                source="simplemem_compress",
                importance=0.6,
                scope="global",
            )
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Compression write failed: {exc}",
            "entries_before": entries_before,
            "entries_after": entries_before,
            "compressed": 0,
        }

    return {
        "status": "ok",
        "message": f"Compressed {entries_before} entries into {entries_after}",
        "entries_before": entries_before,
        "entries_after": entries_after,
        "compressed": compressed_count,
        "bank": bank,
        "target": target,
    }


# ---------------------------------------------------------------------------
# MCP Server Setup
# ---------------------------------------------------------------------------


def create_app() -> FastMCP:
    """Create and configure the FastMCP application with all tools."""
    if not _MCP_AVAILABLE:
        raise RuntimeError(
            "MCP SDK not installed. Run: pip install 'mcp>=1.28.0'"
        )
    if not _MNEMOSYNE_AVAILABLE:
        raise RuntimeError(
            "Mnemosyne not installed. Run: pip install mnemosyne-memory"
        )

    mcp = FastMCP("kensei-super-memory", log_level="WARNING")

    # -----------------------------------------------------------------------
    # Tool: mem0_extract
    # -----------------------------------------------------------------------
    @mcp.tool(
        name="mem0_extract",
        description=(
            "Extract entities and fact triples from text using LLM-based extraction. "
            "Takes raw text, runs it through an LLM to identify named entities and "
            "subject-predicate-object fact triples, then stores the triples in "
            "Mnemosyne's temporal knowledge graph. Returns extracted entities and "
            "the count of triples stored. Use this to convert unstructured conversation "
            "into structured, queryable facts."
        ),
    )
    def mem0_extract(
        text: str,
        bank: str = _DEFAULT_BANK,
        source: str = "mem0_extract",
        confidence: float = 0.8,
    ) -> str:
        result = _handle_mem0_extract(
            text=text,
            bank=bank,
            source=source,
            confidence=confidence,
        )
        return json.dumps(result, indent=2, default=str)

    # -----------------------------------------------------------------------
    # Tool: cognee_cognify
    # -----------------------------------------------------------------------
    @mcp.tool(
        name="cognee_cognify",
        description=(
            "Run a multi-step ETL pipeline on text to enrich the knowledge graph. "
            "Step 1: Extract named entities with types and descriptions. "
            "Step 2: Extract relations between entities as triples. "
            "Step 3: Store entity descriptions in the canonical store and relations "
            "in the triple store. Returns counts of entities and relations stored. "
            "Use this for deeper, more structured enrichment than mem0_extract."
        ),
    )
    def cognee_cognify(
        text: str,
        bank: str = _DEFAULT_BANK,
        source: str = "cognee_cognify",
        confidence: float = 0.7,
    ) -> str:
        result = _handle_cognee_cognify(
            text=text,
            bank=bank,
            source=source,
            confidence=confidence,
        )
        return json.dumps(result, indent=2, default=str)

    # -----------------------------------------------------------------------
    # Tool: zep_temporal_query
    # -----------------------------------------------------------------------
    @mcp.tool(
        name="zep_temporal_query",
        description=(
            "Query the temporal knowledge graph with validity interval awareness. "
            "Supports as-of-timestamp queries (ISO date string) to retrieve facts "
            "that were valid at a specific point in time. Use include_invalidated=True "
            "to also return superseded facts. Parameters are optional — omit to list "
            "all facts, or filter by subject/predicate/object patterns."
        ),
    )
    def zep_temporal_query(
        subject: str = "",
        predicate: str = "",
        object: str = "",
        as_of: str = "",
        bank: str = _DEFAULT_BANK,
        include_invalidated: bool = False,
    ) -> str:
        result = _handle_zep_temporal_query(
            subject=subject,
            predicate=predicate,
            object=object,
            as_of=as_of,
            bank=bank,
            include_invalidated=include_invalidated,
        )
        return json.dumps(result, indent=2, default=str)

    # -----------------------------------------------------------------------
    # Tool: simplemem_compress
    # -----------------------------------------------------------------------
    @mcp.tool(
        name="simplemem_compress",
        description=(
            "Compress memory entries by merging related facts and removing redundancy. "
            "Reads recent working (or episodic) memories, uses LLM-guided semantic merging "
            "to combine related entries into fewer, shorter entries that preserve all unique "
            "information. Use dry_run=True to preview what would be compressed without "
            "writing changes. Similar to Mnemosyne's sleep cycle but with explicit "
            "LLM-guided semantic merging."
        ),
    )
    def simplemem_compress(
        bank: str = _DEFAULT_BANK,
        target: str = "working",
        max_entries: int = 50,
        dry_run: bool = False,
    ) -> str:
        result = _handle_simplemem_compress(
            bank=bank,
            target=target,
            max_entries=max_entries,
            dry_run=dry_run,
        )
        return json.dumps(result, indent=2, default=str)

    return mcp


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Kensei Super-Memory MCP server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Kensei Super-Memory MCP Server — enhanced memory tools"
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Run in SSE mode (default: stdio)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address for SSE mode (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Port for SSE mode (default: 8081)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    mcp = create_app()

    if args.sse:
        logger.info("Starting Kensei Super-Memory MCP server on %s:%d", args.host, args.port)
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.info("Starting Kensei Super-Memory MCP server (stdio)")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
