#!/usr/bin/env python3
"""
Context Graph Manager — Wraps Graphiti + Kuzu for persistent personal knowledge graph.

Provides temporal entity/relationship/episode storage with semantic search.
Uses Hermes's auxiliary LLM client for graph ingestion (entity extraction,
deduplication, edge detection). Search requires zero LLM calls.

Storage: ~/.hermes/context-graph/kuzu_db/
LLM: Configured via auxiliary.context_graph in cli-config.yaml or
     AUXILIARY_CONTEXT_GRAPH_MODEL / AUXILIARY_CONTEXT_GRAPH_PROVIDER env vars.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class GraphManager:
    """Wrapper around Graphiti + Kuzu for personal context graph operations.

    Lazy-initializes on first use (~1s startup). All public methods are async.
    """

    def __init__(self, db_path: Path, llm_config: Optional[Dict[str, Any]] = None):
        self._db_path = Path(db_path)
        self._llm_config = llm_config or {}
        self._graphiti = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Lazy init — creates Graphiti + Kuzu on first use."""
        if self._initialized:
            return

        try:
            from graphiti_core import Graphiti
            from graphiti_core.driver.kuzu_driver import KuzuDriver
            from graphiti_core.llm_client import OpenAIClient
            from graphiti_core.llm_client.config import LLMConfig
            from graphiti_core.embedder import OpenAIEmbedder
            from graphiti_core.embedder.openai import OpenAIEmbedderConfig
            from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        except ImportError as e:
            raise RuntimeError(
                f"graphiti-core[kuzu] not installed: {e}. "
                "Install with: pip install 'graphiti-core[kuzu]'"
            ) from e

        # Create DB directory
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Resolve LLM provider via Hermes auxiliary client
        llm_client = self._build_llm_client(LLMConfig, OpenAIClient)
        embedder = self._build_embedder(OpenAIEmbedderConfig, OpenAIEmbedder)
        cross_encoder = self._build_cross_encoder(LLMConfig, OpenAIRerankerClient)

        # Create Kuzu driver (embedded, file-based)
        driver = KuzuDriver(db=str(self._db_path))

        self._graphiti = Graphiti(
            graph_driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=cross_encoder,
            store_raw_episode_content=True,
        )

        # Build indices (idempotent)
        await self._graphiti.build_indices_and_constraints()
        self._initialized = True
        logger.info("Context graph initialized at %s", self._db_path)

    def _build_llm_client(self, LLMConfig, OpenAIClient):
        """Build a Graphiti-compatible LLM client using Hermes's provider chain."""
        provider, model, base_url, api_key = self._resolve_provider()

        config = LLMConfig(
            api_key=api_key or "no-key-required",
            model=model,
            base_url=base_url,
        )

        return OpenAIClient(config=config)

    def _build_cross_encoder(self, LLMConfig, OpenAIRerankerClient):
        """Build a Graphiti-compatible cross-encoder/reranker client."""
        _provider, _model, base_url, api_key = self._resolve_provider()

        config = LLMConfig(
            api_key=api_key or "no-key-required",
            base_url=base_url,
        )

        return OpenAIRerankerClient(config=config)

    def _resolve_embedder_provider(self):
        """Resolve an embedding-capable provider.

        Not all LLM providers support embeddings (e.g. Anthropic doesn't).
        Priority: OpenRouter > OpenAI > Ollama > same as LLM provider.
        """
        # Check env var overrides
        embed_base = os.environ.get("AUXILIARY_CONTEXT_GRAPH_EMBED_BASE_URL", "").strip()
        embed_key = os.environ.get("AUXILIARY_CONTEXT_GRAPH_EMBED_API_KEY", "").strip()
        if embed_base and embed_key:
            return embed_base, embed_key

        # Try OpenRouter (supports embeddings)
        or_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if or_key:
            return "https://openrouter.ai/api/v1", or_key

        # Try OpenAI directly (native embedding support)
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if openai_key:
            return None, openai_key  # None = default OpenAI URL

        # Ollama (supports embeddings locally)
        provider, _model, base_url, api_key = self._resolve_provider()
        if provider in ("ollama", "lmstudio", "vllm", "llamacpp", "custom"):
            return base_url, api_key or "no-key-required"

        # Anthropic doesn't support embeddings — give a clear error
        if provider == "anthropic":
            raise RuntimeError(
                "Context graph requires an embedding-capable provider (OpenRouter, OpenAI, or Ollama). "
                "Anthropic does not support embeddings. Set OPENROUTER_API_KEY or OPENAI_API_KEY, "
                "or configure auxiliary.context_graph.provider in cli-config.yaml to use Ollama."
            )

        # Fallback: use whatever the LLM provider is
        return base_url, api_key

    def _build_embedder(self, OpenAIEmbedderConfig, OpenAIEmbedder):
        """Build embedder for vector search.

        Routes to an embedding-capable provider (OpenRouter, OpenAI, Ollama).
        Anthropic doesn't support embeddings, so we never route there.
        """
        embed_base_url, embed_api_key = self._resolve_embedder_provider()

        config = OpenAIEmbedderConfig(
            api_key=embed_api_key or "no-key-required",
            base_url=embed_base_url,
            embedding_model="text-embedding-3-small",
        )

        return OpenAIEmbedder(config=config)

    def _resolve_provider(self):
        """Resolve LLM provider/model/base_url/api_key for the context_graph task.

        Priority:
        1. AUXILIARY_CONTEXT_GRAPH_* env vars
        2. cli-config.yaml auxiliary.context_graph.* section
        3. Auto-detection via Hermes provider chain
        """
        # Check env var overrides first
        provider = os.environ.get("AUXILIARY_CONTEXT_GRAPH_PROVIDER", "").strip().lower()
        model = os.environ.get("AUXILIARY_CONTEXT_GRAPH_MODEL", "").strip()
        base_url = os.environ.get("AUXILIARY_CONTEXT_GRAPH_BASE_URL", "").strip()
        api_key = os.environ.get("AUXILIARY_CONTEXT_GRAPH_API_KEY", "").strip()

        # Check config
        aux_config = self._llm_config.get("auxiliary", {})
        if isinstance(aux_config, dict):
            graph_aux = aux_config.get("context_graph", {})
            if isinstance(graph_aux, dict):
                provider = provider or graph_aux.get("provider", "")
                model = model or graph_aux.get("model", "")
                base_url = base_url or graph_aux.get("base_url", "")
                api_key = api_key or graph_aux.get("api_key", "")

        # Handle Ollama / local server aliases
        if provider in ("ollama", "lmstudio", "vllm", "llamacpp"):
            base_url = base_url or "http://localhost:11434/v1"
            api_key = api_key or "no-key-required"
            model = model or "llama3.2"
            return provider, model, base_url, api_key

        # OpenRouter (default cloud path)
        if not provider or provider in ("auto", "openrouter"):
            or_key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
            if or_key:
                return (
                    "openrouter",
                    model or "google/gemini-2.5-flash",
                    base_url or "https://openrouter.ai/api/v1",
                    or_key,
                )

        # Custom endpoint
        if provider == "custom" or base_url:
            api_key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
            return provider or "custom", model, base_url, api_key

        # Anthropic fallback
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if anthropic_key:
            return (
                "anthropic",
                model or "claude-haiku-4-5-20251001",
                base_url or "https://api.anthropic.com/v1",
                anthropic_key,
            )

        # Last resort — try OpenAI key
        openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if openai_key:
            return "openai", model or "gpt-4o-mini", base_url, openai_key

        raise RuntimeError(
            "No LLM provider configured for context graph. "
            "Set OPENROUTER_API_KEY, AUXILIARY_CONTEXT_GRAPH_PROVIDER, "
            "or configure auxiliary.context_graph in cli-config.yaml"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_episode(
        self,
        content: str,
        source_type: str = "text",
        name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        group_id: str = "personal",
    ) -> Dict[str, Any]:
        """Ingest a text episode into the graph.

        Runs Graphiti's 5-stage pipeline: context retrieval → node resolution
        → attribute extraction → edge extraction → contradiction detection.
        Requires LLM calls (~2-10s).

        Args:
            content: The text to ingest (decision trace, learning, conversation summary)
            source_type: "text", "message", or "json"
            name: Short label for the episode (auto-generated if empty)
            metadata: Optional metadata dict
            group_id: Namespace for the episode (default: "personal")

        Returns:
            Dict with episode info, extracted entities, and edges
        """
        await self._ensure_initialized()

        from graphiti_core.nodes import EpisodeType

        type_map = {
            "text": EpisodeType.text,
            "message": EpisodeType.message,
            "json": EpisodeType.json,
        }
        episode_type = type_map.get(source_type, EpisodeType.text)

        now = datetime.now(timezone.utc)
        episode_name = name or f"episode-{now.strftime('%Y%m%d-%H%M%S')}"
        source_desc = (metadata or {}).get("source_description", "hermes-agent session")

        result = await self._graphiti.add_episode(
            name=episode_name,
            episode_body=content,
            source_description=source_desc,
            reference_time=now,
            source=episode_type,
            group_id=group_id,
        )

        # Serialize result for JSON response
        entities = []
        for node in (result.nodes or []):
            entities.append({
                "uuid": node.uuid,
                "name": node.name,
                "summary": getattr(node, "summary", ""),
            })

        edges = []
        for edge in (result.edges or []):
            edges.append({
                "uuid": edge.uuid,
                "name": edge.name,
                "fact": edge.fact,
                "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
            })

        # Feature 2: Reinforce derived knowledge edges
        if edges:
            edge_uuids = [e["uuid"] for e in edges]
            await self.reinforce_edges(edge_uuids)

        return {
            "episode_uuid": result.episode.uuid if result.episode else None,
            "entities_extracted": len(entities),
            "edges_extracted": len(edges),
            "entities": entities,
            "edges": edges,
        }

    async def add_academic_episode(
        self,
        content: str,
        name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        group_id: str = "research",
    ) -> Dict[str, Any]:
        """Ingest a research episode, verifying citations via Federated Router before Graphiti ingestion.
        
        Parses the text for a 'References' or 'Bibliography' section, verifies each citation against
        arXiv or OpenAlex, and reconstructs the text with only factually grounded citations before
        passing it to the LLM Knowledge Graph constructor to prevent hallucination caching.
        """
        import re
        from .research_utils import CitationVerifier
        
        verifier = CitationVerifier()
        
        # 1. Parse content for references section
        ref_split = re.split(r'\n(?:#+\s+)?(?:References|Bibliography|Citations)\s*\n', content, flags=re.IGNORECASE)
        sanitized_content = ref_split[0]
        
        verified_refs = []
        if len(ref_split) > 1:
            raw_refs = ref_split[1].strip().split('\n')
            for ref in raw_refs:
                ref = ref.strip()
                if not ref: continue
                # Remove leading bullets or numberings
                clean_ref = re.sub(r'^(\d+\.|\[\d+\]|-|\*)\s*', '', ref)
                if not clean_ref: continue
                
                # 2. Verify against Federated Router
                v = verifier.verify(clean_ref, context=name)
                if v:
                    authors = ", ".join(v['authors'])
                    verified_refs.append(f"[{v['source'].upper()}] {v['title']} by {authors} ({v['year']}). URL: {v['url']}")
                else:
                    logger.debug("Filtered out hallucinated or unverifiable citation: %s", clean_ref)
        
        # 3. Reconstruct text with only grounded facts
        if verified_refs:
            sanitized_content += "\n\n## Verified References\n" + "\n".join(f"- {r}" for r in verified_refs)
            
        # 4. Standard Graphiti pipeline
        return await self.add_episode(
            content=sanitized_content,
            source_type="text",
            name=name,
            metadata=metadata,
            group_id=group_id
        )

    async def add_academic_pdf(
        self,
        pdf_path: str,
        name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        group_id: str = "research",
    ) -> Dict[str, Any]:
        """Ingest a raw PDF academic paper.

        Uses opendataloader-pdf to extract highly accurate, multi-column resolved Markdown,
        then passes it to add_academic_episode for citation verification and graph ingestion.
        """
        import tempfile
        try:
            import opendataloader_pdf
        except ImportError as e:
            raise RuntimeError("opendataloader-pdf is required for PDF ingestion. Install with: pip install opendataloader-pdf") from e

        # Use a temporary directory so we don't clutter the workspace with markdown/json sidecars
        # since the GraphManager primarily cares about extracting relations into the DB.
        pdf_file = Path(pdf_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Runs the fast, standalone deterministic extraction (0.015s/page)
                opendataloader_pdf.convert(
                    input_path=[str(pdf_file)],
                    output_dir=tmpdir,
                    format="markdown"
                )
                
                # The output file is named <original_stem>.md
                md_path = Path(tmpdir) / f"{pdf_file.stem}.md"
                if not md_path.exists():
                    raise FileNotFoundError(f"opendataloader_pdf failed to generate markdown for {pdf_file.name}")
                
                extracted_content = md_path.read_text(encoding="utf-8")
                
            except Exception as e:
                logger.error("Failed to extract PDF %s: %s", pdf_file.name, e)
                raise
        
        # Determine a name if none was provided
        episode_name = name or pdf_file.stem
        
        # Inject source info into metadata for provenance tracking
        meta = metadata or {}
        meta["source_description"] = f"academic_pdf_ingestion:{pdf_file.name}"
        meta["original_pdf_path"] = str(pdf_path)

        # Pass the cleanly extracted Markdown to the federated academic pipeline
        return await self.add_academic_episode(
            content=extracted_content,
            name=episode_name,
            metadata=meta,
            group_id=group_id,
        )

    def reciprocal_rank_fusion(self, list_of_rankings: List[List[Any]], k: int = 60) -> List[Any]:
        """Reciprocal Rank Fusion (RRF) for Hybrid Search.
        Fuses multiple ranked lists into a single relevance-sorted list.
        """
        rrf_scores = {}
        items_by_id = {}
        
        for ranking in list_of_rankings:
            for rank, item in enumerate(ranking, 1):
                item_id = item.uuid
                if item_id not in rrf_scores:
                    rrf_scores[item_id] = 0.0
                    items_by_id[item_id] = item
                rrf_scores[item_id] += 1.0 / (k + rank)
                
        # Sort items mathematically by their fused rank
        fused_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        return [items_by_id[i] for i in fused_ids]

    async def search(
        self,
        query: str,
        limit: int = 10,
        group_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Hybrid Search. Returns matching edges fusing Vector Similarity & Graph Traversal (via Graphiti) 
        and Keyword Match (via Kuzu) using Reciprocal Rank Fusion (RRF).

        Args:
            query: Natural language search query
            limit: Maximum results
            group_ids: Filter by namespace(s)

        Returns:
            Dict with fused edges and their connected entities
        """
        await self._ensure_initialized()
        driver = self._graphiti.driver

        # Stream 1: Graphiti Native Search (Vector Embedding + Graph Traversal)
        try:
            vector_graph_edges = await self._graphiti.search(
                query=query,
                num_results=limit,
                group_ids=group_ids or ["personal"],
            )
        except Exception:
            vector_graph_edges = []

        # Stream 2: Keyword / BM25 Naive Surrogate via Kuzu wildcard matching
        keyword_edges = []
        try:
            tables_res = await driver.execute_query("CALL show_tables() RETURN name, type")
            edge_tables = [t["name"] for t in tables_res if t["type"] == "REL"]
            terms = [t for t in query.lower().split() if len(t) > 3]
            
            # Fetch matching edges manually and mock a Graphiti edge object
            class MockEdge:
                def __init__(self, e):
                    self.uuid = e.get("uuid")
                    self.name = e.get("name")
                    self.fact = e.get("fact")
                    self.source_node_uuid = e.get("source_node_uuid")
                    self.target_node_uuid = e.get("target_node_uuid")
                    from datetime import datetime, timezone
                    now_str = datetime.now(timezone.utc).isoformat()
                    # Safely handle Kuzu TIMESTAMP parsing, fallback None
                    self.valid_at = self._parse_tsz(e.get("valid_at"))
                    self.invalid_at = self._parse_tsz(e.get("invalid_at"))
                    self.expired_at = self._parse_tsz(e.get("expired_at"))
                    self.episodes = []
                
                def _parse_tsz(self, val):
                    if not val: return None
                    from datetime import datetime
                    if isinstance(val, str):
                        try:
                            return datetime.fromisoformat(val)
                        except (ValueError, TypeError) as exc:
                            logger.debug("graph timestamp parse failed for %r: %s", val, exc)
                    return None

            for table in edge_tables:
                for term in terms[:3]: # limit query explosion
                    # Kuzu string functions: contains
                    res = await driver.execute_query(
                        f"MATCH ()-[e:{table}]->() WHERE lower(e.fact) CONTAINS $term "
                        "RETURN e.uuid as uuid, e.name as name, e.fact as fact, "
                        "e.valid_at as valid_at, e.invalid_at as invalid_at, "
                        "e.expired_at as expired_at LIMIT 10",
                        term=term
                    )
                    for r in res:
                        keyword_edges.append(MockEdge(r))
        except Exception as e:
            logger.debug("Keyword search stream failed: %s", e)

        # Mathematical RRF Fusion
        fused_edges = self.reciprocal_rank_fusion([vector_graph_edges, keyword_edges])
        fused_edges = fused_edges[:limit]

        results = []
        for edge in fused_edges:
            results.append({
                "uuid": edge.uuid,
                "name": getattr(edge, "name", ""),
                "fact": getattr(edge, "fact", ""),
                "source_node": getattr(edge, "source_node_uuid", ""),
                "target_node": getattr(edge, "target_node_uuid", ""),
                "valid_at": edge.valid_at.isoformat() if getattr(edge, "valid_at", None) else None,
                "invalid_at": edge.invalid_at.isoformat() if getattr(edge, "invalid_at", None) else None,
                "expired_at": edge.expired_at.isoformat() if getattr(edge, "expired_at", None) else None,
                "episodes": getattr(edge, "episodes", []),
            })

        return {
            "query": query,
            "count": len(results),
            "results": results,
            "fusion_method": "RRF (Vector + Graph + Keyword)",
        }

    async def get_episodes(
        self,
        last_n: int = 10,
        group_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent episodes from the graph.

        Args:
            last_n: Number of recent episodes to return
            group_ids: Filter by namespace(s)

        Returns:
            List of episode dicts
        """
        await self._ensure_initialized()

        episodes = await self._graphiti.retrieve_episodes(
            reference_time=datetime.now(timezone.utc),
            last_n=last_n,
            group_ids=group_ids or ["personal"],
        )

        results = []
        for ep in episodes:
            results.append({
                "uuid": ep.uuid,
                "name": ep.name,
                "content": ep.content,
                "source": ep.source.value if hasattr(ep.source, "value") else str(ep.source),
                "source_description": ep.source_description,
                "valid_at": ep.valid_at.isoformat() if ep.valid_at else None,
                "created_at": ep.created_at.isoformat() if ep.created_at else None,
            })

        return results

    async def get_nodes_by_episode(
        self,
        episode_uuid: str,
    ) -> Dict[str, Any]:
        """Get all entities and edges extracted from a specific episode.

        Args:
            episode_uuid: UUID of the episode

        Returns:
            Dict with nodes and edges
        """
        await self._ensure_initialized()

        result = await self._graphiti.get_nodes_and_edges_by_episode(episode_uuid)
        nodes = result.get("nodes", []) if isinstance(result, dict) else []
        edges = result.get("edges", []) if isinstance(result, dict) else []

        return {
            "episode_uuid": episode_uuid,
            "nodes": [
                {"uuid": n.uuid, "name": n.name, "summary": getattr(n, "summary", "")}
                for n in nodes
            ],
            "edges": [
                {"uuid": e.uuid, "name": e.name, "fact": e.fact}
                for e in edges
            ],
        }

    # ------------------------------------------------------------------
    # Feature 2: Memory Lifecycle (Ebbinghaus Decay)
    # ------------------------------------------------------------------

    async def _setup_confidence_schema(self, table_name: str):
        """Idempotently add confidence metadata schema to a Kuzu relationship table."""
        driver = self._graphiti.driver
        if not hasattr(driver, "execute_query"):
            return
        try:
            await driver.execute_query(f"ALTER TABLE {table_name} ADD confidence_score DOUBLE DEFAULT 1.0")
        except Exception:
            pass
        try:
            await driver.execute_query(f"ALTER TABLE {table_name} ADD last_reinforced_at STRING DEFAULT ''")
        except Exception:
            pass

    async def reinforce_edges(self, edge_uuids: List[str]):
        """Reinforce the given edges (reset their decay timer and maximize confidence)."""
        if not edge_uuids:
            return
        driver = self._graphiti.driver
        if not hasattr(driver, "execute_query"):
            return
        now = datetime.now(timezone.utc).isoformat()
        
        try:
            tables_res = await driver.execute_query("CALL show_tables() RETURN name, type")
            edge_tables = [t["name"] for t in tables_res if t["type"] == "REL"]
            for table in edge_tables:
                await self._setup_confidence_schema(table)
        except Exception as e:
            logger.debug("Failed to fetch tables for reinforcement schema: %s", e)
            return

        for u in edge_uuids:
            try:
                # Kuzu does not support dynamic table names in standard MATCH without knowing the table,
                # but we can query across known edge tables if we don't know the exact one.
                # A safer approach is to check all edge tables.
                for table in edge_tables:
                    await driver.execute_query(
                        f"MATCH ()-[e:{table}]->() WHERE e.uuid = $uuid "
                        "SET e.confidence_score = 1.0, e.last_reinforced_at = $now",
                        uuid=u, now=now
                    )
            except Exception as e:
                logger.debug("Failed to reinforce edge %s: %s", u, e)

    async def decay_knowledge_graph(self, half_life_days: int = 365, threshold: float = 0.2) -> int:
        """Active memory decay and forgetting curve for the context graph.
        Decays edge confidence. If confidence drops below threshold, tombstone it.
        """
        await self._ensure_initialized()
        driver = self._graphiti.driver
        if not hasattr(driver, "execute_query"):
            return 0
            
        import math
        k = math.log(2) / half_life_days
        now = datetime.now(timezone.utc)
        archived = 0

        try:
            tables_res = await driver.execute_query("CALL show_tables() RETURN name, type")
            edge_tables = [t["name"] for t in tables_res if t["type"] == "REL"]
        except Exception:
            return 0
        
        for table in edge_tables:
            await self._setup_confidence_schema(table)
            
            try:
                edges = await driver.execute_query(f"MATCH ()-[e:{table}]->() RETURN e.uuid, e.confidence_score, e.last_reinforced_at")
                for record in edges:
                    u = record.get("e.uuid")
                    conf = record.get("e.confidence_score")
                    last_t = record.get("e.last_reinforced_at")
                    
                    if conf is None: conf = 1.0
                    
                    try:
                        last_time = datetime.fromisoformat(last_t) if (last_t and last_t.strip()) else now
                        if last_time.tzinfo is None:
                            last_time = last_time.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError, AttributeError) as exc:
                        logger.debug("last_t parse failed for %r, using now: %s", last_t, exc)
                        last_time = now
                    
                    delta_days = (now - last_time).days
                    if delta_days > 0:
                        new_conf = conf * math.exp(-k * delta_days)
                        if new_conf < threshold:
                            # Archive/Tombstone the edge
                            await driver.execute_query(
                                f"MATCH ()-[e:{table}]->() WHERE e.uuid=$u SET e.expired_at = $now",
                                u=u, now=now.isoformat()
                            )
                            archived += 1
                        else:
                            await driver.execute_query(
                                f"MATCH ()-[e:{table}]->() WHERE e.uuid=$u SET e.confidence_score = $c",
                                u=u, c=new_conf
                            )
            except Exception as e:
                logger.debug("Decay pass failed on table %s: %s", table, e)

        logger.info("Decayed memory graph. Archived %s stale facts.", archived)
        return archived

    async def export_json(self) -> str:
        """Export the full graph as JSON for backup purposes.

        Returns:
            JSON string with all episodes (last 1000)
        """
        await self._ensure_initialized()

        episodes = await self._graphiti.retrieve_episodes(
            reference_time=datetime.now(timezone.utc),
            last_n=1000,
            group_ids=None,
        )

        export = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "db_path": str(self._db_path),
            "episode_count": len(episodes),
            "episodes": [
                {
                    "uuid": ep.uuid,
                    "name": ep.name,
                    "content": ep.content,
                    "source": ep.source.value if hasattr(ep.source, "value") else str(ep.source),
                    "source_description": ep.source_description,
                    "valid_at": ep.valid_at.isoformat() if ep.valid_at else None,
                    "created_at": ep.created_at.isoformat() if ep.created_at else None,
                }
                for ep in episodes
            ],
        }

        return json.dumps(export, indent=2, ensure_ascii=False)

    async def close(self):
        """Close the graph driver and clean up resources."""
        if self._graphiti:
            try:
                await self._graphiti.close()
            except Exception as e:
                logger.debug("Error closing graph: %s", e)
            self._graphiti = None
            self._initialized = False
