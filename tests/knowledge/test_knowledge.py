"""Tests for the Hermes Knowledge Retrieval (RAG) subsystem.

Covers the Step 10 checklist: search, retrieval, citations, sync,
provider swap, caching, failure handling and retries.
"""
from __future__ import annotations

import json
import os
import sys
import time
import unittest
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from packages.knowledge import (  # noqa: E402
    AnythingLLMProvider,
    Document,
    KnowledgeConfig,
    KnowledgeProvider,
    KnowledgeService,
    LocalProvider,
    build_provider,
    register_provider,
)
from packages.knowledge.sync import DocumentSynchronizer  # noqa: E402
from packages.knowledge.types import (  # noqa: E402
    Chunk,
    Citation,
    HealthStatus,
    IndexResult,
    SearchResult,
)

import tempfile
import shutil


def cfg(tmp: str, **kw) -> KnowledgeConfig:
    base = dict(provider="local", db_path=os.path.join(tmp, "k.db"),
                cache_ttl=60, retries=1, retry_backoff=0.01, timeout=5,
                min_score=0.0, top_k=5)
    base.update(kw)
    return KnowledgeConfig(**base)


def doc(i: str, title: str, content: str, path: str = "", source="markdown") -> Document:
    return Document(id=i, title=title, content=content,
                    path=path or f"/vault/{title}.md", source=source)


CORPUS = [
    doc("d1", "Kafka Basics",
        "Kafka is a distributed commit log. Topics are partitioned and replicated. "
        "Consumer groups let you scale horizontally across partitions."),
    doc("d2", "ADR 0007 Authentication",
        "Architecture decision record: we adopt OAuth2 with PKCE for authentication. "
        "Rejected: session cookies. Spring Security enforces the resource server."),
    doc("d3", "Event Sourcing Notes",
        "Event sourcing stores state as an append-only sequence of events. "
        "Projections rebuild read models. Works well with Kafka as the log."),
    doc("d4", "Standup Meeting Notes",
        "Meeting notes: discussed the authentication migration and Kafka retention."),
]


# ----------------------------------------------------------------- basics
class TestLocalProviderSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.p = LocalProvider(os.path.join(self.tmp, "k.db"))
        for d in CORPUS:
            self.p.index(d)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_search_works(self):
        res = self.p.search("kafka partitions consumer groups", limit=3)
        self.assertTrue(res.chunks, "expected hits")
        self.assertEqual(res.chunks[0].document_id, "d1")
        self.assertGreater(res.confidence, 0)

    def test_retrieval_works(self):
        res = self.p.retrieve("what is event sourcing", limit=2)
        self.assertTrue(res.answer)
        self.assertIn("event", res.answer.lower())

    def test_citations_returned(self):
        res = self.p.search("oauth2 pkce authentication decision", limit=2)
        cit = res.chunks[0].citation
        self.assertIsInstance(cit, Citation)
        for field in ("title", "file", "path", "score", "workspace", "chunk_id"):
            self.assertTrue(hasattr(cit, field))
        self.assertTrue(cit.path.endswith(".md"))
        self.assertTrue(cit.chunk_id)
        self.assertEqual(cit.workspace, "default")

    def test_index_is_idempotent_by_checksum(self):
        r = self.p.index(CORPUS[0])
        self.assertEqual(r.action, "skipped")

    def test_update_and_delete(self):
        d = doc("d1", "Kafka Basics", "Completely rewritten content about Kafka Streams.")
        self.assertEqual(self.p.update(d).action, "updated")
        self.assertTrue(self.p.search("Kafka Streams").chunks)
        self.assertEqual(self.p.delete("d1").action, "deleted")
        hits = [c for c in self.p.search("Kafka Streams").chunks if c.document_id == "d1"]
        self.assertFalse(hits)

    def test_find_similar(self):
        res = self.p.find_similar("d1", limit=3)
        self.assertTrue(res.chunks)
        self.assertNotIn("d1", {c.document_id for c in res.chunks})

    def test_health(self):
        h = self.p.health()
        self.assertTrue(h.healthy)
        self.assertEqual(h.provider, "local")


# ---------------------------------------------------------------- service
class TestKnowledgeService(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.svc = KnowledgeService(config=cfg(self.tmp))
        for d in CORPUS:
            self.svc.index(d)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_search_and_rerank(self):
        res = self.svc.search("authentication architecture decision", limit=3)
        self.assertEqual(res.chunks[0].document_id, "d2")

    def test_cached_retrieval(self):
        a = self.svc.search("kafka topics", limit=3)
        self.assertFalse(a.cached)
        b = self.svc.search("kafka topics", limit=3)
        self.assertTrue(b.cached)
        self.assertEqual(self.svc.stats["cache_hits"], 1)
        self.assertEqual([c.id for c in a.chunks], [c.id for c in b.chunks])

    def test_cache_invalidated_on_write(self):
        self.svc.search("kafka topics", limit=3)
        self.svc.index(doc("d9", "New Note", "Kafka topics tiered storage note."))
        self.assertFalse(self.svc.search("kafka topics", limit=3).cached)

    def test_retrieve_with_sources_shape(self):
        out = self.svc.retrieve_with_sources("kafka", limit=2)
        for key in ("answer", "sources", "chunks", "confidence", "provider",
                    "elapsedTime", "workspace"):
            self.assertIn(key, out)
        self.assertTrue(out["sources"])
        self.assertIn("chunk_id", out["sources"][0])

    def test_find_relevant_context_is_prompt_ready(self):
        ctx = self.svc.find_relevant_context("event sourcing projections", limit=2)
        self.assertIn("<knowledge_context>", ctx)
        self.assertIn("path=", ctx)
        self.assertIn("chunk=", ctx)

    def test_requires_retrieval_heuristic(self):
        self.assertTrue(KnowledgeService.requires_retrieval("What do I know about Kafka?"))
        self.assertTrue(KnowledgeService.requires_retrieval(
            "Find every architecture decision about authentication"))
        self.assertFalse(KnowledgeService.requires_retrieval("hi"))
        self.assertFalse(KnowledgeService.requires_retrieval("thanks a lot"))

    def test_health_reports_providers_and_cache(self):
        h = self.svc.health()
        self.assertTrue(h["healthy"])
        self.assertIn("cache", h)
        self.assertEqual(h["providers"][0]["provider"], "local")


# ------------------------------------------------------- swap / resilience
class FakeProvider(KnowledgeProvider):
    """Deterministic in-memory provider used for swap + failure tests."""

    name = "fake"

    def __init__(self, fail_times: int = 0, always_fail: bool = False,
                 slow: float = 0.0, payload: str = "fake knowledge about kafka"):
        self.calls = 0
        self.fail_times = fail_times
        self.always_fail = always_fail
        self.slow = slow
        self.payload = payload

    def _maybe_fail(self):
        self.calls += 1
        if self.slow:
            time.sleep(self.slow)
        if self.always_fail or self.calls <= self.fail_times:
            raise RuntimeError("backend exploded")

    def search(self, query, limit=5, workspace=None, filters=None):
        self._maybe_fail()
        c = Chunk(id="f1", text=self.payload, score=0.9, document_id="fdoc",
                  citation=Citation("Fake Doc", "fake.md", "/fake/fake.md", 0.9,
                                    workspace or "default", "f1", "fdoc", self.name))
        return SearchResult(query=query, chunks=[c], provider=self.name,
                            workspace=workspace or "default", confidence=0.9)

    def retrieve(self, query, limit=5, workspace=None, filters=None):
        r = self.search(query, limit, workspace, filters)
        r.answer = "fake answer"
        return r

    def index(self, document): return IndexResult(True, document.id, "indexed")
    def update(self, document): return IndexResult(True, document.id, "updated")
    def delete(self, document_id, workspace=None): return IndexResult(True, document_id, "deleted")
    def health(self): return HealthStatus(True, self.name, "ok")


class TestProviderSwapAndResilience(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_provider_can_be_swapped_without_touching_hermes(self):
        svc = KnowledgeService(config=cfg(self.tmp))
        for d in CORPUS:
            svc.index(d)
        self.assertEqual(svc.search("kafka").provider, "local")
        svc.set_provider(FakeProvider())
        out = svc.search("kafka")
        self.assertEqual(out.provider, "fake")
        self.assertEqual(out.chunks[0].citation.provider, "fake")

    def test_retries_implemented(self):
        p = FakeProvider(fail_times=1)
        svc = KnowledgeService(config=cfg(self.tmp, retries=2), provider=p)
        res = svc.search("kafka")
        self.assertTrue(res.chunks)
        self.assertEqual(p.calls, 2)
        self.assertEqual(svc.stats["retries"], 1)

    def test_failures_handled_gracefully(self):
        svc = KnowledgeService(config=cfg(self.tmp, retries=1),
                               provider=FakeProvider(always_fail=True))
        res = svc.search("kafka")
        self.assertEqual(res.chunks, [])
        self.assertIn("backend exploded", res.error)
        self.assertEqual(res.provider, "none")
        self.assertGreaterEqual(svc.stats["failures"], 1)

    def test_timeout_enforced(self):
        svc = KnowledgeService(config=cfg(self.tmp, retries=0, timeout=0.05),
                               provider=FakeProvider(slow=0.5))
        res = svc.search("kafka")
        self.assertEqual(res.chunks, [])
        self.assertEqual(svc.stats["timeouts"], 1)

    def test_fallback_provider_used_when_primary_fails(self):
        primary = FakeProvider(always_fail=True)
        backup = LocalProvider(os.path.join(self.tmp, "b.db"))
        for d in CORPUS:
            backup.index(d)
        svc = KnowledgeService(config=cfg(self.tmp, retries=0),
                               providers=[primary, backup])
        res = svc.search("kafka partitions")
        self.assertTrue(res.chunks)
        self.assertIn("local", res.provider)

    def test_merge_dedupes_and_orders(self):
        a = SearchResult(query="q", chunks=[Chunk("x", "t", 0.3)])
        b = SearchResult(query="q", chunks=[Chunk("x", "t", 0.8), Chunk("y", "u", 0.5)])
        merged = KnowledgeService.merge([a, b], 5)
        self.assertEqual([c.id for c in merged], ["x", "y"])
        self.assertEqual(merged[0].score, 0.8)

    def test_registry_supports_new_backends(self):
        for name in ("local", "anythingllm", "qdrant", "weaviate", "chroma", "pgvector"):
            from packages.knowledge.providers import PROVIDER_REGISTRY
            self.assertIn(name, PROVIDER_REGISTRY)
        register_provider("custom", lambda **kw: FakeProvider())
        self.assertEqual(build_provider("custom").name, "fake")
        with self.assertRaises(ValueError):
            build_provider("nope")


# ------------------------------------------------------------ anythingllm
class _FakeHTTP:
    """Minimal urlopen stand-in returning canned JSON per path."""

    def __init__(self, routes: Dict[str, Any]):
        self.routes = routes
        self.seen: List[str] = []

    def __call__(self, req, timeout=None):
        path = req.full_url.split("/api/v1", 1)[1]
        self.seen.append(f"{req.get_method()} {path}")
        body = json.dumps(self.routes.get(path, {})).encode()

        class _Resp:
            def read(self_inner): return body
            def __enter__(self_inner): return self_inner
            def __exit__(self_inner, *a): return False
        return _Resp()


class TestAnythingLLMProvider(unittest.TestCase):
    def setUp(self):
        self.http = _FakeHTTP({
            "/auth": {"authenticated": True},
            "/workspace/default/vector-search": {"results": [{
                "id": "c-1", "docId": "doc-1", "score": 0.82,
                "text": "Kafka retention is configured per topic.",
                "metadata": {"title": "Kafka Ops", "url": "file:///vault/Kafka Ops.md"},
            }]},
            "/workspace/default/chat": {
                "textResponse": "Kafka retention is per-topic.",
                "sources": [{"id": "c-1", "docId": "doc-1", "score": 0.82,
                             "text": "Kafka retention...",
                             "metadata": {"title": "Kafka Ops",
                                          "url": "file:///vault/Kafka Ops.md"}}],
            },
            "/document/raw-text": {"documents": [{"location": "custom/kafka.json"}]},
            "/workspace/default/update-embeddings": {"workspace": {}},
            "/system/remove-documents": {"success": True},
        })
        self.p = AnythingLLMProvider("http://allm:3001", "key", opener=self.http)

    def test_search_maps_citations(self):
        res = self.p.search("kafka retention", limit=3)
        self.assertEqual(res.provider, "anythingllm")
        cit = res.chunks[0].citation
        self.assertEqual(cit.title, "Kafka Ops")
        self.assertEqual(cit.path, "/vault/Kafka Ops.md")
        self.assertEqual(cit.chunk_id, "c-1")
        self.assertAlmostEqual(cit.score, 0.82)

    def test_retrieve_returns_answer_and_sources(self):
        res = self.p.retrieve("kafka retention")
        self.assertEqual(res.answer, "Kafka retention is per-topic.")
        self.assertTrue(res.sources)

    def test_index_uploads_and_embeds(self):
        r = self.p.index(doc("d1", "Kafka Basics", "content"))
        self.assertTrue(r.ok)
        self.assertIn("POST /document/raw-text", self.http.seen)
        self.assertIn("POST /workspace/default/update-embeddings", self.http.seen)

    def test_delete_calls_removal_endpoints(self):
        r = self.p.delete("doc-1")
        self.assertTrue(r.ok)
        self.assertIn("DELETE /system/remove-documents", self.http.seen)

    def test_health(self):
        self.assertTrue(self.p.health().healthy)

    def test_transport_failure_surfaces_as_unhealthy(self):
        def boom(req, timeout=None):
            raise OSError("connection refused")
        bad = AnythingLLMProvider("http://down:3001", opener=boom)
        h = bad.health()
        self.assertFalse(h.healthy)
        self.assertIn("connection refused", h.detail)

    def test_service_over_anythingllm_end_to_end(self):
        tmp = tempfile.mkdtemp()
        try:
            svc = KnowledgeService(config=cfg(tmp, provider="anythingllm"), provider=self.p)
            out = svc.retrieve_with_sources("kafka retention", limit=2)
            self.assertTrue(out["sources"])
            self.assertEqual(out["sources"][0]["provider"], "anythingllm")
            self.assertTrue(out["answer"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# -------------------------------------------------------------------- sync
class TestSync(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.vault = os.path.join(self.tmp, "vault")
        os.makedirs(os.path.join(self.vault, "notes"))
        self._write("notes/kafka.md", "# Kafka\nDistributed log notes.")
        self._write("notes/auth.md", "# Auth ADR\nOAuth2 PKCE decision.")
        self.svc = KnowledgeService(config=cfg(self.tmp))
        self.syncer = DocumentSynchronizer(
            self.svc, manifest_path=os.path.join(self.tmp, "manifest.json"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, rel, text):
        p = os.path.join(self.vault, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(text)
        return p

    def test_sync_detects_new_updated_deleted(self):
        r1 = self.syncer.sync_obsidian(self.vault)
        self.assertEqual(len(r1.added), 2)
        self.assertEqual(len(r1.updated), 0)

        r2 = self.syncer.sync_obsidian(self.vault)
        self.assertEqual(len(r2.added), 0)
        self.assertEqual(len(r2.unchanged), 2, "unchanged files must not be re-indexed")

        self._write("notes/kafka.md", "# Kafka\nNow with tiered storage details.")
        r3 = self.syncer.sync_obsidian(self.vault)
        self.assertEqual(len(r3.updated), 1)
        self.assertEqual(len(r3.unchanged), 1)

        os.remove(os.path.join(self.vault, "notes/auth.md"))
        r4 = self.syncer.sync_obsidian(self.vault)
        self.assertEqual(len(r4.deleted), 1)
        self.assertFalse([c for c in self.svc.search("OAuth2 PKCE").chunks
                          if "auth.md" in (c.citation.path if c.citation else "")])

    def test_synced_content_is_searchable_with_titles(self):
        self.syncer.sync_obsidian(self.vault)
        res = self.svc.search("tiered? distributed log", limit=3)
        self.assertTrue(res.chunks)
        titles = {c.citation.title for c in res.chunks if c.citation}
        self.assertIn("Kafka", titles)

    def test_git_repo_sync_includes_code(self):
        repo = os.path.join(self.tmp, "repo")
        os.makedirs(os.path.join(repo, "src"))
        with open(os.path.join(repo, "README.md"), "w") as fh:
            fh.write("# Service\nSpring Security resource server.")
        with open(os.path.join(repo, "src", "auth.py"), "w") as fh:
            fh.write("def verify_token(tok):\n    return decode_jwt(tok)\n")
        rep = self.syncer.sync_git_repo(repo, include_code=True)
        self.assertEqual(len(rep.added), 2)
        hits = self.svc.search("verify_token decode_jwt", limit=3)
        self.assertTrue(hits.chunks)

    def test_skips_ignored_directories(self):
        os.makedirs(os.path.join(self.vault, "node_modules"))
        self._write("node_modules/junk.md", "should not be indexed")
        rep = self.syncer.sync_obsidian(self.vault)
        self.assertTrue(all("node_modules" not in p for p in rep.added))


# -------------------------------------------------------------------- tool
class TestKnowledgeTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        import packages.knowledge.service as service_mod
        self.svc = KnowledgeService(config=cfg(self.tmp))
        for d in CORPUS:
            self.svc.index(d)
        service_mod._SINGLETON = self.svc

    def tearDown(self):
        import packages.knowledge.service as service_mod
        service_mod._SINGLETON = None
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_tool_output_contract(self):
        from tools.knowledge_tools import knowledge_search

        out = json.loads(knowledge_search("what do I know about kafka", limit=3))
        for key in ("answer", "sources", "chunks", "confidence", "provider", "elapsedTime"):
            self.assertIn(key, out)
        self.assertTrue(out["success"])
        self.assertTrue(out["sources"])
        src = out["sources"][0]
        for key in ("title", "file", "path", "score", "workspace", "chunk_id"):
            self.assertIn(key, src)

    def test_tool_rejects_empty_query(self):
        from tools.knowledge_tools import knowledge_search

        self.assertIn("error", json.loads(knowledge_search("")))

    def test_tool_similar_mode(self):
        from tools.knowledge_tools import knowledge_search

        out = json.loads(knowledge_search("d1", mode="similar", limit=3))
        self.assertTrue(out["success"])

    def test_health_tool(self):
        from tools.knowledge_tools import knowledge_health

        out = json.loads(knowledge_health())
        self.assertTrue(out["healthy"])

    def test_sync_tool(self):
        from tools.knowledge_tools import knowledge_sync

        d = os.path.join(self.tmp, "docs")
        os.makedirs(d)
        with open(os.path.join(d, "x.md"), "w") as fh:
            fh.write("# Redis\nCaching notes.")
        out = json.loads(knowledge_sync(d, source="markdown"))
        self.assertTrue(out["success"])
        self.assertEqual(out["report"]["counts"]["added"], 1)

    def test_tool_is_registered_in_registry_and_toolset(self):
        import tools.knowledge_tools  # noqa: F401
        from tools.registry import registry
        from toolsets import TOOLSETS

        self.assertIn("knowledge_search", registry.get_all_tool_names())
        self.assertEqual(registry.get_toolset_for_tool("knowledge_search"), "knowledge")
        self.assertIn("knowledge_search", TOOLSETS["knowledge"]["tools"])


class TestPromptWiring(unittest.TestCase):
    def test_retrieval_guidance_exists(self):
        from agent.prompt_builder import KNOWLEDGE_RETRIEVAL_GUIDANCE

        g = KNOWLEDGE_RETRIEVAL_GUIDANCE
        self.assertIn("knowledge_search", g)
        self.assertIn("does this require external knowledge?", g)
        self.assertIn("[n]", g)

    def test_guidance_is_gated_on_tool_presence(self):
        import inspect
        import agent.system_prompt as sp

        src = inspect.getsource(sp)
        self.assertIn('if "knowledge_search" in agent.valid_tool_names:', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
