from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _config(tmp_path: Path) -> dict:
    return {
        "workspace": {
            "enabled": True,
            "path": str(tmp_path / "workspace"),
            "auto_create": True,
            "persist_gateway_uploads": "ask",
        },
        "knowledgebase": {
            "enabled": True,
            "path": str(tmp_path / "knowledgebase"),
            "roots": [],
            "retrieval_mode": "off",
            "auto_index": True,
            "watch_for_changes": False,
            "max_injected_chunks": 6,
            "max_injected_tokens": 3200,
            "dense_top_k": 40,
            "sparse_top_k": 40,
            "fused_top_k": 30,
            "final_top_k": 8,
            "min_fused_score": 0.0,
            "injection_format": "sourced_note",
            "chunking": {
                "default_tokens": 512,
                "overlap_tokens": 80,
                "code_strategy": "structural",
                "markdown_strategy": "headings",
            },
            "embeddings": {
                "provider": "local",
                "model": "google/embeddinggemma-300m",
                "dimensions": 768,
            },
            "reranker": {
                "enabled": False,
                "provider": "local",
                "model": "bge-reranker-v2-m3",
            },
            "indexing": {
                "respect_gitignore": True,
                "respect_hermesignore": True,
                "include_hidden": False,
                "max_file_mb": 10,
            },
        },
    }


class TestWorkspacePaths:
    def test_get_workspace_paths_creates_expected_directories(self, tmp_path):
        from agent.workspace import get_workspace_paths

        paths = get_workspace_paths(_config(tmp_path), ensure=True)

        assert paths.workspace_root == tmp_path / "workspace"
        assert paths.knowledgebase_root == tmp_path / "knowledgebase"
        for subdir in ("docs", "notes", "data", "code", "uploads", "media"):
            assert (paths.workspace_root / subdir).is_dir()
        assert paths.indexes_dir.is_dir()
        assert paths.manifests_dir.is_dir()
        assert paths.cache_dir.is_dir()


class TestWorkspaceManifest:
    def test_build_workspace_manifest_writes_summary(self, tmp_path):
        from agent.workspace import build_workspace_manifest

        cfg = _config(tmp_path)
        workspace = Path(cfg["workspace"]["path"])
        (workspace / "docs").mkdir(parents=True)
        (workspace / "notes").mkdir(parents=True)
        (workspace / "docs" / "a.md").write_text("alpha\n", encoding="utf-8")
        (workspace / "notes" / "b.txt").write_text("beta\n", encoding="utf-8")

        manifest = build_workspace_manifest(cfg)

        assert manifest["success"] is True
        assert manifest["file_count"] == 2
        assert manifest["manifest_path"].endswith("workspace.json")
        assert Path(manifest["manifest_path"]).exists()
        paths = {entry["relative_path"] for entry in manifest["files"]}
        assert paths == {"docs/a.md", "notes/b.txt"}

        saved = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
        assert saved["file_count"] == 2


class TestWorkspaceSearch:
    def test_workspace_search_finds_text_matches_and_respects_ignore(self, tmp_path):
        from agent.workspace import workspace_search

        cfg = _config(tmp_path)
        workspace = Path(cfg["workspace"]["path"])
        (workspace / "docs").mkdir(parents=True)
        (workspace / "docs" / "keep.md").write_text("Hermes likes retrieval\n", encoding="utf-8")
        (workspace / "docs" / "skip.md").write_text("Hermes hidden\n", encoding="utf-8")
        (workspace / ".hermesignore").write_text("docs/skip.md\n", encoding="utf-8")
        (workspace / "docs" / "blob.bin").write_bytes(b"\x00\x01\x02Hermes")

        result = workspace_search("Hermes", config=cfg)

        assert result["success"] is True
        assert result["count"] == 1
        match = result["matches"][0]
        assert match["relative_path"] == "docs/keep.md"
        assert match["line"] == 1

    def test_workspace_search_supports_file_glob(self, tmp_path):
        from agent.workspace import workspace_search

        cfg = _config(tmp_path)
        workspace = Path(cfg["workspace"]["path"])
        (workspace / "docs").mkdir(parents=True)
        (workspace / "docs" / "a.md").write_text("deploy target\n", encoding="utf-8")
        (workspace / "docs" / "a.txt").write_text("deploy target\n", encoding="utf-8")

        result = workspace_search("deploy", config=cfg, file_glob="*.md")

        assert result["success"] is True
        assert result["count"] == 1
        assert result["matches"][0]["relative_path"] == "docs/a.md"


class TestWorkspaceEmbedder:
    def test_local_embeddinggemma_uses_sentence_transformers_when_available(self, tmp_path, monkeypatch):
        from agent.workspace import WorkspaceEmbedder

        calls = {}

        class FakeVector(list):
            def tolist(self):
                return list(self)

        class FakeModel:
            def __init__(self, model_id, **kwargs):
                calls["model_id"] = model_id
                calls["kwargs"] = kwargs

            def encode_query(self, text, **kwargs):
                calls["query"] = (text, kwargs)
                return FakeVector([0.1, 0.2, 0.3])

            def encode_document(self, texts, **kwargs):
                calls["documents"] = (list(texts), kwargs)
                return [FakeVector([0.4, 0.5, 0.6]) for _ in texts]

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeModel))

        embedder = WorkspaceEmbedder(_config(tmp_path))
        docs = embedder.embed_documents(["alpha doc"])
        query = embedder.embed_query("alpha query")

        assert embedder.backend == "sentence-transformers"
        assert calls["model_id"] == "google/embeddinggemma-300m"
        assert calls["documents"][0] == ["alpha doc"]
        assert calls["query"][0] == "alpha query"
        assert docs == [[0.4, 0.5, 0.6]]
        assert query == [0.1, 0.2, 0.3]


class TestWorkspaceChunking:
    def test_markdown_chunking_prefers_headings(self, tmp_path):
        from agent.workspace import _chunk_text

        cfg = _config(tmp_path)
        text = "# Intro\n\nAlpha overview.\n\n## Deploy\n\nBlue green rollout plan.\n\n## Rollback\n\nRollback steps.\n"
        chunks = _chunk_text(text, Path("docs/plan.md"), cfg)

        assert len(chunks) >= 3
        assert any("deploy" in chunk["content"].lower() for chunk in chunks)
        assert any("rollback" in chunk["content"].lower() for chunk in chunks)

    def test_code_chunking_prefers_symbol_boundaries(self, tmp_path):
        from agent.workspace import _chunk_text

        cfg = _config(tmp_path)
        text = "def alpha():\n    return 'a'\n\n\ndef beta():\n    return 'b'\n"
        chunks = _chunk_text(text, Path("code/example.py"), cfg)

        assert len(chunks) >= 2
        assert any("def alpha" in chunk["content"] for chunk in chunks)
        assert any("def beta" in chunk["content"] for chunk in chunks)


class TestWorkspaceReranker:
    def test_local_cross_encoder_reranker_reorders_candidates(self, tmp_path, monkeypatch):
        from agent.workspace import WorkspaceReranker

        calls = {}

        class FakeCrossEncoder:
            def __init__(self, model_name, **kwargs):
                calls["model_name"] = model_name
                calls["kwargs"] = kwargs

            def predict(self, pairs, **kwargs):
                calls["pairs"] = pairs
                calls["predict_kwargs"] = kwargs
                return [0.1, 0.9]

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder))

        cfg = _config(tmp_path)
        cfg["knowledgebase"]["reranker"]["enabled"] = True
        cfg["knowledgebase"]["reranker"]["provider"] = "local"
        cfg["knowledgebase"]["reranker"]["model"] = "cross-encoder/ms-marco-MiniLM-L6-v2"

        reranker = WorkspaceReranker(cfg)
        ranked = reranker.rerank(
            "rollback plan",
            [
                {"content": "deployment overview", "rrf_score": 0.9, "dense_score": 0.9},
                {"content": "rollback plan details", "rrf_score": 0.3, "dense_score": 0.2},
            ],
        )

        assert reranker.backend == "cross-encoder"
        assert calls["model_name"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
        assert ranked[0]["content"] == "rollback plan details"


class TestWorkspaceRetrieval:
    def test_index_workspace_builds_chunk_db_and_retrieves_ranked_chunks(self, tmp_path):
        from agent.workspace import index_workspace_knowledgebase, workspace_retrieve

        cfg = _config(tmp_path)
        workspace = Path(cfg["workspace"]["path"])
        (workspace / "docs").mkdir(parents=True)
        (workspace / "docs" / "arch.md").write_text(
            "# Deployment\n\nThe deployment architecture uses blue green rollout and staged health checks.\n",
            encoding="utf-8",
        )
        (workspace / "notes").mkdir(parents=True)
        (workspace / "notes" / "random.txt").write_text("buy groceries\n", encoding="utf-8")

        indexed = index_workspace_knowledgebase(cfg)
        assert indexed["success"] is True
        assert indexed["chunk_count"] >= 1
        assert Path(indexed["index_path"]).exists()

        retrieved = workspace_retrieve("deployment architecture", config=cfg, limit=3)
        assert retrieved["success"] is True
        assert retrieved["count"] >= 1
        assert retrieved["results"][0]["relative_path"] == "docs/arch.md"
        assert "blue green" in retrieved["results"][0]["content"].lower()

    def test_workspace_retrieve_reports_backend_metadata(self, tmp_path):
        from agent.workspace import index_workspace_knowledgebase, workspace_retrieve

        cfg = _config(tmp_path)
        workspace = Path(cfg["workspace"]["path"])
        (workspace / "docs").mkdir(parents=True)
        (workspace / "docs" / "plan.md").write_text("blue green rollout plan\n", encoding="utf-8")

        index_workspace_knowledgebase(cfg)
        retrieved = workspace_retrieve("blue green rollout", config=cfg, limit=2)

        assert "dense_backend" in retrieved
        assert "rerank_backend" in retrieved

    def test_workspace_context_for_turn_formats_sources_and_respects_gating(self, tmp_path):
        from agent.workspace import index_workspace_knowledgebase, workspace_context_for_turn

        cfg = _config(tmp_path)
        cfg["knowledgebase"]["retrieval_mode"] = "always"
        workspace = Path(cfg["workspace"]["path"])
        (workspace / "docs").mkdir(parents=True)
        (workspace / "docs" / "plan.md").write_text(
            "Deployment plan includes canary analysis and rollback checkpoints.\n",
            encoding="utf-8",
        )

        index_workspace_knowledgebase(cfg)
        context = workspace_context_for_turn("summarize the deployment plan", config=cfg)
        assert "workspace context was retrieved for this turn only" in context.lower()
        assert "[source: relative/path]" in context.lower()
        assert "docs/plan.md" in context

        cfg["knowledgebase"]["retrieval_mode"] = "gated"
        assert workspace_context_for_turn("thanks", config=cfg) == ""
