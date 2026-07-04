"""Tests for the Brain RAG memory provider."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def brain_rag_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_brain_rag_hybrid_search(brain_rag_env):
    from plugins.memory.brain_rag.store import BrainRAGStore
    from plugins.memory.brain_rag.retrieval import BrainRAGRetriever

    db = brain_rag_env / "brain-rag.db"
    store = BrainRAGStore(db)
    retriever = BrainRAGRetriever(store)

    store.ingest_text(
        "Python asyncio event loops handle concurrent I/O for automation tasks.",
        source="docs",
        title="asyncio",
    )
    store.remember("User prefers pytest for testing.", category="preference")

    hits = retriever.search("asyncio automation", limit=5)
    assert hits
    contents = " ".join(h["content"] for h in hits).lower()
    assert "asyncio" in contents or "pytest" in contents

    ctx = retriever.format_context(hits)
    assert "asyncio" in ctx.lower() or "pytest" in ctx.lower()
    store.close()


def test_brain_rag_provider_tools(brain_rag_env):
    from plugins.memory.brain_rag import BrainRAGProvider

    provider = BrainRAGProvider()
    provider.initialize("test-session", hermes_home=str(brain_rag_env))

    ingest = provider.handle_tool_call(
        "brain_rag_ingest",
        {"text": "Deploy pipeline runs on GitHub Actions every night.", "title": "ci"},
    )
    assert json.loads(ingest)["success"] is True

    remember = provider.handle_tool_call(
        "brain_rag_remember",
        {"content": "Project uses FastAPI for the API layer.", "category": "code"},
    )
    assert json.loads(remember)["success"] is True

    search = provider.handle_tool_call(
        "brain_rag_search",
        {"query": "deploy pipeline"},
    )
    results = json.loads(search)
    assert results["success"] is True
    assert results["results"]

    provider.shutdown()


def test_brain_rag_discoverable():
    from plugins.memory import discover_memory_providers

    names = [n for n, _desc, _avail in discover_memory_providers()]
    assert "brain_rag" in names


def test_gateway_disabled_by_default():
    from hermes_cli.config import DEFAULT_CONFIG, cfg_get

    assert cfg_get(DEFAULT_CONFIG, "gateway", "enabled") is False
    assert cfg_get(DEFAULT_CONFIG, "agent", "product") == "ai-brain"
    assert cfg_get(DEFAULT_CONFIG, "memory", "provider") == "brain_rag"


def test_ai_brain_skin_exists():
    from hermes_cli.skin_engine import load_skin

    skin = load_skin("ai-brain")
    assert skin.get_branding("agent_name") == "AI Brain"
