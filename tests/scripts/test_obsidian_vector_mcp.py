"""Tests for scripts/obsidian_vector_mcp.py — focused, no Ollama/network required.

All tests use an in-process fake backend (hash embeddings) and a temporary
SQLite index so they run offline and deterministically.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import sqlite3
import time
from pathlib import Path
from types import ModuleType

import pytest

# ---------------------------------------------------------------------------
# Load module under test
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "obsidian_vector_mcp.py"
_DUMMY_BACKEND = Path("/dummy/backend.py")
_TEST_MODEL = "mxbai-embed-large"  # must match the default in _run_search


def _load_mcp_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("obsidian_vector_mcp", _SCRIPT)
    assert spec and spec.loader, f"cannot load {_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Minimal fake backend — no dependency on external backend file or Ollama
# ---------------------------------------------------------------------------


def _norm(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n else vec


def _hash_embed(text: str, dims: int = 384) -> list[float]:
    vec = [0.0] * dims
    for token in re.findall(r"[\w\-]+", text.lower()):
        h = hashlib.blake2b(token.encode(), digest_size=8).digest()
        n = int.from_bytes(h, "little")
        vec[n % dims] += 1.0 if ((n >> 9) & 1) else -1.0
    return _norm(vec)


def _make_fake_backend() -> ModuleType:
    """Return a module-like object implementing the backend surface used by _run_search."""
    mod = ModuleType("_fake_backend")
    mod.__file__ = __file__

    def connect(index_path: Path | str) -> sqlite3.Connection:
        p = Path(index_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(p)
        db.execute("pragma journal_mode=wal")
        db.execute(
            """
            create table if not exists chunks (
                id integer primary key,
                path text not null,
                heading text not null,
                chunk_index integer not null,
                text text not null,
                embedding_json text not null,
                file_mtime real not null,
                file_hash text not null,
                model text not null,
                updated_at real not null,
                unique(path, chunk_index, model)
            )
            """
        )
        return db

    def embed_texts(texts, base_url, model):  # noqa: ARG001
        return [_hash_embed(t) for t in texts]

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    mod.connect = connect
    mod.embed_texts = embed_texts
    mod.cosine = cosine
    return mod


def _populate_index(db_path: Path, bk: ModuleType, model: str = _TEST_MODEL) -> None:
    db = bk.connect(db_path)
    now = time.time()
    entries = [
        (
            "transformers/attention.md",
            "Attention Mechanism",
            0,
            "Transformers rely on self-attention to relate tokens across positions.",
        ),
        (
            "transformers/attention.md",
            "Attention Mechanism",
            1,
            "Multi-head attention applies the mechanism in parallel across subspaces.",
        ),
        (
            "cnn/filters.md",
            "Convolutional Filters",
            0,
            "CNNs use learned filters to detect local patterns in images.",
        ),
        (
            "rl/policy.md",
            "Policy Gradient",
            0,
            "Policy gradient methods optimize expected cumulative reward.",
        ),
    ]
    for path, heading, cidx, text in entries:
        vec = _hash_embed(text)
        db.execute(
            "insert into chunks(path,heading,chunk_index,text,embedding_json,"
            "file_mtime,file_hash,model,updated_at) values(?,?,?,?,?,?,?,?,?)",
            (path, heading, cidx, text, json.dumps(vec), now, f"h{cidx}", model, now),
        )
    db.commit()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _hash_mode(monkeypatch):
    """Force hash embedding mode for every test in this module."""
    monkeypatch.setenv("LLM_WIKI_EMBEDDING_MODE", "hash")


@pytest.fixture()
def fake_bk() -> ModuleType:
    return _make_fake_backend()


@pytest.fixture()
def mcp_mod() -> ModuleType:
    mod = _load_mcp_module()
    mod._backend_cache.clear()
    return mod


@pytest.fixture()
def temp_index(tmp_path: Path, fake_bk: ModuleType) -> Path:
    idx = tmp_path / "test.sqlite"
    _populate_index(idx, fake_bk)
    return idx


@pytest.fixture()
def ctx(mcp_mod: ModuleType, fake_bk: ModuleType, temp_index: Path):
    """Pre-wired trio: mcp_mod with fake backend cached, and a populated temp index."""
    mcp_mod._backend_cache[str(_DUMMY_BACKEND)] = fake_bk
    return mcp_mod, temp_index


# ---------------------------------------------------------------------------
# Tests: JSON shape
# ---------------------------------------------------------------------------


class TestJsonShape:
    def test_top_level_keys(self, ctx):
        mod, idx = ctx
        data = json.loads(mod._run_search("attention", 3, None, False, _DUMMY_BACKEND, idx))
        assert data["query"] == "attention"
        assert isinstance(data["count"], int)
        assert isinstance(data["results"], list)
        assert data["count"] == len(data["results"])

    def test_result_item_required_fields(self, ctx):
        mod, idx = ctx
        data = json.loads(mod._run_search("attention", 1, None, False, _DUMMY_BACKEND, idx))
        assert data["results"], "expected at least one result"
        r = data["results"][0]
        for key in ("title", "path", "heading", "chunk_index", "snippet", "score", "source", "metadata"):
            assert key in r, f"missing field: {key}"
        assert "content" not in r  # include_content=False

    def test_title_derives_from_heading(self, ctx):
        mod, idx = ctx
        data = json.loads(mod._run_search("attention mechanism", 1, None, False, _DUMMY_BACKEND, idx))
        r = data["results"][0]
        assert r["title"] != ""
        assert r["title"] == r["heading"]

    def test_title_falls_back_to_path_stem(self, mcp_mod, fake_bk, tmp_path):
        """When heading is empty, title is the path stem."""
        mcp_mod._backend_cache[str(_DUMMY_BACKEND)] = fake_bk
        idx = tmp_path / "noheading.sqlite"
        db = fake_bk.connect(idx)
        now = time.time()
        text = "some content without a heading"
        db.execute(
            "insert into chunks(path,heading,chunk_index,text,embedding_json,"
            "file_mtime,file_hash,model,updated_at) values(?,?,?,?,?,?,?,?,?)",
            ("my-note.md", "", 0, text, json.dumps(_hash_embed(text)), now, "h0", _TEST_MODEL, now),
        )
        db.commit()
        data = json.loads(mcp_mod._run_search("content", 1, None, False, _DUMMY_BACKEND, idx))
        assert data["results"][0]["title"] == "my-note"

    def test_snippet_whitespace_normalised(self, mcp_mod, fake_bk, tmp_path):
        mcp_mod._backend_cache[str(_DUMMY_BACKEND)] = fake_bk
        messy = "line1\n\n\nline2\t\ttab\n   indent"
        idx = tmp_path / "messy.sqlite"
        db = fake_bk.connect(idx)
        now = time.time()
        db.execute(
            "insert into chunks(path,heading,chunk_index,text,embedding_json,"
            "file_mtime,file_hash,model,updated_at) values(?,?,?,?,?,?,?,?,?)",
            ("messy.md", "Messy", 0, messy, json.dumps(_hash_embed(messy)), now, "h0", _TEST_MODEL, now),
        )
        db.commit()
        data = json.loads(mcp_mod._run_search("line", 1, None, False, _DUMMY_BACKEND, idx))
        snippet = data["results"][0]["snippet"]
        assert "\n" not in snippet
        assert "\t" not in snippet
        assert "  " not in snippet

    def test_snippet_capped_at_limit(self, mcp_mod, fake_bk, tmp_path):
        mcp_mod._backend_cache[str(_DUMMY_BACKEND)] = fake_bk
        long_text = ("word " * 600).strip()
        idx = tmp_path / "long.sqlite"
        db = fake_bk.connect(idx)
        now = time.time()
        db.execute(
            "insert into chunks(path,heading,chunk_index,text,embedding_json,"
            "file_mtime,file_hash,model,updated_at) values(?,?,?,?,?,?,?,?,?)",
            ("long.md", "Long", 0, long_text, json.dumps(_hash_embed(long_text)), now, "h0", _TEST_MODEL, now),
        )
        db.commit()
        data = json.loads(mcp_mod._run_search("word", 1, None, False, _DUMMY_BACKEND, idx))
        snippet = data["results"][0]["snippet"]
        assert len(snippet) <= mcp_mod._SNIPPET_CHARS + 1  # +1 for the ellipsis char


# ---------------------------------------------------------------------------
# Tests: include_content
# ---------------------------------------------------------------------------


class TestIncludeContent:
    def test_content_present_when_true(self, ctx):
        mod, idx = ctx
        data = json.loads(mod._run_search("attention", 2, None, True, _DUMMY_BACKEND, idx))
        assert all("content" in r for r in data["results"])

    def test_content_absent_when_false(self, ctx):
        mod, idx = ctx
        data = json.loads(mod._run_search("attention", 2, None, False, _DUMMY_BACKEND, idx))
        assert all("content" not in r for r in data["results"])

    def test_content_matches_full_text(self, ctx):
        mod, idx = ctx
        data = json.loads(mod._run_search("attention mechanism", 1, None, True, _DUMMY_BACKEND, idx))
        r = data["results"][0]
        assert r["content"] != ""
        # content should be the unshortenend original, longer than the snippet
        assert len(r["content"]) >= len(r["snippet"].rstrip("…"))


# ---------------------------------------------------------------------------
# Tests: limit coercion
# ---------------------------------------------------------------------------


class TestLimitCoercion:
    @pytest.mark.parametrize("limit", [1, 2, 3])
    def test_limit_caps_results(self, ctx, limit):
        mod, idx = ctx
        data = json.loads(mod._run_search("attention", limit, None, False, _DUMMY_BACKEND, idx))
        assert data["count"] <= limit

    def test_limit_zero_returns_at_least_one(self, ctx):
        """Limit <= 0 is coerced to 1 so search always returns something."""
        mod, idx = ctx
        data = json.loads(mod._run_search("attention", 0, None, False, _DUMMY_BACKEND, idx))
        assert data["count"] >= 1

    def test_large_limit_doesnt_exceed_corpus(self, ctx):
        mod, idx = ctx
        data = json.loads(mod._run_search("the", 999, None, False, _DUMMY_BACKEND, idx))
        assert data["count"] <= 4  # only 4 chunks in fixture


# ---------------------------------------------------------------------------
# Tests: source label
# ---------------------------------------------------------------------------


class TestSourceLabel:
    def test_custom_source_applied(self, ctx):
        mod, idx = ctx
        data = json.loads(mod._run_search("attention", 2, "my-vault", False, _DUMMY_BACKEND, idx))
        assert all(r["source"] == "my-vault" for r in data["results"])

    def test_default_source_is_llm_wiki(self, ctx):
        mod, idx = ctx
        data = json.loads(mod._run_search("attention", 2, None, False, _DUMMY_BACKEND, idx))
        assert all(r["source"] == "llm-wiki" for r in data["results"])


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_empty_query_error(self, mcp_mod, fake_bk, temp_index):
        mcp_mod._backend_cache[str(_DUMMY_BACKEND)] = fake_bk
        data = json.loads(mcp_mod._run_search("", 8, None, False, _DUMMY_BACKEND, temp_index))
        assert "error" in data

    def test_whitespace_only_query_error(self, mcp_mod, fake_bk, temp_index):
        mcp_mod._backend_cache[str(_DUMMY_BACKEND)] = fake_bk
        data = json.loads(mcp_mod._run_search("   ", 8, None, False, _DUMMY_BACKEND, temp_index))
        assert "error" in data

    def test_missing_backend_returns_json_error(self, mcp_mod, temp_index):
        """Backend not in cache and path nonexistent → JSON error, not exception."""
        bad = Path("/nonexistent/backend.py")
        data = json.loads(mcp_mod._run_search("attention", 4, None, False, bad, temp_index))
        assert "error" in data
        assert "backend" in data["error"].lower() or "not found" in data["error"].lower()

    def test_missing_index_returns_json_error(self, mcp_mod, fake_bk, tmp_path):
        mcp_mod._backend_cache[str(_DUMMY_BACKEND)] = fake_bk
        bad_idx = tmp_path / "nonexistent.sqlite"
        data = json.loads(mcp_mod._run_search("attention", 4, None, False, _DUMMY_BACKEND, bad_idx))
        assert "error" in data
        assert "index" in data["error"].lower()

    def test_empty_index_returns_json_error(self, mcp_mod, fake_bk, tmp_path):
        """Index file exists with schema but zero rows → JSON error."""
        mcp_mod._backend_cache[str(_DUMMY_BACKEND)] = fake_bk
        empty_idx = tmp_path / "empty.sqlite"
        fake_bk.connect(empty_idx)  # creates schema, no rows
        data = json.loads(mcp_mod._run_search("attention", 4, None, False, _DUMMY_BACKEND, empty_idx))
        assert "error" in data
        assert "empty" in data["error"].lower() or "index" in data["error"].lower()

    def test_error_response_is_valid_json(self, mcp_mod, tmp_path):
        """All error paths produce valid JSON, not raised exceptions."""
        bad = Path("/nonexistent/backend.py")
        bad_idx = tmp_path / "nonexistent.sqlite"
        for result_str in [
            mcp_mod._run_search("", 8, None, False, bad, bad_idx),
            mcp_mod._run_search("q", 8, None, False, bad, bad_idx),
        ]:
            parsed = json.loads(result_str)  # must not raise
            assert isinstance(parsed, dict)
