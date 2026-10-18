"""Hermetic regression tests for the optional wiki-search skill."""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import urllib.error
from pathlib import Path

import pytest
import yaml


SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "research"
    / "wiki-search"
)
SCRIPT_PATH = SKILL_DIR / "scripts" / "wiki_search.py"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    source = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    match = re.search(r"^---\n(.*?)\n---", source, re.DOTALL)
    assert match, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(match.group(1))


def test_skill_metadata_and_python_source_are_compliant(frontmatter: dict) -> None:
    assert SKILL_DIR.is_dir()
    assert frontmatter["name"] == "wiki-search"
    assert len(frontmatter["description"]) <= 60
    assert frontmatter["description"].endswith(".")
    assert "PINKIIILQWQ" in frontmatter["author"]
    assert frontmatter["license"] == "MIT"
    ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))


def load_module():
    spec = importlib.util.spec_from_file_location("wiki_search", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_namespace_isolated_by_root_and_model(tmp_path: Path) -> None:
    module = load_module()
    cache_dir = tmp_path / "cache"
    root_a = tmp_path / "wiki-a"
    root_b = tmp_path / "wiki-b"
    root_a.mkdir()
    root_b.mkdir()

    path_a = module.index_path(cache_dir, root_a, "all-minilm")
    path_b = module.index_path(cache_dir, root_b, "all-minilm")
    path_other_model = module.index_path(cache_dir, root_a, "nomic-embed-text")

    assert path_a != path_b
    assert path_a != path_other_model

    index_a = module.fresh_index(root_a, "all-minilm")
    index_a["files"] = {"only-a.md": {"sections": []}}
    module.save_index(cache_dir, root_a, "all-minilm", index_a)

    index_b = module.fresh_index(root_b, "all-minilm")
    index_b["files"] = {"only-b.md": {"sections": []}}
    module.save_index(cache_dir, root_b, "all-minilm", index_b)

    assert set(module.load_index(cache_dir, root_a, "all-minilm")["files"]) == {"only-a.md"}
    assert set(module.load_index(cache_dir, root_b, "all-minilm")["files"]) == {"only-b.md"}


def test_load_index_discards_incompatible_metadata(tmp_path: Path) -> None:
    module = load_module()
    cache_dir = tmp_path / "cache"
    root = tmp_path / "wiki"
    root.mkdir()
    path = module.index_path(cache_dir, root, "all-minilm")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "schema": module.CACHE_SCHEMA_VERSION,
                    "root": str(root.resolve()),
                    "model": "nomic-embed-text",
                    "dimension": 768,
                },
                "files": {"stale.md": {"sections": []}},
            }
        ),
        encoding="utf-8",
    )

    loaded = module.load_index(cache_dir, root, "all-minilm")

    assert loaded["files"] == {}
    assert loaded["metadata"]["model"] == "all-minilm"
    assert loaded["metadata"]["dimension"] is None


def test_load_index_discards_stored_vectors_with_the_wrong_dimension(tmp_path: Path) -> None:
    module = load_module()
    cache_dir = tmp_path / "cache"
    root = tmp_path / "wiki"
    root.mkdir()
    corrupt = module.fresh_index(root, "all-minilm")
    corrupt["metadata"]["dimension"] = 2
    corrupt["files"] = {
        "wrong.md": {
            "sections": [
                {"heading": "Wrong", "content": "A corrupt vector entry.", "embedding": [1.0, 0.0, 0.0]}
            ]
        }
    }
    path = module.index_path(cache_dir, root, "all-minilm")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(corrupt), encoding="utf-8")

    loaded = module.load_index(cache_dir, root, "all-minilm")

    assert loaded["files"] == {}
    assert loaded["metadata"]["dimension"] is None


def test_indexing_prunes_deleted_markdown_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    cache_dir = tmp_path / "cache"
    root = tmp_path / "wiki"
    root.mkdir()
    (root / "one.md").write_text("# One\n\nUseful content for the first page.", encoding="utf-8")
    gone = root / "gone.md"
    gone.write_text("# Gone\n\nUseful content for the deleted page.", encoding="utf-8")
    monkeypatch.setattr(module, "ollama_embed", lambda text, model: [1.0, 0.0])

    module.cmd_index(root, model="all-minilm", cache_dir=cache_dir)
    gone.unlink()
    module.cmd_index(root, model="all-minilm", cache_dir=cache_dir)

    index = module.load_index(cache_dir, root, "all-minilm")
    assert set(index["files"]) == {"one.md"}


def test_indexing_rejects_sections_with_incompatible_vector_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    cache_dir = tmp_path / "cache"
    root = tmp_path / "wiki"
    root.mkdir()
    (root / "two.md").write_text(
        "# Two\n\nFirst section has useful content.\n\n## Three\n\nSecond section has useful content.",
        encoding="utf-8",
    )
    vectors = iter(([1.0, 0.0], [1.0, 0.0, 0.0]))
    monkeypatch.setattr(module, "ollama_embed", lambda text, model: next(vectors))

    module.cmd_index(root, model="all-minilm", cache_dir=cache_dir)

    index = module.load_index(cache_dir, root, "all-minilm")
    sections = index["files"]["two.md"]["sections"]
    assert index["metadata"]["dimension"] == 2
    assert sections[0]["embedding"] == [1.0, 0.0]
    assert sections[1]["embedding"] is None


def test_reindex_rebuilds_a_persisted_namespace_after_embedding_dimension_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    cache_dir = tmp_path / "cache"
    root = tmp_path / "wiki"
    root.mkdir()
    (root / "entry.md").write_text("# Entry\n\nA durable note that is long enough to embed.", encoding="utf-8")

    monkeypatch.setattr(module, "ollama_embed", lambda text, model: [1.0, 0.0])
    module.cmd_index(root, model="all-minilm", cache_dir=cache_dir)

    monkeypatch.setattr(module, "ollama_embed", lambda text, model: [1.0, 0.0, 0.0])
    module.cmd_index(root, model="all-minilm", cache_dir=cache_dir, force=True)

    rebuilt = module.load_index(cache_dir, root, "all-minilm")
    embedding = rebuilt["files"]["entry.md"]["sections"][0]["embedding"]

    assert rebuilt["metadata"]["dimension"] == 3
    assert embedding == [1.0, 0.0, 0.0]
    results, semantic_available = module.semantic_search("durable note", rebuilt, "all-minilm")
    assert semantic_available is True
    assert results[0]["file"] == "entry.md"


def test_cosine_similarity_rejects_mismatched_vectors_and_ollama_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )

    assert module.cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) is None
    assert module.ollama_embed("query", "all-minilm") is None


def test_ollama_embed_honors_configured_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    requested_urls: list[str] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self) -> bytes:
            return b'{"embedding": [1, 0]}'

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        return Response()

    monkeypatch.setenv("OLLAMA_URL", "http://ollama.example:11434/")
    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)

    assert module.ollama_embed("query", "all-minilm") == [1.0, 0.0]
    assert requested_urls == ["http://ollama.example:11434/api/embeddings"]


def test_search_falls_back_to_keyword_results_when_embedding_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    cache_dir = tmp_path / "cache"
    root = tmp_path / "wiki"
    root.mkdir()
    index = module.fresh_index(root, "all-minilm")
    index["files"] = {
        "notes.md": {
            "title": "Notes",
            "sections": [
                {
                    "heading": "Useful note",
                    "content": "The green comet appears only in winter.",
                    "embedding": None,
                }
            ],
        }
    }
    module.save_index(cache_dir, root, "all-minilm", index)
    monkeypatch.setattr(module, "ollama_embed", lambda text, model: None)

    result = module.cmd_search(
        "green comet", root, model="all-minilm", cache_dir=cache_dir, mode="semantic"
    )

    assert result["fallback"] == "keyword"
    assert [item["file"] for item in result["results"]] == ["notes.md"]


def test_json_cli_emits_a_stable_search_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_module()
    cache_dir = tmp_path / "cache"
    root = tmp_path / "wiki"
    root.mkdir()
    index = module.fresh_index(root, "all-minilm")
    index["files"] = {
        "notes.md": {
            "title": "Notes",
            "sections": [
                {"heading": "Useful note", "content": "The green comet appears in winter.", "embedding": None}
            ],
        }
    }
    module.save_index(cache_dir, root, "all-minilm", index)
    monkeypatch.setenv("WIKI_SEARCH_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(module, "ollama_embed", lambda text, model: None)
    monkeypatch.setattr(sys, "argv", ["wiki-search", "--json", "--wiki", str(root), "green comet"])

    module.main()

    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"schema_version", "command", "wiki", "model", "mode", "fallback", "results"}
    assert payload["schema_version"] == 1
    assert payload["command"] == "search"
    assert payload["fallback"] == "keyword"
    assert payload["results"][0]["file"] == "notes.md"
