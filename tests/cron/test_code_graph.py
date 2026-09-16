"""Tests for the per-service code knowledge graph builder (``cron/code_graph.py``).

Covers the parts that must hold regardless of whether ``graphifyy`` is installed:
the graphify→wire grammar mapping, structural kind derivation, browse-root
containment + read caps, the content-digest cache (a hit must not spawn the
subprocess; a byte change must rebuild; the code-control revision participates in
the key), and the cheap node stamp. A single real-``graphify`` integration test
is guarded by an import check so CI without the package still passes.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cron.code_graph as code_graph  # noqa: E402
from cron.code_graph import (  # noqa: E402
    CodeGraphUnavailable,
    _assemble,
    _derive_kind,
    _edge_class,
    build_service_code_graph,
    eligible_source_files,
    service_code_graph_stamp,
    source_files_digest,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _use_root(monkeypatch, root: Path) -> None:
    """Point the source-file resolver at a single ``repo`` root under tmp."""
    import cron.jobs as jobs

    monkeypatch.setattr(jobs, "_source_file_roots", lambda: {"repo": root.resolve()})


def _entry(root: Path, rel: str, *, exists: bool = True, root_name: str = "repo") -> dict:
    return {
        "path": str((root / rel).resolve()),
        "declared": rel,
        "role": "declared",
        "root": root_name,
        "rel": rel,
        "exists": exists,
    }


# --------------------------------------------------------------------------- #
# Grammar + kind
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "relation,expected",
    [
        ("imports", "flow"),
        ("imports_from", "flow"),
        ("calls", "flow"),
        ("contains", "structure"),
        ("implements", "structure"),
        ("inherits", "structure"),
        ("references", "reference"),
        ("rationale_for", "reference"),
        ("anything_else", "reference"),
    ],
)
def test_edge_class_mapping(relation, expected):
    assert _edge_class(relation) == expected


@pytest.mark.parametrize(
    "node,expected",
    [
        ({"source_file": "", "label": "Any"}, "external"),
        ({"source_file": "a.py", "label": "server.py"}, "module"),
        ({"source_file": "a.py", "label": "handle()", "_callable": True}, "func"),
        ({"source_file": "a.py", "label": "Widget"}, "class"),
        ({"source_file": "a.py", "label": "count"}, "symbol"),
    ],
)
def test_derive_kind(node, expected):
    assert _derive_kind(node) == expected


def test_assemble_maps_sorts_and_resolves():
    source_index = {"app/server.py": {"root": "repo", "rel": "app/server.py"}}
    raw = {
        "nodes": [
            {"id": "z", "label": "server.py", "source_file": "app/server.py", "source_location": "L1"},
            {"id": "a", "label": "handle()", "source_file": "app/server.py",
             "source_location": "L10", "_callable": True},
            {"id": "ext", "label": "json", "source_file": "", "source_location": ""},
        ],
        "edges": [
            {"source": "z", "target": "a", "relation": "contains", "confidence": "EXTRACTED"},
            {"source": "a", "target": "ext", "relation": "imports", "confidence": "EXTRACTED"},
        ],
        "communities": {1: ["a", "z"], 0: ["ext"]},
    }
    out = _assemble("svc", "digest123", {"repository": "o/r"}, source_index, raw)

    assert out["service"] == "svc"
    assert out["digest"] == "digest123"
    assert out["code_control"] == {"repository": "o/r"}
    # Nodes sorted by id; edges sorted by (source, target, type).
    assert [n["id"] for n in out["nodes"]] == ["a", "ext", "z"]
    assert [(e["source"], e["target"]) for e in out["edges"]] == [("a", "ext"), ("z", "a")]
    # Resolution + kind + community.
    server = next(n for n in out["nodes"] if n["id"] == "z")
    assert server["kind"] == "module" and server["root"] == "repo"
    assert server["rel"] == "app/server.py" and server["path"] == "app/server.py"
    assert server["line"] == 1 and server["community"] == "1"
    ext = next(n for n in out["nodes"] if n["id"] == "ext")
    assert ext["kind"] == "external" and ext["root"] is None and ext["rel"] is None
    # Edge classes carried through.
    classes = {(e["source"], e["target"]): e["class"] for e in out["edges"]}
    assert classes[("z", "a")] == "structure"
    assert classes[("a", "ext")] == "flow"
    # Communities canonicalised (string keys, sorted members).
    assert out["communities"] == {"0": ["ext"], "1": ["a", "z"]}


# --------------------------------------------------------------------------- #
# Eligibility: containment, caps, binary
# --------------------------------------------------------------------------- #
def test_eligible_filters_containment_size_binary(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _use_root(monkeypatch, root)

    (root / "good.py").write_text("x = 1\n", encoding="utf-8")
    (root / "binary.bin").write_bytes(b"\x00\x01\x02data")
    (root / "big.py").write_bytes(b"# " + b"a" * (1024 * 1024 + 10))
    outside = tmp_path / "outside.py"
    outside.write_text("y = 2\n", encoding="utf-8")

    entries = [
        _entry(root, "good.py"),
        _entry(root, "binary.bin"),
        _entry(root, "big.py"),
        _entry(root, "missing.py", exists=False),
        _entry(root, "good.py", root_name="secrets"),  # non-allowlisted root claim
        {  # outside any browse root — re-verification drops it
            "path": str(outside), "declared": "outside.py", "role": "declared",
            "root": "repo", "rel": "outside.py", "exists": True,
        },
    ]
    eligible = eligible_source_files(entries)
    assert [e["rel"] for e in eligible] == ["good.py"]
    assert eligible[0]["root"] == "repo"


def test_eligible_never_recurses_directories(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("z = 3\n", encoding="utf-8")
    _use_root(monkeypatch, root)
    # A directory entry must not be walked into — only explicit files count.
    entries = [{"path": str(root / "pkg"), "declared": "pkg", "role": "declared",
                "root": "repo", "rel": "pkg", "exists": True}]
    assert eligible_source_files(entries) == []


# --------------------------------------------------------------------------- #
# Digests
# --------------------------------------------------------------------------- #
def test_content_digest_changes_on_edit(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _use_root(monkeypatch, root)
    target = root / "a.py"
    target.write_text("v = 1\n", encoding="utf-8")
    entries = [_entry(root, "a.py")]

    first = source_files_digest(entries)
    assert first == source_files_digest(entries)  # stable for unchanged content
    target.write_text("v = 2\n", encoding="utf-8")
    assert source_files_digest(entries) != first


def test_stamp_changes_on_edit_and_is_none_without_files(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _use_root(monkeypatch, root)
    assert service_code_graph_stamp([]) is None

    target = root / "a.py"
    target.write_text("v = 1\n", encoding="utf-8")
    entries = [_entry(root, "a.py")]
    first = service_code_graph_stamp(entries)
    assert first is not None
    target.write_text("v = 22\n", encoding="utf-8")  # size + mtime change
    assert service_code_graph_stamp(entries) != first


# --------------------------------------------------------------------------- #
# Cache behaviour (subprocess-free via a stubbed runner)
# --------------------------------------------------------------------------- #
@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _use_root(monkeypatch, root)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (root / "a.py").write_text("v = 1\n", encoding="utf-8")

    calls = {"n": 0}

    def fake_run(files, *, root, timeout):
        calls["n"] += 1
        return {"nodes": [], "edges": [], "communities": {}}

    monkeypatch.setattr(code_graph, "_run_graphify", fake_run)
    return root, calls


def test_cache_hit_skips_subprocess_and_edit_rebuilds(cache_env):
    root, calls = cache_env
    entries = [_entry(root, "a.py")]

    build_service_code_graph("svc", source_files=entries)
    build_service_code_graph("svc", source_files=entries)
    assert calls["n"] == 1  # second call served from cache

    (root / "a.py").write_text("v = 999\n", encoding="utf-8")
    build_service_code_graph("svc", source_files=entries)
    assert calls["n"] == 2  # content changed → rebuild


def test_revision_participates_in_cache_key(cache_env):
    root, calls = cache_env
    entries = [_entry(root, "a.py")]
    verified = {"status": "verified", "repository": "o/r", "revision": "aaa",
                "pull_request": {"number": 1}}

    build_service_code_graph("svc", source_files=entries, code_control=verified)
    build_service_code_graph("svc", source_files=entries, code_control=verified)
    assert calls["n"] == 1

    verified2 = dict(verified, revision="bbb")
    build_service_code_graph("svc", source_files=entries, code_control=verified2)
    assert calls["n"] == 2  # same files, new revision → new cache key


def test_no_readable_files_raises(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _use_root(monkeypatch, root)
    with pytest.raises(CodeGraphUnavailable):
        build_service_code_graph("svc", source_files=[])


# --------------------------------------------------------------------------- #
# Real graphify (skipped when no resolvable interpreter has the package)
# --------------------------------------------------------------------------- #
def _graphify_runnable() -> bool:
    """Whether the interpreter the builder would use can import ``graphify``.

    Honours ``HERMES_GRAPHIFY_PYTHON`` (a venv) exactly as the builder does, so
    the integration test runs locally with the env set and in CI where
    ``graphifyy`` is installed in the gateway env — and skips otherwise.
    """
    import subprocess

    try:
        return subprocess.run(
            [code_graph._graphify_python(), "-c", "import graphify"],
            capture_output=True, timeout=30,
        ).returncode == 0
    except Exception:
        return False


@pytest.mark.skipif(not _graphify_runnable(), reason="graphifyy not installed")
def test_real_graphify_end_to_end(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    _use_root(monkeypatch, root)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    (root / "app" / "server.py").write_text(
        "import json\n\n\nclass Server:\n    def handle(self):\n        return json.dumps({})\n",
        encoding="utf-8",
    )
    entries = [_entry(root, "app/server.py")]
    graph = build_service_code_graph("svc", source_files=entries, use_cache=False)

    assert graph["nodes"] and graph["edges"]
    kinds = {n["kind"] for n in graph["nodes"]}
    assert "module" in kinds
    # The import edge is a flow edge; a resolved node deep-links to its file.
    assert any(e["class"] == "flow" for e in graph["edges"])
    assert any(n["rel"] == "app/server.py" and n["line"] for n in graph["nodes"])
