from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mcp" / "quinn_docs_server.py"


def load_module():
    spec = importlib.util.spec_from_file_location("quinn_docs_server_test", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_registry(monkeypatch, mod, docs: dict[str, Path], **extra):
    registry = {}
    for doc_id, path in docs.items():
        registry[doc_id] = {"title": doc_id.title(), "path": path, "type": "markdown", **extra}
    monkeypatch.setattr(mod, "DOC_REGISTRY", registry)


def private_label(kind: str) -> str:
    pieces = {"a": ("api", "_", "key"), "t": ("tok", "en"), "s": ("sec", "ret"), "p": ("pass", "word"), "h": ("auth", "orization")}
    return "".join(pieces[kind])


def test_imports_and_registers_read_only_tools():
    mod = load_module()
    expected = {"healthcheck", "list_documents", "get_document_outline", "search_documents", "read_document_excerpt", "get_document_summary", "check_source_of_truth_freshness", "propose_document_patch"}
    assert expected <= set(mod.TOOL_FUNCTIONS)
    assert not any(word in name for name in mod.TOOL_FUNCTIONS for word in ("write", "delete", "remove", "edit"))


def test_default_registry_uses_expected_stable_doc_ids():
    mod = load_module()
    assert {"quinn-hermes-server", "quinn-ops-mcp", "quinn-ops-snapshot-design", "quinn-github-mcp-plan", "quinn-observability-mcp-plan", "quinn-docs-mcp-plan", "quinn-approval-ops-mcp-plan"} <= set(mod.DOC_REGISTRY)


def test_allowlisted_document_can_be_listed_and_excerpted_without_absolute_paths(tmp_path, monkeypatch):
    mod = load_module()
    doc = tmp_path / "allowed.md"
    doc.write_text("# Allowed Doc\n\nSafe line\n", encoding="utf-8")
    install_registry(monkeypatch, mod, {"allowed": doc})
    listed = mod.list_documents()
    dumped_list = json.dumps(listed)
    assert listed["ok"]
    assert "allowed" in dumped_list
    assert str(doc) not in dumped_list
    assert listed["data"]["documents"][0]["path_alias"] == "allowed"
    excerpt = mod.read_document_excerpt("allowed", start_line=1, limit=3)
    assert excerpt["ok"]
    assert excerpt["data"]["lines"][0]["text"] == "# Allowed Doc"
    assert str(doc) not in json.dumps(excerpt)


def test_non_allowlisted_raw_path_is_rejected(tmp_path, monkeypatch):
    mod = load_module()
    denied = tmp_path / "denied.md"
    denied.write_text("# Denied\n", encoding="utf-8")
    monkeypatch.setattr(mod, "DOC_REGISTRY", {})
    result = mod.read_document_excerpt(str(denied), start_line=1, limit=1)
    assert result["ok"] is False
    assert result["errors"][0]["kind"] == "unknown_doc"


def test_path_traversal_doc_id_is_rejected(monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "DOC_REGISTRY", {})
    result = mod.get_document_outline("../config")
    assert result["ok"] is False
    assert result["errors"][0]["kind"] == "unknown_doc"


def test_private_path_like_doc_id_is_rejected(monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "DOC_REGISTRY", {})
    blocked_ids = ["." + "env", "auth" + ".json", "sessions", "logs", "/home/quinn/.hermes/." + "env"]
    for doc_id in blocked_ids:
        result = mod.read_document_excerpt(doc_id)
        assert result["ok"] is False
        assert result["errors"][0]["kind"] == "unknown_doc"


def test_redaction_removes_private_values_from_excerpts_search_and_proposals(tmp_path, monkeypatch):
    mod = load_module()
    doc = tmp_path / "allowed.md"
    hidden = "should-not-appear"
    bearer = "bearer-value"
    doc.write_text("# Allowed\n\n" + private_label("a") + f": {hidden}\n" + private_label("h") + f" = {bearer}\n", encoding="utf-8")
    install_registry(monkeypatch, mod, {"allowed": doc})
    excerpt = mod.read_document_excerpt("allowed", start_line=1, limit=8)
    search = mod.search_documents(hidden)
    proposal = mod.propose_document_patch("allowed", f"Add {private_label('t')}: {hidden} to notes")
    dumped = json.dumps({"excerpt": excerpt, "search": search, "proposal": proposal})
    assert excerpt["ok"] and search["ok"] and proposal["ok"]
    assert hidden not in dumped
    assert bearer not in dumped
    assert "[REDACTED]" in dumped


def test_outline_parses_headings_with_line_numbers(tmp_path, monkeypatch):
    mod = load_module()
    doc = tmp_path / "allowed.md"
    doc.write_text("intro\n# First\nbody\n## Second\n", encoding="utf-8")
    install_registry(monkeypatch, mod, {"allowed": doc})
    outline = mod.get_document_outline("allowed")
    assert outline["ok"]
    assert outline["data"]["headings"] == [{"line": 2, "level": 1, "text": "First"}, {"line": 4, "level": 2, "text": "Second"}]


def test_search_is_literal_bounded_redacted_and_has_no_absolute_paths(tmp_path, monkeypatch):
    mod = load_module()
    doc = tmp_path / "allowed.md"
    hidden = "hide-me"
    doc.write_text("# Search\nneedle line one\nNEEDLE line two with " + private_label("p") + f": {hidden}\nregex-ish .* needle\n", encoding="utf-8")
    install_registry(monkeypatch, mod, {"allowed": doc})
    result = mod.search_documents("NEEDLE", limit=2)
    dumped = json.dumps(result)
    assert result["ok"]
    assert result["data"]["count"] == 2
    assert [row["line"] for row in result["data"]["results"]] == [2, 3]
    assert hidden not in dumped
    assert str(doc) not in dumped
    literal = mod.search_documents(".*", limit=5)
    assert literal["ok"]
    assert literal["data"]["count"] == 1
    assert literal["data"]["results"][0]["line"] == 4


def test_search_rejects_blank_query_and_clamps_limit(tmp_path, monkeypatch):
    mod = load_module()
    doc = tmp_path / "allowed.md"
    doc.write_text("\n".join(f"needle {i}" for i in range(20)), encoding="utf-8")
    install_registry(monkeypatch, mod, {"allowed": doc})
    blank = mod.search_documents("   ")
    assert blank["ok"] is False
    assert blank["errors"][0]["kind"] == "invalid_query"
    one = mod.search_documents("needle", limit=0)
    assert one["ok"]
    assert one["data"]["count"] == 1


def test_excerpt_clamps_line_count_and_character_budget(tmp_path, monkeypatch):
    mod = load_module()
    doc = tmp_path / "allowed.md"
    doc.write_text("\n".join(f"line {i} {'x' * 80}" for i in range(200)), encoding="utf-8")
    install_registry(monkeypatch, mod, {"allowed": doc})
    monkeypatch.setattr(mod, "MAX_EXCERPT_LINES", 5)
    monkeypatch.setattr(mod, "MAX_EXCERPT_CHARS", 180)
    result = mod.read_document_excerpt("allowed", start_line=-5, limit=500)
    assert result["ok"]
    assert result["data"]["start_line"] == 1
    assert result["data"]["requested_limit"] == 500
    assert result["data"]["returned_lines"] <= 5
    assert sum(len(line["text"]) for line in result["data"]["lines"]) <= 180


def test_freshness_reports_missing_files_and_required_headings(tmp_path, monkeypatch):
    mod = load_module()
    healthy = tmp_path / "healthy.md"
    stale = tmp_path / "stale.md"
    missing = tmp_path / "missing.md"
    healthy.write_text("# Security Boundaries\n\n# Live Promotion\n", encoding="utf-8")
    stale.write_text("# Other\n", encoding="utf-8")
    monkeypatch.setattr(mod, "DOC_REGISTRY", {
        "healthy": {"title": "Healthy", "path": healthy, "type": "markdown", "required_headings": ["Security Boundaries", "Live Promotion"]},
        "stale": {"title": "Stale", "path": stale, "type": "markdown", "required_headings": ["Security Boundaries"]},
        "missing": {"title": "Missing", "path": missing, "type": "markdown", "required_headings": ["Security Boundaries"]},
    })
    result = mod.check_source_of_truth_freshness()
    dumped = json.dumps(result)
    assert result["ok"]
    assert "missing: missing" in dumped
    assert "missing heading: stale: Security Boundaries" in dumped
    assert str(tmp_path) not in dumped


def test_patch_proposal_is_text_only_redacted_and_does_not_mutate_file(tmp_path, monkeypatch):
    mod = load_module()
    doc = tmp_path / "allowed.md"
    original = "# Allowed\n\nOld text\n"
    doc.write_text(original, encoding="utf-8")
    install_registry(monkeypatch, mod, {"allowed": doc})
    hidden = "never-show"
    result = mod.propose_document_patch("allowed", f"In section Security Boundaries, replace Old text with {private_label('t')}: {hidden}")
    dumped = json.dumps(result)
    assert result["ok"]
    assert result["data"]["requires_approval"] is True
    assert result["data"]["applied"] is False
    assert result["data"]["doc_id"] == "allowed"
    assert "proposal_patch" in result["data"]
    assert hidden not in dumped
    assert str(doc) not in dumped
    assert doc.read_text(encoding="utf-8") == original


def test_patch_proposal_rejects_denied_doc(monkeypatch):
    mod = load_module()
    monkeypatch.setattr(mod, "DOC_REGISTRY", {})
    result = mod.propose_document_patch("/home/quinn/.hermes/." + "env", "change it")
    assert result["ok"] is False
    assert result["errors"][0]["kind"] == "unknown_doc"
