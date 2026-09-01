# content_engine/tests/test_kb_retrieve.py
import kb_retrieve as kb


def test_returns_snippets_from_brain(tmp_path, monkeypatch):
    brain = tmp_path / "brain"; brain.mkdir()
    (brain / "routing.md").write_text("# Model routing\nI prefer cheap-first with a fallback chain.")
    (brain / "other.md").write_text("unrelated note about football")
    monkeypatch.setattr(kb, "BRAIN_DIR", brain)
    out = kb.retrieve("model routing fallback", limit=3)
    assert any("cheap-first" in s for s in out)

def test_graceful_when_sources_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(kb, "BRAIN_DIR", tmp_path / "nope")
    monkeypatch.setattr(kb, "WIKI_DIR", tmp_path / "nope_wiki")
    assert kb.retrieve("anything") == []


def test_wiki_root_defaults_to_canonical_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("WIKI_PATH", raising=False)

    assert kb._wiki_dir() == tmp_path / "docs" / "wiki"


def test_wiki_path_override_wins(monkeypatch, tmp_path):
    override = tmp_path / "override-wiki"
    monkeypatch.setenv("WIKI_PATH", str(override))

    assert kb._wiki_dir() == override
