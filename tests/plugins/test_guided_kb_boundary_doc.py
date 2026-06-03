from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "plans" / "guided-kb-review-sessions-boundary.md"


def test_guided_kb_boundary_doc_tracks_current_refs_and_extraction_status():
    body = DOC.read_text(encoding="utf-8")

    assert "Current Evidence, 2026-06-03" in body
    assert "03c96b5407e85b098e3576bcf2ed1c22f99b8b8a" in body
    assert "v2026.5.29.2" in body
    assert "51f432685e4bc73379abe70367f196400dd44054" in body
    assert "e223503b0303b6e257f6e264bcb0815dde8528b0" in body
    assert "plugins/kb_journeys/__init__.py" in body
    assert "tests/plugins/test_kb_journeys.py" in body


def test_guided_kb_boundary_doc_keeps_plugin_and_fork_debt_open_until_safe():
    body = DOC.read_text(encoding="utf-8")

    assert "Hermes #35 should stay open" in body
    assert "Hermes #38 should stay open" in body
    assert "User-installed plugins override bundled plugins" in body
    assert "without relying on bundled Hermes fork semantics" in body
    assert "Rename or wrap `tools/kb_callback_registry.py` only after" in body
    assert "NOC installs and enables the out-of-tree plugin on helix" in body
