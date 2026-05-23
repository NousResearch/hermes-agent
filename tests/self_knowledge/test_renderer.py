from pathlib import Path

from hermes_cli.self_knowledge.renderer import (
    render_self_knowledge,
    refresh_self_knowledge,
)


def test_renderer_preserves_handwritten_sections(tmp_path, monkeypatch):
    doc = tmp_path / "self.md"
    doc.write_text(
        "Before\n<!-- AUTO-START: known -->\nold\n<!-- AUTO-END: known -->\nAfter\n"
    )
    monkeypatch.setattr(
        "hermes_cli.self_knowledge.renderer.GENERATORS",
        {"known": lambda: "new"},
    )

    rendered = render_self_knowledge(doc)

    assert rendered == "Before\n<!-- AUTO-START: known -->\nnew\n<!-- AUTO-END: known -->\nAfter\n"


def test_renderer_unknown_block_gets_unavailable_placeholder(tmp_path, monkeypatch):
    doc = tmp_path / "self.md"
    doc.write_text("<!-- AUTO-START: missing -->\nold\n<!-- AUTO-END: missing -->\n")
    monkeypatch.setattr("hermes_cli.self_knowledge.renderer.GENERATORS", {})

    rendered = render_self_knowledge(doc)

    assert "_unavailable: no generator registered for `missing`_" in rendered


def test_refresh_self_knowledge_writes_only_when_changed(tmp_path, monkeypatch):
    doc = tmp_path / "self.md"
    doc.write_text("<!-- AUTO-START: known -->\nold\n<!-- AUTO-END: known -->\n")
    monkeypatch.setattr(
        "hermes_cli.self_knowledge.renderer.GENERATORS",
        {"known": lambda: "new"},
    )

    assert refresh_self_knowledge(doc) is True
    assert "new" in doc.read_text()
    assert refresh_self_knowledge(doc) is False


def test_render_self_knowledge_accepts_path_objects(tmp_path, monkeypatch):
    doc = Path(tmp_path / "self.md")
    doc.write_text("<!-- AUTO-START: known -->\nold\n<!-- AUTO-END: known -->\n")
    monkeypatch.setattr(
        "hermes_cli.self_knowledge.renderer.GENERATORS",
        {"known": lambda: "new"},
    )

    assert "new" in render_self_knowledge(doc)
