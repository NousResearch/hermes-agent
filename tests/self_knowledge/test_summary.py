from pathlib import Path

from hermes_cli.self_knowledge.summary import build_slim_summary


def test_build_slim_summary_extracts_identity_principles_and_capabilities(tmp_path):
    doc = tmp_path / "self.md"
    doc.write_text(
        "# Hermes Agent Self-Knowledge\n\n"
        "## Identity\n\nHermes is an agent.\n\n"
        "## Core Principles\n\n- Be accurate\n- Use tools\n\n"
        "## Capabilities at a Glance\n\n"
        "<!-- AUTO-START: capabilities -->\n"
        "| Tool | Toolset | Description |\n"
        "|---|---|---|\n"
        "| web_search | web | Search the web |\n"
        "| terminal | terminal | Run commands |\n"
        "<!-- AUTO-END: capabilities -->\n\n"
        "## Open Questions / Unknowns\n\nNone.\n"
    )

    summary = build_slim_summary(doc, max_capabilities=1)

    assert "Hermes self-knowledge" in summary
    assert "Hermes is an agent." in summary
    assert "- Be accurate" in summary
    assert "Capabilities: web_search" in summary
    assert "terminal" not in summary


def test_build_slim_summary_missing_doc_returns_empty(tmp_path):
    assert build_slim_summary(tmp_path / "missing.md") == ""


def test_build_slim_summary_can_be_disabled(monkeypatch, tmp_path):
    doc = tmp_path / "self.md"
    doc.write_text("## Identity\n\nHello\n")
    monkeypatch.setenv("HERMES_SELF_KNOWLEDGE_PROMPT", "off")

    assert build_slim_summary(doc) == ""
