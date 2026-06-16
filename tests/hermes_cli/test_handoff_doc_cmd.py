from __future__ import annotations

from pathlib import Path

from hermes_cli.handoff_doc_cmd import (
    build_handoff_document,
    consume_handoff_markdown_text,
    handle_handoff_document_command,
    parse_handoff_args,
    parse_handoff_markdown,
)


def _history() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "Investigate auth drift in /srv/app and keep it narrow."},
        {"role": "assistant", "content": "I found the likely cause in /srv/app/config.yaml and https://example.com/docs."},
        {"role": "user", "content": "Create a clean handoff for a fresh session."},
    ]


class TestParseHandoffArgs:
    def test_inline_mode(self):
        mode, path, mission = parse_handoff_args("/handoff inline investigate auth drift")
        assert mode == "inline"
        assert path is None
        assert mission == "investigate auth drift"

    def test_save_mode_with_explicit_path(self):
        mode, path, mission = parse_handoff_args("/handoff save /tmp/handoff.md fix auth drift")
        assert mode == "save"
        assert path == "/tmp/handoff.md"
        assert mission == "fix auth drift"

    def test_save_mode_without_explicit_path(self):
        mode, path, mission = parse_handoff_args("/handoff save fix auth drift")
        assert mode == "save"
        assert path is None
        assert mission == "fix auth drift"

    def test_consume_requires_path(self):
        try:
            parse_handoff_args("/handoff consume")
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "Usage: /handoff consume <path>" in str(exc)


class TestBuildHandoffDocument:
    def test_document_contains_required_sections(self):
        doc = build_handoff_document(
            mission="investigate auth drift",
            conversation_history=_history(),
            session_id="sess-123",
            workdir="/root/project",
        )
        assert "## Purpose of next session" in doc.markdown
        assert "## Current status" in doc.markdown
        assert "## Relevant artifacts" in doc.markdown
        assert "## Constraints and non-goals" in doc.markdown
        assert "## Exact first prompt" in doc.markdown
        assert "## Success criteria" in doc.markdown
        assert doc.next_mode == "fresh session"
        assert doc.suggested_filename.startswith("handoff-")

    def test_parse_generated_markdown_round_trips_required_sections(self):
        doc = build_handoff_document(
            mission="review this repo",
            conversation_history=_history(),
            session_id="sess-234",
            workdir="/root/project",
        )
        parsed = parse_handoff_markdown(doc.markdown)
        assert "review this repo" in parsed["Purpose of next session"]
        assert "fresh session" not in parsed["Purpose of next session"].lower() or parsed["Purpose of next session"]


class TestConsumePastedHandoffMarkdown:
    def test_detects_and_queues_pasted_handoff(self):
        text = (
            "# Handoff: auth drift\n\n"
            "## Purpose of next session\nFix auth drift.\n\n"
            "## Current status\n- drift confirmed\n\n"
            "## Relevant artifacts\n- workdir: /root/project\n\n"
            "## Constraints and non-goals\n- stay narrow\n\n"
            "## Exact first prompt\nValidate the config and fix the drift.\n\n"
            "## Success criteria\n- [ ] config fixed\n"
        )
        result = consume_handoff_markdown_text(text, source_label="<paste>")
        assert result is not None
        assert result.agent_seed is not None
        assert "<paste>" in result.agent_seed
        assert "Detected handoff markdown" in result.text

    def test_ignores_non_handoff_text(self):
        assert consume_handoff_markdown_text("hello there") is None


class TestHandleHandoffDocumentCommand:
    def test_save_defaults_under_hermes_home(self, tmp_path):
        hermes_home = tmp_path / ".hermes"
        result = handle_handoff_document_command(
            cmd="/handoff save investigate auth drift",
            conversation_history=_history(),
            session_id="sess-1",
            workdir="/root/project",
            hermes_home=hermes_home,
        )
        assert result.saved_path is not None
        saved = Path(result.saved_path)
        assert saved.exists()
        assert saved.is_absolute()
        assert saved.parent == hermes_home / "sessions" / "handoffs"

    def test_consume_returns_agent_seed(self, tmp_path):
        handoff = tmp_path / "handoff.md"
        handoff.write_text(
            "# Handoff: auth drift\n\n"
            "## Purpose of next session\nFix auth drift.\n\n"
            "## Current status\n- drift confirmed\n\n"
            "## Relevant artifacts\n- workdir: /root/project\n\n"
            "## Constraints and non-goals\n- stay narrow\n\n"
            "## Exact first prompt\nValidate the config and fix the drift.\n\n"
            "## Success criteria\n- [ ] config fixed\n",
            encoding="utf-8",
        )
        result = handle_handoff_document_command(
            cmd=f"/handoff consume {handoff}",
            conversation_history=_history(),
            session_id="sess-1",
            workdir="/root/project",
            hermes_home=tmp_path / ".hermes",
        )
        assert result.agent_seed is not None
        assert str(handoff) in result.agent_seed
        assert "Validate the config and fix the drift." in result.agent_seed
