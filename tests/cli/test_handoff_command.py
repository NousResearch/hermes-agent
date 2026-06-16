from __future__ import annotations

from pathlib import Path

from hermes_cli.cli_commands_mixin import CLICommandsMixin


class _FakeCLI(CLICommandsMixin):
    pass


def _fake_cli(tmp_path):
    output: list[str] = []
    cli = _FakeCLI()
    cli.conversation_history = [
        {"role": "user", "content": "Please create a handoff for reviewing /srv/app."},
        {"role": "assistant", "content": "I inspected /srv/app/config.yaml."},
    ]
    cli.session_id = "sess-cli-1"
    cli._pending_agent_seed = None
    cli._console_print = output.append
    cli._session_db = None
    cli._agent_running = False
    cli._should_exit = False
    cli._output = output
    return cli


def test_handoff_inline_document_mode_prints_markdown(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    cli = _fake_cli(tmp_path)

    keep_running = CLICommandsMixin._handle_handoff_command(
        cli,
        "/handoff inline review the repo",
    )

    assert keep_running is True
    rendered = "\n".join(cli._output)
    assert "# Handoff:" in rendered
    assert "Suggested filename:" in rendered
    assert cli._pending_agent_seed is None


def test_handoff_consume_document_mode_sets_pending_seed(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    handoff = tmp_path / "handoff.md"
    handoff.write_text(
        "# Handoff: repo review\n\n"
        "## Purpose of next session\nReview the repo.\n\n"
        "## Current status\n- scope defined\n\n"
        "## Relevant artifacts\n- workdir: /srv/app\n\n"
        "## Constraints and non-goals\n- stay narrow\n\n"
        "## Exact first prompt\nRead the key files and start the review.\n\n"
        "## Success criteria\n- [ ] review produced\n",
        encoding="utf-8",
    )
    cli = _fake_cli(tmp_path)

    keep_running = CLICommandsMixin._handle_handoff_command(
        cli,
        f"/handoff consume {handoff}",
    )

    assert keep_running is True
    assert cli._pending_agent_seed is not None
    assert str(handoff) in cli._pending_agent_seed
    rendered = "\n".join(cli._output)
    assert "Loaded handoff from:" in rendered
