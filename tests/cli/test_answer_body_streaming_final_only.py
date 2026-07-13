from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.fixture
def cli_stub(monkeypatch):
    from cli import HermesCLI
    import cli as cli_mod

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = False
    cli.final_response_markdown = "raw"
    cli.show_timestamps = False
    cli.assistant_body_streaming = False
    cli._reset_stream_state()

    emitted: list[str] = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda text: emitted.append(text))
    monkeypatch.setattr(cli_mod, "_terminal_width_for_streaming", lambda: 74)
    return cli, emitted


def test_answer_body_stream_deltas_are_buffered_not_rendered_by_default(cli_stub) -> None:
    cli, emitted = cli_stub

    visible = cli._stream_delta("draft body that must wait for canonical final\n")
    cli._flush_stream()

    plain = _strip_ansi("\n".join(emitted))
    assert visible is False
    assert "draft body that must wait" not in plain
    assert cli._stream_box_opened is False


def test_final_only_streaming_still_surfaces_reasoning_when_enabled(monkeypatch) -> None:
    from cli import HermesCLI
    import cli as cli_mod

    cli = HermesCLI.__new__(HermesCLI)
    cli.show_reasoning = True
    cli.final_response_markdown = "raw"
    cli.show_timestamps = False
    cli.assistant_body_streaming = False
    cli._reset_stream_state()

    emitted: list[str] = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda text: emitted.append(text))
    monkeypatch.setattr(cli_mod, "_terminal_width_for_streaming", lambda: 74)

    visible = cli._stream_delta("<think>reasoning stays live</think>\nfinal body waits\n")
    cli._flush_stream()

    plain = _strip_ansi("\n".join(emitted))
    assert visible is False
    assert "reasoning stays live" in plain
    assert "final body waits" not in plain


def test_legacy_visible_stream_reports_every_visible_delta(cli_stub) -> None:
    cli, _emitted = cli_stub
    cli.assistant_body_streaming = True

    first_visible = cli._stream_delta("first ")
    second_visible = cli._stream_delta("second\n")

    assert first_visible is True
    assert second_visible is True
