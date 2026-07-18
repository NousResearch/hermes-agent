"""Integration test: streaming reasoning honors the /reasoning clamp.

Drives the real HermesCLI._stream_reasoning_delta / _close_reasoning_box with
_cprint captured, proving the streaming path now clamps to the shared
threshold (issue #53529) instead of flooding every line to the terminal.
"""

import pytest


def _make_cli():
    import cli
    obj = object.__new__(cli.HermesCLI)
    obj.reasoning_full = False
    # Shadow the width helper so the box borders render without a real terminal.
    obj._scrollback_box_width = lambda *a, **k: 60
    return obj


def _capture(monkeypatch):
    import cli
    printed = []
    monkeypatch.setattr(cli, "_cprint", lambda *a, **k: printed.append(a[0] if a else ""))
    return printed


def test_streaming_reasoning_is_clamped(monkeypatch):
    printed = _capture(monkeypatch)
    obj = _make_cli()

    obj._stream_reasoning_delta("".join(f"line{i}\n" for i in range(25)))
    obj._close_reasoning_box()

    text = "\n".join(printed)
    assert "line9" in text          # 10th line still shown
    assert "line10" not in text     # 11th line clamped away
    assert "15 more lines" in text  # 25 - 10 hidden
    assert "/reasoning full" in text


def test_streaming_reasoning_full_shows_everything(monkeypatch):
    printed = _capture(monkeypatch)
    obj = _make_cli()
    obj.reasoning_full = True

    obj._stream_reasoning_delta("".join(f"line{i}\n" for i in range(25)))
    obj._close_reasoning_box()

    text = "\n".join(printed)
    assert "line24" in text         # nothing clamped
    assert "more lines" not in text  # no footer


def test_streaming_reasoning_short_block_no_footer(monkeypatch):
    printed = _capture(monkeypatch)
    obj = _make_cli()

    obj._stream_reasoning_delta("".join(f"line{i}\n" for i in range(4)))
    obj._close_reasoning_box()

    text = "\n".join(printed)
    assert "line3" in text
    assert "more lines" not in text
