"""Tests for ``hermes gateway simulate-delivery`` (real adapter oracles).

Drives the REAL ``format_message``/``truncate_message`` adapter code —
same construction as ``scripts/generate_conformance_vectors.py`` — so
these tests double as a guard against silently swapping in a
reimplementation.
"""

from __future__ import annotations

import argparse
import json

import pytest

from hermes_cli.gateway_delivery_simulator import (
    SUPPORTED_PLATFORMS,
    cmd_simulate_delivery,
    simulate,
)


def test_supported_platforms_are_stable():
    assert set(SUPPORTED_PLATFORMS) == {"telegram", "discord", "slack", "whatsapp"}


def test_telegram_escapes_markdownv2_reserved_chars():
    sim = simulate("telegram", "Price 3.50 (was 4.00) save ~12%!")
    # Real MarkdownV2 escaping from TelegramAdapter.format_message — the
    # exact bug class reported in #78524 (preview/final divergence).
    assert "\\." in sim.formatted
    assert "\\(" in sim.formatted
    assert "\\~" in sim.formatted
    assert "\\!" in sim.formatted


def test_discord_passes_bold_through_unchanged():
    sim = simulate("discord", "hello **world**")
    assert sim.formatted == "hello **world**"


def test_unsupported_platform_raises_value_error():
    with pytest.raises(ValueError, match="unsupported platform"):
        simulate("irc", "hi")


def test_no_network_no_side_effects_just_pure_computation():
    # Calling simulate() must never require credentials/config — both
    # renderers below construct their adapter class without __init__.
    sim = simulate("whatsapp", "plain text")
    assert sim.platform == "whatsapp"
    assert sim.splits_natively is False


def test_chunking_splits_content_over_the_adapter_limit():
    # Discord's real limit is 2000 chars; build content well past it.
    long_text = "line\n" * 1000  # 5000 chars
    sim = simulate("discord", long_text)
    assert sim.exceeds_limit is True
    assert sim.chunks is not None
    assert len(sim.chunks) > 1
    for chunk in sim.chunks:
        assert len(chunk) <= sim.max_length


def test_telegram_uses_utf16_length_for_the_limit_check():
    # A short string under both code-point and UTF-16 length is fine;
    # this just asserts the length reported matches utf16_len, not len().
    from gateway.platforms.base import utf16_len

    sim = simulate("telegram", "hello")
    formatted_expected_len = utf16_len(sim.formatted)
    assert sim.length == formatted_expected_len


def test_cmd_simulate_delivery_json_output(capsys):
    args = argparse.Namespace(
        platform="slack", input=None, text="hi *there*", json=True
    )
    rc = cmd_simulate_delivery(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["platform"] == "slack"
    assert out["chunks"] == [out["formatted"]]


def test_cmd_simulate_delivery_reads_from_input_file(tmp_path, capsys):
    input_file = tmp_path / "message.md"
    input_file.write_text("hello from file", encoding="utf-8")
    args = argparse.Namespace(
        platform="discord", input=str(input_file), text=None, json=True
    )
    rc = cmd_simulate_delivery(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["formatted"] == "hello from file"


def test_cmd_simulate_delivery_unknown_platform_returns_error(capsys):
    args = argparse.Namespace(platform="irc", input=None, text="hi", json=False)
    rc = cmd_simulate_delivery(args)
    assert rc == 1
    assert "unsupported platform" in capsys.readouterr().err


def test_cmd_simulate_delivery_missing_input_file_returns_error(capsys):
    args = argparse.Namespace(
        platform="telegram", input="/no/such/file.txt", text=None, json=False
    )
    rc = cmd_simulate_delivery(args)
    assert rc == 1
    assert "cannot read" in capsys.readouterr().err
