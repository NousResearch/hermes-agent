import json
import subprocess

import pytest

from gateway.stock_research_router import (
    StockResearchRouterConfig,
    build_injected_text,
    classify_mode,
    config_from_extra,
    extract_topic,
    maybe_build_injected_text,
    route_for_text,
    should_route,
)


def test_classifies_deep_research_phrases():
    assert classify_mode("deep dive NVDA") == "deep"
    assert classify_mode("write a full memo on 2454.TW") == "deep"
    assert classify_mode("institutional equity report for $ASML") == "deep"
    assert classify_mode("research NVDA") == "standard"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("deep dive $NVDA", "NVDA"),
        ("research 2454.TW", "2454.TW"),
        ("can you look at AIXA.DE", "AIXA.DE"),
        ("research on hpe", "HPE"),
        ("what do we think?", None),
    ],
)
def test_extract_topic_conservatively(text, expected):
    assert extract_topic(text) == expected


def test_should_route_only_when_enabled_and_channel_matches():
    cfg = StockResearchRouterConfig(enabled=True, channels=frozenset({"stock", "parent"}))
    assert should_route(cfg, ["thread", "parent"])
    assert should_route(cfg, ["stock", None])
    assert not should_route(cfg, ["other", None])
    assert not should_route(cfg, ["stock"], is_dm=True)
    assert not should_route(StockResearchRouterConfig(enabled=False, channels=frozenset({"stock"})), ["stock"])


def test_config_from_nested_extra(monkeypatch):
    for key in list(__import__('os').environ):
        if key.startswith('DISCORD_STOCK_RESEARCH_'):
            monkeypatch.delenv(key, raising=False)
    cfg = config_from_extra({
        "stock_research_router": {
            "enabled": True,
            "channels": ["1504156141293404271"],
            "workdir": "/tmp/rt",
            "timeout_seconds": 12,
        }
    })
    assert cfg.enabled is True
    assert cfg.channels == frozenset({"1504156141293404271"})
    assert cfg.workdir == "/tmp/rt"
    assert cfg.timeout_seconds == 12


def test_route_for_text_builds_rt_command():
    route = route_for_text("please deep dive $NVDA")
    assert route is not None
    assert route.mode == "deep"
    assert route.command_alias == "research-deep"
    assert route.command == ("npm", "--silent", "run", "rt", "--", "research-deep", "NVDA")


def test_maybe_build_injected_text_uses_mocked_packet(monkeypatch, tmp_path):
    for key in list(__import__('os').environ):
        if key.startswith('DISCORD_STOCK_RESEARCH_'):
            monkeypatch.delenv(key, raising=False)
    packet = {
        "kind": "athena-research-packet",
        "version": 2,
        "mode": "deep",
        "topic": "NVDA",
        "packet_path": "/tmp/packet.json",
        "prompt_files": [".claude/commands/r-research.md", "docs/prompts/equity-deep-dive-v3.md"],
        "freshness_guard": {"status_command": "npm --silent run rt -- reliability"},
        "preflight_commands": ["npm --silent run rt -- reliability"],
        "routing": {"deterministic_gateway_router_enabled": False},
        "execution": {"packet_only": True},
        "safety": {"no_trading_or_account_actions": True},
    }

    def fake_run(command, cwd, text, capture_output, timeout, check):
        assert command == ["npm", "--silent", "run", "rt", "--", "research-deep", "NVDA"]
        assert cwd == str(tmp_path)
        assert text is True
        assert capture_output is True
        assert check is False
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(packet), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    injected = maybe_build_injected_text(
        "deep dive $NVDA",
        channel_ids=["1504156141293404271"],
        extra={
            "stock_research_router": {
                "enabled": True,
                "channels": ["1504156141293404271"],
                "workdir": str(tmp_path),
            }
        },
    )
    assert injected is not None
    assert "Deterministic stock-research packet" in injected
    assert "research-deep NVDA" in injected
    assert "no report generation, trades, transfers, or account actions" in injected
    assert "athena-research-packet" in injected


def test_build_injected_text_keeps_original_request():
    route = route_for_text("research $TER")
    packet = {"kind": "athena-research-packet", "version": 2, "topic": "TER", "safety": {"no_trading_or_account_actions": True}}
    text = build_injected_text("research $TER", route, packet)
    assert "Original user request:\nresearch $TER" in text
    assert "Packet JSON:" in text
