"""Tests for ``hermes mcp fixtures record/replay``.

Spawns a REAL MCP stdio server subprocess (tests/hermes_cli/_mcp_toy_server.py)
for both recording and replay-check — the whole point of this feature is to
exercise genuine protocol framing/lifecycle instead of mocking the
transport, matching ``tests/conformance``'s no-mock discipline.
"""

from __future__ import annotations

import json
import sys

import pytest

from hermes_cli.mcp_fixtures import (
    _parse_call_args,
    _record_async,
    _replay_check_async,
    record_fixture,
    write_fixture,
)

TOY_SERVER_CFG = {
    "command": sys.executable,
    "args": ["-m", "tests.hermes_cli._mcp_toy_server"],
    "env": {},
}


def test_parse_call_args_valid():
    calls = _parse_call_args(['echo={"text": "hi"}', "noop="])
    assert calls == [("echo", {"text": "hi"}), ("noop", {})]


def test_parse_call_args_rejects_missing_equals():
    with pytest.raises(ValueError, match="expected TOOL=JSON_ARGS"):
        _parse_call_args(["echo"])


def test_parse_call_args_rejects_invalid_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_call_args(["echo=not-json"])


def test_record_fixture_against_real_toy_server():
    fixture = record_fixture(
        "toy", TOY_SERVER_CFG, [("echo", {"text": "hello"})], timeout=15
    )
    assert fixture["schema_version"] == 1
    assert fixture["server_name"] == "toy"
    assert fixture["initialize"]["server_name"] == "hermes-mcp-toy-server"
    assert {t["name"] for t in fixture["tools"]} == {"echo", "attachment"}
    assert fixture["calls"] == [
        {
            "name": "echo",
            "arguments": {"text": "hello"},
            "content": [{"type": "text", "text": "hello"}],
            "is_error": False,
        }
    ]


def test_record_fixture_captures_tool_side_error_as_is_error():
    # The toy server's echo tool treats {"fail": true} as an intentional
    # failure — MCP servers report tool errors as isError content, not a
    # transport exception, so the fixture should capture that shape.
    fixture = record_fixture(
        "toy", TOY_SERVER_CFG, [("echo", {"fail": True})], timeout=15
    )
    call = fixture["calls"][0]
    assert call["is_error"] is True
    assert "intentional failure" in call["content"][0]["text"]


def test_record_fixture_filters_non_text_content():
    # Non-text content (images, audio, ...) would round-trip as
    # `text: null` through the replay stub, which only ever serves
    # TextContent — so it must be filtered out at record time instead.
    fixture = record_fixture(
        "toy", TOY_SERVER_CFG, [("attachment", {})], timeout=15
    )
    call = fixture["calls"][0]
    assert call["content"] == [{"type": "text", "text": "caption"}]


def test_record_fixture_requires_command():
    with pytest.raises(ValueError, match="has no 'command'"):
        record_fixture("toy", {}, [], timeout=5)


def test_write_fixture_redacts_secret_shaped_fields(tmp_path):
    fixture = {
        "schema_version": 1,
        "server_name": "toy",
        "calls": [
            {
                "name": "echo",
                "arguments": {"api_key": "sk-super-secret-value-12345"},
                "content": [{"type": "text", "text": "ok"}],
            }
        ],
    }
    output = tmp_path / "fixture.json"
    write_fixture(fixture, output)
    written = output.read_text(encoding="utf-8")
    assert "sk-super-secret-value-12345" not in written


@pytest.mark.asyncio
async def test_replay_round_trip_matches_recorded_calls(tmp_path):
    fixture = await _record_async(
        "toy",
        TOY_SERVER_CFG,
        [
            ("echo", {"text": "round trip"}),
            ("echo", {"fail": True}),
            ("attachment", {}),
        ],
        timeout=15,
    )
    fixture_path = tmp_path / "fixture.json"
    write_fixture(fixture, fixture_path)

    report = await _replay_check_async(fixture_path)
    assert report["initialize_ok"] is True
    assert report["tools_ok"] is True
    assert all(call["ok"] for call in report["calls"])


def test_cmd_mcp_fixtures_record_and_replay_cli(tmp_path, monkeypatch, capsys):
    import argparse

    from hermes_cli.mcp_fixtures import cmd_mcp_fixtures

    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    config_path = home / "config.yaml"
    config_path.write_text(
        json.dumps(
            {
                "mcp_servers": {
                    "toy": {
                        "command": sys.executable,
                        "args": ["-m", "tests.hermes_cli._mcp_toy_server"],
                        "env": {},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "fixture.json"
    record_args = argparse.Namespace(
        mcp_fixtures_action="record",
        name="toy",
        output=str(output),
        call=['echo={"text": "via cli"}'],
        timeout=15,
    )
    rc = cmd_mcp_fixtures(record_args)
    assert rc == 0
    assert output.exists()
    capsys.readouterr()

    replay_args = argparse.Namespace(
        mcp_fixtures_action="replay", fixture=str(output)
    )
    rc = cmd_mcp_fixtures(replay_args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "initialize: ok" in out
    assert "list_tools: ok" in out


def test_cmd_mcp_fixtures_record_unknown_server(tmp_path, monkeypatch, capsys):
    import argparse

    from hermes_cli.mcp_fixtures import cmd_mcp_fixtures

    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    args = argparse.Namespace(
        mcp_fixtures_action="record",
        name="does-not-exist",
        output=str(tmp_path / "out.json"),
        call=[],
        timeout=5,
    )
    rc = cmd_mcp_fixtures(args)
    assert rc == 1
    assert "no mcp_servers.does-not-exist" in capsys.readouterr().err


def test_cmd_mcp_fixtures_replay_missing_file(tmp_path, capsys):
    import argparse

    from hermes_cli.mcp_fixtures import cmd_mcp_fixtures

    args = argparse.Namespace(
        mcp_fixtures_action="replay", fixture=str(tmp_path / "nope.json")
    )
    rc = cmd_mcp_fixtures(args)
    assert rc == 1
    assert "not found" in capsys.readouterr().err
