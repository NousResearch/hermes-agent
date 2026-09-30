"""Tests for the security-records skill and the mithril-fund MCP server."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "skills" / "security" / "security-records" / "scripts" / "records.py"
SERVER = ROOT / "mcp" / "mithril-fund" / "server.py"


@pytest.fixture(scope="module")
def records():
    spec = importlib.util.spec_from_file_location("security_records", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cmdb_reports_duplicates_unknowns_isolates_and_cycles(records) -> None:
    result = records.check_cmdb(
        {
            "assets": [
                {"id": "a"},
                {"id": "a"},
                {"id": "b"},
                {"id": "alone"},
            ],
            "dependencies": [
                {"from": "a", "to": "b"},
                {"from": "b", "to": "a"},
                {"from": "a", "to": "b"},
                {"from": "a", "to": "missing"},
            ],
        }
    )
    assert result["duplicateAssets"] == ["a"]
    assert result["duplicateDependencies"] == [{"from": "a", "to": "b"}]
    assert result["unknownReferences"] == [{"from": "a", "to": "missing"}]
    assert result["isolated"] == ["alone"]
    assert result["cycles"] == [["a", "b"]]


def test_cmdb_rejects_an_empty_asset_list(records) -> None:
    with pytest.raises(ValueError, match="invalid asset list"):
        records.check_cmdb({"assets": [], "dependencies": []})


def test_email_signals_distinguish_ppap_and_double_extensions(records) -> None:
    result = records.email_signals(
        {
            "messages": [
                {
                    "id": "ppap",
                    "bodyMentionsPassword": True,
                    "attachments": [{"name": "secret.zip", "encrypted": True}],
                },
                {
                    "id": "decoy",
                    "attachments": [{"name": "invoice.pdf.exe"}],
                },
                {"id": "plain", "attachments": [{"name": "notes.txt"}]},
            ]
        }
    )
    by_id = {message["id"]: message["signals"] for message in result["messages"]}
    assert "ppap-pattern" in by_id["ppap"]
    assert "password-protected-archive" in by_id["ppap"]
    assert "double-extension" in by_id["decoy"]
    assert by_id["plain"] == []
    assert result["flagged"] == 2


def test_csf_fit_is_committed_and_other_frameworks_are_not(records) -> None:
    csf = records.compliance_fit("nist-csf-20")
    assert csf["framework"]["controlMapping"] == "committed"
    assert "ID.RA-01" in next(item["controls"] for item in csf["services"] if item["service"] == "ctem")
    other = records.compliance_fit("iso-27001")
    assert other["framework"]["controlMapping"] == "not-committed"
    assert other["services"][0]["controls"] is None
    with pytest.raises(ValueError, match="unknown framework"):
        records.compliance_fit("not-a-framework")


def test_catalog_and_spear_phishing_are_description_only(records) -> None:
    catalog = records.service_catalog()
    assert {item["id"] for item in catalog["services"]} == {
        "ctem",
        "dast",
        "sast",
        "spear-phishing",
        "vm",
        "grc",
        "ir",
        "dr",
    }
    assert "does not scan" in catalog["boundary"]
    defence = records.spear_phishing_defence()
    assert "private authority" in defence["boundary"]
    assert defence["controls"]


def _frame(payload: dict) -> bytes:
    raw = json.dumps(payload).encode()
    return f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw


def _read_frame(stream) -> dict:
    headers = {}
    while True:
        line = stream.readline()
        if line in (b"\r\n", b"\n"):
            break
        key, value = line.decode().split(":", 1)
        headers[key.strip().lower()] = value.strip()
    length = int(headers["content-length"])
    return json.loads(stream.read(length))


def test_mcp_lists_tools_and_checks_cmdb() -> None:
    proc = subprocess.Popen(
        [sys.executable, str(SERVER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert proc.stdin and proc.stdout
    proc.stdin.write(_frame({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}))
    proc.stdin.write(_frame({"jsonrpc": "2.0", "method": "notifications/initialized"}))
    proc.stdin.write(_frame({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
    proc.stdin.write(
        _frame(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "cmdb_check",
                    "arguments": {"assets": [{"id": "a"}], "dependencies": []},
                },
            }
        )
    )
    proc.stdin.close()
    initialized = _read_frame(proc.stdout)
    listed = _read_frame(proc.stdout)
    called = _read_frame(proc.stdout)
    proc.wait(timeout=10)
    assert initialized["result"]["serverInfo"]["name"] == "mithril-fund"
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert {"cmdb_check", "domain_check", "sandbox_status", "billing_status"} <= names
    body = json.loads(called["result"]["content"][0]["text"])
    assert body["isolated"] == ["a"]
    assert called["result"]["isError"] is False


def test_mcp_refuses_an_unlisted_api_origin() -> None:
    proc = subprocess.Popen(
        [sys.executable, str(SERVER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "MITHRIL_API_ORIGIN": "https://evil.example"},
    )
    assert proc.stdin and proc.stdout
    proc.stdin.write(
        _frame(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "domain_check", "arguments": {"domain": "example.com"}},
            }
        )
    )
    proc.stdin.close()
    response = _read_frame(proc.stdout)
    proc.wait(timeout=10)
    assert response["result"]["isError"] is True
    assert "not an allowed" in response["result"]["content"][0]["text"]
