"""Tests for optional-skills/devops/plasmalogin-greeter-monitor/scripts/greeter_outputs.py"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "devops"
    / "plasmalogin-greeter-monitor"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import greeter_outputs as go


def _edid(mfr: str, name: str) -> bytes:
    raw = bytearray(128)
    raw[:8] = b"\x00\xff\xff\xff\xff\xff\xff\x00"
    word = 0
    for ch in mfr:
        word = (word << 5) | (ord(ch) - 64)
    raw[8], raw[9] = word >> 8, word & 0xFF
    block = b"\x00\x00\x00\xfc\x00" + (name.encode() + b"\x0a").ljust(13, b" ")
    raw[54:72] = block
    return bytes(raw)


CONFIG = [
    {"name": "outputs", "data": [{"connectorName": "DP-1"}, {"connectorName": "DP-2"}]},
    {
        "name": "setups",
        "data": [
            {"outputs": [
                {"enabled": True, "outputIndex": 0, "priority": 2},
                {"enabled": True, "outputIndex": 1, "priority": 1},
            ]},
            {"outputs": [{"enabled": True, "outputIndex": 1, "priority": 0}]},
        ],
    },
]


def test_parse_edid_reads_manufacturer_and_name():
    assert go.parse_edid(_edid("GSM", "LG ULTRAGEAR")) == {
        "manufacturer": "GSM",
        "name": "LG ULTRAGEAR",
    }


@pytest.mark.parametrize("raw", [b"", b"\x00" * 128, b"junk" * 40])
def test_parse_edid_rejects_invalid(raw):
    assert go.parse_edid(raw) == {}


def test_scan_maps_connectors(tmp_path):
    a = tmp_path / "card1-DP-1"
    b = tmp_path / "card1-HDMI-A-1"
    for d in (a, b):
        d.mkdir()
    (a / "status").write_text("connected\n")
    (a / "edid").write_bytes(_edid("ACI", "TEST MONITOR"))
    (b / "status").write_text("disconnected\n")
    (b / "edid").write_bytes(b"")
    rows = go.scan(tmp_path)
    assert rows == [
        {"connector": "DP-1", "status": "connected", "manufacturer": "ACI", "name": "TEST MONITOR"},
        {"connector": "HDMI-A-1", "status": "disconnected"},
    ]


def test_primaries_follow_output_index():
    assert go.primaries(CONFIG) == [
        {"setup": 0, "connectors": ["DP-1", "DP-2"], "primary": "DP-2"},
        {"setup": 1, "connectors": ["DP-2"], "primary": None},
    ]


def test_primaries_tolerate_bad_index():
    cfg = [
        {"name": "outputs", "data": []},
        {"name": "setups", "data": [{"outputs": [{"outputIndex": 5, "priority": 1}]}]},
    ]
    assert go.primaries(cfg) == [{"setup": 0, "connectors": [None], "primary": None}]


def test_stage_copies_and_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(go.tempfile, "tempdir", str(tmp_path))
    src = tmp_path / "src.json"
    src.write_text(json.dumps(CONFIG))
    path, digest = go.stage(src)
    assert path.read_bytes() == src.read_bytes()
    assert digest == hashlib.sha256(src.read_bytes()).hexdigest()


def test_cli_stage_refuses_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        go.main(["stage", str(bad)])
