"""Tests for optional-skills/hardware/black-wire-reference/scripts/bwm.py.

Offline: the catalog is a local fixture; the one URL-loading test mocks urlopen.
"""
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SKILL_SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "hardware"
    / "black-wire-reference"
    / "scripts"
)
sys.path.insert(0, str(SKILL_SCRIPTS))

import bwm  # noqa: E402


def _asset(**overrides):
    asset = {
        "id": "PDF-NEW-0002-P05",
        "label": "Pinout reference - PDF page 5",
        "type": "pinout image",
        "hash": "a" * 64,
        "file": "media/" + "a" * 64 + ".png",
        "url": "https://example.com/board.pdf",
        "rights": "Not established; source attribution retained",
        "review": "Reviewed source",
        "revision": "Not identified",
    }
    asset.update(overrides)
    return asset


def _board(**overrides):
    board = {
        "id": "rakwireless-rak13002",
        "name": "RAK13002",
        "brand": "RAKwireless",
        "family": "WisBlock",
        "processor": "RAK4631",
        "aliases": ["RAK-13002"],
        "revision": "Not identified",
        "coverage": "Pinout image collected",
        "partial": False,
        "sources": ["https://example.com/rak13002"],
        "assets": [_asset()],
        "scope": "curated",
        "preview": "",
        "pinouts": 1,
        "searchText": "",
        "powerIds": [],
        "pinoutCoverage": {
            "status": "pinout-source",
            "physicalCount": 1,
            "functionTableCount": 0,
            "functionSheetCount": 0,
            "purposeRows": 0,
            "completeDeviceApproved": False,
            "scope": "Source pinout available; check its connector scope and revision.",
            "missing": ["Independent all-connector completeness review"],
        },
        "pinConnectors": {
            "method": "image-transcription",
            "review": "ocr-checked",
            "connectors": [
                {
                    "id": "P1",
                    "name": "Main header",
                    "rows": 1,
                    "pins": [
                        {"position": 1, "label": "GND", "functions": []},
                        {"position": 2, "label": "GPIO04", "functions": ["4"]},
                    ],
                }
            ],
        },
    }
    board.update(overrides)
    if not board.get("searchText"):
        board["searchText"] = " ".join(
            [
                board.get("name", ""),
                board.get("brand", ""),
                board.get("family", ""),
                board.get("processor", ""),
            ]
        )
    return board


def _catalog():
    return {
        "version": 1,
        "edition": "First Edition · 2026 · Collection 2026.11.1",
        "generatedAt": "2026-11-08T00:58:53Z",
        "boards": [
            _board(),
            _board(
                id="rakwireless-rak13003",
                name="RAK13003 WisBlock IO Module",
                aliases=[],
                assets=[],
                pinouts=0,
                pinConnectors=None,
                pinoutCoverage={"status": "missing", "completeDeviceApproved": False},
            ),
            _board(
                id="espressif-esp32-s3-devkitc-1",
                name="ESP32-S3-DevKitC-1",
                brand="Espressif",
                family="ESP32-S3",
                processor="ESP32-S3",
                aliases=[],
                assets=[],
                pinouts=0,
                pinConnectors=None,
                pinoutCoverage={"status": "reference-only"},
            ),
        ],
        "power": [],
        "makerParts": [],
        "wiringGuides": [
            {
                "id": "rs485-bus",
                "name": "RS-485 bus",
                "category": "Buses",
                "aliases": ["RS485", "RS-485"],
                "summary": "Differential half-duplex bus wiring.",
                "scope": "Not a termination design guide.",
                "columns": ["Signal", "Direction"],
                "rows": [["A", "non-inverting"], ["B", "inverting"]],
                "steps": ["Wire A to A and B to B."],
                "checks": ["Measure continuity."],
                "sources": [{"title": "Example", "url": "https://example.com/rs485"}],
                "review": "Source documents inspected; independent electrical review pending",
            }
        ],
        "stats": {"trackedBoards": 3},
        "boardIdAliases": {},
    }


def test_exact_model_match_beats_numeric_sibling():
    catalog = _catalog()
    hits = bwm.search_boards(catalog, "RAK-13002")
    assert [board["id"] for _, board in hits] == ["rakwireless-rak13002"]
    # the printed code and the dashed alias land on the same record
    assert bwm.search_boards(catalog, "RAK13002")[0][1]["id"] == "rakwireless-rak13002"


def test_exact_flag_keeps_identifier_matches_only():
    catalog = _catalog()
    assert bwm.search_boards(catalog, "RAK", exact=True) == []
    assert [
        board["id"] for _, board in bwm.search_boards(catalog, "RAK13002", exact=True)
    ] == ["rakwireless-rak13002"]


def test_exact_matches_a_model_token_inside_a_longer_name():
    catalog = _catalog()
    assert [
        board["id"] for _, board in bwm.search_boards(catalog, "RAK13003", exact=True)
    ] == ["rakwireless-rak13003"]


def test_status_labels_use_the_catalog_vocabulary():
    catalog = _catalog()
    by_id = {board["id"]: board for board in catalog["boards"]}
    assert bwm.status_label(by_id["rakwireless-rak13002"]) == "pinout source"
    assert bwm.status_label(by_id["rakwireless-rak13003"]) == "documentation needed"
    assert bwm.status_label(by_id["espressif-esp32-s3-devkitc-1"]) == "pinout needed"


def test_resolve_prints_raw_and_deep_links(capsys):
    catalog = _catalog()
    args = bwm.build_parser().parse_args(["resolve", "rakwireless-rak13002"])
    assert bwm.cmd_resolve(args, catalog) == 0
    out = capsys.readouterr().out
    assert "https://valleytech-black-wire-guide.pages.dev/?board=rakwireless-rak13002" in out
    assert (bwm.COLLECTION_RAW_BASE + "media/" + "a" * 64 + ".png") in out


def test_resolve_adds_board_page_from_paths_file(tmp_path, capsys):
    catalog = _catalog()
    paths = tmp_path / "board-paths.json"
    paths.write_text(
        json.dumps(
            {"rakwireless-rak13002": "boards/rakwireless/wisblock/rak13002--abc/README.md"}
        )
    )
    args = bwm.build_parser().parse_args(
        ["--board-paths", str(paths), "resolve", "rakwireless-rak13002"]
    )
    assert bwm.cmd_resolve(args, catalog) == 0
    out = capsys.readouterr().out
    assert "boards/rakwireless/wisblock/rak13002--abc/README.md" in out
    assert (
        "https://raw.githubusercontent.com/valleytechsolutions/black-wire-pinouts/main/"
        "boards/rakwireless/wisblock/rak13002--abc/README.md" in out
    )


def test_pins_prints_transcribed_connector_rows(capsys):
    catalog = _catalog()
    args = bwm.build_parser().parse_args(["pins", "rakwireless-rak13002"])
    assert bwm.cmd_pins(args, catalog) == 0
    out = capsys.readouterr().out
    assert "P1 - Main header" in out
    assert "GPIO04" in out
    assert "not electrical verification" in out


def test_pins_falls_back_to_pinout_assets_without_transcription(capsys):
    catalog = _catalog()
    args = bwm.build_parser().parse_args(["pins", "espressif-esp32-s3-devkitc-1"])
    assert bwm.cmd_pins(args, catalog) == 1
    assert "no transcribed pin connectors" in capsys.readouterr().out


def test_wiring_search_matches_alias_and_prints_rows(capsys):
    catalog = _catalog()
    args = bwm.build_parser().parse_args(["wiring", "RS-485"])
    assert bwm.cmd_wiring(args, catalog) == 0
    out = capsys.readouterr().out
    assert "rs485-bus - RS-485 bus" in out
    assert "non-inverting" in out
    assert "https://example.com/rs485" in out


def test_show_reports_coverage_and_gaps(capsys):
    catalog = _catalog()
    args = bwm.build_parser().parse_args(["show", "rakwireless-rak13002"])
    assert bwm.cmd_show(args, catalog) == 0
    out = capsys.readouterr().out
    assert "id: rakwireless-rak13002" in out
    assert "pinout source" in out
    assert "Independent all-connector completeness review" in out
    assert "complete device approved: False" in out


def test_load_catalog_downloads_once_then_reads_cache(tmp_path):
    payload = json.dumps(_catalog()).encode("utf-8")
    with mock.patch.object(bwm.urllib.request, "urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value.read.return_value = payload
        first = bwm.load_catalog(
            "https://example.com/catalog.json", cache_dir=str(tmp_path)
        )
        second = bwm.load_catalog(
            "https://example.com/catalog.json", cache_dir=str(tmp_path)
        )
    assert first == second == _catalog()
    assert (tmp_path / "bwm-catalog.json").exists()
    assert urlopen.call_count == 1


def test_load_catalog_reads_a_local_path(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(_catalog()))
    assert bwm.load_catalog(str(path))["edition"].startswith("First Edition")
