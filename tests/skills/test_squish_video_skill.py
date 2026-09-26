from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "media"
    / "squish-video"
)
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPT_PATH = SKILL_DIR / "scripts" / "squish_video.py"


def load_module():
    spec = importlib.util.spec_from_file_location("squish_video_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def contract_payload(video: str) -> dict:
    return {
        "input": video,
        "duration": 20.275,
        "frames": 9,
        "sheets": 1,
        "files": ["/abs/path/clip.sheet-1.jpg"],
        "warnings": [],
        "contract": "squish-cli-v0",
    }


# --- SKILL.md authoring standards ---


def test_description_is_one_sentence_max_60_chars():
    m = re.search(r"^description: (.*)$", SKILL_MD.read_text(), re.MULTILINE)
    assert m is not None
    desc = m.group(1).strip().strip('"')
    assert len(desc) <= 60, len(desc)
    assert desc.endswith(".")


def test_body_uses_modern_section_order():
    text = SKILL_MD.read_text()
    headings = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [text.index(h) for h in headings]
    assert positions == sorted(positions)


# --- command construction (whitespace-safe by design) ---


def test_build_command_keeps_whitespace_path_as_single_argument():
    mod = load_module()
    path = "/tmp/My Clips/screen recording.mov"
    cmd = mod.build_command(path)
    assert cmd[:3] == ["npx", "-y", "@getsquish/squish"]
    assert cmd[3] == path
    assert "--json" in cmd


def test_build_command_appends_density_window_and_out():
    mod = load_module()
    cmd = mod.build_command(
        "clip.mov", density="5x5", start="1:00", end="1:30", out="/tmp/out"
    )
    for pair in (
        ["--density", "5x5"],
        ["--start", "1:00"],
        ["--end", "1:30"],
        ["--out", "/tmp/out"],
    ):
        i = cmd.index(pair[0])
        assert cmd[i + 1] == pair[1]


def test_build_command_rejects_unknown_density():
    mod = load_module()
    with pytest.raises(ValueError):
        mod.build_command("clip.mov", density="9x9")


# --- contract parsing ---


def test_parse_contract_accepts_v0_payload():
    mod = load_module()
    payload = mod.parse_contract(json.dumps(contract_payload("clip.mov")))
    assert payload["files"] == ["/abs/path/clip.sheet-1.jpg"]


def test_parse_contract_rejects_wrong_contract():
    mod = load_module()
    bad = contract_payload("clip.mov") | {"contract": "squish-cli-v999"}
    with pytest.raises(ValueError):
        mod.parse_contract(json.dumps(bad))


def test_parse_contract_rejects_empty_files():
    mod = load_module()
    bad = contract_payload("clip.mov") | {"files": []}
    with pytest.raises(ValueError):
        mod.parse_contract(json.dumps(bad))


# --- main(), fully hermetic ---


def test_main_passes_video_as_single_argv_and_prints_contract(tmp_path, capsys):
    mod = load_module()
    video = tmp_path / "clip with spaces.mov"
    video.write_bytes(b"\x00")
    fake = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps(contract_payload(str(video))),
        stderr="",
    )
    with patch.object(mod.shutil, "which", return_value="/usr/bin/npx"), patch.object(
        mod.subprocess, "run", return_value=fake
    ) as run:
        rc = mod.main([str(video)])

    assert rc == 0
    sent = run.call_args[0][0]
    assert sent[3] == str(video)  # one argv element, whitespace intact
    out = json.loads(capsys.readouterr().out)
    assert out["contract"] == "squish-cli-v0"
    assert out["files"]


def test_main_errors_when_npx_is_missing(tmp_path, capsys):
    mod = load_module()
    video = tmp_path / "clip.mov"
    video.write_bytes(b"\x00")
    with patch.object(mod.shutil, "which", return_value=None), patch.object(
        mod.subprocess, "run"
    ) as run:
        rc = mod.main([str(video)])

    assert rc == 1
    run.assert_not_called()
    assert "npx" in capsys.readouterr().err


def test_main_errors_on_missing_video_without_running_npx(tmp_path, capsys):
    mod = load_module()
    with patch.object(mod.shutil, "which", return_value="/usr/bin/npx"), patch.object(
        mod.subprocess, "run"
    ) as run:
        rc = mod.main([str(tmp_path / "nope.mov")])

    assert rc == 1
    run.assert_not_called()
    assert "no such video" in capsys.readouterr().err


def test_main_relays_cli_failure(tmp_path, capsys):
    mod = load_module()
    video = tmp_path / "clip.mov"
    video.write_bytes(b"\x00")
    fake = subprocess.CompletedProcess(
        args=[], returncode=2, stdout="", stderr="ffmpeg exploded\n"
    )
    with patch.object(mod.shutil, "which", return_value="/usr/bin/npx"), patch.object(
        mod.subprocess, "run", return_value=fake
    ):
        rc = mod.main([str(video)])

    assert rc == 2
    assert "ffmpeg exploded" in capsys.readouterr().err
