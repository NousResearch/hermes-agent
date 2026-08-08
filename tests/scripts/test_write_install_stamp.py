import pytest

from scripts.write_install_stamp import build_stamp


def test_build_stamp_keeps_provenance_separate_from_distribution():
    stamp = build_stamp(commit="a" * 40, source="ci", distribution="docker")

    assert stamp["source"] == "ci"
    assert stamp["distribution"] == "docker"


def test_thin_build_carries_no_payload_regardless_of_tag(monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP_BUNDLED", raising=False)
    monkeypatch.setenv("HERMES_PAYLOAD_TAG", "v9.9.9")

    stamp = build_stamp(commit="a" * 40)

    assert stamp["payload"] is False
    assert stamp["tag"] is None


def test_bundled_build_records_payload_and_tag(monkeypatch):
    monkeypatch.setenv("HERMES_DESKTOP_BUNDLED", "1")
    monkeypatch.setenv("HERMES_PAYLOAD_TAG", "v0.18.0")

    stamp = build_stamp(commit="b" * 40)

    assert stamp["payload"] is True
    assert stamp["tag"] == "v0.18.0"


def test_bundled_build_without_tag_stops_the_build(monkeypatch):
    monkeypatch.setenv("HERMES_DESKTOP_BUNDLED", "1")
    monkeypatch.delenv("HERMES_PAYLOAD_TAG", raising=False)

    with pytest.raises(SystemExit, match="HERMES_PAYLOAD_TAG"):
        build_stamp(commit="b" * 40)


def test_desktop_app_is_a_valid_distribution():
    stamp = build_stamp(commit="c" * 40, source="ci", distribution="desktop-app")

    assert stamp["distribution"] == "desktop-app"


def test_distribution_defaults_to_null():
    stamp = build_stamp(commit="c" * 40)

    assert stamp["distribution"] is None


def test_cli_accepts_desktop_app_distribution(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    out = tmp_path / "stamp.json"
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "write_install_stamp.py"),
            "--output", str(out),
            "--commit", "d" * 40,
            "--distribution", "desktop-app",
        ],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["distribution"] == "desktop-app"
