import pytest

from scripts.write_install_stamp import build_stamp


def test_build_stamp_keeps_provenance_separate_from_distribution():
    stamp = build_stamp(commit="a" * 40, source="ci", distribution="docker", update_mechanism="external")

    assert stamp["source"] == "ci"
    assert stamp["distribution"] == "docker"


def test_default_build_is_a_bootstrap_artifact_regardless_of_tag(monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP_VARIANT", raising=False)
    monkeypatch.setenv("HERMES_PAYLOAD_TAG", "v9.9.9")

    stamp = build_stamp(commit="a" * 40, update_mechanism="self")

    assert stamp["payload"] == "bootstrap"
    assert stamp["tag"] is None


def test_explicit_bootstrap_variant_matches_the_default(monkeypatch):
    monkeypatch.setenv("HERMES_DESKTOP_VARIANT", "bootstrap")
    monkeypatch.setenv("HERMES_PAYLOAD_TAG", "v9.9.9")

    stamp = build_stamp(commit="a" * 40, update_mechanism="self")

    assert stamp["payload"] == "bootstrap"
    assert stamp["tag"] is None


def test_bundled_variant_records_payload_and_tag(monkeypatch):
    monkeypatch.setenv("HERMES_DESKTOP_VARIANT", "bundled")
    monkeypatch.setenv("HERMES_PAYLOAD_TAG", "v0.18.0")

    stamp = build_stamp(commit="b" * 40, update_mechanism="self")

    assert stamp["payload"] == "bundled"
    assert stamp["tag"] == "v0.18.0"


def test_light_variant_records_payload_and_tag(monkeypatch):
    monkeypatch.setenv("HERMES_DESKTOP_VARIANT", "light")
    monkeypatch.setenv("HERMES_PAYLOAD_TAG", "v0.18.0")

    stamp = build_stamp(commit="b" * 40, update_mechanism="self")

    assert stamp["payload"] == "light"
    assert stamp["tag"] == "v0.18.0"


@pytest.mark.parametrize("variant", ["bundled", "light"])
def test_self_updating_variants_without_tag_stop_the_build(monkeypatch, variant):
    monkeypatch.setenv("HERMES_DESKTOP_VARIANT", variant)
    monkeypatch.delenv("HERMES_PAYLOAD_TAG", raising=False)

    with pytest.raises(SystemExit, match="HERMES_PAYLOAD_TAG"):
        build_stamp(commit="b" * 40, update_mechanism="self")


def test_unknown_variant_stops_the_build(monkeypatch):
    monkeypatch.setenv("HERMES_DESKTOP_VARIANT", "chonky")
    monkeypatch.setenv("HERMES_PAYLOAD_TAG", "v0.18.0")

    with pytest.raises(SystemExit, match="unknown HERMES_DESKTOP_VARIANT"):
        build_stamp(commit="b" * 40, update_mechanism="self")


def test_desktop_app_is_a_valid_distribution():
    stamp = build_stamp(commit="c" * 40, source="ci", distribution="desktop-app", update_mechanism="electron-updater")

    assert stamp["distribution"] == "desktop-app"


def test_distribution_defaults_to_null():
    stamp = build_stamp(commit="c" * 40, update_mechanism="self")

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
            "--update-mechanism", "electron-updater",
        ],
        capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text(encoding="utf-8-sig"))["distribution"] == "desktop-app"
