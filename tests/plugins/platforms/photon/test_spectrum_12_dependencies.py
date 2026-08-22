"""Dependency contract for the Photon Spectrum 12 migration."""

from __future__ import annotations

import json
from pathlib import Path


_SIDECAR = Path("plugins/platforms/photon/sidecar")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_photon_sidecar_uses_current_spectrum_stack() -> None:
    package = _json(_SIDECAR / "package.json")

    assert package["dependencies"]["spectrum-ts"] == "12.8.0"
    assert package["dependencies"]["ffmpeg-static"] == "^5.3.0"
    assert package["engines"]["node"] == ">=22.22.0"


def test_photon_sidecar_drops_spectrum_8_install_mutations() -> None:
    package = _json(_SIDECAR / "package.json")
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "postinstall" not in package.get("scripts", {})
    assert not (_SIDECAR / "patch-spectrum-mixed-attachments.mjs").exists()
    assert "patch-spectrum-mixed-attachments.mjs" not in dockerfile


def test_photon_sidecar_keeps_security_overrides_compatible_with_spectrum_12() -> None:
    package = _json(_SIDECAR / "package.json")

    assert package["overrides"] == {
        "protobufjs": "^8.7.1",
        "@opentelemetry/otlp-transformer": "0.218.0",
        "@opentelemetry/otlp-exporter-base": "0.218.0",
        "@opentelemetry/exporter-trace-otlp-http": "0.218.0",
        "@opentelemetry/exporter-logs-otlp-http": "0.218.0",
        "@opentelemetry/core": "2.10.0",
    }


def test_photon_lockfile_resolves_expected_transport_versions() -> None:
    lock = _json(_SIDECAR / "package-lock.json")
    packages = lock["packages"]

    assert packages["node_modules/spectrum-ts"]["version"] == "12.8.0"
    assert packages["node_modules/@spectrum-ts/imessage"]["version"] == "12.8.0"
    assert packages["node_modules/@photon-ai/advanced-imessage"]["version"] == "2.1.0"
    assert "node_modules/ffmpeg-static" in packages
