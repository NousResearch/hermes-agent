"""Regression: the [edge-tts] extra must stay at >= 7.3.0 (#137944).

Microsoft rotated the Edge Voice DRM token scheme (Sec-MS-GEC) on
2026-10-12. Every edge-tts release before 7.3.0 fails every synthesis
with HTTP 403 against the live endpoint, which kills the default TTS
provider for all installs. The pin plus its exclude-newer-package
exception in pyproject.toml is the fix; this test pins the contract so
a dependency refresh cannot silently roll the extra back below 7.3.0.
"""

from pathlib import Path
import tomllib


def _edge_tts_pin():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8-sig"))
    specs = metadata["project"]["optional-dependencies"]["edge-tts"]
    pin = next(spec for spec in specs if spec.startswith("edge-tts=="))
    return pin, metadata


def test_edge_tts_extra_pin_clears_the_drm_rotation():
    pin, _ = _edge_tts_pin()
    version = pin.split("==", 1)[1]
    assert tuple(int(part) for part in version.split(".")) >= (7, 3, 0), (
        "edge-tts < 7.3.0 403s on every synthesis after Microsoft's "
        "2026-10-12 Sec-MS-GEC token rotation (#137944)"
    )


def test_release_age_quarantine_has_the_edge_tts_exception():
    _, metadata = _edge_tts_pin()
    exceptions = metadata["tool"]["uv"].get("exclude-newer-package", {})
    assert "edge-tts" in exceptions, (
        "edge-tts 7.3.0 was published inside the 3-day exclude-newer "
        "window; without the per-package exception `uv lock` cannot "
        "resolve the pin (#137944)"
    )
