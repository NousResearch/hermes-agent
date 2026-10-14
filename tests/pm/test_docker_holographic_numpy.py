"""Packaging contract for issue #136707.

The holographic catalog memory provider (``itpartypattaya/hermes-plugin-holographic``)
needs NumPy for its HRR algebra. The plugin pins NumPy itself, but on a sealed
Docker image plugin dependencies install lazily into ``/opt/data/installs`` at
first use — impossible on an offline/air-gapped container. The result was a
silent FTS-only degradation (``FactRetriever`` redistributed ``hrr_weight``
from 0.3 to 0.0) with green logs everywhere.

The image bakes a small ``[holographic]`` extra carrying NumPy, exactly like
it bakes ``[matrix]`` / ``[google-chat]`` for providers whose deps cannot be
prepared on demand in a container. These tests are static (no Docker daemon)
so the packaging contract is enforced on every machine and in CI.
"""

from __future__ import annotations

from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def _optional_dependencies() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["optional-dependencies"]


def _sealed_venv_build() -> str:
    """The Dockerfile ``pm.build_env`` stanza that assembles the sealed venv."""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    start = text.index("python3 -m pm.build_env")
    end = text.index("\n\n", start)  # one backslash-continued RUN stanza
    return text[start:end]


def test_holographic_extra_is_declared_and_pins_numpy():
    extras = _optional_dependencies()
    assert "holographic" in extras, "pyproject must declare a [holographic] extra carrying NumPy"
    numpy_pins = [
        s for s in extras["holographic"]
        if s.split(";", 1)[0].strip().startswith("numpy")
    ]
    assert numpy_pins == ["numpy==2.4.3"], (
        "the [holographic] extra must pin the same NumPy as [voice]/[wake] "
        f"(one-version contract); got {numpy_pins}"
    )


def test_holographic_extra_has_a_numpy_anchor():
    from pm import extras

    # ANCHORS names real pyproject extras and is the fast availability probe
    # PM/doctor use without a resolver.
    assert extras.ANCHORS.get("holographic") == "numpy"


def test_holographic_extra_is_excluded_from_all():
    """Provider dep baked for the image / lazily installed, not in [all]."""
    extras = _optional_dependencies()
    assert not any("hermes-agent[holographic]" in spec for spec in extras["all"])


def test_docker_image_bakes_the_holographic_extra_into_the_sealed_venv():
    bake = _sealed_venv_build()
    assert "--sealed" in bake and "--record-selection" in bake, (
        "test must target the sealed, selection-recording venv build"
    )
    assert "--extra holographic" in bake, (
        "the sealed image venv must bake --extra holographic so an offline "
        "container has NumPy and HRR structural recall works out of the box (#136707)"
    )
