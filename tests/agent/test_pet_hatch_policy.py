"""Hatch retry-policy contracts: what a failed row extraction costs.

Fast, hermetic, no paid calls — image generation is mocked with synthetic
strips. These pin the #87739 retry-amplification fixes: a row whose art is
structurally unsegmentable skips the remaining strict (paid) re-rolls and
salvages the SAME strip leniently, while a row whose sliced frames collapsed
to slivers keeps the normal strict ladder instead (a fresh roll can still
segment cleanly).

The full image-processing hatch suite was opt-in behind
``HERMES_RUN_SLOW_PET_TESTS`` and was purged from the default suite; these
contracts decide whether a row costs another paid image call, so they live in
the default suite.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from agent.pet.generate import atlas
from agent.pet.generate import imagegen, orchestrate


def _strip(n_blobs: int, *, size=(208, 208)) -> Image.Image:
    """One row strip with *n_blobs* opaque ellipses on a transparent canvas."""
    img = Image.new("RGBA", (size[0] * n_blobs, size[1]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i in range(n_blobs):
        cx = i * size[0] + size[0] // 2
        draw.ellipse((cx - 60, 34, cx + 60, 174), fill=(60, 80, 200, 255))
    return img


def _hatch_mocks(monkeypatch, tmp_path, *, extract) -> dict[str, int]:
    """Wire hermetic generation mocks; returns the per-prefix paid-call counter."""
    attempts: dict[str, int] = {}

    def fake_generate(prompt, *, n=1, reference_images=None, provider=None, prefix="pet", aspect_ratio="square"):
        attempts[prefix] = attempts.get(prefix, 0) + 1
        state = prefix.replace("pet_row_", "")
        count = dict((s, c) for s, _, c in atlas.ROW_SPECS).get(state, 6)
        path = tmp_path / f"{prefix}_{attempts[prefix]}.png"
        _strip(count).save(path)
        return [path]

    monkeypatch.setattr(imagegen, "resolve_provider", lambda **_: object())
    monkeypatch.setattr(imagegen, "generate", fake_generate)
    monkeypatch.setattr(atlas, "extract_strip_frames", extract)
    return attempts


def test_hatch_pet_skips_strict_retries_on_unsegmentable_row(monkeypatch, tmp_path):
    """A row whose poses are merged (unsegmentable under strict ``components``)
    goes straight to lenient ``auto`` on the SAME strip instead of paying for
    more strict re-rolls — the #87739 retry-amplification case."""
    base = tmp_path / "base.png"
    _strip(1).save(base)

    idle_methods: list[str] = []
    real_extract = atlas.extract_strip_frames

    def tracking_extract(strip, count, *args, method="auto", **kwargs):
        if Path(strip).name.startswith("pet_row_idle"):
            idle_methods.append(method)
            if method == "components":
                raise atlas.UnsegmentableStripError("could not segment 6 padded sprites from strip")
        return real_extract(strip, count, *args, method=method, **kwargs)

    attempts = _hatch_mocks(monkeypatch, tmp_path, extract=tracking_extract)

    result = orchestrate.hatch_pet(base_image=base, slug="unseg-skip", concept="a fox")

    # One paid call for idle: strict failed → lenient salvaged the SAME strip.
    assert attempts["pet_row_idle"] == 1
    assert idle_methods == ["components", "auto"]
    assert "idle" in result.states
    assert not list(tmp_path.glob("pet_row_*"))  # strips still cleaned up


def test_hatch_pet_retries_row_whose_frames_collapse_to_slivers(monkeypatch, tmp_path):
    """Lenient slicing can \"succeed\" with thin slivers of the body. Those pass the
    frame-relative checks, then sink the WHOLE atlas at compose (every state
    shares one normalized scale) after every row has been paid for. The gate
    retries that row instead — the failure the live hatch hit on 2026-09-10
    (running-right median 32x145px vs global 92x145px)."""
    base = tmp_path / "base.png"
    _strip(1).save(base)  # reference silhouette: one full-size ellipse

    idle_extractions: list[str] = []
    real_extract = atlas.extract_strip_frames

    def sliver_then_real(strip, count, *args, method="auto", **kwargs):
        frames = real_extract(strip, count, *args, method=method, **kwargs)
        name = Path(strip).name if isinstance(strip, (str, Path)) else ""
        if name.startswith("pet_row_idle"):
            idle_extractions.append(method)
            if len(idle_extractions) == 1:
                # First extraction "succeeds" but with a 4th-width sliver: the
                # exact shape of the live failure.
                return [f.crop((0, 0, f.width // 4, f.height)) for f in frames]
        return frames

    attempts = _hatch_mocks(monkeypatch, tmp_path, extract=sliver_then_real)

    result = orchestrate.hatch_pet(base_image=base, slug="sliver-gate", concept="a fox")

    assert attempts["pet_row_idle"] == 2, "sliver row must be re-rolled, not accepted"
    assert "idle" in result.states

    # And the gate itself: a sliver row is reported, a whole-body row is not.
    whole = real_extract(_strip(6), 6, fit=False)
    assert atlas.row_frames_collapsed(whole, atlas.silhouette_box(base)) is None
    slivers = [f.crop((0, 0, f.width // 4, f.height)) for f in whole]
    assert "sliver" in (atlas.row_frames_collapsed(slivers, atlas.silhouette_box(base)) or "")


def test_collapsed_row_keeps_the_normal_retry_ladder(monkeypatch, tmp_path):
    """A collapse is NOT an unsegmentable strip: the strip sliced fine, so a
    fresh roll deserves the normal strict retry. Pins the policy that a
    collapsed row does not take the skip-strict shortcut."""
    base = tmp_path / "base.png"
    _strip(1).save(base)

    idle_methods: list[str] = []
    real_extract = atlas.extract_strip_frames

    def collapse_once(strip, count, *args, method="auto", **kwargs):
        frames = real_extract(strip, count, *args, method=method, **kwargs)
        name = Path(strip).name if isinstance(strip, (str, Path)) else ""
        if name.startswith("pet_row_idle"):
            idle_methods.append(method)
            if len(idle_methods) == 1:
                return [f.crop((0, 0, f.width // 4, f.height)) for f in frames]
        return frames

    attempts = _hatch_mocks(monkeypatch, tmp_path, extract=collapse_once)

    result = orchestrate.hatch_pet(base_image=base, slug="collapse-policy", concept="a fox")

    # Two paid calls, both strict — the collapse did not skip to lenient.
    assert attempts["pet_row_idle"] == 2
    assert idle_methods == ["components", "components"]
    assert "idle" in result.states
