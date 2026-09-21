"""Tests for baoyu_loader's runtime resolution of MIT on-box skill refs.

The on-box skill at ~/.hermes/skills/creative/baoyu-article-illustrator/
is the source of truth at runtime; the vendored refs in baoyu_refs/ are
the offline fallback when the on-box dir is absent.
"""
import baoyu_loader as bl


def test_preset_lookup_known():
    out = bl.preset("versus")
    assert out.get("type") == "comparison"
    assert out.get("style") == "vector-illustration"


def test_preset_lookup_unknown_returns_empty():
    assert bl.preset("nope-not-a-real-preset") == {}


def test_list_presets_includes_canonical_names():
    names = bl.list_presets()
    for required in (
        "tech-explainer", "system-design", "architecture", "science-paper",
        "versus", "ink-notes-compare", "edu-visual", "hero",
    ):
        assert required in names, f"missing preset: {required}"


def test_runtime_prefers_on_box_skill_refs(monkeypatch, tmp_path):
    """When the on-box skill is present, its refs are read, not the vendored."""
    fake_root = tmp_path / "skill"
    refs = fake_root / "references"
    (refs / "styles").mkdir(parents=True)
    (refs / "palettes").mkdir(parents=True)
    (refs / "styles" / "screen-print.md").write_text(
        "# FROM ON-BOX\nSentinel-on-box-12345\n", encoding="utf-8"
    )
    (refs / "palettes" / "warm.md").write_text(
        "# FROM ON-BOX PALETTE\nSentinel-palette-on-box-999\n", encoding="utf-8"
    )
    (refs / "style-presets.md").write_text("# sentinel presets\n", encoding="utf-8")
    (refs / "prompt-construction.md").write_text("# sentinel pcm\n", encoding="utf-8")

    monkeypatch.setattr(bl, "_ON_BOX_REFS", refs)
    # Lru_cache busts needed.
    bl._read_first_available.cache_clear()
    style = bl.style_block("screen-print")
    pal = bl.palette_block("warm")
    assert "Sentinel-on-box-12345" in style
    assert "Sentinel-palette-on-box-999" in pal


def test_runtime_falls_back_to_vendored_when_on_box_absent(monkeypatch):
    """When the on-box path is missing, vendored refs are used."""
    monkeypatch.setattr(bl, "_ON_BOX_REFS", None)
    bl._read_first_available.cache_clear()
    style = bl.style_block("screen-print")
    # Vendored refs already have "screen" in the screen-print style block.
    assert "screen" in style.lower()
    assert len(style) > 50


def test_prompt_construction_block_returns_string():
    txt = bl.prompt_construction_block()
    assert isinstance(txt, str)
    # Real content if refs present (either on-box or vendored).
    assert len(txt) > 100
