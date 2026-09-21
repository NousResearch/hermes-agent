import baoyu_loader as bl


def test_style_block_loads_known_style():
    block = bl.style_block("screen-print")
    assert "screen" in block.lower()
    assert len(block) > 50


def test_palette_block_loads_known_palette():
    block = bl.palette_block("warm")
    assert "#" in block


def test_universal_rules_present():
    r = bl.universal_rules()
    assert "do NOT display" in r
    assert "white space" in r.lower()


def test_unknown_style_falls_back():
    assert bl.style_block("does-not-exist") == bl.style_block(bl.DEFAULT_STYLE)
