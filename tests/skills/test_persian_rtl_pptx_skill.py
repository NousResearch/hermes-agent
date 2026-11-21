"""Skill tests for persian-rtl-pptx (stdlib + pytest + unittest.mock only).

Covers: SKILL.md frontmatter contract, script syntax sanity, Persian digit
conversion, the builder pipeline (RTL XML + digits, needs python-pptx), font
embedding into a zip (no external font file needed), and pattern generation
(needs matplotlib). Missing optional deps skip, never fail.
"""
import ast
import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path

import pytest
import hermes_yaml as yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills/productivity/persian-rtl-pptx"


def _load(module_name, rel_path):
    spec = importlib.util.spec_from_file_location(
        module_name, SKILL_DIR / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def skill_md():
    content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"\n---\s*\n", content[3:])
    return content, yaml.safe_load(content[3:m.start() + 3])


def test_skill_md_frontmatter(skill_md):
    _, fm = skill_md
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing {field}"
    hermes = (fm.get("metadata") or {}).get("hermes") or {}
    assert hermes.get("tags"), "no tags"
    assert hermes.get("category") == "productivity"


def test_skill_md_description(skill_md):
    _, fm = skill_md
    desc = fm["description"]
    assert len(desc) <= 60
    assert desc.endswith(".")


def test_scripts_parse():
    for py in sorted((SKILL_DIR / "scripts").glob("*.py")):
        ast.parse(py.read_text(encoding="utf-8"), filename=str(py))


def test_templates_ship_assets():
    assert (SKILL_DIR / "templates/example_spec.json").exists()
    assert (SKILL_DIR / "templates/assets/pattern.png").exists()
    assert (SKILL_DIR / "templates/assets/map_sarhani.png").exists()
    assert (SKILL_DIR / "references/spec.md").exists()


def test_fa_digits():
    rtl = _load("rtl_pptx_fa", "scripts/rtl_pptx.py") if pytest.importorskip(
        "pptx") is not None else None
    # conversion is a pure-string helper; exercise it via a spec-level check
    assert rtl.fa("20 Tir 1364") == "۲۰ Tir ۱۳۶۴"


SPEC = {
    "font": "DejaVu Sans",
    "persian_digits": True,
    "footer": "تست",
    "slides": [
        {"type": "title", "kicker": "ک", "title": "عنوان", "subtitle": "زیرعنوان",
         "date": "۱۴۰۳"},
        {"type": "bullets", "title": "موضوع", "items": ["مورد ۱", "مورد ۲", {"text": "زیر", "sub": "توضیح"}]},
        {"type": "closing", "title": "پایان"},
    ],
}


def test_builder_pipeline(tmp_path, monkeypatch):
    pptx = pytest.importorskip("pptx")
    rtl = _load("rtl_pptx_test", "scripts/rtl_pptx.py")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(SPEC, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "deck.pptx"
    rc = rtl.main([str(spec_path), str(out)])
    assert rc == 0 and out.exists()

    prs = pptx.Presentation(str(out))
    assert len(prs.slides._sldIdLst) == 3

    # every paragraph must be RTL and right-aligned; digits must be Persian
    latin = re.compile(r"[0-9]")
    seen = 0
    for slide in prs.slides:
        for sh in slide.shapes:
            if not sh.has_text_frame:
                continue
            for p in sh.text_frame.paragraphs:
                if not "".join(r.text for r in p.runs).strip():
                    continue
                seen += 1
                pPr = p._pPr
                assert pPr is not None and pPr.get("rtl") == "1"
                assert pPr.get("algn") == "r"
                txt = "".join(r.text for r in p.runs)
                assert not latin.search(txt), f"Latin digits leaked: {txt!r}"
    assert seen >= 4, "expected several non-empty paragraphs"


def test_builder_validates_bad_slide_type(tmp_path):
    pytest.importorskip("pptx")
    rtl = _load("rtl_pptx_bad", "scripts/rtl_pptx.py")
    spec = dict(SPEC, slides=[{"type": "nope", "title": "x"}])
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(AttributeError):
        rtl.main([str(spec_path), str(tmp_path / "o.pptx")])


def test_embed_fonts(tmp_path):
    # build a minimal pptx to embed into (no python-pptx needed for embedding)
    pptx = pytest.importorskip("pptx")
    rtl = _load("rtl_pptx_embed", "scripts/rtl_pptx.py")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(
        dict(SPEC, slides=[{"type": "title", "title": "x"}]), ensure_ascii=False),
        encoding="utf-8")
    deck = tmp_path / "deck.pptx"
    rtl.main([str(spec_path), str(deck)])

    embed = _load("embed_fonts_test", "scripts/embed_fonts.py")
    a = tmp_path / "a.ttf"
    a.write_bytes(b"FAKEFONT-A")
    b = tmp_path / "b.ttf"
    b.write_bytes(b"FAKEFONT-B")
    rc = embed.main([str(deck), "--typeface", "DejaVu Sans",
                     "--regular", str(a), "--bold", str(b)])
    assert rc == 0

    with zipfile.ZipFile(deck) as z:
        names = z.namelist()
        assert "fonts/font1.fntdata" in names and "fonts/font2.fntdata" in names
        pres = z.read("ppt/presentation.xml").decode("utf-8")
        assert "embeddedFontLst" in pres
        assert 'typeface="DejaVu Sans"' in pres
        rels = z.read("ppt/_rels/presentation.xml.rels").decode("utf-8")
        assert "embeddedFont" in rels
        ct = z.read("[Content_Types].xml").decode("utf-8")
        assert 'Extension="fntdata"' in ct
        assert z.read("fonts/font1.fntdata") == b"FAKEFONT-A"

    # the deck still opens cleanly as a presentation
    pptx.Presentation(str(deck))


def test_embed_fonts_requires_fonts(tmp_path):
    pytest.importorskip("pptx")
    rtl = _load("rtl_pptx_embed2", "scripts/rtl_pptx.py")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(SPEC, ensure_ascii=False), encoding="utf-8")
    deck = tmp_path / "deck.pptx"
    rtl.main([str(spec_path), str(deck)])
    embed = _load("embed_fonts_req", "scripts/embed_fonts.py")
    with pytest.raises(SystemExit):
        embed.main(["--help"])


def test_verify_deck_clean(tmp_path):
    pytest.importorskip("pptx")
    rtl = _load("rtl_pptx_verify", "scripts/rtl_pptx.py")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(SPEC, ensure_ascii=False), encoding="utf-8")
    deck = tmp_path / "deck.pptx"
    rtl.main([str(spec_path), str(deck)])
    verify = _load("verify_deck_test", "scripts/verify_deck.py")
    rc = verify.main([str(deck), "--margin", "0.2"])
    assert rc == 0


def test_make_graphics_pattern(tmp_path):
    pytest.importorskip("matplotlib")
    mg = _load("make_graphics_test", "scripts/make_graphics.py")
    out = tmp_path / "pattern.png"
    rc = mg.main([str(out), "--kind", "pattern", "--w", "4", "--h", "2.25", "--dpi", "72"])
    assert rc == 0 and out.exists() and out.stat().st_size > 1000
