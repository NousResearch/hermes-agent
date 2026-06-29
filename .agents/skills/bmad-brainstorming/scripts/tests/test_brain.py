# /// script
# requires-python = ">=3.10"
# dependencies = ["pytest>=8.0"]
# ///
"""Tests for brain.py. Run: uv run -m pytest scripts/tests/test_brain.py"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import brain  # noqa: E402

CSV = """category,technique_name,description,detail
collaborative,Yes And Building,Build on every idea with "yes and" to keep momentum,
wild,Quantum Superposition,Hold contradictory ideas as simultaneously true,techniques/quantum.md
structured,SCAMPER Method,Run the idea through seven transformation lenses,
wild,Anti-Solution,Brainstorm how to make the problem worse then invert,
"""

DETAIL = "# Quantum Superposition\nFull multi-step instructions for the complex technique."


@pytest.fixture
def lib(tmp_path):
    csv_path = tmp_path / "brain-methods.csv"
    csv_path.write_text(CSV, encoding="utf-8")
    (tmp_path / "techniques").mkdir()
    (tmp_path / "techniques" / "quantum.md").write_text(DETAIL, encoding="utf-8")
    return csv_path


def test_load_normalizes_detail(lib):
    rows = brain.load(lib)
    assert len(rows) == 4
    assert rows[0]["detail"] == ""
    assert rows[1]["detail"] == "techniques/quantum.md"


def test_categories_counts_sorted(lib):
    assert brain.categories(brain.load(lib)) == [("collaborative", 1), ("structured", 1), ("wild", 2)]


def test_filter_is_case_insensitive(lib):
    rows = brain.filter_cats(brain.load(lib), ["WILD"])
    assert {r["technique_name"] for r in rows} == {"Quantum Superposition", "Anti-Solution"}


def test_filter_none_returns_all(lib):
    assert len(brain.filter_cats(brain.load(lib), None)) == 4


def test_find_hits_and_misses(lib):
    found, missing = brain.find(brain.load(lib), ["scamper method", "Nope"])
    assert [r["technique_name"] for r in found] == ["SCAMPER Method"]
    assert missing == ["Nope"]


def test_resolve_detail_present(lib):
    row = next(r for r in brain.load(lib) if r["detail"])
    assert "multi-step instructions" in brain.resolve_detail(row, lib.parent)


def test_resolve_detail_absent_is_none(lib):
    row = next(r for r in brain.load(lib) if not r["detail"])
    assert brain.resolve_detail(row, lib.parent) is None


def test_resolve_detail_missing_file_warns_not_fatal(lib, capsys):
    rows = brain.load(lib)
    rows[1]["detail"] = "techniques/gone.md"
    assert brain.resolve_detail(rows[1], lib.parent) is None
    assert "not found" in capsys.readouterr().err


def test_show_inlines_detail(lib, capsys):
    assert brain.main(["--file", str(lib), "show", "Quantum Superposition"]) == 0
    out = capsys.readouterr().out
    assert "multi-step instructions" in out and "[wild]" in out


def test_show_simple_has_no_detail(lib, capsys):
    brain.main(["--file", str(lib), "show", "SCAMPER Method"])
    out = capsys.readouterr().out
    assert "transformation lenses" in out


def test_show_all_missing_returns_1(lib):
    assert brain.main(["--file", str(lib), "show", "Ghost"]) == 1


def test_list_filtered_text(lib, capsys):
    brain.main(["--file", str(lib), "list", "--category", "structured"])
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("structured\tSCAMPER Method\t")


def test_list_bare_is_refused(lib, capsys):
    # the footgun: bare `list` must NOT dump the catalog into context
    assert brain.main(["--file", str(lib), "list"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""  # nothing leaked to stdout
    assert "--category" in captured.err and "--all" in captured.err


def test_list_all_dumps_everything(lib, capsys):
    assert brain.main(["--file", str(lib), "list", "--all"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 4  # the deliberate full-catalog escape hatch


def test_json_output(lib, capsys):
    import json
    brain.main(["--file", str(lib), "--json", "categories"])
    data = json.loads(capsys.readouterr().out)
    assert {"category": "wild", "count": 2} in data


def test_random_respects_n_and_category(lib, capsys):
    brain.main(["--file", str(lib), "random", "--category", "wild", "-n", "5"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2  # only 2 wild exist, n capped
    assert all(line.startswith("wild\t") for line in lines)


def test_random_negative_n_does_not_crash(lib, capsys):
    # a negative -n is clamped to 0, not passed to random.sample (which would raise)
    assert brain.main(["--file", str(lib), "random", "-n", "-1"]) == 0
    assert capsys.readouterr().out.strip() == ""


def test_missing_file_returns_2(tmp_path):
    assert brain.main(["--file", str(tmp_path / "nope.csv"), "categories"]) == 2


# --- html selection page ------------------------------------------------

def test_html_requires_out(lib, capsys):
    # never dump the catalog to stdout — writing to a file is the whole point
    assert brain.main(["--file", str(lib), "html"]) == 2
    assert "--out" in capsys.readouterr().err


def test_html_writes_selection_page(lib, tmp_path):
    out = tmp_path / "sel.html"
    assert brain.main(["--file", str(lib), "html", "--out", str(out)]) == 0
    doc = out.read_text(encoding="utf-8")
    assert doc.startswith("<!DOCTYPE html>")
    assert "BMad Method Brainstorming Selection" in doc
    for r in brain.load(lib):
        assert r["technique_name"] in doc  # every technique is selectable
    assert "&quot;yes and&quot;" in doc  # quotes in a description are escaped, not raw


def test_html_creates_missing_parent(lib, tmp_path):
    out = tmp_path / "nested" / "deep" / "sel.html"
    assert brain.main(["--file", str(lib), "html", "--out", str(out)]) == 0
    assert out.is_file()


# --- --extra overlay (customize.toml additional_techniques) -------------

EXTRA = (
    '[{"category": "domain-specific", "technique_name": "Regulatory Inversion", '
    '"description": "Start from the compliance constraint and brainstorm what it unlocks."}, '
    '{"category": "wild", "technique_name": "Extra Wild One", "description": "An added wild method."}]'
)


@pytest.fixture
def extra(tmp_path):
    p = tmp_path / "extra.json"
    p.write_text(EXTRA, encoding="utf-8")
    return p


def test_extra_merges_into_categories(lib, extra, capsys):
    brain.main(["--file", str(lib), "--extra", str(extra), "categories"])
    out = capsys.readouterr().out
    assert "domain-specific\t1" in out  # a brand-new category appears
    assert "wild\t3" in out  # the extra wild one is counted alongside the shipped two


def test_extra_appears_in_list_and_random(lib, extra, capsys):
    brain.main(["--file", str(lib), "--extra", str(extra), "list", "--category", "domain-specific"])
    assert "Regulatory Inversion" in capsys.readouterr().out


def test_extra_is_first_class_in_html(lib, extra, tmp_path):
    out = tmp_path / "sel.html"
    assert brain.main(["--file", str(lib), "--extra", str(extra), "html", "--out", str(out)]) == 0
    doc = out.read_text(encoding="utf-8")
    # custom technique is selectable and its new category renders without crashing (fallback glyph/hue)
    assert "Regulatory Inversion" in doc
    assert "Domain Specific" in doc


def test_extra_missing_file_returns_2(lib, tmp_path):
    assert brain.main(["--file", str(lib), "--extra", str(tmp_path / "nope.json"), "categories"]) == 2


def test_unknown_category_style_uses_fallback_glyph():
    hue, glyph = brain.category_style("totally-made-up-category")
    assert hue.startswith("#") and len(hue) == 7  # valid derived hex
    assert glyph == brain._FALLBACK_GLYPH


def test_shipped_selector_is_in_sync_with_catalog():
    # foolproofing: if someone edits brain-methods.csv they must regenerate the page.
    # Regenerate with: uv run brain.py html --out assets/brain-selector.html
    asset = brain.DEFAULT_FILE.parent / "brain-selector.html"
    assert asset.is_file(), "missing assets/brain-selector.html — generate it"
    expected = brain.html_doc(brain.load(brain.DEFAULT_FILE))
    assert asset.read_text(encoding="utf-8") == expected, (
        "assets/brain-selector.html is stale; regenerate: "
        "uv run brain.py html --out assets/brain-selector.html"
    )
