"""Tests for the grounded-citations bundled skill.

Covers the SKILL.md authoring standards (frontmatter shape, ≤60-char
description) and the behavior of ``scripts/sources.py`` — the citation ledger
that assigns stable ``url -> [n]`` ids, renders Sources blocks, and verifies a
draft's citations. The verify path is the load-bearing piece: it is what
catches a hallucinated or renumbered citation before delivery.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "research" / "grounded-citations"
SCRIPT = SKILL_DIR / "scripts" / "sources.py"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


@pytest.fixture(scope="module")
def sources_mod():
    spec = importlib.util.spec_from_file_location("gc_sources", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    return tmp_path / "ledger.json"


# ---------------------------------------------------------------------------
# Authoring standards
# ---------------------------------------------------------------------------


def test_skill_files_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()
    assert SCRIPT.is_file()
    assert (SKILL_DIR / "references" / "citation-formats.md").is_file()
    assert (SKILL_DIR / "references" / "grounding-rationale.md").is_file()


def test_description_within_limit(frontmatter: dict) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (limit 60): {desc!r}"
    assert desc.endswith(".")


def test_required_frontmatter_fields(frontmatter: dict) -> None:
    assert frontmatter["name"] == "grounded-citations"
    for field in ("version", "author", "license", "platforms"):
        assert frontmatter.get(field), f"missing frontmatter field: {field}"
    assert frontmatter["metadata"]["hermes"]["category"] == "research"


def test_skill_body_has_modern_sections() -> None:
    body = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    for heading in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert heading in body, f"SKILL.md missing section: {heading}"


# ---------------------------------------------------------------------------
# Ledger identity
# ---------------------------------------------------------------------------


def test_ids_are_stable_and_sequential(sources_mod, ledger: Path) -> None:
    first = sources_mod.add_sources(ledger, ["https://a.example"])
    second = sources_mod.add_sources(ledger, ["https://b.example"])
    assert (first[0]["id"], second[0]["id"]) == (1, 2)
    again = sources_mod.add_sources(ledger, ["https://a.example"])
    assert again[0]["id"] == 1


def test_url_normalization_collapses_fragment_and_trailing_slash(sources_mod, ledger: Path) -> None:
    base = sources_mod.add_sources(ledger, ["https://x.example/page"])[0]["id"]
    for variant in ("https://x.example/page/", "https://x.example/page#part"):
        assert sources_mod.add_sources(ledger, [variant])[0]["id"] == base


def test_query_string_is_significant(sources_mod, ledger: Path) -> None:
    a = sources_mod.add_sources(ledger, ["https://x.example/s?q=1"])[0]["id"]
    b = sources_mod.add_sources(ledger, ["https://x.example/s?q=2"])[0]["id"]
    assert a != b


def test_title_backfills_without_changing_id(sources_mod, ledger: Path) -> None:
    first = sources_mod.add_sources(ledger, ["https://t.example"])[0]
    assert first["title"] == ""
    second = sources_mod.add_sources(ledger, ["https://t.example"], title="Later title")[0]
    assert (second["id"], second["title"]) == (first["id"], "Later title")


def test_ingest_walks_search_and_extract_payloads(sources_mod) -> None:
    payload = {
        "data": {"web": [{"title": "One", "url": "https://n.example/1"}]},
        "results": [
            {"url": "https://n.example/1", "title": "One again"},
            {"url": "https://n.example/2", "title": "Two"},
        ],
    }
    pairs = sources_mod.urls_from_json(payload)
    assert [u for u, _ in pairs] == ["https://n.example/1", "https://n.example/2"]


def test_ingest_ignores_non_http_values(sources_mod) -> None:
    payload = {"url": "file:///etc/passwd", "nested": {"link": "mailto:a@b.c"}}
    assert sources_mod.urls_from_json(payload) == []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _seed(sources_mod, ledger: Path) -> list[dict]:
    sources_mod.add_sources(ledger, ["https://a.example"], title="Alpha")
    sources_mod.add_sources(ledger, ["https://b.example"])
    sources_mod.add_sources(ledger, ["https://c.example"], title="Gamma")
    return json.loads(ledger.read_text(encoding="utf-8"))["sources"]


def test_render_markdown_lists_ids_and_urls(sources_mod, ledger: Path) -> None:
    block = sources_mod.render_sources(_seed(sources_mod, ledger))
    assert block.startswith("## Sources")
    assert "[1] https://a.example — Alpha" in block
    assert "[2] https://b.example" in block


def test_render_only_subset_and_ordering(sources_mod, ledger: Path) -> None:
    block = sources_mod.render_sources(_seed(sources_mod, ledger), style="plain", only={3, 1})
    lines = [ln for ln in block.splitlines() if ln.startswith("[")]
    assert lines[0].startswith("[1]") and lines[1].startswith("[3]")
    assert not any(ln.startswith("[2]") for ln in lines)


def test_render_bibtex_keys_match_ids(sources_mod, ledger: Path) -> None:
    block = sources_mod.render_sources(_seed(sources_mod, ledger), style="bibtex", only={1})
    assert "@misc{source1," in block
    assert r"\url{https://a.example}" in block


def test_render_empty_selection_is_empty_string(sources_mod, ledger: Path) -> None:
    assert sources_mod.render_sources(_seed(sources_mod, ledger), only=set()) == ""


# ---------------------------------------------------------------------------
# Verification — the guarantee
# ---------------------------------------------------------------------------


def _verify(sources_mod, ledger: Path, tmp_path: Path, text: str, **kw):
    draft = tmp_path / "draft.md"
    draft.write_text(text, encoding="utf-8")
    sources = json.loads(ledger.read_text(encoding="utf-8"))["sources"]
    return sources_mod.verify_draft(draft, sources, **kw)


def test_well_formed_draft_passes(sources_mod, ledger: Path, tmp_path: Path) -> None:
    _seed(sources_mod, ledger)
    text = (
        "Ice is less dense than liquid water and floats.[1][2]\n\n"
        "Sources:\n[1] https://a.example\n[2] https://b.example\n"
    )
    code, errors, _ = _verify(sources_mod, ledger, tmp_path, text)
    assert (code, errors) == (0, [])


def test_unknown_citation_id_fails(sources_mod, ledger: Path, tmp_path: Path) -> None:
    _seed(sources_mod, ledger)
    text = "A claim with an invented source id here.[42]\n\nSources:\n[42] https://fake.example\n"
    code, errors, _ = _verify(sources_mod, ledger, tmp_path, text)
    assert code == 1
    # Must be flagged as an inline citation the ledger never issued — not merely
    # as a stray Sources-block line, which is a separate (weaker) error.
    assert any("hallucinated or renumbered" in e for e in errors), errors


def test_sources_block_url_must_match_ledger(sources_mod, ledger: Path, tmp_path: Path) -> None:
    _seed(sources_mod, ledger)
    text = "A real claim carrying a real id.[1]\n\nSources:\n[1] https://wrong.example\n"
    code, errors, _ = _verify(sources_mod, ledger, tmp_path, text)
    assert code == 1
    assert any("does not match the ledger" in e for e in errors)


def test_missing_sources_block_fails(sources_mod, ledger: Path, tmp_path: Path) -> None:
    _seed(sources_mod, ledger)
    code, errors, _ = _verify(sources_mod, ledger, tmp_path, "A real claim carrying an id.[1]\n")
    assert code == 1
    assert any("no `Sources:` block" in e for e in errors)


def test_cited_but_absent_from_block_fails(sources_mod, ledger: Path, tmp_path: Path) -> None:
    _seed(sources_mod, ledger)
    text = (
        "First claim about the topic at hand.[1]\n"
        "Second claim about the topic at hand.[2]\n\n"
        "Sources:\n[1] https://a.example\n"
    )
    code, errors, _ = _verify(sources_mod, ledger, tmp_path, text)
    assert code == 1
    assert any("absent from the Sources block" in e for e in errors)


def test_brackets_inside_code_fences_are_not_citations(sources_mod, ledger: Path, tmp_path: Path) -> None:
    _seed(sources_mod, ledger)
    text = "Prose with no external claims in it.\n\n```python\nvalue = arr[42]\n```\n"
    code, errors, _ = _verify(sources_mod, ledger, tmp_path, text)
    assert (code, errors) == (0, [])


def test_markdown_links_are_not_citations(sources_mod, ledger: Path, tmp_path: Path) -> None:
    _seed(sources_mod, ledger)
    text = "See the [docs](https://x.example) for the full option list.\n"
    code, errors, _ = _verify(sources_mod, ledger, tmp_path, text)
    assert (code, errors) == (0, [])


def test_min_coverage_gate(sources_mod, ledger: Path, tmp_path: Path) -> None:
    _seed(sources_mod, ledger)
    text = (
        "A cited claim about the subject matter.[1]\n"
        "An uncited claim about the subject matter.\n"
        "Another uncited claim about the subject matter.\n"
        "A third uncited claim about the subject matter.\n\n"
        "Sources:\n[1] https://a.example\n"
    )
    ok_code, _, _ = _verify(sources_mod, ledger, tmp_path, text)
    assert ok_code == 0
    code, errors, _ = _verify(sources_mod, ledger, tmp_path, text, min_coverage=0.5)
    assert code == 1
    assert any("coverage" in e for e in errors)


def test_over_citation_warns_without_failing(sources_mod, ledger: Path, tmp_path: Path) -> None:
    sources_mod.add_sources(
        ledger, [f"https://s{i}.example" for i in range(1, 5)]
    )
    text = (
        "One sentence leaning on far too many sources at once.[1][2][3][4]\n\n"
        "Sources:\n"
        + "".join(f"[{i}] https://s{i}.example\n" for i in range(1, 5))
    )
    code, errors, warnings = _verify(sources_mod, ledger, tmp_path, text)
    assert (code, errors) == (0, [])
    assert any("more than 3 citations" in w for w in warnings)


def test_strict_mode_promotes_warnings_to_failure(sources_mod, ledger: Path, tmp_path: Path) -> None:
    _seed(sources_mod, ledger)
    text = "A cited claim about the subject.[1]\n\nSources:\n[1] https://a.example\n"
    assert _verify(sources_mod, ledger, tmp_path, text)[0] == 0
    assert _verify(sources_mod, ledger, tmp_path, text, strict=True)[0] == 1


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_add_render_verify_roundtrip(sources_mod, tmp_path: Path, capsys) -> None:
    ledger = tmp_path / "cli.json"
    args = ["--ledger", str(ledger)]
    assert sources_mod.main(args + ["add", "https://a.example", "https://b.example"]) == 0
    assert "[1] https://a.example" in capsys.readouterr().out

    draft = tmp_path / "d.md"
    draft.write_text("Only the first source is used here.[1]\n", encoding="utf-8")
    assert sources_mod.main(args + ["render", "--cited-in", str(draft)]) == 0
    block = capsys.readouterr().out
    assert "[1] https://a.example" in block and "[2]" not in block

    with draft.open("a", encoding="utf-8") as fh:
        fh.write("\n" + block)
    assert sources_mod.main(args + ["verify", str(draft)]) == 0


def test_cli_reset_empties_the_ledger(sources_mod, tmp_path: Path, capsys) -> None:
    ledger = tmp_path / "r.json"
    args = ["--ledger", str(ledger)]
    sources_mod.main(args + ["add", "https://a.example"])
    capsys.readouterr()
    assert sources_mod.main(args + ["reset"]) == 0
    capsys.readouterr()
    assert sources_mod.main(args + ["add", "https://z.example"]) == 0
    assert "[1] https://z.example" in capsys.readouterr().out


def test_cli_verify_missing_draft_returns_2(sources_mod, tmp_path: Path) -> None:
    ledger = tmp_path / "m.json"
    code = sources_mod.main(["--ledger", str(ledger), "verify", str(tmp_path / "nope.md")])
    assert code == 2


def test_cli_ledger_path_prefers_flag_over_env(sources_mod, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_CITATION_LEDGER", str(tmp_path / "env.json"))
    flagged = tmp_path / "flag.json"
    assert sources_mod.resolve_ledger_path(str(flagged)) == flagged
    assert sources_mod.resolve_ledger_path(None) == tmp_path / "env.json"


def test_cli_ledger_path_defaults_under_hermes_home(sources_mod, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HERMES_CITATION_LEDGER", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    path = sources_mod.resolve_ledger_path(None)
    assert path.parts[-3:] == ("cache", "citations", "ledger.json")
    assert str(tmp_path) in str(path)


def test_corrupt_ledger_raises_actionable_error(sources_mod, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        sources_mod.load_ledger(bad)
    assert "reset" in str(exc.value)


# ---------------------------------------------------------------------------
# Fact-checking: evidence quotes and [unverified] markers
# ---------------------------------------------------------------------------

_PAGE = (
    "Water expands when it freezes.\n"
    "Ice is about 9% less   dense than liquid water,\n"
    "which is why icebergs float.\n"
)


def test_quote_verbatim_match_is_whitespace_and_case_insensitive(sources_mod, ledger: Path) -> None:
    sources_mod.add_sources(ledger, ["https://a.example"])
    entry = sources_mod.attach_quote(
        ledger, 1, "ice is about 9% less dense than liquid water,", _PAGE
    )
    assert len(entry["quotes"]) == 1


def test_quote_rejects_paraphrase(sources_mod, ledger: Path) -> None:
    sources_mod.add_sources(ledger, ["https://a.example"])
    with pytest.raises(SystemExit) as exc:
        sources_mod.attach_quote(ledger, 1, "Frozen water is roughly 9% lighter", _PAGE)
    assert "not found verbatim" in str(exc.value)


def test_quote_rejects_unknown_id_and_short_text(sources_mod, ledger: Path) -> None:
    sources_mod.add_sources(ledger, ["https://a.example"])
    with pytest.raises(SystemExit) as exc:
        sources_mod.attach_quote(ledger, 7, "which is why icebergs float.", _PAGE)
    assert "no source [7]" in str(exc.value)
    with pytest.raises(SystemExit) as exc:
        sources_mod.attach_quote(ledger, 1, "icebergs float.", _PAGE)
    assert "too short" in str(exc.value)


def test_quote_is_idempotent(sources_mod, ledger: Path) -> None:
    sources_mod.add_sources(ledger, ["https://a.example"])
    sources_mod.attach_quote(ledger, 1, "Water expands when it freezes.", _PAGE)
    entry = sources_mod.attach_quote(ledger, 1, "water  expands when it freezes.", _PAGE)
    assert len(entry["quotes"]) == 1


def test_verify_evidence_gate_requires_quotes(sources_mod, ledger: Path, tmp_path: Path) -> None:
    _seed(sources_mod, ledger)
    text = (
        "A claim supported by the first source.[1]\n\n"
        "Sources:\n[1] https://a.example\n"
    )
    code, errors, _ = _verify(sources_mod, ledger, tmp_path, text, require_evidence=True)
    assert code == 1
    assert any("no verbatim evidence quote" in e for e in errors)

    sources_mod.attach_quote(ledger, 1, "Water expands when it freezes.", _PAGE)
    code, errors, _ = _verify(sources_mod, ledger, tmp_path, text, require_evidence=True)
    assert (code, errors) == (0, [])


def test_evidence_gate_only_applies_to_cited_sources(sources_mod, ledger: Path, tmp_path: Path) -> None:
    _seed(sources_mod, ledger)
    sources_mod.attach_quote(ledger, 1, "Water expands when it freezes.", _PAGE)
    # [2] and [3] have no quotes but are not cited — the gate must not fail on them.
    text = "A claim supported by the first source.[1]\n\nSources:\n[1] https://a.example\n"
    code, errors, _ = _verify(sources_mod, ledger, tmp_path, text, require_evidence=True)
    assert (code, errors) == (0, [])


def test_unverified_marker_counts_toward_coverage(sources_mod, ledger: Path, tmp_path: Path) -> None:
    _seed(sources_mod, ledger)
    text = (
        "A cited claim about the subject matter.[1]\n"
        "A model-knowledge claim declared as such.[unverified]\n\n"
        "Sources:\n[1] https://a.example\n"
    )
    code, errors, _ = _verify(sources_mod, ledger, tmp_path, text, min_coverage=0.9)
    assert (code, errors) == (0, [])


def test_unverified_marker_does_not_hide_uncited_sentences(sources_mod, ledger: Path, tmp_path: Path) -> None:
    _seed(sources_mod, ledger)
    text = (
        "A cited claim about the subject matter.[1]\n"
        "An uncited, unmarked claim about the subject.\n"
        "Another uncited, unmarked claim about the subject.\n\n"
        "Sources:\n[1] https://a.example\n"
    )
    code, errors, _ = _verify(sources_mod, ledger, tmp_path, text, min_coverage=0.9)
    assert code == 1
    assert any("coverage" in e for e in errors)


def test_render_evidence_style_includes_quotes(sources_mod, ledger: Path) -> None:
    _seed(sources_mod, ledger)
    sources_mod.attach_quote(ledger, 1, "Water expands when it freezes.", _PAGE)
    sources = json.loads(ledger.read_text(encoding="utf-8"))["sources"]
    block = sources_mod.render_sources(sources, style="evidence", only={1})
    assert "[1] https://a.example" in block
    assert '> "Water expands when it freezes."' in block
    plain = sources_mod.render_sources(sources, style="markdown", only={1})
    assert "Water expands" not in plain


def test_cli_quote_and_evidence_verify_roundtrip(sources_mod, tmp_path: Path, capsys) -> None:
    ledger = tmp_path / "ev.json"
    page = tmp_path / "page.txt"
    page.write_text(_PAGE, encoding="utf-8")
    args = ["--ledger", str(ledger)]
    assert sources_mod.main(args + ["add", "https://a.example"]) == 0
    capsys.readouterr()

    assert (
        sources_mod.main(
            args + ["quote", "1", "--text", "Water expands when it freezes.", "--from", str(page)]
        )
        == 0
    )
    assert "evidence attached" in capsys.readouterr().out

    draft = tmp_path / "d.md"
    draft.write_text(
        "A claim resting on the source page.[1]\n\nSources:\n[1] https://a.example\n",
        encoding="utf-8",
    )
    assert sources_mod.main(args + ["verify", str(draft), "--evidence"]) == 0
    capsys.readouterr()
    assert sources_mod.main(args + ["render", "--style", "evidence"]) == 0
    assert '> "Water expands when it freezes."' in capsys.readouterr().out
