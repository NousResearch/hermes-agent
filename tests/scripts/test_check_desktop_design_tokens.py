"""scripts/check_desktop_design_tokens.py: redesign CSS restating a design value is flagged; token
reads, classic CSS, tokens/ and exempted declarations are not; the ratchet only goes down."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_desktop_design_tokens.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_desktop_design_tokens", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module while the class is built
    spec.loader.exec_module(mod)
    return mod


GUARD = _load()
GATED = ":root[data-shell='portal'] .x {{ {} }}"


def _kinds(css: str, rel: str = "app/chat/stage.css") -> list[str]:
    return [f.kind for f in GUARD.file_findings(rel, css)]


@pytest.mark.parametrize(
    ("decl", "kind"),
    [
        ("color: #a22a20;", "raw-color"),
        ("box-shadow: 0 8px 24px rgb(0 0 0 / 0.5);", "raw-color"),
        ("background: white;", "raw-color"),
        ("background: color-mix(in srgb, var(--ui-base) 5%, transparent);", "inline-mix"),
        ("border-color: color-mix(in srgb, var(--portal-brand-fill) 25%, transparent);", "inline-mix"),
        ("border-radius: 6px;", "radius"),
        ("border-top-left-radius: 0.5rem;", "radius"),
        ("font-size: 0.8125rem;", "font-size"),
        ("font: 300 1.875rem/1.1 var(--portal-font-display);", "font-size"),
        ("letter-spacing: 0.06em;", "letter-spacing"),
        ("line-height: 17px;", "line-height"),
        ("transition: background-color 150ms ease;", "duration"),
        ("z-index: 20;", "z-index"),
    ],
)
def test_a_restated_value_in_a_gated_rule_is_flagged(decl, kind):
    assert kind in _kinds(GATED.format(decl))


@pytest.mark.parametrize(
    "decl",
    [
        "color: var(--portal-text-primary);",
        "background: transparent;",
        "border-color: currentColor;",
        "border-radius: var(--portal-radius-card);",
        "border-radius: 50%;",
        "border-radius: 0;",
        "font: 400 var(--portal-type-ui-size)/1.4 var(--portal-font-body);",
        "line-height: 1.4;",
        "letter-spacing: 0;",
        "transition: background-color var(--portal-motion-fast) var(--portal-ease);",
        "transition-duration: 0s;",
        "z-index: 1;",
        "font-family: 'Oswald', var(--dt-font-sans);",
        "grid-template-areas: 'tan red';",
        "background: var(--portal-raw-ink-5);",
    ],
)
def test_token_reads_and_neutral_keywords_pass(decl):
    assert _kinds(GATED.format(decl)) == []


def test_classic_rules_are_not_judged_but_redesign_only_files_are_judged_whole():
    classic = ".card { border-radius: 6px; color: #fff; }"
    assert _kinds(classic, "app/chat/stage.css") == []
    assert _kinds(":root.dark .card { border-radius: 6px; }", "components/ui/button.css") == []
    assert sorted(_kinds(classic, "app/chat/sidebar/portal/portal-sidebar.css")) == ["radius", "raw-color"]


def test_the_tokens_folder_is_where_values_live():
    tokens = ":root[data-shell='portal'] { --portal-raw-danger: #a22a20; --portal-radius-card: 4px; }"
    assert _kinds(tokens, "app/redesign/tokens/raw.css") == []
    assert _kinds(tokens, "app/redesign/portal-shell.css") == ["raw-color", "token-literal", "token-literal"]


def test_portal_tokens_outside_tokens_must_alias_a_token():
    css = ":root[data-shell='portal'] { --portal-x-gap: 1.25rem; --portal-x-fill: var(--portal-raw-ink-5); }"
    assert _kinds(css, "app/redesign/portal-shell.css") == ["token-literal"]


def test_token_exempt_marks_one_declaration_or_a_whole_rule():
    before = GATED.format("/* token-exempt: Figma optical offset */ border-radius: 23px; font-size: 1.1rem;")
    assert _kinds(before) == ["font-size"]
    trailing = ":root[data-shell='portal'] .x {\n  border-radius: 23px; /* token-exempt: optical */\n  z-index: 9;\n}"
    assert _kinds(trailing) == ["z-index"]
    rule = "/* token-exempt: hero art crop */\n:root[data-shell='portal'] .art { border-radius: 23px; z-index: 9; }"
    assert _kinds(rule) == []


def test_nested_at_rules_keep_the_gate_and_report_their_line():
    css = "@media (min-width: 40rem) {\n  :root[data-shell='portal'] .x {\n    color: #000;\n  }\n}\n"
    [finding] = GUARD.file_findings("app/chat/stage.css", css)
    assert (finding.kind, finding.line) == ("raw-color", 3)


def test_ratchet_fails_growth_and_advises_shrinkage():
    over, under = GUARD.ratchet({"a.css": 3, "b.css": 1, "new.css": 1}, {"a.css": 3, "b.css": 2})
    assert over == ["new.css"]
    assert under == ["b.css"]


def test_undefined_portal_references_are_found_across_css_and_tsx(tmp_path):
    (tmp_path / "tokens.css").write_text(":root[data-shell='portal'] { --portal-radius-card: 4px; }")
    (tmp_path / "a.css").write_text(".x { border-radius: var(--portal-radius-card); color: var(--portal-radius-crad); }")
    (tmp_path / "b.tsx").write_text(
        "const s = { '--portal-sidebar-width': '20rem' }\n"
        "export const C = () => <div className='w-(--portal-sidebar-width) bg-(--portal-nope)' />\n"
    )
    defined, refs = GUARD.references(tmp_path)
    missing = sorted(name for name in refs if name not in defined)
    assert missing == ["--portal-nope", "--portal-radius-crad"]


def test_docs_tables_list_dark_and_light_values_per_token(tmp_path):
    (tmp_path / "shape.css").write_text(
        ":root[data-shell='portal'] {\n  /* Cards, tables, menus. */\n  --portal-radius-card: 4px;\n"
        "  --portal-shadow-floating: 0 8px 24px var(--portal-raw-shadow-50);\n}\n"
        ":root[data-shell='portal']:not(.dark) {\n  --portal-shadow-floating: none;\n}\n"
    )
    docs = GUARD.render_docs(tmp_path)
    assert "| `--portal-radius-card` | `4px` | ″ | Cards, tables, menus. |" in docs
    assert "| `--portal-shadow-floating` | `0 8px 24px var(--portal-raw-shadow-50)` | `none` |" in docs
    assert GUARD.docs_block(f"intro\n{docs}\noutro") == docs


def test_a_tree_without_the_tokens_folder_has_nothing_to_judge(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(GUARD, "SRC", tmp_path / "apps" / "desktop" / "src")
    monkeypatch.setattr(GUARD, "DESIGN_MD", tmp_path / "apps" / "desktop" / "DESIGN.md")

    assert GUARD.main([]) == 0
    assert "nothing to check" in capsys.readouterr().out
