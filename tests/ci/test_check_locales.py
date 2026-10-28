"""Tests for scripts/check_locales.py.

The checker's job is surfacing catalog drift that the runtime hides:
YAML lookups and defineLocale() both fall back to English, so a missing
key is invisible until a user sees mixed-language UI. These tests pin the
key-extraction behavior for each catalog format and the collect() layout,
including the not-yet-merged hybrid JSON catalogs (#38846) staying absent
until their directory exists.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_locales.py"
_spec = importlib.util.spec_from_file_location("check_locales", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load check_locales.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_yaml_leaf_keys_nested_mappings(tmp_path: Path) -> None:
    catalog = tmp_path / "en.yaml"
    catalog.write_text(
        "\n".join(
            [
                "# comment",
                "greeting: hello",
                "menu:",
                "  file:",
                "    open: Open",
                "    close: 'Close: now'",
                "  edit: Edit",
                "empty:",
            ]
        ),
        encoding="utf-8",
    )
    assert _mod.yaml_leaf_keys(catalog) == {
        "greeting",
        "menu.file.open",
        "menu.file.close",
        "menu.edit",
    }


def test_ts_leaf_keys_nested_inline_and_quoted(tmp_path: Path) -> None:
    catalog = tmp_path / "en.ts"
    catalog.write_text(
        "\n".join(
            [
                "export const en = {",
                "  title: 'Hermes',",
                "  // comment",
                "  panel: {",
                "    'quoted-key': 'value: with colon',",
                "    inline: { label: 'x', help: 'y' },",
                "    count: (n: number) => `${n} items`,",
                "  },",
                "}",
            ]
        ),
        encoding="utf-8",
    )
    keys = _mod.ts_leaf_keys(catalog)
    assert "title" in keys
    assert "panel.quoted-key" in keys
    assert "panel.inline.label" in keys
    assert "panel.inline.help" in keys
    assert "panel.count" in keys
    assert "panel" not in keys


def test_json_leaf_keys_flat_and_nested_agree(tmp_path: Path) -> None:
    flat = tmp_path / "flat.json"
    flat.write_text('{"menu.file.open": "Open", "greeting": "hi"}', encoding="utf-8")
    nested = tmp_path / "nested.json"
    nested.write_text(
        '{"menu": {"file": {"open": "Open"}}, "greeting": "hi"}', encoding="utf-8"
    )
    expected = {"menu.file.open", "greeting"}
    assert _mod.json_leaf_keys(flat) == expected
    assert _mod.json_leaf_keys(nested) == expected


def _fake_repo(root: Path, with_json: bool) -> None:
    yaml_dir = root / "locales"
    yaml_dir.mkdir(parents=True)
    (yaml_dir / "en.yaml").write_text("a: x\nb: y\n", encoding="utf-8")
    (yaml_dir / "zz.yaml").write_text("a: x\n", encoding="utf-8")

    ts_dir = root / "apps" / "desktop" / "src" / "i18n"
    ts_dir.mkdir(parents=True)
    (ts_dir / "en.ts").write_text("export const en = {\n  a: 'x',\n}\n", encoding="utf-8")
    (ts_dir / "zz.ts").write_text("export const zz = {\n  a: 'x',\n}\n", encoding="utf-8")
    (ts_dir / "index.ts").write_text("export {}\n", encoding="utf-8")

    if with_json:
        json_dir = root / "apps" / "desktop" / "src" / "locales"
        json_dir.mkdir(parents=True)
        (json_dir / "en.json").write_text('{"a": "x", "b": "y"}', encoding="utf-8")
        (json_dir / "zz.json").write_text('{"a": "x"}', encoding="utf-8")


def test_collect_skips_json_catalogs_until_directory_exists(tmp_path: Path) -> None:
    _fake_repo(tmp_path, with_json=False)
    labels = set(_mod.collect(tmp_path))
    assert labels == {"locales/zz.yaml", "desktop/zz.ts"}


def test_collect_reports_json_drift_when_present(tmp_path: Path) -> None:
    _fake_repo(tmp_path, with_json=True)
    results = _mod.collect(tmp_path)
    assert set(results) == {"locales/zz.yaml", "desktop/zz.ts", "desktop-json/zz.json"}
    baseline, keys = results["desktop-json/zz.json"]
    assert baseline - keys == {"b"}


def test_collect_excludes_machinery_and_baselines(tmp_path: Path) -> None:
    _fake_repo(tmp_path, with_json=True)
    labels = set(_mod.collect(tmp_path))
    for baseline_label in ("locales/en.yaml", "desktop/en.ts", "desktop-json/en.json"):
        assert baseline_label not in labels
    assert "desktop/index.ts" not in labels
