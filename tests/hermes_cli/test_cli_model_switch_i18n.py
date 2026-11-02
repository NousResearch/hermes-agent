"""The /model switch mixin's CLI copy resolves through the i18n catalog at call time."""

from __future__ import annotations

import shutil

import pytest

from agent import i18n
from hermes_cli import cli_model_switch_mixin as ms


@pytest.fixture
def overlay_locales(tmp_path, monkeypatch):
    fake = tmp_path / "locales"
    fake.mkdir()
    shutil.copy(i18n._locales_dir() / "en.yaml", fake / "en.yaml")
    (fake / "zh.yaml").write_text(
        "cli:\n  model:\n    effort_keep_current: 保持当前\n", encoding="utf-8")
    monkeypatch.setattr(i18n, "_locales_dir", lambda: fake)
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    i18n.reset_language_cache()
    yield fake
    i18n.reset_language_cache()


def test_picker_effort_rows_follow_language_at_call_time(overlay_locales, monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    i18n.reset_language_cache()
    en_rows = ms._picker_reasoning_rows()
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    i18n.reset_language_cache()
    zh_rows = ms._picker_reasoning_rows()
    # Values (identifiers) are stable; the keep-current label moved with the language.
    assert [v for v, _ in en_rows] == [v for v, _ in zh_rows]
    assert zh_rows[-1] == ("", "保持当前")
    assert en_rows[-1][1] != zh_rows[-1][1]
    # A key the zh overlay lacks falls back to English.
    assert zh_rows[-2][1] == en_rows[-2][1]


def test_model_usage_block_lists_every_form_with_catalog_description(monkeypatch):
    """No authenticated providers → the usage block: one aligned row per /model form."""
    import cli as cli_mod
    printed: list[str] = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda line: printed.append(line))

    class _Cli:
        def _open_model_picker(self, *a, **kw):  # pragma: no cover - not reached
            raise AssertionError("picker must not open without providers")

    ms._show_model_picker(_Cli(), None, force_refresh=False)  # ctx=None → no providers
    assert printed[0].strip() == i18n.t("cli.model.no_authenticated_providers")
    rows = printed[2:]
    assert len(rows) == len(ms._MODEL_USAGE_ROWS) == 8
    for (form, key), line in zip(ms._MODEL_USAGE_ROWS, rows):
        assert line.startswith(f"  {form}")
        assert line.endswith(i18n.t(key))
        assert i18n.t(key) != key  # present in en.yaml
    # Description column is aligned across rows.
    assert len({line.index(i18n.t(key)) for (_, key), line in zip(ms._MODEL_USAGE_ROWS, rows)}) == 1
