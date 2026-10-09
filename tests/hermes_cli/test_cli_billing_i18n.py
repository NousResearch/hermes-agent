"""The billing/subscription CLI copy resolves through the i18n catalog at call time.

Menu labels and outcome copy used to live in import-time module tuples/dicts, which froze
English before ``display.language`` was known. These tests pin the behaviour (not the
wording): a language switch after import changes what the mixin renders.
"""

from __future__ import annotations

import shutil

import pytest

import agent.i18n as i18n
from cli import HermesCLI
from hermes_cli import cli_billing_mixin as bm


@pytest.fixture
def cli():
    obj = HermesCLI.__new__(HermesCLI)
    obj._app = None
    return obj


@pytest.fixture
def overlay_locales(tmp_path, monkeypatch):
    """Real bundled en.yaml + a one-key ``zh`` catalog, served from a temp locales dir."""
    fake = tmp_path / "locales"
    fake.mkdir()
    shutil.copy(i18n._locales_dir() / "en.yaml", fake / "en.yaml")
    (fake / "zh.yaml").write_text(
        "cli:\n  billing:\n    choice_add_funds: 充值\n    charge_failed_card_declined: 卡被拒绝\n",
        encoding="utf-8")
    monkeypatch.setattr(i18n, "_locales_dir", lambda: fake)
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    i18n.reset_language_cache()
    yield fake
    i18n.reset_language_cache()


def _use(monkeypatch, lang: str) -> None:
    monkeypatch.setenv("HERMES_LANGUAGE", lang)
    i18n.reset_language_cache()


def test_menu_choices_follow_language_switch_after_import(overlay_locales, monkeypatch):
    _use(monkeypatch, "en")
    en_label = bm._topup_menu_choices()[0][1]
    _use(monkeypatch, "zh")
    zh_label = bm._topup_menu_choices()[0][1]
    assert zh_label == "充值"
    assert en_label == i18n.t("cli.billing.choice_add_funds", lang="en")
    assert zh_label != en_label
    # Choice VALUES are identifiers and never move with the language.
    assert [c[0] for c in bm._topup_menu_choices()] == ["buy", "auto", "limit", "portal", "cancel"]


def test_charge_failed_copy_resolves_per_reason_at_call_time(cli, overlay_locales, monkeypatch, capsys):
    class _State:
        portal_url = ""

    _use(monkeypatch, "zh")
    cli._billing_render_charge_failed(_State(), "card_declined")
    out = capsys.readouterr().out
    assert "卡被拒绝" in out
    # A zh miss falls back to the English line for the same reason.
    cli._billing_render_charge_failed(_State(), "payment_method_expired")
    assert i18n.t("cli.billing.charge_failed_payment_method_expired", lang="en") in capsys.readouterr().out
    # Unknown reasons use the generic template carrying the raw reason.
    cli._billing_render_charge_failed(_State(), "weird_reason")
    assert i18n.t("cli.billing.charge_failed_generic", reason="weird_reason") in capsys.readouterr().out


def test_logged_out_block_uses_catalog(cli, capsys):
    class _State:
        error = ""

    cli._print_logged_out(_State(), i18n.t("cli.subscription.load_failed_label"), "/subscription")
    out = capsys.readouterr().out
    assert i18n.t("cli.billing.not_logged_in") in out
    assert i18n.t("cli.billing.run_portal_then", cmd="/subscription") in out
