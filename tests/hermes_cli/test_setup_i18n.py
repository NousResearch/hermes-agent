"""Setup wizard strings resolve through agent.i18n and keep the banner box intact in every language."""

import pytest

from agent import i18n
from hermes_cli import setup as setup_mod

_BORDER = "┌─────────────────────────────────────────────────────────┐"


@pytest.mark.parametrize("lang", list(i18n.SUPPORTED_LANGUAGES))
def test_setup_banners_fit_the_box_in_every_language(lang):
    """A translated title longer than the box would push the right border out of line."""
    rows = [setup_mod._box_line(i18n.t("setup.wizard.banner_title", lang=lang), indent=13),
            setup_mod._box_line(i18n.t("setup.wizard.banner_intro", lang=lang)),
            setup_mod._box_line(i18n.t("setup.wizard.banner_exit_hint", lang=lang)),
            setup_mod._box_line(i18n.t("setup.portal.banner", lang=lang), indent=5)]
    rows += [setup_mod._box_line(i18n.t("setup.wizard.section_banner", lang=lang, label=i18n.t(key, lang=lang)),
                                 indent=5)
             for _, key, _ in setup_mod.SETUP_SECTIONS]
    for row in rows:
        assert len(row) == len(_BORDER), (lang, row)


def test_setup_follows_display_language(monkeypatch):
    """The wizard speaks the configured language instead of hard-coded English."""
    monkeypatch.setenv("HERMES_LANGUAGE", "it")
    i18n.reset_language_cache()
    try:
        labels = [setup_mod.t(key) for _, key, _ in setup_mod.SETUP_SECTIONS]
        assert labels[0] == "Modello e provider"
        assert setup_mod.t("setup.mode.question") == "Come vuoi configurare Hermes?"
    finally:
        i18n.reset_language_cache()
