"""Regression for #102336: a profile overlays the bundled i18n catalog from HERMES_HOME.

``<HERMES_HOME>/locales/<lang>.yaml`` is merged *over* ``locales/<lang>.yaml``: only the keys it
lists change, everything else keeps falling through to the bundled wording, and the overlay is
scoped to the profile that wrote it (one gateway process serves several profiles).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import i18n


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fresh temp HERMES_HOME with the catalog cache cleared around the test."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    i18n.reset_language_cache()
    try:
        yield home
    finally:
        i18n.reset_language_cache()


def _write_overlay(home: Path, lang: str, text: str) -> None:
    locales = home / "locales"
    locales.mkdir(exist_ok=True)
    (locales / f"{lang}.yaml").write_text(text, encoding="utf-8")


def test_overlay_wins_for_listed_keys_and_the_rest_falls_through(home):
    """A two-line overlay restyles one message; keys it omits keep the bundled wording."""
    bundled_rate_limit = i18n.t("gateway.provider_error.rate_limit", lang="en")
    bundled_auth = i18n.t("gateway.provider_error.auth", lang="en")
    # Guard the comparison below: a missing-on-both-sides key would echo the key itself.
    assert bundled_rate_limit != "gateway.provider_error.rate_limit"

    _write_overlay(home, "en", 'gateway:\n  provider_error:\n    rate_limit: "busy right now, please resend"\n')
    i18n.reset_language_cache()

    assert i18n.t("gateway.provider_error.rate_limit", lang="en") == "busy right now, please resend"
    assert i18n.t("gateway.provider_error.auth", lang="en") == bundled_auth


def test_overlay_belongs_to_the_asking_profile_not_the_first_one(tmp_path, monkeypatch):
    """A→B→A with no cache reset between lookups: each home answers with its own overlay.

    The bundled catalog is shared, so the cache key has to include the home — otherwise the
    first profile's overlay would answer for every other profile in a multiplexed gateway.
    """
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    _write_overlay(home_a, "en", 'gateway:\n  provider_error:\n    generic: "wording from profile A"\n')

    monkeypatch.setenv("HERMES_HOME", str(home_b))
    bundled = i18n.t("gateway.provider_error.generic", lang="en")
    assert bundled != "wording from profile A"

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    assert i18n.t("gateway.provider_error.generic", lang="en") == "wording from profile A"
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    assert i18n.t("gateway.provider_error.generic", lang="en") == bundled
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    assert i18n.t("gateway.provider_error.generic", lang="en") == "wording from profile A"
