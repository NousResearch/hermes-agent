"""The /restart drain message needs no plural form and keeps LTR fragments out of RTL sentences.

Decision (also in the comment at the guard in ``gateway/slash_commands.py``):

* **No ``{count}`` in the catalog.** A count noun needs CLDR plural selection — Arabic has six
  categories (zero/one/two/few/many/other), Hebrew four (one/two/many/other), Russian and Ukrainian
  three — and ``t()`` only formats a flat string, so a translated "3 task(s)" can only be wrong in
  those catalogues. The counts therefore travel on a second, code-built line whose labels
  (``agent``/``cron``/``api``/``deferred``) are identifiers no catalog translates.
* **A locale may name no counts at all** (fixed phrase): ``t()`` never sees a placeholder, so a
  translation without them still renders — the code-built line carries the numbers.
* **Direction.** Nothing LTR sits inside the translatable sentence, so an RTL paragraph can never
  reorder a number through Arabic/Hebrew punctuation. Hebrew has no bundled catalog
  (``locales/he.yaml`` does not exist and ``he`` is not a supported id): it falls back to English,
  and a pack that adds it must flag ``rtl``.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import hermes_yaml as yaml
from agent import i18n
from agent.i18n import t
from agent.i18n_languages import BUNDLED_LANGUAGE_INFO, describe_language
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source

KEY = "gateway.restart.blocked"

# Bidi controls: clients render them inconsistently and they leak into copy-paste and logs.
BIDI_CONTROLS = frozenset("\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")

# One live agent run per count: ar one(1)/two(2)/few(3)/many(11)/other(100), he one/two/other/many.
COUNTS = (1, 2, 3, 11, 100)


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=make_restart_source(),
        message_id="m1",
    )


def _first_strong(text: str) -> str | None:
    for char in text:
        direction = unicodedata.bidirectional(char)
        if direction in {"L", "R", "AL"}:
            return direction
    return None


@pytest.fixture(autouse=True)
def _hermetic_restart_context(monkeypatch):
    for var in ("INVOCATION_ID", "XPC_SERVICE_NAME", "HERMES_S6_SUPERVISED_CHILD",
                "HERMES_GATEWAY_EXTERNAL_SUPERVISOR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("gateway.restart.is_container_restart_context", lambda: False)
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    i18n.reset_language_cache()
    yield
    i18n.reset_language_cache()


def test_no_bundled_catalog_leaves_a_placeholder_or_a_pseudo_plural():
    for lang in i18n.SUPPORTED_LANGUAGES:
        text = i18n._load_bundled(lang)[KEY]
        assert "{" not in text and "}" not in text, lang
        assert "task(s)" not in text, lang


@pytest.mark.parametrize("count", COUNTS)
@pytest.mark.asyncio
async def test_arabic_sentence_is_identical_at_every_count_and_the_count_rides_its_own_line(
        count, monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "ar")
    i18n.reset_language_cache()
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    runner._drain_work_counts = MagicMock(return_value=(count, 0, 0, 0))

    sentence, detail = (await runner._handle_message(_event("/restart"))).split("\n")

    assert sentence == t(KEY, lang="ar")  # same form for every count: no plural category to select
    assert not any(char.isdigit() for char in sentence)
    assert detail == f"{count} agent"
    runner.request_restart.assert_not_called()


def test_arabic_is_flagged_rtl_and_leads_with_rtl_text():
    assert BUNDLED_LANGUAGE_INFO["ar"][1] is True
    assert describe_language("ar", pack=None, overlay=False)["rtl"] is True
    assert _first_strong(i18n._load_bundled("ar")[KEY]) == "AL"


def test_no_catalog_sentence_carries_bidi_marks_or_leads_with_ltr_text_in_an_rtl_locale():
    for lang in i18n.SUPPORTED_LANGUAGES:
        text = i18n._load_bundled(lang)[KEY]
        assert not (set(text) & BIDI_CONTROLS), lang
        if _first_strong(text) in {"R", "AL"}:
            assert BUNDLED_LANGUAGE_INFO[lang][1] is True, lang


def test_hebrew_has_no_bundled_catalog_and_falls_back_to_english():
    assert "he" not in i18n.SUPPORTED_LANGUAGES
    assert not (Path(i18n._locales_dir()) / "he.yaml").exists()
    assert t(KEY, lang="he") == t(KEY, lang="en")


def test_a_hebrew_pack_is_flagged_rtl_so_the_same_rule_applies_when_it_lands():
    pack = {"endonym": "עברית", "rtl": True, "source": "plugin:hermes-lang-he"}
    assert describe_language("he", pack=pack, overlay=False)["rtl"] is True


@pytest.mark.asyncio
async def test_a_locale_that_names_no_counts_renders_its_fixed_phrase_plus_the_detail_line(
        tmp_path, monkeypatch):
    """Documented fallback: omit the counts from the sentence; the code-built line still reports them."""
    fixed = "⚠️ Nie restartuję: trwa jeszcze praca."
    locales = tmp_path / "locales"
    locales.mkdir()
    (locales / "pl.yaml").write_text(
        yaml.safe_dump({"gateway": {"restart": {"blocked": fixed}}}, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_LANGUAGE", "pl")
    i18n.reset_language_cache()
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    runner._drain_work_counts = MagicMock(return_value=(2, 1, 0, 0))

    assert await runner._handle_message(_event("/restart")) == f"{fixed}\n2 agent, 1 cron"
    runner.request_restart.assert_not_called()


@pytest.mark.parametrize("lang", i18n.SUPPORTED_LANGUAGES)
@pytest.mark.asyncio
async def test_the_blocked_reply_is_each_shipped_pack_s_own_sentence(lang, monkeypatch):
    """Call-site parity, per pack: a locale missing the key would reply with the bare dotted key."""
    monkeypatch.setenv("HERMES_LANGUAGE", lang)
    i18n.reset_language_cache()
    runner, _adapter = make_restart_runner()
    runner.request_restart = MagicMock(return_value=True)
    runner._drain_work_counts = MagicMock(return_value=(2, 1, 0, 0))

    result = await runner._handle_message(_event("/restart"))

    assert result == f"{i18n._load_bundled(lang)[KEY]}\n2 agent, 1 cron"
    assert KEY not in result
    runner.request_restart.assert_not_called()
