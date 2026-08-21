"""Tests for agent.i18n -- catalog parity, fallback, language resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent import i18n


LOCALES_DIR = Path(__file__).resolve().parents[2] / "locales"


def _load_raw(lang: str) -> dict:
    with (LOCALES_DIR / f"{lang}.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten(d, prefix="") -> dict:
    flat = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


# ---------------------------------------------------------------------------
# Catalog completeness -- this is the key invariant test.  If someone adds a
# new key to en.yaml they MUST add it to every other locale, else runtime
# falls back to English for those users and defeats the feature.
# ---------------------------------------------------------------------------



@pytest.mark.parametrize("lang", [l for l in i18n.SUPPORTED_LANGUAGES if l != "en"])
def test_catalog_keys_match_english(lang: str):
    """Every non-English catalog must have exactly the same key set as English."""
    en_keys = set(_flatten(_load_raw("en")).keys())
    lang_keys = set(_flatten(_load_raw(lang)).keys())
    missing = en_keys - lang_keys
    extra = lang_keys - en_keys
    assert not missing, f"{lang}.yaml missing keys: {sorted(missing)}"
    assert not extra, f"{lang}.yaml has keys not in en.yaml: {sorted(extra)}"


# ---------------------------------------------------------------------------
# gateway.media.* (i18n-ization of the hardcoded "user sent" injection notes
# prepended to user messages when an attachment arrives).  These strings
# reach the LLM verbatim, so if the user's active language is non-English
# they MUST be in that language.  Reject the case where a translator
# mistakenly left the English placeholder values in a locale.
# ---------------------------------------------------------------------------

# Keys where the English source is a self-contained system note
# (no platform-specific data inside the string itself).  The placeholder
# values, when substituted, still come from the platform adapter.
MEDIA_L10N_KEYS = [
    "gateway.media.doc_text_note",
    "gateway.media.doc_binary_note",
    "gateway.media.audio_attachment_note",
    "gateway.media.video_attachment_note",
    "gateway.media.image_seen",
    "gateway.media.image_blurry",
    "gateway.media.image_error",
    "gateway.media.voice_empty_transcript",
    "gateway.media.voice_transcribe_unavailable",
    "gateway.media.voice_transcribe_error",
    "gateway.media.empty_placeholder",
    "gateway.media.sticker",
    "gateway.media.animated_sticker_emoji",
    "gateway.media.animated_sticker_no_emoji",
]


@pytest.mark.parametrize("key", MEDIA_L10N_KEYS)
def test_zh_translation_is_actually_chinese(key):
    """zh.yaml must carry real Chinese translation for every media note, not
    a copy-paste of the English placeholder.  Catches translators who add
    the key (so parity passes) but leave the value as English.
    """
    flat = _flatten(_load_raw("zh"))
    value = flat.get(key, "")
    # CJK Unified Ideographs + the U+FF1A fullwidth colon and other common
    # Chinese punctuation cover the bulk of our translations.  A handful
    # of bracketed punctuation like 【】、《》is also fine.
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in value)
    assert has_cjk, (
        f"zh.yaml key={key!r} doesn't contain CJK characters -- "
        f"value={value!r}"
    )


@pytest.mark.parametrize("key", MEDIA_L10N_KEYS)
def test_zh_translation_differs_from_english(key):
    """zh.yaml translation must actually differ from the English source --
    guards against the 'added key but forgot to translate' failure mode
    where a missing key would have been caught by the catalog parity test
    but a same-as-English value would slip through.
    """
    en_value = _flatten(_load_raw("en")).get(key, "")
    zh_value = _flatten(_load_raw("zh")).get(key, "")
    assert en_value != zh_value, (
        f"zh.yaml key={key!r} is identical to en.yaml -- looks untranslated"
    )


@pytest.mark.parametrize("lang", list(i18n.SUPPORTED_LANGUAGES))
def test_catalog_placeholders_match_english(lang: str):
    """Every translated value must use the same {placeholder} tokens as English.

    A mistranslated placeholder (e.g. ``{description}`` typoed as ``{descricao}``)
    would either raise KeyError at runtime or silently drop the interpolated
    value.  Pin parity at the test layer.
    """
    import re
    placeholder_re = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
    en_flat = _flatten(_load_raw("en"))
    lang_flat = _flatten(_load_raw(lang))
    for key, en_value in en_flat.items():
        en_placeholders = set(placeholder_re.findall(en_value))
        lang_value = lang_flat.get(key, "")
        lang_placeholders = set(placeholder_re.findall(lang_value))
        assert en_placeholders == lang_placeholders, (
            f"{lang}.yaml key={key!r}: placeholders {lang_placeholders} "
            f"don't match English {en_placeholders}"
        )


# ---------------------------------------------------------------------------
# Language resolution
# ---------------------------------------------------------------------------











def test_default_when_nothing_set(monkeypatch):
    """With no env var and no config override, falls back to English."""
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    # Force config lookup to return None -- patch the cached reader.
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_config_language_cached", lambda: None)
    assert i18n.get_language() == "en"


# ---------------------------------------------------------------------------
# t() semantics
# ---------------------------------------------------------------------------







def test_t_missing_key_in_non_english_falls_back_to_english(tmp_path, monkeypatch):
    """If a key exists in English but not in the target locale, fall back."""
    # Stand up a fake incomplete locale under a temp locales dir.
    fake_locales = tmp_path / "locales"
    fake_locales.mkdir()
    (fake_locales / "en.yaml").write_text("foo: English Foo\n", encoding="utf-8")
    (fake_locales / "zh.yaml").write_text("# intentionally empty\n", encoding="utf-8")
    monkeypatch.setattr(i18n, "_locales_dir", lambda: fake_locales)
    i18n.reset_language_cache()
    try:
        assert i18n.t("foo", lang="zh") == "English Foo"
    finally:
        # Clear the cache on teardown so subsequent tests don't see the
        # fake "foo: English Foo" catalog instead of the real locales/*.yaml.
        i18n.reset_language_cache()




# ---------------------------------------------------------------------------
# _locales_dir resolution ladder -- regression for #23943 / #27632 / #35374.
# Sealed installs (Nix store venv, pip wheel) have no source tree next to
# agent/, so _locales_dir must resolve via env override or the data scheme.
# ---------------------------------------------------------------------------



def test_locales_dir_env_override_ignored_when_missing(tmp_path, monkeypatch):
    """A bogus HERMES_BUNDLED_LOCALES falls through to source/wheel resolution
    instead of returning a path that doesn't exist."""
    monkeypatch.setenv("HERMES_BUNDLED_LOCALES", str(tmp_path / "does-not-exist"))
    result = i18n._locales_dir()
    assert result != tmp_path / "does-not-exist"
    # In a source checkout this is the repo-root locales dir.
    assert result.name == "locales"


