"""Lightweight internationalization (i18n) for Hermes static user-facing messages.

Scope (thin slice, by design): only the highest-impact static strings shown
to the user by Hermes itself -- approval prompts, a handful of gateway slash
command replies, restart-drain notices.  Agent-generated output, log lines,
low-level terminal noise, and model responses are explicitly excluded.

Usage
-----
    from agent.i18n import t

    print(t("approval.denied"))          # uses the configured language
    print(t("approval.denied", lang="pt"))  # override

Tags and paths
--------------
Catalog files live in ``locales/<lang>.yaml`` where ``<lang>`` is a
BCP-47 language tag (``en``, ``zh``, ``zh-hant``, ``pt``, etc.).

Language resolution (``resolve_language``)
------------------------------------------
1. Explicit ``lang`` kwarg
2. ``display.language`` in the active config
3. ``HERMES_LANGUAGE`` environment variable
4. ``DEFAULT_LANGUAGE`` (``"en"``)

Accepted aliases for step 2/3:
  - ``zh-cn``, ``chinese``, ``mandarin``, ``zh-hans`` -> ``zh``
  - ``zh-tw``, ``zh-hk``, ``zh-mo``, ``traditional-chinese`` -> ``zh-hant``
  - ``ja``, ``jp``, ``japanese`` -> ``ja``
  - ``de``, ``deutsch``, ``german`` -> ``de``
  - ``es``, ``espanol``, ``espa\u00f1ol``, ``spanish`` -> ``es``
  - ``fr``, ``francais``, ``fran\u00e7ais``, ``french`` -> ``fr``
  - ``tr``, ``turkish`` -> ``tr``
  - ``uk``, ``ukrainian`` -> ``uk``
  - ``af``, ``afrikaans`` -> ``af``
  - ``ko``, ``korean`` -> ``ko``
  - ``it``, ``italian`` -> ``it``
  - ``ga``, ``gaeilge``, ``irish`` -> ``ga``
  - ``pt``, ``portuguese``, ``pt-pt`` -> ``pt`` (European Portuguese)
  - ``pt-br``, ``brazilian``, ``brasileiro`` -> ``pt-BR`` (Brazilian Portuguese)
  - ``ru``, ``russian`` -> ``ru``
  - ``hu``, ``hungarian`` -> ``hu``
"""

SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "en", "zh", "zh-hant", "ja", "de", "es", "fr", "tr", "uk",
    "af", "ko", "it", "ga", "pt", "pt-BR", "ru", "hu",
)
DEFAULT_LANGUAGE = "en"

# Accept a few natural aliases so users who type "chinese" / "zh-CN" / "jp"
# get the right catalog instead of silently falling back to English.
_LANGUAGE_ALIASES: dict[str, str] = {
    "english": "en", "en-us": "en", "en-gb": "en",
    # Simplified Chinese --- explicit codes route here; bare "chinese" / "mandarin"
    # also default to Simplified since that's the larger user base.
    "chinese": "zh", "mandarin": "zh", "zh-cn": "zh", "zh-hans": "zh", "zh-sg": "zh",
    # Traditional Chinese --- distinct catalog.  Cover Taiwan / Hong Kong / Macau
    # locale tags plus the common "traditional" alias.
    "traditional-chinese": "zh-hant", "traditional_chinese": "zh-hant",
    "zh-tw": "zh-hant", "zh-hk": "zh-hant", "zh-mo": "zh-hant",
    "japanese": "ja", "jp": "ja", "ja-jp": "ja",
    "german": "de", "deutsch": "de", "de-de": "de", "de-at": "de", "de-ch": "de",
    "spanish": "es", "espa\u00f1ol": "es", "espanol": "es", "es-es": "es", "es-mx": "es", "es-ar": "es",
    "french": "fr", "fran\u00e7ais": "fr", "france": "fr", "fr-fr": "fr", "fr-be": "fr", "fr-ca": "fr", "fr-ch": "fr",
    "ukrainian": "uk", "ukrainisch": "uk", "\u0443\u043a\u0440\u0430\u0457\u043d\u0441\u044c\u043a\u0430": "uk", "uk-ua": "uk", "ua": "uk",
    "turkish": "tr", "t\u00fcrk\u00e7e": "tr", "tr-tr": "tr",
    # Afrikaans --- South African Dutch-derived language; "af-ZA" is the common BCP-47 tag.
    "afrikaans": "af", "af-za": "af",
    # Korean
    "korean": "ko", "\ud55c\uad6d\uc5b4": "ko", "ko-kr": "ko",
    # Italian
    "italian": "it", "italiano": "it", "it-it": "it", "it-ch": "it",
    # Irish (Gaeilge) --- ga is the BCP-47 code
    "irish": "ga", "gaeilge": "ga", "ga-ie": "ga",
    # Portuguese --- "portuguese" routes to European Portuguese;
    # "pt-br" and "brazilian" route to Brazilian Portuguese catalog.
    "portuguese": "pt", "portugu\u00eas": "pt", "portugues": "pt",
    "pt-pt": "pt", "pt-br": "pt-BR", "brazilian": "pt-BR", "brasileiro": "pt-BR",
    # Russian
    "russian": "ru", "\u0440\u0443\u0441\u0441\u043a\u0438\u0439": "ru", "ru-ru": "ru",
    # Hungarian
    "hungarian": "hu", "magyar": "hu", "hu-hu": "hu",
}

# ---------------------------------------------------------------------------
# Load all locale YAML files at import time.  Each file is parsed once and
# cached in ``_CACHES``.  A missing/exhausted locale file raises a clear
# diagnostic so maintainers spot skew before a user does.
#
# The thin-slice contract helps here: with only ~270 keys, any missing
# key in a PR is immediately visible in test coverage.
# ---------------------------------------------------------------------------