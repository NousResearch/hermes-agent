"""Supported languages for the Hermes CLI setup wizard.

Each entry is a tuple of (locale_code, display_label).
The display label is shown as-is in the language picker.
Order matters: the first entry is the default selection.

To add a new language:
  1. Create hermes_cli/i18n/<locale>/LC_MESSAGES/setup.po
  2. Add an entry to LANGUAGES below
  3. That's it — no code changes needed.

Select the locale code from IANA subtag registry:
  https://www.iana.org/assignments/language-subtag-registry/language-subtag-registry
"""

# Ordered so most-requested / highest-population languages appear first.
LANGUAGES = [
    ("en",    "English"),
    ("zh_CN", "简体中文 (Chinese - Simplified)"),
    ("ja_JP", "日本語 (Japanese)"),
    ("es_ES", "Español (Spanish)"),
    ("fr_FR", "Français (French)"),
    ("de_DE", "Deutsch (German)"),
]
