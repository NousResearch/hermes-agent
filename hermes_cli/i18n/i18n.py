"""
Internationalization (i18n) support for Hermes Agent CLI.
"""

import os
import json
from typing import Dict, Optional, Any

DEFAULT_LOCALE = "en"
SUPPORTED_LOCALES = ["en", "zh"]

# Get user's locale from environment or default to en
def get_user_locale() -> str:
    locale = os.environ.get("LANG", "").split(".")[0].split("_")[0].lower()
    if locale in SUPPORTED_LOCALES:
        return locale
    return DEFAULT_LOCALE

# Load translations for a given locale
def load_translations(locale: str = DEFAULT_LOCALE) -> Dict[str, Any]:
    translations = {}
    
    # Load base translations
    base_file = os.path.join(os.path.dirname(__file__), f"{DEFAULT_LOCALE}.json")
    if os.path.exists(base_file):
        with open(base_file, 'r', encoding='utf-8') as f:
            translations = json.load(f)
    
    # Load locale-specific translations
    if locale != DEFAULT_LOCALE:
        locale_file = os.path.join(os.path.dirname(__file__), f"{locale}.json")
        if os.path.exists(locale_file):
            with open(locale_file, 'r', encoding='utf-8') as f:
                locale_translations = json.load(f)
                # Merge locale translations into base
                _deep_merge(translations, locale_translations)
    
    return translations

def _deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Deep merge source into target."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value

# Global translations instance
_current_locale = get_user_locale()
_translations = load_translations(_current_locale)

def set_locale(locale: str) -> bool:
    """Set the current locale."""
    global _current_locale, _translations
    if locale in SUPPORTED_LOCALES:
        _current_locale = locale
        _translations = load_translations(locale)
        return True
    return False

def get_locale() -> str:
    """Get the current locale."""
    return _current_locale

def t(key: str, **kwargs) -> str:
    """Translate a key with optional formatting."""
    # Split the key into parts
    parts = key.split('.')
    value = _translations
    
    # Traverse the translation dictionary
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            # Key not found, return the key itself
            return key
    
    # Format the value if it's a string and kwargs are provided
    if isinstance(value, str) and kwargs:
        try:
            return value.format(**kwargs)
        except KeyError:
            pass
    
    return value if isinstance(value, str) else key
