"""``web.exa_search_type`` is a first-class config key, off by default.

Two failure modes are guarded here, both silent in the config layer:

- The key is not registered in ``DEFAULT_CONFIG``, so ``hermes config set
  web.exa_search_type lite`` writes it but warns "not a recognized config key" — the user
  cannot tell a typo from a supported setting (#34067 class).
- The default is non-empty, which would reprice and re-rank every existing Exa user's searches
  the moment they upgrade. The default is ``""``: no argument is sent at all.

The provider-side behaviour (the argument actually reaching the SDK, and the keyless tier
ignoring it) is covered by ``tests/plugins/web/test_web_search_provider_plugins.py``.
"""

from __future__ import annotations


def test_exa_search_type_is_recognized_and_off_by_default() -> None:
    from hermes_cli.config import _validate_config_key
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["web"]["exa_search_type"] == ""
    assert _validate_config_key("web.exa_search_type") == (True, None)
