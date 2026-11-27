"""Google model-provider plugin: registers both the ``gemini`` (API key) and ``solstice`` (OAuth)
profiles from one package.

Both profiles are registered directly from ``__init__.py`` (not from submodules) because
``providers.bundled_profiles()`` re-executes only the package ``__init__.py`` on later passes while
submodules stay cached in ``sys.modules``. ``gemini`` registers before importing ``account`` so an
import error on the account side cannot prevent ``gemini`` from registering, and transport modules
stay lazy (never imported here).
"""

from __future__ import annotations

from providers import register_provider

from .api_key import gemini

register_provider(gemini)

from .account import solstice

register_provider(solstice)
