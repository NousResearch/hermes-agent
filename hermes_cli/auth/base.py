"""Shared auth helpers used by every provider module (future home).

F-C2 step 1 scaffold — this module is intentionally empty. Subsequent
audit commits (F-C2 step N) will move the per-provider resolution
functions out of ``hermes_cli.auth`` (formerly ``hermes_cli/auth.py``,
now ``hermes_cli/auth/__init__.py``) into dedicated modules
(``anthropic_auth.py``, ``google_auth.py``, …), and the shared
machinery they all rely on will land here:

- ``ProviderAuth`` base class (common OAuth device-flow, API-key
  env-var resolution, cache/refresh coordination).
- ``resolve_api_key()`` / ``load_dotenv_layered()`` helpers.
- Retry / backoff primitives used across providers.

Until that migration lands, every symbol still lives in
``hermes_cli/auth/__init__.py`` and is importable at the package root
(``from hermes_cli.auth import PROVIDER_REGISTRY``, etc.) to preserve
the public surface for the ~73 call sites + test monkeypatches.
"""
