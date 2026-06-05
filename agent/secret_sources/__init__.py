"""External secret source integrations.

A secret source is anything that can supply environment-variable-shaped
credentials at process startup, _after_ ~/.hermes/.env has loaded.  By
default sources are non-destructive: they only set values for env vars
that aren't already present, so .env and shell exports continue to win.

Currently shipped:

  - ``bitwarden`` — Bitwarden Secrets Manager (`bws` CLI).  See
    ``agent.secret_sources.bitwarden`` for the integration and
    ``hermes_cli.secrets_cli`` for the user-facing setup wizard.

  - ``onepassword`` — 1Password Service Account (``onepassword-sdk``).
    See ``agent.secret_sources.onepassword`` for the integration and
    ``hermes_cli.secrets_cli`` for the user-facing setup wizard.
    Uses the Python SDK instead of the ``op`` CLI daemon.
"""
