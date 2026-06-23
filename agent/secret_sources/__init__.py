"""External secret source integrations.

A secret source is anything that can supply environment-variable-shaped
credentials at process startup, _after_ ~/.hermes/.env has loaded.  By
default sources are non-destructive: they only set values for env vars
that aren't already present, so .env and shell exports continue to win.

Currently shipped:

  - ``bitwarden`` — Bitwarden Secrets Manager (`bws` CLI).  See
    ``agent.secret_sources.bitwarden`` for the integration and
    ``hermes_cli.secrets_cli`` for the user-facing setup wizard.
  - ``onepassword`` — 1Password CLI (`op`) reference resolver.  See
    ``agent.secret_sources.onepassword``; it resolves ``op://`` links into
    process-local environment variables without writing secret values to disk.
  - ``keychain`` — macOS Keychain generic-password resolver.  See
    ``agent.secret_sources.keychain``; it reads machine-local Keychain items
    into process-local environment variables without putting values in
    ``~/.hermes/.env`` or wrapper scripts.
"""
