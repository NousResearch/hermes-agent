"""External secret source integrations.

A secret source supplies env-var-shaped credentials at process startup, after
~/.hermes/.env has loaded. Contract: :class:`base.SecretSource`; orchestrator
(ordering, mapped-beats-bulk, first-claim-wins, provenance): :func:`registry.apply_all`.
Bundled: ``bitwarden``, ``onepassword``, ``command``. The set is deliberately
closed — third-party managers ship as plugins that subclass ``SecretSource`` and
register through ``PluginContext.register_secret_source()``.
"""

from agent.secret_sources.base import (  # noqa: F401
    SECRET_SOURCE_API_VERSION,
    ErrorKind,
    FetchResult,
    SecretSource,
    build_minimal_provider_env,
    normalize_provider_output,
    redact_provider_output,
    sanitize_provider_version,
    is_valid_env_name,
    run_secret_cli,
    scrub_ansi,
)
