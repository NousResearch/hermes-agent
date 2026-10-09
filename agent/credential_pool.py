"""External model-provider plugin compatibility; internal owner is auth."""
# ---- BEGIN EXTERNAL-AUTH-CONTRACT (see PHASE6_AUTH_POOL.md) ----
from auth.credential_pool import AUTH_TYPE_OAUTH, PooledCredential  # noqa: F401

def load_pool(provider):
    """Preserve the documented plugin login-handler API at the CLI boundary."""
    from auth.credential_pool import load_pool as canonical_load_pool
    from hermes_cli.config_credentials import credential_pool_environment
    return canonical_load_pool(provider, environment=credential_pool_environment())
# ---- END EXTERNAL-AUTH-CONTRACT ----
