"""Safe, actionable diagnostics for an explicitly selected CLI route.

Resolver exceptions can contain keys or endpoint credentials. Preserve only known
structured failure categories here; arbitrary exception text is not an egress format.
"""


def safe_runtime_store_error(exc):
    """Keep known policy refusal codes and fixed details, not arbitrary client text."""
    from hermes_state_runtime import RuntimeStoreError

    reasons = {'launch_credentials_unavailable', 'invalid_params', 'stale_epoch', 'not_found', 'admission_conflict',
               'queue_full', 'unknown_execution', 'stale_generation', 'admission_started', 'session_idle',
               'revision_conflict', 'permission_denied'}
    reason = exc.reason if isinstance(exc.reason, str) and exc.reason in reasons else 'runtime_store_error'
    fields = {'model.api_key', 'model.api', 'model.api_key_from', 'config_secret_ref', 'home', 'profile'}
    field = exc.field if exc.field in fields else ('provider credentials' if exc.field else None)
    details = {'removed from config.yaml since this session started',
               'changed in config.yaml since this session started',
               'its endpoint moved in the session policy without a re-bind',
               'stamped by an older gateway for another session policy',
               'the frozen config credentials belong to another home',
               'the frozen config credentials belong to another profile'}
    detail = exc.detail if exc.detail in details else None
    return RuntimeStoreError(reason, field=field, detail=detail)


def pinned_route_message(exc):
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error
    from hermes_cli.runtime_provider_custom import LocalRuntimeUnavailableError

    if isinstance(exc, LocalRuntimeUnavailableError):
        if exc.retryable:
            return "The local model server is unavailable. Start the server and try again."
        return "The local model server is unavailable. Check its configuration before trying again."
    if isinstance(exc, AuthError):
        if exc.code == 'invalid_provider':
            return "Unknown provider. Check the provider name with `hermes model` or run `hermes doctor`."
        if is_rate_limited_auth_error(exc):
            wait = exc.retry_after
            if isinstance(wait, (int, float)) and not isinstance(wait, bool) and 0 <= wait <= 31_536_000:
                return f"The selected provider's quota is exhausted; retry after {int(wait)}s or choose another provider."
            return "The selected provider's quota is exhausted; retry after its usage limit resets or choose another provider."
        if exc.code in ('upstream_blocked', 'temporarily_unavailable'):
            return "The provider is temporarily unavailable; credentials were retained. Try again later."
        if exc.relogin_required:
            return "Provider authentication expired. Run `hermes model` to re-authenticate."
        return "Provider credentials could not be resolved. Check the credentials or run `hermes model`."
    return "Requested route could not be resolved; check the model and provider credentials."
