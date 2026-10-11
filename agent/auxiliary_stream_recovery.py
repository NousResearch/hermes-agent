"""Recover failed auxiliary stream creation before any output reaches the caller."""

from typing import Any


def create_stream_with_pool_recovery(
    request: Any, retry_context: dict[str, Any]
) -> Any:
    from agent import auxiliary_client as aux

    client, kwargs = request.client, request.kwargs
    attempted_keys: set[str] = set()
    while True:
        try:
            return aux._relay_sync_stream(
                client,
                kwargs,
                provider=request.request_provider,
                api_mode=request.resolved_api_mode,
            )
        except Exception as exc:
            provider = aux._recoverable_pool_provider(
                request.resolved_provider,
                client,
                main_runtime=retry_context["main_runtime"],
            )
            if not provider or not aux._credential_rung_accepts(exc):
                raise
            failed_key = str(getattr(client, "api_key", "") or "")
            entries = aux.load_pool(provider).entries()
            # An explicit standalone key is not permission to consume a different account.
            if (
                failed_key in attempted_keys
                or len(attempted_keys) >= len(entries)
                or not any(entry.runtime_api_key == failed_key for entry in entries)
            ):
                raise
            attempted_keys.add(failed_key)
            if not aux._recover_provider_pool(provider, exc, failed_api_key=failed_key):
                raise
            replacement = aux.load_pool(provider).select()
            if replacement is None or replacement.runtime_api_key in attempted_keys:
                raise
            context: dict[str, Any] = dict(retry_context)
            context["resolved_api_key"] = replacement.runtime_api_key
            client, rebuilt = aux._prepare_same_provider_retry(
                resolved_provider=request.resolved_provider,
                resolved_model=request.resolved_model,
                async_mode=False,
                **context,
            )
            # The ordinary retry builder creates a complete-response request. Preserve the
            # streaming wire and attribution while replacing only the failed credential.
            rebuilt["stream"] = True
            if "stream_options" in kwargs:
                rebuilt["stream_options"] = kwargs["stream_options"]
            kwargs = rebuilt
