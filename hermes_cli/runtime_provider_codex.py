"""Native Codex runtime without a mutable credential pool."""


def resolve_readonly_codex_runtime(
    requested_provider: str, explicit_api_key: str | None, explicit_base_url: str | None,
) -> dict:
    from hermes_cli import runtime_provider as rp

    if explicit_api_key:
        return rp._runtime("openai-codex", "codex_responses",
                           explicit_base_url or rp.DEFAULT_CODEX_BASE_URL, explicit_api_key,
                           source="explicit", requested_provider=requested_provider)
    creds = rp.resolve_codex_runtime_credentials(read_only=True)
    return rp._runtime("openai-codex", "codex_responses",
                       explicit_base_url or creds["base_url"], creds["api_key"],
                       source=creds["source"], last_refresh=creds.get("last_refresh"),
                       requested_provider=requested_provider)
