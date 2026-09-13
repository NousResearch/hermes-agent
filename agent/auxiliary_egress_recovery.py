"""Recover a denied auxiliary request only through an explicitly local route."""

from agent.llm_egress_firewall import DestinationClass, classify_destination


def local_fallback_steps(route, step_factory):
    """Try configured fallbacks, accepting only local-process or loopback routes."""
    # Resolve through the caller module so its routing/cache seams remain authoritative.
    from agent import auxiliary_client as auxiliary

    failed_provider = route.resolved_provider or "auto"
    failed_model = route.final_model
    failed_base_url = route.base_info
    sources = [
        auxiliary._try_configured_fallback_chain,
    ]
    if route.resolved_provider in {"", "auto"}:
        # Auto routes have two user-configured fallback layers. The main-agent
        # model is the final explicit-provider fallback, after the top-level
        # fallback_providers policy has been exhausted.
        sources.append(auxiliary._try_main_fallback_chain)
    sources.append(auxiliary._try_main_agent_model_fallback)
    visited: set[tuple[str, str, str, str]] = set()

    for source in sources:
        while True:
            if source is auxiliary._try_main_agent_model_fallback:
                client, model, label = source(
                    failed_provider,
                    route.task,
                    reason="egress blocked",
                    failed_model=failed_model,
                    failed_base_url=failed_base_url,
                    main_runtime=route.main_runtime,
                    excluded_identities=visited,
                    async_mode=route.async_mode,
                )
            elif source is auxiliary._try_main_fallback_chain:
                client, model, label = source(
                    route.task,
                    failed_provider,
                    reason="egress blocked",
                    failed_model=failed_model,
                    failed_base_url=failed_base_url,
                    excluded_identities=visited,
                    async_mode=route.async_mode,
                )
            else:
                client, model, label = source(
                    route.task,
                    failed_provider,
                    reason="egress blocked",
                    failed_model=failed_model,
                    failed_base_url=failed_base_url,
                    excluded_identities=visited,
                    async_mode=route.async_mode,
                )
            if client is None:
                break

            destination = auxiliary._fallback_destination(
                route.task, client, model, label
            )
            provider = destination.provider or auxiliary._fallback_provider_from_label(label)
            base_url = destination.base_url or str(getattr(client, "base_url", "") or "")
            api_mode = destination.api_mode or getattr(client, "api_mode", None)
            identity = (provider, model or destination.model or "", base_url, api_mode or "")
            if identity in visited:
                # A selector that ignores the failed identity made no
                # progress. Stop this layer instead of cycling forever.
                break
            visited.add(identity)
            classification = classify_destination(provider, base_url, api_mode)
            if classification in {DestinationClass.LOCAL_PROCESS, DestinationClass.LOOPBACK}:
                auxiliary._record_route_info(route.route_info, provider, model)
                response = yield step_factory("fallback", (client, model, label))
                if response is not None:
                    return response
                # A local candidate may have been quarantined by the call
                # step. Feed its complete identity back so the selector can
                # continue to the next healthy local candidate.
                failed_provider = provider
                failed_model = model or destination.model
                failed_base_url = base_url
                continue

            # _try_* returns the first usable candidate. Feed its complete
            # identity back as the failed route so the next call scans past
            # that remote candidate rather than returning it repeatedly.
            failed_provider = provider
            failed_model = model or destination.model
            failed_base_url = base_url
    return None



def stream_with_local_recovery(req, retry_kwargs, candidate_kwargs, *, task,
                               stream_options=None, route_info=None):
    """Recover a denied stream before returning it to its reassembly owner."""
    from agent import auxiliary_client as auxiliary
    from agent.llm_egress_firewall import EgressBlocked

    def send(client, kwargs, provider, api_mode):
        kwargs = dict(kwargs, stream=True)
        if stream_options:
            kwargs["stream_options"] = stream_options
        if task == "moa_aggregator" and isinstance(client, auxiliary.CodexAuxiliaryClient):
            return client.chat.completions.create(**kwargs)
        return auxiliary._relay_sync_stream(client, kwargs, provider=provider, api_mode=api_mode)

    try:
        return send(req.client, req.kwargs, req.request_provider, req.resolved_api_mode)
    except EgressBlocked as first_err:
        def perform(step):
            client, model, label = step.args
            destination, kwargs, _ = auxiliary._plan_fallback_candidate(
                client, model, label, apply_fast_lane=True, **candidate_kwargs)
            return send(client, kwargs, destination.provider, destination.api_mode)

        result = auxiliary._drive_ladder(
            auxiliary._start_recovery_ladder(
                first_err, req, retry_kwargs, task=task, async_mode=False, route_info=route_info),
            perform,
        )
        if result is auxiliary._RERAISE_ORIGINAL:
            raise
        return result
