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

    for source in sources:
        while True:
            if source is auxiliary._try_main_agent_model_fallback:
                client, model, label = source(
                    failed_provider,
                    route.task,
                    reason="egress blocked",
                    failed_model=failed_model,
                    failed_base_url=failed_base_url,
                )
            elif source is auxiliary._try_main_fallback_chain:
                client, model, label = source(
                    route.task,
                    failed_provider,
                    reason="egress blocked",
                    failed_model=failed_model,
                    failed_base_url=failed_base_url,
                )
            else:
                client, model, label = source(
                    route.task,
                    failed_provider,
                    reason="egress blocked",
                    failed_model=failed_model,
                    failed_base_url=failed_base_url,
                )
            if client is None:
                break

            destination = auxiliary._fallback_destination(
                route.task, client, model, label
            )
            provider = destination.provider or auxiliary._fallback_provider_from_label(label)
            base_url = destination.base_url or str(getattr(client, "base_url", "") or "")
            api_mode = destination.api_mode or getattr(client, "api_mode", None)
            classification = classify_destination(provider, base_url, api_mode)
            if classification in {DestinationClass.LOCAL_PROCESS, DestinationClass.LOOPBACK}:
                auxiliary._record_route_info(route.route_info, provider, model)
                response = yield step_factory("fallback", (client, model, label))
                if response is not None:
                    return response
                break

            # _try_* returns the first usable candidate. Feed its complete
            # identity back as the failed route so the next call scans past
            # that remote candidate rather than returning it repeatedly.
            failed_provider = provider
            failed_model = model or destination.model
            failed_base_url = base_url
    return None
