"""Recover a denied auxiliary request only through an explicitly local route."""
from agent.llm_egress_firewall import DestinationClass, classify_destination


def local_fallback_steps(route, step_factory):
    # Resolve through the caller module so its routing/cache seams remain authoritative.
    from agent import auxiliary_client as auxiliary

    candidates = (
        lambda: auxiliary._try_configured_fallback_chain(
            route.task, route.resolved_provider or "auto", reason="egress blocked",
            failed_model=route.final_model,
        ),
        lambda: auxiliary._try_main_agent_model_fallback(
            route.resolved_provider, route.task, reason="egress blocked",
            failed_model=route.final_model,
        ),
    )
    for candidate in candidates:
        client, model, label = candidate()
        if client is None:
            continue
        provider = auxiliary._fallback_provider_from_label(label)
        destination = classify_destination(
            provider, str(getattr(client, "base_url", "") or ""), "chat_completions",
        )
        if destination not in {DestinationClass.LOCAL_PROCESS, DestinationClass.LOOPBACK}:
            continue
        auxiliary._record_route_info(route.route_info, provider, model)
        response = yield step_factory("fallback", (client, model, label))
        if response is not None:
            return response
    return None
