"""Request options shared by the chat-completions and Responses builders."""


def consume_ephemeral_max_output(agent):
    """Pop the one-shot output cap so only the next request consumes it."""
    ephemeral_out = getattr(agent, "_ephemeral_max_output_tokens", None)
    if ephemeral_out is not None:
        agent._ephemeral_max_output_tokens = None
    return ephemeral_out


def message_target_model(agent):
    """Use the real MoA aggregator for message replay, not its virtual request model."""
    if agent.provider != "moa":
        return None
    aggregator = getattr(getattr(agent, "client", None), "last_aggregator_slot", None)
    return aggregator.get("model") if isinstance(aggregator, dict) else None
