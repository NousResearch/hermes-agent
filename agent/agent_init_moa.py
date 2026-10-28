"""MoA initialization for the AIAgent facade."""


def init_moa_client(agent, api_key):
    """provider == "moa": virtual Mixture-of-Agents facade, no real HTTP client."""
    from agent.moa_loop import bind_moa_runtime
    # build_moa_facade relays "moa.*" events through tool_progress_callback so every surface
    # shows each reference's answer before the aggregator acts. Display-only; shared with
    # fallback-restore so a restored facade keeps emitting.
    # build_moa_facade wires the reference relay that routes reference-model outputs to the agent's
    # tool_progress_callback so every surface that already consumes it (CLI spinner/scrollback, TUI,
    # desktop, gateway) can show each reference's answer as a labelled block before the aggregator acts. The
    # facade emits "moa.reference", "moa.progress", "moa.phase", and "moa.aggregating" events, forwarded
    # through the same callback the tool lifecycle uses. Best-effort and cache-safe — display-only events,
    # they never touch the message history. See #53802.
    bind_moa_runtime(agent, agent.model, api_key)
    if not agent.quiet_mode:
        print(f"🤖 AI Agent initialized with MoA preset: {agent.model}")
