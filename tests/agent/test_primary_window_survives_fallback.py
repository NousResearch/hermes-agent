"""A context window the session learned on its primary survives a fallback excursion.

The window is corrected mid-session (turn_overflow adopts a provider-reported limit with
``update_model``); restoring the primary after a fallback turn must bring back that window,
not the construction-time guess held by ``_primary_runtime``.
"""

from run_agent import AIAgent

# Loopback discard port: every probe is refused at once, nothing leaves the machine.
PRIMARY_URL = "http://127.0.0.1:9/primary/v1"
FALLBACK_URL = "http://127.0.0.1:9/fallback/v1"


def test_restore_after_fallback_keeps_the_primary_window_learned_mid_session():
    agent = AIAgent(
        model="primary-model", provider="custom", base_url=PRIMARY_URL, api_key="primary-key",
        quiet_mode=True, enabled_toolsets=[], skip_context_files=True, skip_memory=True,
        fallback_model=[{"provider": "custom", "model": "fallback-model",
                         "base_url": FALLBACK_URL, "api_key": "fallback-key"}],
    )
    compressor = agent.context_compressor
    learned = 65_536
    assert compressor.context_length != learned
    compressor.update_model(
        model=agent.model, context_length=learned, base_url=agent.base_url,
        api_key=agent.api_key, provider=agent.provider, api_mode=agent.api_mode,
    )

    assert agent._try_activate_fallback()
    assert agent.model == "fallback-model"
    assert agent._restore_primary_runtime()

    assert (agent.model, compressor.context_length) == ("primary-model", learned)
