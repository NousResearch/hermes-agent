"""session.create model×provider gate vs. the Copilot providers (#107391).

``static_model_provider_conflict`` runs offline on every ``session.create`` against the
curated ``_PROVIDER_MODELS`` catalogs. Two failure modes blocked every Copilot session
creation:

* ``copilot-acp``'s only static "model" is the provider-default sentinel (the ACP client
  treats the id ``copilot-acp`` as "use the session default"), so the gate read it as a
  one-model catalog and rejected every real model id — including ones the ACP subprocess
  advertises as enabled.
* ``copilot``'s static list lagged the live GitHub catalog, so newly enabled models
  (``kimi-k3``, the ``gpt-5.6``/``gpt-6`` tiers, ``grok-4.x``) were rejected as foreign.
"""

from hermes_cli.models_validate import static_model_provider_conflict

# Ids the Copilot CLI's ACP session advertised as enabled at the time of writing.
LIVE_COPILOT_MODELS = [
    "claude-sonnet-5", "claude-opus-5", "claude-opus-5.5", "gpt-5.6-luna", "gpt-5.3-codex",
    "gemini-3.8-flash", "kimi-k3", "gpt-6-luna", "gpt-6-sol", "grok-4.7",
]


def test_copilot_acp_gate_is_permissive_for_live_models():
    """copilot-acp's real catalog is discovered from the ACP subprocess at runtime; the
    offline gate must not judge it by the provider-default sentinel entry."""
    for model in LIVE_COPILOT_MODELS:
        assert static_model_provider_conflict(model, "copilot-acp") is None, model


def test_copilot_acp_gate_accepts_provider_default_sentinel():
    assert static_model_provider_conflict("copilot-acp", "copilot-acp") is None


def test_copilot_static_catalog_covers_live_models():
    """The API-key provider validates offline against ``_PROVIDER_MODELS['copilot']``;
    models the live catalog serves must not be rejected as foreign."""
    for model in LIVE_COPILOT_MODELS:
        assert static_model_provider_conflict(model, "copilot") is None, model
