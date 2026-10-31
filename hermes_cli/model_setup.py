"""Public helpers for a provider plugin's own ``hermes model`` step (``ProviderProfile.model_setup_flow``).

Documented in website/docs/developer-guide/model-provider-plugin.md: these are the contract a plugin
flow calls instead of re-implementing core's config rules.
"""

from __future__ import annotations


def persist_model_selection(model: str, provider: str, *, base_url: str | None = None,
                            api_mode: str | None = None) -> dict:
    """Save *model* on *provider* as the default: writes the ``model:`` section (``provider``, then
    ``base_url`` and ``api_mode`` when given), scrubs inline endpoint credentials from it, saves
    config.yaml and deactivates the previously active OAuth provider. Returns the saved section."""
    from hermes_cli.model_setup_flows_common import _persist_model

    return _persist_model(model, provider, base_url=base_url, api_mode=api_mode)


def note_setup_failure(failure_class: str) -> None:
    """Record why the flow ends without a pick (``"auth"``, ``"no_models"``, ...) for the opt-in
    provider-setup metric; a no-op outside a tracked ``hermes model`` / setup flow."""
    from hermes_cli.model_setup_flows_common import _note_setup_failure

    _note_setup_failure(failure_class)
