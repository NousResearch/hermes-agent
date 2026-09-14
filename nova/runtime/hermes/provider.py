"""Translating NOVA's provider vocabulary into the runtime's.

NOVA says *endpoint*, *credential variable*, *context window*. Hermes says
``custom_providers[].base_url``, ``key_env``, ``context_length``. Every word of that
translation lives here, because ``custom_providers`` is Hermes vocabulary and an ``AgentSpec``
that contained it would make the second adapter a rewrite rather than an addition.

Two properties this file is responsible for.

**No credential is ever written.** The runtime resolves ``key_env`` through
``agent/auxiliary_client.py::_custom_provider_credential`` →
``agent/secret_scope.py::get_secret``, so only the *name* reaches the config and the value is
fetched at the point of use — under multiplexing, without borrowing another profile's. That is
strictly better than ``api_key: ${VAR}``, where expansion puts the secret into the loaded
config object. NOVA therefore emits ``key_env`` and never ``api_key``.

**A named provider is addressed by its name.** A ``custom_providers`` entry called ``scripted``
is selected by ``provider: scripted`` — not by ``provider: custom``, which resolves to nothing
and fails as "No LLM provider configured". Discovered the hard way during the live run, so the
naming is done in one place instead of being each caller's problem.
"""

from __future__ import annotations

from typing import Any, Optional

from nova.spec.deployment import ProviderSpec, env_references

#: Providers the runtime resolves natively. Anything else is a ``custom_providers`` entry.
#: Listed rather than detected because the distinction is about how the runtime *routes* a
#: name, and an endpoint is not what decides that: ``bedrock`` takes a region and no URL.
NATIVE_PROVIDERS = frozenset(
    {
        "anthropic", "openai", "bedrock", "vertex", "gemini", "azure-foundry", "openrouter",
        "deepseek", "xai", "nous", "fireworks", "ollama", "router",
    }
)

#: The runtime refuses a model whose declared window is below this
#: (``agent/agent_init.py``: "below the minimum 64,000 required"). Surfaced as a NOVA-side
#: warning so it is caught at apply, not when the first task fails with a message about
#: context windows that reads as unrelated to the config someone just changed.
MINIMUM_CONTEXT_WINDOW = 64_000


def build_provider_config(provider: ProviderSpec) -> tuple[dict[str, Any], list[str]]:
    """``(config fragment, warnings)`` for one resolved provider.

    The fragment is merged into the agent's ``config.yaml``. An undeclared provider produces
    an empty fragment, which leaves the runtime's own configuration untouched — the state
    every deployment before this file existed was in.
    """
    if not provider.declared:
        return {}, []

    warnings: list[str] = []
    config: dict[str, Any] = {}

    model_section: dict[str, Any] = {}
    if provider.model:
        model_section["model"] = provider.model
    if provider.context_window is not None:
        model_section["context_length"] = provider.context_window
        if provider.context_window < MINIMUM_CONTEXT_WINDOW:
            warnings.append(
                f"context_window {provider.context_window:,} is below the "
                f"{MINIMUM_CONTEXT_WINDOW:,} the runtime requires; agents will refuse to "
                "start. Set the model's real window if the server under-reports it"
            )

    is_custom = bool(provider.provider) and provider.provider not in NATIVE_PROVIDERS
    if is_custom:
        entry: dict[str, Any] = {"name": provider.provider}
        if provider.endpoint:
            entry["base_url"] = provider.endpoint
        if provider.api_key_env:
            # The name, never the value. See the module docstring.
            entry["key_env"] = provider.api_key_env
        if provider.model:
            entry["model"] = provider.model
            entry["models"] = [provider.model]
        if provider.context_window is not None:
            entry["context_length"] = provider.context_window
        config["custom_providers"] = [entry]

        if not provider.endpoint:
            warnings.append(
                f"provider {provider.provider!r} is not one the runtime resolves natively "
                "and no endpoint is declared, so there is nowhere to send a request. Set "
                "deployment.provider.endpoint"
            )
    else:
        if provider.endpoint:
            warnings.append(
                f"provider {provider.provider!r} is resolved natively by the runtime and "
                "ignores a declared endpoint; remove it, or use a custom provider name if "
                "you meant to point at your own gateway"
            )
        if provider.api_key_env:
            # A native provider reads its own conventional variable. Naming one here would
            # look configured and change nothing, which is worse than saying so.
            warnings.append(
                f"provider {provider.provider!r} resolves its own credential; "
                f"api_key_env {provider.api_key_env!r} is recorded but the runtime will not "
                "read it. Provide the provider's own variable in <profile>/.env instead"
            )

    # The provider key the runtime routes by is the custom entry's NAME, not the literal
    # word "custom" — that resolves to nothing and fails as "No LLM provider configured".
    if provider.provider:
        model_section["provider"] = provider.provider
    if provider.region:
        model_section["region"] = provider.region
    if model_section:
        config["model"] = model_section

    return config, warnings


def required_env(provider: ProviderSpec, runtime_config: Optional[dict[str, Any]] = None) -> tuple[str, ...]:
    """Variables that must exist in the worker's environment for this agent to run.

    A native provider's ``api_key_env`` is excluded: the runtime resolves those credentials
    through its own conventional variables and never reads the name declared here, so
    demanding it would send an operator to provision a variable nothing will read — and a
    readiness report that is wrong in the safe direction still trains people to ignore it.
    ``build_provider_config`` warns about the same declaration for the same reason.
    """
    names = list(provider.required_env)
    if provider.provider in NATIVE_PROVIDERS and provider.api_key_env in names:
        names.remove(provider.api_key_env)
    for value in _strings(runtime_config or {}):
        names.extend(env_references(value, required_only=True))
    return tuple(dict.fromkeys(names))


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)
