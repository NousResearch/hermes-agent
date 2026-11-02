"""A provider key the active profile's secret scope does not have, said before any request.

Custom resolution substitutes the ``no-key-required`` placeholder for an empty key so keyless local
servers (Ollama, llama.cpp, vLLM) work. When the endpoint's config DECLARES a key variable
(``key_env`` / ``api_key_env``) and it resolves to nothing, the request can only be refused, and the
refusal reads "rejected your API key" although no key was sent. Under a multiplexing server a named
profile reads keys only from its own ``.env``, never the server's process environment (secret scope
is per profile by design), so a key exported in the server's shell is invisible to it. The error
names the provider, the variable, the profile and the command that puts the key where it is read.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from agent.secret_scope import get_secret_str

NO_KEY_PLACEHOLDER = "no-key-required"


def _declared_key_env(entry: Optional[dict[str, Any]]) -> str:
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("key_env") or entry.get("api_key_env") or "").strip()


def _declared_custom_key(requested_provider: str, base_url: str, model_cfg: dict[str, Any],
                         config: Optional[dict[str, Any]]) -> tuple[str, str]:
    """``(label, key_env)`` of the config block that declared a key variable for this custom
    endpoint, else ``("", "")``: the owning ``providers:`` / ``custom_providers:`` entry (by name,
    then URL), then ``model.key_env`` when ``model.base_url`` is this endpoint."""
    from agent.custom_entry_pool import custom_entry_for_endpoint
    name = requested_provider if requested_provider not in ("", "auto", "custom") else None
    match = custom_entry_for_endpoint(base_url, name, config)
    if match is not None and (key_env := _declared_key_env(match[1])):
        return f"custom provider '{match[1].get('name') or match[0]}'", key_env
    cfg_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
    if cfg_url and cfg_url == base_url.strip().rstrip("/") and (key_env := _declared_key_env(model_cfg)):
        return "the custom endpoint in config.yaml (model.key_env)", key_env
    return "", ""


def where_to_set_key(env_var: str) -> str:
    """Sentence naming the profile whose ``.env`` must hold ``env_var`` and the command that writes
    it there. Says so when the variable exists only in this process's environment while a routed
    (named) profile is served: that is the copy users hit after exporting keys in an SSH shell."""
    from agent.secret_scope import serves_routed_profile
    from hermes_constants import display_hermes_home, get_hermes_home, profile_cli_selector, profile_name_for_home
    profile = profile_name_for_home(get_hermes_home()) or "default"
    env_path = f"{display_hermes_home()}/.env"
    hint = (f"Add it to profile '{profile}': run `hermes {profile_cli_selector()}config set {env_var} <your key>` "
            f"(stored in {env_path}).")
    if serves_routed_profile() and os.environ.get(env_var):
        hint = (f"{env_var} is set in this server's environment, but profile '{profile}' reads keys only "
                f"from its own .env. {hint}")
    return hint


def named_profile_key_hint(env_vars) -> str:
    """``" <where_to_set_key>"`` for a registry provider's missing key while a NAMED profile is
    active, else ``""``: the default profile also reads the process environment, so the plain
    "Set X." already tells it everything."""
    from hermes_constants import get_hermes_home, profile_name_for_home
    names = [str(v) for v in (env_vars or ()) if str(v or "").strip()]
    if not names or profile_name_for_home(get_hermes_home()) in (None, "default"):
        return ""
    return f" {where_to_set_key(names[0])}"


def raise_if_declared_key_missing(requested_provider: Optional[str], runtime: dict[str, Any], model_cfg: dict[str, Any],
                                  config: Optional[dict[str, Any]] = None) -> None:
    """Refuse a custom runtime carrying the placeholder when its config declared a key variable
    that resolves to nothing in a ROUTED profile's scope (a profile served by a process launched
    for another, which never reads that process's environment). A single-profile process keeps
    the placeholder plus ``_key_env_secret``'s warning: there the variable is read from the
    process environment too, and callers rely on the placeholder rather than an error. A block
    that declares no key variable keeps the placeholder: that IS the keyless local-server
    configuration. ``config`` is a frozen
    route's config (the entry that owned the endpoint when the session froze it). Typed ``AuthError``
    (``missing_api_key``) so every caller's fallback chain still advances."""
    from agent.secret_scope import serves_routed_profile
    if runtime.get("provider") != "custom" or runtime.get("api_key") != NO_KEY_PLACEHOLDER:
        return
    if not serves_routed_profile():
        return
    label, key_env = _declared_custom_key(str(requested_provider or "").strip().lower(),
                                          str(runtime.get("base_url") or ""), model_cfg, config)
    if not key_env or get_secret_str(key_env, "").strip():
        return
    from hermes_cli.auth import AuthError
    raise AuthError(f"No API key for {label}: it reads {key_env}, which is not set for this profile, so the "
                    f"request was not sent. {where_to_set_key(key_env)}",
                    provider=str(requested_provider or "custom"), code="missing_api_key")
