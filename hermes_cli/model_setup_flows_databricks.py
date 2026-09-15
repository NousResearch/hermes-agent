"""Databricks Unity Gateway setup for ``hermes setup`` and ``hermes model``."""

from __future__ import annotations

import copy
import json
import subprocess
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
from urllib.parse import urlparse

from hermes_cli.urllib_security import open_credentialed_url

_CLI_TIMEOUT_SECONDS = 20
_CHAT_API_TYPE = "mlflow/v1/chat/completions"
_EXCLUDED_MODELS = frozenset({"system.ai.gpt-oss-20b", "system.ai.gpt-oss-120b"})
_MAX_DISCOVERY_PAGES = 10


class _GatewayRequestError(RuntimeError):
    def __init__(self, status: int | None = None) -> None:
        self.status = status


def _is_transient_status(status: int | None) -> bool:
    return status is None or status in {408, 429, 499} or (
        isinstance(status, int) and 500 <= status <= 599
    )


def _urlopen_json(request: urllib.request.Request, *, retries: int = 1) -> dict[str, Any]:
    for attempt in range(retries):
        try:
            with open_credentialed_url(request, timeout=30) as response:
                payload = json.loads(response.read())
            if isinstance(payload, dict):
                return payload
            raise _GatewayRequestError
        except urllib.error.HTTPError as exc:
            if _is_transient_status(exc.code) and attempt + 1 < retries:
                time.sleep(0.25 * (2 ** attempt))
                continue
            raise _GatewayRequestError(exc.code) from None
        except (OSError, TimeoutError, ValueError):
            if attempt + 1 < retries:
                time.sleep(0.25 * (2 ** attempt))
                continue
            raise _GatewayRequestError from None
    raise _GatewayRequestError


def valid_u2m_profiles(payload: Any) -> list[dict[str, str]]:
    """Valid Databricks CLI U2M profiles with structurally safe HTTPS hosts."""
    rows = payload.get("profiles") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    profiles: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("valid") is not True or row.get("auth_type") != "databricks-cli":
            continue
        name, host = row.get("name"), row.get("host")
        if not isinstance(name, str) or not name or not isinstance(host, str):
            continue
        try:
            parsed = urlparse(host)
            valid_origin = bool(parsed.hostname) and (parsed.port is None or parsed.port > 0)
        except ValueError:
            continue
        if (
            parsed.scheme != "https" or not valid_origin or parsed.username or parsed.password
            or parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment
        ):
            continue
        profiles.append({"name": name, "host": host.rstrip("/")})
    return profiles


def _run_databricks_json(argv: list[str]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        completed = subprocess.run(
            argv, shell=False, capture_output=True, text=True, timeout=_CLI_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return None, "Databricks CLI was not found. Install and authenticate it first."
    except subprocess.TimeoutExpired:
        return None, "Databricks CLI timed out."
    except OSError:
        return None, "Databricks CLI could not be executed."
    if completed.returncode != 0:
        return None, "Databricks CLI authentication failed. Run `databricks auth login` and try again."
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError):
        return None, "Databricks CLI returned invalid JSON."
    return (payload, None) if isinstance(payload, dict) else (None, "Databricks CLI returned invalid JSON.")


def load_u2m_profiles() -> tuple[list[dict[str, str]], str | None]:
    payload, error = _run_databricks_json(
        ["databricks", "auth", "profiles", "--output", "json"]
    )
    return (valid_u2m_profiles(payload), None) if payload is not None else ([], error)


def token_argv(profile: str) -> list[str]:
    return ["databricks", "auth", "token", "--profile", profile, "--output", "json"]


def preflight_profile(profile: str) -> tuple[str | None, str | None]:
    payload, error = _run_databricks_json(token_argv(profile))
    if payload is None:
        return None, error
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        return None, "Databricks CLI returned no access token."
    return token, None


def discover_models(host: str, token: str) -> tuple[list[str], str | None]:
    rows: list[Any] = []
    page_token = ""
    seen_tokens: set[str] = set()
    page_count = 0
    while True:
        page_count += 1
        if page_count > _MAX_DISCOVERY_PAGES:
            return [], "Databricks model-service discovery exceeded its page limit."
        params = {"parent": "schemas/system.ai", "page_size": "100"}
        if page_token:
            params["page_token"] = page_token
        request = urllib.request.Request(
            f"{host.rstrip('/')}/api/2.1/unity-catalog/model-services?{urllib.parse.urlencode(params)}",
            headers={"Authorization": f"Bearer {token}"},
        )
        try:
            payload = _urlopen_json(request, retries=3)
        except _GatewayRequestError:
            return [], "Databricks model-service discovery failed."
        page_rows = payload.get("model_services")
        if not isinstance(page_rows, list):
            return [], "Databricks model-service discovery returned invalid data."
        rows.extend(page_rows)
        next_token = payload.get("next_page_token")
        if not isinstance(next_token, str) or not next_token:
            break
        if next_token in seen_tokens:
            return [], "Databricks model-service discovery returned a repeated page token."
        seen_tokens.add(next_token)
        page_token = next_token
    models: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        api_types = row.get("supported_api_types")
        name = row.get("name")
        if (
            not isinstance(api_types, list) or _CHAT_API_TYPE not in api_types
            or not isinstance(name, str) or not name.startswith("model-services/")
        ):
            continue
        model = name.removeprefix("model-services/").strip()
        if model and model not in _EXCLUDED_MODELS:
            models.append(model)
    return list(dict.fromkeys(models)), None


def _api_mode_for_model(model: str) -> str:
    """Resolve Databricks' route from its provider profile."""
    import providers as provider_registry
    profile = getattr(provider_registry, "get_provider_profile")("databricks")
    return profile.resolve_api_mode(model, "chat_completions") if profile else "chat_completions"


def _base_url_for_model(host: str, model: str) -> str:
    """Resolve the selected service's native or unified Databricks route."""
    import providers as provider_registry
    configured = f"{host.rstrip('/')}/ai-gateway/mlflow/v1"
    profile = getattr(provider_registry, "get_provider_profile")("databricks")
    return profile.resolve_base_url(model, configured) if profile else configured


def preflight_model(host: str, token: str, model: str) -> tuple[bool, str | None]:
    """Verify the selected service can produce a forced function call."""
    function_name = "hermes_setup_probe"
    api_mode = _api_mode_for_model(model)
    base_url = _base_url_for_model(host, model)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if api_mode == "codex_responses":
        payload = {
            "model": model,
            "input": "Call the provided function.",
            "tools": [{
                "type": "function",
                "name": function_name,
                "description": "Return setup compatibility status.",
                "parameters": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
                "strict": True,
            }],
            "tool_choice": {"type": "function", "name": function_name},
            "max_output_tokens": 512,
            "stream": False,
        }
        url = f"{base_url}/responses"
    elif api_mode == "anthropic_messages":
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Call the provided function."}],
            "tools": [{
                "name": function_name,
                "description": "Return setup compatibility status.",
                "input_schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            }],
            "tool_choice": {"type": "tool", "name": function_name},
            "max_tokens": 512,
        }
        url = f"{base_url}/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
    elif base_url.rstrip("/").endswith("/ai-gateway/gemini/v1beta"):
        payload = {
            "contents": [{
                "role": "user", "parts": [{"text": "Call the provided function."}],
            }],
            "tools": [{"functionDeclarations": [{
                "name": function_name,
                "description": "Return setup compatibility status.",
                "parameters": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                },
            }]}],
            "toolConfig": {"functionCallingConfig": {
                "mode": "ANY", "allowedFunctionNames": [function_name],
            }},
            "generationConfig": {"maxOutputTokens": 512},
        }
        encoded_model = urllib.parse.quote(model, safe="")
        url = f"{base_url}/models/{encoded_model}:generateContent"
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Call the provided function."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": function_name,
                    "description": "Return setup compatibility status.",
                    "parameters": {
                        "type": "object",
                        "properties": {"status": {"type": "string"}},
                        "required": ["status"],
                        "additionalProperties": False,
                    },
                },
            }],
            "tool_choice": {"type": "function", "function": {"name": function_name}},
            "max_tokens": 512,
            "stream": False,
        }
        url = f"{base_url}/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        result = _urlopen_json(request, retries=3)
    except _GatewayRequestError as exc:
        if _is_transient_status(exc.status):
            raise
        return False, "Selected model service failed the tool preflight."
    if api_mode == "codex_responses":
        output = result.get("output", []) if isinstance(result, dict) else []
        called_name = next((item.get("name", "") for item in output
                            if isinstance(item, dict) and item.get("type") == "function_call"), "")
    elif api_mode == "anthropic_messages":
        content = result.get("content", []) if isinstance(result, dict) else []
        called_name = next((item.get("name", "") for item in content
                            if isinstance(item, dict) and item.get("type") == "tool_use"), "")
    elif base_url.rstrip("/").endswith("/ai-gateway/gemini/v1beta"):
        try:
            parts = result["candidates"][0]["content"]["parts"]
            called_name = next((item["functionCall"]["name"] for item in parts
                                if isinstance(item, dict) and isinstance(item.get("functionCall"), dict)), "")
        except (KeyError, IndexError, TypeError):
            called_name = ""
    else:
        try:
            tool_calls = result["choices"][0]["message"]["tool_calls"]
            called_name = tool_calls[0]["function"]["name"]
        except (KeyError, IndexError, TypeError):
            called_name = ""
    if called_name != function_name:
        return False, "Selected model service did not return the required tool call."
    return True, None


def ensure_databricks_model_verified(
    *, requested_provider: str, model: str, credential: Any,
) -> tuple[bool, str | None]:
    """Preflight one hot-switched model and persist only its compatibility receipt."""
    if str(requested_provider or "").strip().lower() != "custom:databricks":
        return True, None
    from agent.command_token_source import materialize_probe_api_key
    from hermes_cli.config import load_config, save_config
    from hermes_cli.model_switch import _declared_model_ids

    config = load_config()
    providers = config.get("providers") if isinstance(config, dict) else None
    if not isinstance(providers, dict):
        return False, "Databricks provider configuration is unavailable."
    provider_key = next((
        key for key, entry in providers.items()
        if isinstance(entry, dict)
        and str(entry.get("provider") or "").strip().lower() == "custom:databricks"
    ), None)
    if provider_key is None:
        return False, "Databricks provider configuration is unavailable."
    provider_config = providers[provider_key]
    verified_models = _declared_model_ids(provider_config.get("models"))
    if model.lower() in {item.lower() for item in verified_models}:
        return True, None

    configured_base = str(provider_config.get("base_url") or "").strip()
    try:
        parsed = urlparse(configured_base)
        valid_origin = (
            parsed.scheme == "https" and bool(parsed.hostname)
            and not parsed.username and not parsed.password
        )
    except ValueError:
        valid_origin = False
        parsed = None
    token = materialize_probe_api_key(credential)
    if not valid_origin or not token or parsed is None:
        return False, "Databricks model compatibility could not be verified."
    host = f"{parsed.scheme}://{parsed.netloc}"
    try:
        model_ok, _ = preflight_model(host, token, model)
    except _GatewayRequestError:
        return False, "Databricks model compatibility is temporarily unavailable."
    if not model_ok:
        return False, "Selected Databricks model is not compatible with Hermes tools."

    provider_config["models"] = [*verified_models, model]
    try:
        save_config(config)
    except Exception:
        return False, "Databricks compatibility receipt could not be saved."
    return True, None


def build_databricks_config(
    config: dict[str, Any],
    *,
    profile: str,
    host: str,
    model: str,
    catalog_models: list[str] | None = None,
) -> dict[str, Any]:
    """Return a secret-free one-provider config without mutating *config*."""
    result = copy.deepcopy(config)
    model_config = result.get("model")
    if not isinstance(model_config, dict):
        model_config = {}
        result["model"] = model_config
    from hermes_cli.config import clear_model_endpoint_credentials
    clear_model_endpoint_credentials(model_config, clear_api_mode=True)
    model_config.pop("base_url", None)
    model_config["default"] = model
    model_config["provider"] = "custom:databricks"
    providers = result.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        result["providers"] = providers
    configured_base = f"{host.rstrip('/')}/ai-gateway/mlflow/v1"
    credential_argv = token_argv(profile)
    available_models: list[str] = []
    existing = providers.get("databricks")
    if (
        isinstance(existing, dict)
        and str(existing.get("base_url") or "").rstrip("/") == configured_base
        and existing.get("key_cmd") == credential_argv
        and isinstance(existing.get("models"), list)
    ):
        available_models.extend(
            item.strip() for item in existing["models"]
            if isinstance(item, str)
            and item.strip()
            and item.strip() not in _EXCLUDED_MODELS
        )
    available_models = list(dict.fromkeys(available_models))
    if model not in available_models:
        available_models.append(model)
    provider_config = {
        "provider": "custom:databricks",
        "name": "Databricks Unity Gateway",
        "base_url": configured_base,
        "transport": _api_mode_for_model(model),
        "default_model": model,
        "models": available_models,
        "discover_models": False,
        "key_cmd": credential_argv,
    }
    if catalog_models is not None:
        provider_config["catalog_models"] = list(dict.fromkeys(
            item.strip() for item in catalog_models
            if isinstance(item, str)
            and item.strip()
            and item.strip() not in _EXCLUDED_MODELS
        ))
    providers["databricks"] = provider_config
    return result


def _model_flow_databricks(config: dict[str, Any], current_model: str = "") -> None:
    from hermes_cli.auth import _prompt_model_selection, deactivate_provider
    from hermes_cli.config import save_config

    profiles, profile_error = load_u2m_profiles()
    if profile_error or not profiles:
        print(profile_error or "No valid Databricks CLI OAuth profiles found.")
        return
    from hermes_cli.model_setup_flows_common import _curses_choice
    profile_rows = [f"{profile['name']} ({profile['host']})" for profile in profiles]
    index = _curses_choice("Select a Databricks CLI OAuth profile:", profile_rows, 0)
    if index is None or index < 0:
        print("No change.")
        return
    selected_profile = profiles[index]
    token, token_error = preflight_profile(selected_profile["name"])
    if token_error or not token:
        print(token_error or "Databricks CLI returned no access token.")
        return
    models, discovery_error = discover_models(selected_profile["host"], token)
    if discovery_error or not models:
        print(discovery_error or "No compatible Databricks model services found.")
        return
    remaining_models = list(models)
    selected_model = ""
    while remaining_models:
        selected_model = _prompt_model_selection(
            remaining_models, current_model=current_model
        )
        if not selected_model:
            print("No change.")
            return
        if selected_model in _EXCLUDED_MODELS:
            print("GPT-OSS model services are not supported by this integration yet.")
            current_model = ""
            continue
        try:
            model_ok, model_error = preflight_model(
                selected_profile["host"], token, selected_model
            )
        except _GatewayRequestError:
            print("Databricks model-service preflight is temporarily unavailable. Try again later.")
            return
        if model_ok:
            break
        print(model_error or "Selected model service is not compatible with Hermes tools.")
        remaining_models = [model for model in remaining_models if model != selected_model]
        current_model = ""
    if not selected_model or not remaining_models:
        print("No compatible Databricks model services remain.")
        return
    save_config(build_databricks_config(
        config,
        profile=selected_profile["name"],
        host=selected_profile["host"],
        model=selected_model,
        catalog_models=remaining_models,
    ))
    deactivate_provider()
    print(f"Default model set to: {selected_model} (via Databricks Unity Gateway)")
