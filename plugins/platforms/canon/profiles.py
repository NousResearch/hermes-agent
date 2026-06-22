"""Canon profile resolution, local registration, and setup UX."""

from __future__ import annotations

import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from plugins.platforms.canon.client import _safe_error
from plugins.platforms.canon.constants import (
    CANON_AGENTS_JSON_BOOTSTRAP_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_STREAM_URL,
    DEFAULT_TIMEOUT_SECONDS,
    REGISTRATION_POLL_INTERVAL_SECONDS,
    REGISTRATION_TIMEOUT_SECONDS,
)
from plugins.platforms.canon.models import CanonApiError, CanonResolvedAgent

def _config_extra_value(config: Any, *keys: str) -> str:
    extra = getattr(config, "extra", {}) or {}
    if not isinstance(extra, dict):
        return ""
    for key in keys:
        value = extra.get(key)
        if value:
            return str(value).strip()
    return ""


def _canon_home() -> Path:
    configured = os.getenv("CANON_HOME", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".canon"


def _canon_agents_path() -> Path:
    configured = (
        os.getenv("CANON_AGENTS_FILE", "").strip()
        or os.getenv("CANON_AGENTS_PATH", "").strip()
    )
    if configured:
        return Path(configured).expanduser()
    return _canon_home() / "agents.json"


def _canon_pending_registrations_path() -> Path:
    return _canon_home() / "pending-registrations.json"


def _load_pending_registrations() -> dict[str, dict[str, Any]]:
    path = _canon_pending_registrations_path()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {
        str(name): value
        for name, value in loaded.items()
        if isinstance(value, dict)
    }


def _write_pending_registrations(pending: dict[str, dict[str, Any]]) -> None:
    path = _canon_pending_registrations_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(pending, indent=2, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def _pending_registration(profile_name: str) -> dict[str, Any]:
    pending = _load_pending_registrations()
    entry = pending.get(profile_name)
    if not isinstance(entry, dict):
        entry = {}
    local_id = str(entry.get("localRegistrationId") or "").strip()
    if not local_id:
        local_id = f"hermes-{uuid.uuid4().hex}"
        entry["localRegistrationId"] = local_id
        pending[profile_name] = entry
        _write_pending_registrations(pending)
    return dict(entry)


def _save_pending_registration(profile_name: str, entry: dict[str, Any]) -> None:
    pending = _load_pending_registrations()
    pending[profile_name] = dict(entry)
    _write_pending_registrations(pending)


def _clear_pending_registration(profile_name: str) -> None:
    pending = _load_pending_registrations()
    if pending.pop(profile_name, None) is not None:
        _write_pending_registrations(pending)


def _parse_agent_profiles(raw: str) -> dict[str, dict[str, Any]]:
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(base64.b64decode(text).decode("utf-8"))
        except Exception as exc:
            raise ValueError(
                f"{CANON_AGENTS_JSON_BOOTSTRAP_ENV} must be agents.json JSON or base64 JSON"
            ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"{CANON_AGENTS_JSON_BOOTSTRAP_ENV} must be a JSON object keyed by profile name"
        )
    profiles: dict[str, dict[str, Any]] = {}
    for raw_name, value in parsed.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"{CANON_AGENTS_JSON_BOOTSTRAP_ENV} contains an empty profile name")
        if not isinstance(value, dict):
            raise ValueError(
                f'{CANON_AGENTS_JSON_BOOTSTRAP_ENV} profile "{name}" must be an object'
            )
        profiles[name] = value
    return profiles


def _write_agent_profiles(path: Path, profiles: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(profiles, indent=2, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)


def _load_agent_profiles() -> dict[str, dict[str, Any]]:
    path = _canon_agents_path()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        loaded = {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must be a JSON object keyed by profile name")

    profiles = {
        str(name): value
        for name, value in loaded.items()
        if isinstance(value, dict)
    }

    bootstrap = os.getenv(CANON_AGENTS_JSON_BOOTSTRAP_ENV, "")
    if bootstrap:
        changed = False
        for name, profile in _parse_agent_profiles(bootstrap).items():
            if name not in profiles:
                profiles[name] = profile
                changed = True
        if changed or not path.exists():
            _write_agent_profiles(path, profiles)

    return profiles


def _profile_slug(name: str) -> str:
    slug_chars: list[str] = []
    prev_dash = False
    for char in name.strip().lower():
        if char.isascii() and char.isalnum():
            slug_chars.append(char)
            prev_dash = False
        elif not prev_dash:
            slug_chars.append("-")
            prev_dash = True
    slug = "".join(slug_chars).strip("-")
    return slug or "hermes"


def _profile_matches_client(profile: dict[str, Any], expected: str = "hermes") -> bool:
    client_type = profile.get("clientType")
    return not client_type or str(client_type) == expected


def _resolved_from_profile(name: str, profile: dict[str, Any]) -> CanonResolvedAgent:
    if not _profile_matches_client(profile):
        raise ValueError(f'Canon profile "{name}" is not registered for Hermes')
    api_key = str(profile.get("apiKey") or "").strip()
    if not api_key:
        raise ValueError(f'Canon profile "{name}" is missing apiKey')
    return CanonResolvedAgent(
        api_key=api_key,
        profile=name,
        agent_id=str(profile.get("agentId") or "").strip() or None,
        agent_name=str(profile.get("agentName") or "").strip() or None,
        base_url=str(profile.get("baseUrl") or "").strip(),
        stream_url=str(profile.get("streamUrl") or "").strip(),
    )


def _resolve_canon_agent(config: Any, *, raise_on_error: bool = True) -> CanonResolvedAgent:
    api_key = _config_value(config, "api_key", "CANON_API_KEY", "")
    if api_key:
        return CanonResolvedAgent(api_key=api_key)

    profile_name = (
        os.getenv("CANON_AGENT", "").strip()
        or _config_extra_value(config, "agent", "profile", "canon_agent", "CANON_AGENT")
    )
    try:
        profiles = _load_agent_profiles()
        if profile_name:
            profile = profiles.get(profile_name)
            if not profile:
                raise ValueError(
                    f'Canon profile "{profile_name}" not found in {_canon_agents_path()}. '
                    f"Set {CANON_AGENTS_JSON_BOOTSTRAP_ENV}, set CANON_API_KEY, or re-run registration."
                )
            return _resolved_from_profile(profile_name, profile)

        viable = [
            (name, profile)
            for name, profile in profiles.items()
            if _profile_matches_client(profile)
        ]
        if len(viable) == 1:
            return _resolved_from_profile(viable[0][0], viable[0][1])
        if len(viable) > 1:
            raise ValueError(
                "Multiple Canon profiles are available. Set CANON_AGENT to choose one."
            )
    except Exception:
        if raise_on_error:
            raise
        return CanonResolvedAgent()
    return CanonResolvedAgent()


def _config_value(config: Any, key: str, env_name: str, default: str = "") -> str:
    env_value = os.getenv(env_name)
    if env_value:
        return env_value.strip()

    if key == "api_key":
        for attr in ("api_key", "token"):
            value = getattr(config, attr, None)
            if value:
                return str(value).strip()

    extra = getattr(config, "extra", {}) or {}
    if isinstance(extra, dict):
        for candidate in (key, env_name.lower(), env_name):
            value = extra.get(candidate)
            if value:
                return str(value).strip()
    return default


def _config_int(config: Any, key: str, env_name: str, default: int) -> int:
    raw = _config_value(config, key, env_name, "")
    if raw == "":
        return default
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return default


def _save_canon_profile(
    profile_name: str,
    *,
    api_key: str,
    agent_id: str,
    agent_name: str,
    base_url: str = "",
    stream_url: str = "",
) -> None:
    profiles = _load_agent_profiles()
    entry: dict[str, Any] = {
        "apiKey": api_key,
        "agentId": agent_id,
        "agentName": agent_name,
        "registeredAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clientType": "hermes",
    }
    if base_url and base_url.rstrip("/") != DEFAULT_BASE_URL:
        entry["baseUrl"] = base_url.rstrip("/")
    if stream_url and stream_url.rstrip("/") != DEFAULT_STREAM_URL:
        entry["streamUrl"] = stream_url.rstrip("/")
    profiles[profile_name] = entry
    _write_agent_profiles(_canon_agents_path(), profiles)


def _post_registration_request(
    *,
    base_url: str,
    name: str,
    description: str,
    owner_phone: str,
    local_registration_id: str,
    requested_agent_id: Optional[str] = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "description": description,
        "ownerPhone": owner_phone,
        "developerInfo": "Hermes gateway platform adapter",
        "clientType": "hermes",
        "localRegistrationId": local_registration_id,
    }
    if requested_agent_id:
        body["requestedAgentId"] = requested_agent_id

    res = httpx.post(
        f"{base_url.rstrip('/')}/agents/register",
        headers={"Content-Type": "application/json"},
        json=body,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    if res.status_code >= 400:
        raise CanonApiError(res.status_code, res.text)
    data = res.json()
    if not isinstance(data, dict) or not data.get("requestId"):
        raise ValueError("Canon registration did not return a requestId")
    return data


def _get_registration_status(
    *,
    base_url: str,
    request_id: str,
    poll_token: Optional[str],
) -> dict[str, Any]:
    headers = {"x-canon-poll-token": poll_token} if poll_token else None
    res = httpx.get(
        f"{base_url.rstrip('/')}/agents/status/{request_id}",
        headers=headers,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    if res.status_code >= 400:
        raise CanonApiError(res.status_code, res.text)
    data = res.json()
    if not isinstance(data, dict):
        raise ValueError("Canon registration status response was not an object")
    return data


def _ack_registration_status(
    *,
    base_url: str,
    request_id: str,
    poll_token: Optional[str],
) -> None:
    headers = {"x-canon-poll-token": poll_token} if poll_token else None
    res = httpx.post(
        f"{base_url.rstrip('/')}/agents/status/{request_id}/ack",
        headers=headers,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    if res.status_code >= 400:
        raise CanonApiError(res.status_code, res.text)


def _wait_for_registration_approval(
    *,
    base_url: str,
    request_id: str,
    poll_token: Optional[str],
    on_poll: Optional[Callable[[dict[str, Any]], None]] = None,
    timeout_seconds: float = REGISTRATION_TIMEOUT_SECONDS,
    poll_interval_seconds: float = REGISTRATION_POLL_INTERVAL_SECONDS,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        time.sleep(poll_interval_seconds)
        status = _get_registration_status(
            base_url=base_url,
            request_id=request_id,
            poll_token=poll_token,
        )
        if on_poll:
            on_poll(status)
        if status.get("status") in {"approved", "rejected"}:
            return status
    return {"status": "timeout"}


def _setup_canon() -> None:
    from hermes_cli.config import get_env_value, remove_env_value, save_env_value
    from hermes_cli.setup import (
        Colors,
        color,
        print_error,
        print_info,
        print_success,
        print_warning,
        prompt,
        prompt_choice,
        prompt_yes_no,
    )

    print()
    print(color("  --- Canon Setup ---", Colors.CYAN))
    print()
    print_info("  Canon can use an existing agent profile or register a new Hermes agent.")
    print_info("  Registration sends an approval request to the Canon owner app.")

    existing_key = get_env_value("CANON_API_KEY") or ""
    existing_agent = get_env_value("CANON_AGENT") or ""
    if existing_key or existing_agent or validate_config(None):
        print()
        if existing_agent:
            print_success(f"Canon is already configured with profile '{existing_agent}'.")
        else:
            print_success("Canon is already configured.")
        if not prompt_yes_no("  Reconfigure Canon?", False):
            return

    choices = [
        "Register/reconnect a Hermes agent with Canon (recommended)",
        "Use an existing Canon profile from ~/.canon/agents.json",
        "Paste an existing Canon API key (advanced/headless)",
    ]
    choice = prompt_choice("  How would you like to configure Canon?", choices, 0)

    if choice == 2:
        print()
        print_info("  Use this only for headless deployments that cannot use a saved profile.")
        api_key = prompt("  Canon API key", password=True)
        if not api_key:
            print_warning("  Skipped — Canon needs CANON_API_KEY or CANON_AGENT.")
            return
        save_env_value("CANON_API_KEY", api_key)
        remove_env_value("CANON_AGENT")
        print_success("  Canon API key saved.")
        return

    if choice == 1:
        try:
            profiles = _load_agent_profiles()
        except Exception as exc:
            print_error(f"  Could not read {_canon_agents_path()}: {_safe_error(exc)}")
            return
        available = [
            name
            for name, profile in sorted(profiles.items())
            if isinstance(profile, dict) and _profile_matches_client(profile)
        ]
        if available:
            default = existing_agent if existing_agent in available else available[0]
            profile_name = prompt("  Canon profile", default=default)
        else:
            print_warning("  No Hermes-compatible profiles found in ~/.canon/agents.json.")
            profile_name = prompt("  Canon profile")
        if not profile_name:
            print_warning("  Skipped — Canon needs a profile name.")
            return
        try:
            _resolved_from_profile(profile_name, profiles[profile_name])
        except KeyError:
            print_error(f"  Profile '{profile_name}' was not found in {_canon_agents_path()}.")
            return
        except Exception as exc:
            print_error(f"  Profile '{profile_name}' is not usable: {_safe_error(exc)}")
            return
        save_env_value("CANON_AGENT", profile_name)
        remove_env_value("CANON_API_KEY")
        print_success(f"  Canon profile '{profile_name}' saved.")
        return

    print()
    agent_name = prompt("  Agent display name", default="Hermes")
    if not agent_name:
        print_warning("  Skipped — Canon registration needs an agent name.")
        return
    description = prompt(
        "  Agent description",
        default="Hermes gateway agent",
    )
    owner_phone = prompt("  Canon owner phone number (E.164, exact number from the Canon app)")
    if not owner_phone:
        print_warning("  Skipped — Canon registration needs the owner's phone number.")
        return
    profile_name = prompt("  Local Canon profile name", default=_profile_slug(agent_name))
    if not profile_name:
        print_warning("  Skipped — Canon registration needs a local profile name.")
        return

    base_url = prompt(
        "  Canon API base URL",
        default=get_env_value("CANON_BASE_URL") or DEFAULT_BASE_URL,
    ).rstrip("/")

    requested_agent_id = None
    try:
        requested_agent_id = _load_agent_profiles().get(profile_name, {}).get("agentId")
    except Exception:
        requested_agent_id = None

    pending_registration = _pending_registration(profile_name)
    request_id = str(pending_registration.get("requestId") or "").strip()
    poll_token = pending_registration.get("pollToken")
    if request_id:
        print_info(f"  Reconnecting to pending registration (request ID: {request_id}).")
    else:
        local_registration_id = str(pending_registration["localRegistrationId"])
        try:
            submitted = _post_registration_request(
                base_url=base_url,
                name=agent_name,
                description=description,
                owner_phone=owner_phone,
                local_registration_id=local_registration_id,
                requested_agent_id=requested_agent_id,
            )
        except Exception as exc:
            print_error(f"  Canon registration failed: {_safe_error(exc)}")
            return

        request_id = str(submitted.get("requestId") or "")
        poll_token = submitted.get("pollToken")
        _save_pending_registration(
            profile_name,
            {
                **pending_registration,
                "requestId": request_id,
                "pollToken": poll_token,
                "baseUrl": base_url,
                "submittedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        print_success(f"  Registration submitted (request ID: {request_id}).")
    print_info("  Approve the request in the Canon app. Waiting up to 5 minutes...")

    def _poll_update(status: dict[str, Any]) -> None:
        state = status.get("status", "pending")
        if state == "pending":
            print(".", end="", flush=True)

    try:
        result = _wait_for_registration_approval(
            base_url=base_url,
            request_id=request_id,
            poll_token=str(poll_token) if poll_token else None,
            on_poll=_poll_update,
        )
    except Exception as exc:
        print()
        print_error(f"  Canon approval polling failed: {_safe_error(exc)}")
        return

    print()
    status = result.get("status")
    if status == "rejected":
        _clear_pending_registration(profile_name)
        print_warning("  Canon registration was rejected.")
        return
    if status != "approved":
        print_warning("  Canon registration timed out. Run setup again to retry.")
        return

    api_key = str(result.get("apiKey") or "").strip()
    agent_id = str(result.get("agentId") or "").strip()
    approved_name = str(result.get("agentName") or agent_name).strip()
    if not api_key or not agent_id:
        print_error("  Canon approved the request but did not return a usable API key.")
        return

    try:
        _save_canon_profile(
            profile_name,
            api_key=api_key,
            agent_id=agent_id,
            agent_name=approved_name,
            base_url=base_url,
        )
    except Exception as exc:
        print_error(f"  Could not save Canon profile: {_safe_error(exc)}")
        return

    try:
        _ack_registration_status(
            base_url=base_url,
            request_id=request_id,
            poll_token=str(poll_token) if poll_token else None,
        )
    except Exception as exc:
        print_warning(f"  Canon profile saved, but key-delivery ack failed: {_safe_error(exc)}")

    save_env_value("CANON_AGENT", profile_name)
    remove_env_value("CANON_API_KEY")
    if base_url != DEFAULT_BASE_URL:
        save_env_value("CANON_BASE_URL", base_url)
    _clear_pending_registration(profile_name)

    print_success(f"  Canon agent '{approved_name}' saved as profile '{profile_name}'.")
    print_info("  Hermes will use CANON_AGENT for the Canon platform.")




def validate_config(config) -> bool:
    try:
        return bool(_resolve_canon_agent(config).api_key)
    except Exception:
        return False
