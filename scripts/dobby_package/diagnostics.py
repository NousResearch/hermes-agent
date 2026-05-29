"""Pure local diagnostics for the Dobby package skeleton.

The renderer intentionally does not inspect process environment variables,
network services, shells, or Discord state. Callers pass an env mapping or an
explicit env file path, plus already-parsed config dictionaries.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping


SECRET_ENV_KEYS = frozenset({"OPENAI_API_KEY", "DISCORD_BOT_TOKEN", "WEBHOOK_SECRET"})
BROAD_VALUES = frozenset({"*", "all", "any", "everyone", "@everyone", "public"})
BROAD_INTEGRATION_ENV_KEYS = frozenset(
    {
        "GITHUB_TOKEN",
        "HONCHO_API_KEY",
        "MATRIX_ACCESS_TOKEN",
        "NOTION_TOKEN",
        "SLACK_APP_TOKEN",
        "SLACK_BOT_TOKEN",
        "TELEGRAM_BOT_TOKEN",
    }
)

CHECK_ORDER = (
    "model",
    "discord_allowlist",
    "webhook_policy",
    "browser_automation",
    "broad_integrations",
    "memory_boundary",
    "redaction",
    "hermes_home",
)

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bwhsec_[A-Za-z0-9_=-]{20,}\b"),
)


def load_env_file(env_path: str | Path) -> dict[str, str]:
    """Parse a caller-supplied dotenv file without touching real .env files."""

    values: dict[str, str] = {}
    for raw_line in Path(env_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _strip_quotes(value.strip())
    return values


def render_diagnostics(
    package_root: str | Path,
    *,
    env: Mapping[str, Any] | None = None,
    env_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    tool_policy: Mapping[str, Any] | None = None,
    tool_output: Any | None = None,
) -> dict[str, Any]:
    """Return structured local diagnostics and a concise redacted report."""

    package_root_path = Path(package_root)
    env_values: dict[str, str] = {}
    env_source = "empty"
    if env_path is not None:
        env_values.update(load_env_file(env_path))
        env_source = "explicit_env_path"
    if env is not None:
        env_values.update({str(key): "" if value is None else str(value) for key, value in env.items()})
        env_source = "mapping" if env_source == "empty" else "explicit_env_path+mapping"

    config_data = dict(config or {})
    tool_policy_data = dict(tool_policy or {})

    checks = {
        "model": _check_model(env_values, config_data),
        "discord_allowlist": _check_discord_allowlist(env_values, config_data),
        "webhook_policy": _check_webhook_policy(env_values, config_data),
        "browser_automation": _check_browser_automation(config_data, tool_policy_data),
        "broad_integrations": _check_broad_integrations(env_values, config_data, tool_policy_data),
        "memory_boundary": _check_memory_boundary(config_data, tool_policy_data),
        "redaction": _check_redaction_enabled(env_values, config_data),
        "hermes_home": _check_hermes_home(env_values, package_root_path),
    }

    failed = sum(1 for check in checks.values() if check["status"] == "fail")
    warned = sum(1 for check in checks.values() if check["status"] == "warn")
    status = "blocked" if failed else "degraded" if warned else "ready"

    redacted_tool_output = None
    if tool_output is not None:
        redacted_tool_output = redact_text(_stringify(tool_output), env_values)

    report = _format_report(status, checks)
    result: dict[str, Any] = {
        "status": status,
        "summary": {"passed": len(checks) - failed - warned, "warned": warned, "failed": failed},
        "env_source": env_source,
        "checks": checks,
        "redacted_tool_output": redacted_tool_output,
        "report": report,
    }
    return _redact_any(result, env_values)


def redact_text(text: str, env: Mapping[str, Any] | None = None) -> str:
    """Redact known Dobby package secrets and common token-shaped values."""

    redacted = str(text)
    env_values = env or {}
    for key in SECRET_ENV_KEYS:
        value = str(env_values.get(key, ""))
        if value and not _is_placeholder(value):
            redacted = redacted.replace(value, "[REDACTED]")

    for key in SECRET_ENV_KEYS:
        redacted = re.sub(
            rf"({re.escape(key)}\s*[:=]\s*)([^\s,;]+)",
            rf"\1[REDACTED]",
            redacted,
        )
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _check_model(env: Mapping[str, str], config: Mapping[str, Any]) -> dict[str, Any]:
    endpoint = _first_present(env.get("OPENAI_BASE_URL"), _get(config, "model", "base_url"))
    model = _first_present(env.get("HERMES_MODEL"), _get(config, "model", "default"))
    api_key = env.get("OPENAI_API_KEY")
    ok = all(_is_configured(value) for value in (endpoint, model, api_key))
    return _check(
        "model",
        "BYO model",
        ok,
        "BYO endpoint, model id, and API key are configured.",
        "BYO model endpoint/key is missing or still placeholder.",
        {
            "endpoint": _state(endpoint),
            "model": _state(model),
            "OPENAI_API_KEY": _secret_state(api_key),
        },
    )


def _check_discord_allowlist(env: Mapping[str, str], config: Mapping[str, Any]) -> dict[str, Any]:
    users = _first_non_empty_list(env.get("DISCORD_ALLOWED_USERS"), _get(config, "discord", "allowed_users"))
    channels = _first_non_empty_list(
        env.get("DISCORD_ALLOWED_CHANNELS"),
        _get(config, "discord", "allowed_channels"),
        _get(config, "gateway", "allowed_channels"),
    )
    token = env.get("DISCORD_BOT_TOKEN")
    users_ok = _is_explicit_allowlist(users)
    channels_ok = _is_explicit_allowlist(channels)
    token_ok = _is_configured(token)
    return _check(
        "discord_allowlist",
        "Discord readiness",
        users_ok and channels_ok and token_ok,
        "Discord token and user/channel allowlists are configured.",
        "Discord token and user/channel allowlists must be configured and non-wildcard.",
        {
            "DISCORD_BOT_TOKEN": _secret_state(token),
            "users": _allowlist_state(users),
            "channels": _allowlist_state(channels),
        },
    )


def _check_webhook_policy(env: Mapping[str, str], config: Mapping[str, Any]) -> dict[str, Any]:
    webhook = _get(config, "webhook", default={})
    routes = _as_list(webhook.get("allowed_routes")) if isinstance(webhook, Mapping) else []
    secret = env.get("WEBHOOK_SECRET")
    ok = (
        isinstance(webhook, Mapping)
        and webhook.get("enabled") is True
        and webhook.get("require_signature") is True
        and webhook.get("signature_algorithm") == "hmac-sha256"
        and webhook.get("unsigned_requests", "deny") == "deny"
        and _is_explicit_route_allowlist(routes)
        and _is_configured(secret)
    )
    return _check(
        "webhook_policy",
        "Webhook policy",
        ok,
        "Signed webhook policy is enabled with explicit routes.",
        "Webhook policy must be enabled, signed, secret-backed, and non-wildcard.",
        {
            "enabled": bool(isinstance(webhook, Mapping) and webhook.get("enabled") is True),
            "signed": bool(isinstance(webhook, Mapping) and webhook.get("require_signature") is True),
            "routes": _allowlist_state(routes),
            "WEBHOOK_SECRET": _secret_state(secret),
        },
    )


def _check_browser_automation(config: Mapping[str, Any], tool_policy: Mapping[str, Any]) -> dict[str, Any]:
    browser = _get(config, "browser", default={})
    browser_enabled = isinstance(browser, Mapping) and browser.get("enabled") is True
    browser_policy = _get(tool_policy, "deny_policies", "browser_automation", default={})
    policy_enabled = isinstance(browser_policy, Mapping) and browser_policy.get("default_enabled") is True
    disabled_tools = _get(tool_policy, "capabilities", "disabled_by_default", default={})
    disabled_tool_names = set(disabled_tools.keys()) if isinstance(disabled_tools, Mapping) else set()
    browser_tools_disabled = {"browser_navigate", "browser_click", "browser_type"}.issubset(disabled_tool_names)
    ok = not browser_enabled and not policy_enabled and (not tool_policy or browser_tools_disabled)
    return _check(
        "browser_automation",
        "Browser automation",
        ok,
        "Browser automation is disabled by default.",
        "Browser automation must stay disabled unless explicitly reviewed.",
        {
            "config_enabled": browser_enabled,
            "policy_default_enabled": policy_enabled,
            "browser_tools_disabled": browser_tools_disabled,
        },
    )


def _check_broad_integrations(
    env: Mapping[str, str],
    config: Mapping[str, Any],
    tool_policy: Mapping[str, Any],
) -> dict[str, Any]:
    unexpected_env = sorted(
        key for key in BROAD_INTEGRATION_ENV_KEYS if _is_configured(env.get(key))
    )
    premium_enabled = _get(tool_policy, "premium", "enabled") is True
    experimental_enabled = _get(tool_policy, "experimental", "enabled") is True
    external_memory_enabled = (
        _get(config, "external_memory_providers", "enabled") is True
        or _get(tool_policy, "external_memory_providers", "enabled") is True
    )
    broad_oauth_present = isinstance(_get(tool_policy, "deny_policies", "broad_oauth"), Mapping)
    ok = (
        not unexpected_env
        and not premium_enabled
        and not experimental_enabled
        and not external_memory_enabled
        and (not tool_policy or broad_oauth_present)
    )
    return _check(
        "broad_integrations",
        "Broad integrations",
        ok,
        "Broad integrations are disabled by default.",
        "Broad integrations must remain disabled and token-free by default.",
        {
            "unexpected_env_keys": unexpected_env,
            "premium_enabled": premium_enabled,
            "experimental_enabled": experimental_enabled,
            "external_memory_enabled": external_memory_enabled,
            "broad_oauth_deny_policy": broad_oauth_present,
        },
    )


def _check_memory_boundary(config: Mapping[str, Any], tool_policy: Mapping[str, Any]) -> dict[str, Any]:
    memory = _get(config, "memory", default={})
    honcho = _get(config, "honcho", default={})
    memory_disabled = not isinstance(memory, Mapping) or (
        memory.get("memory_enabled") is not True
        and memory.get("user_profile_enabled") is not True
        and not memory.get("provider")
    )
    honcho_disabled = not isinstance(honcho, Mapping) or not _has_enabled_value(honcho)
    external_memory_disabled = (
        _get(config, "external_memory_providers", "enabled") is not True
        and _get(tool_policy, "external_memory_providers", "enabled") is not True
    )
    ok = memory_disabled and honcho_disabled and external_memory_disabled
    return _check(
        "memory_boundary",
        "Memory/Honcho boundary",
        ok,
        "Memory defaults stay package-local; Honcho is not enabled.",
        "Memory defaults must not auto-enable Honcho or external providers.",
        {
            "native_memory_default_off": memory_disabled,
            "honcho_default_off": honcho_disabled,
            "external_memory_default_off": external_memory_disabled,
        },
    )


def _check_redaction_enabled(env: Mapping[str, str], config: Mapping[str, Any]) -> dict[str, Any]:
    env_enabled = _as_bool(env.get("HERMES_REDACT_SECRETS"))
    config_enabled = _get(config, "privacy", "redact_pii") is True
    ok = env_enabled is True or config_enabled is True
    return _check(
        "redaction",
        "Redaction",
        ok,
        "Redaction is enabled.",
        "Redaction must be enabled before diagnostics/status output.",
        {"env": env_enabled, "config": config_enabled},
    )


def _check_hermes_home(env: Mapping[str, str], package_root: Path) -> dict[str, Any]:
    value = env.get("HERMES_HOME")
    classification = _classify_hermes_home(value, package_root)
    ok = classification == "safe"
    return _check(
        "hermes_home",
        "HERMES_HOME",
        ok,
        "HERMES_HOME points at an isolated package runtime path.",
        "Use a fresh package home outside personal ~/.hermes and the repo.",
        {"classification": classification},
    )


def _check(
    check_id: str,
    label: str,
    ok: bool,
    pass_summary: str,
    fail_summary: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": check_id,
        "label": label,
        "status": "pass" if ok else "fail",
        "summary": pass_summary if ok else fail_summary,
        "details": dict(details),
    }


def _format_report(status: str, checks: Mapping[str, Mapping[str, Any]]) -> str:
    failed = sum(1 for check in checks.values() if check["status"] == "fail")
    lines = [f"Dobby package diagnostics: {status} ({failed} failed)."]
    lines.append("No network, shell, Discord, or live service checks are performed.")
    for check_id in CHECK_ORDER:
        check = checks[check_id]
        marker = "OK" if check["status"] == "pass" else "FAIL"
        lines.append(f"- {marker} {check['label']}: {check['summary']}")
    return "\n".join(lines)


def _classify_hermes_home(value: str | None, package_root: Path) -> str:
    if not _is_configured(value):
        return _state(value)

    raw = str(value).strip()
    lowered = raw.lower()
    if raw in {"/", "/root"} or raw.startswith("/root/"):
        return "live_or_system"
    if lowered in {"~/.hermes", "$home/.hermes", "${home}/.hermes"}:
        return "personal_hermes"
    if "/.hermes" in raw and (raw.startswith("/Users/") or raw.startswith("/home/")):
        return "personal_hermes"
    if not raw.startswith("/"):
        return "not_absolute"

    home_path = Path(raw)
    package_abs = package_root.resolve(strict=False)
    repo_abs = package_abs.parents[1] if len(package_abs.parents) > 1 else package_abs
    home_abs = home_path.resolve(strict=False)
    if _is_relative_to(home_abs, package_abs) or _is_relative_to(home_abs, repo_abs):
        return "repo_or_package"
    return "safe"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _get(mapping: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value).strip():
            return value
    return None


def _first_non_empty_list(*values: Any) -> list[str]:
    for value in values:
        entries = _as_list(value)
        if entries:
            return entries
    return []


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;\s]+", value) if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _is_explicit_allowlist(values: list[str]) -> bool:
    return bool(values) and all(not _is_placeholder(value) and value.lower() not in BROAD_VALUES for value in values)


def _is_explicit_route_allowlist(routes: list[str]) -> bool:
    return bool(routes) and all(route not in {"*", "/*"} and "*" not in route for route in routes)


def _allowlist_state(values: list[str]) -> str:
    if not values:
        return "missing"
    if _is_explicit_allowlist(values):
        return "explicit"
    return "placeholder_or_broad"


def _is_configured(value: Any) -> bool:
    return value is not None and bool(str(value).strip()) and not _is_placeholder(str(value))


def _is_placeholder(value: str) -> bool:
    stripped = _strip_quotes(str(value).strip())
    lowered = stripped.lower()
    if not stripped:
        return True
    if stripped.startswith("<") and stripped.endswith(">"):
        return True
    return any(word in lowered for word in ("changeme", "dummy", "example", "fake", "placeholder", "replace-with", "todo", "your-"))


def _state(value: Any) -> str:
    if value is None or not str(value).strip():
        return "missing"
    if _is_placeholder(str(value)):
        return "placeholder"
    return "configured"


def _secret_state(value: Any) -> str:
    return _state(value)


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def _has_enabled_value(value: Mapping[str, Any]) -> bool:
    for nested_value in value.values():
        if isinstance(nested_value, Mapping):
            if _has_enabled_value(nested_value):
                return True
        elif nested_value not in (False, None, "", [], {}):
            return True
    return False


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def _redact_any(value: Any, env: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        redacted = {}
        for key, nested_value in value.items():
            if str(key) in SECRET_ENV_KEYS:
                redacted[key] = _secret_state(nested_value)
            else:
                redacted[key] = _redact_any(nested_value, env)
        return redacted
    if isinstance(value, list):
        return [_redact_any(item, env) for item in value]
    if isinstance(value, str):
        return redact_text(value, env)
    return value
