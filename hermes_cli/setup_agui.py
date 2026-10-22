"""The original standalone AG-UI setup section, using shared setup UI/writers."""

import copy


def _model_overrides(current):
    from hermes_cli import setup as ui

    keys = ("provider", "model", "api_mode", "base_url")
    has_overrides = any(str(current.get(key) or "").strip() for key in keys)
    choice = ui.prompt_choice(
        "Model configuration:",
        [
            "Use the main Hermes model configuration (recommended)",
            "Configure AG-UI-specific model/endpoint overrides",
        ],
        1 if has_overrides else 0,
    )
    if choice != 1:
        return dict.fromkeys(keys, "")
    prompts = (
        "Provider override (Enter keeps; - clears to main config)",
        "Model override (Enter keeps; - clears to main config)",
        "API mode override (Enter keeps; - clears to auto)",
        "Base URL override (Enter keeps; - clears to resolved provider)",
    )
    answers = {
        key: ui.prompt(label, str(current.get(key) or ""))
        for key, label in zip(keys, prompts)
    }
    return {key: "" if value == "-" else value for key, value in answers.items()}


def _configure_token(host):
    from hermes_cli import setup as ui
    from gateway.platforms.base import is_network_accessible
    from agui_adapter.auth import _usable

    if not is_network_accessible(host):
        ui.print_info("Loopback bind selected; a session token is optional.")
        return
    if _usable(ui.get_env_value("HERMES_AGUI_SESSION_TOKEN")):
        ui.print_success("A usable AG-UI session token is already configured.")
        return
    ui.print_warning("A network-accessible bind requires a strong session token.")
    token = ui.prompt("Session token (minimum 16 characters)", password=True)
    if _usable(token):
        ui.save_env_value("HERMES_AGUI_SESSION_TOKEN", token)
        ui.print_success("AG-UI session token saved to .env.")
    else:
        ui.print_warning(
            "No usable token was saved; hermes agui will refuse this network bind."
        )


def setup_agui(config):
    """Configure the optional adapter without discarding lifecycle/future keys."""
    from hermes_cli import setup as ui

    defaults = ui.DEFAULT_CONFIG["agui"]
    current = copy.deepcopy(defaults)
    if isinstance(config.get("agui"), dict):
        current.update(config["agui"])
    ui.print_header("AG-UI Adapter")
    ui.print_info("Expose Hermes to AG-UI clients over HTTP/SSE.")
    ui.print_info(
        "Settings live in config.yaml; the session token is a secret in .env."
    )
    host = ui.prompt("Bind host", str(current.get("host") or defaults["host"]))
    try:
        port = int(current.get("port", defaults["port"]))
        if not 1 <= port <= 65535:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        port = defaults["port"]
    port_input = ui.prompt("Listen port", str(port))
    try:
        candidate = int(port_input)
        if not 1 <= candidate <= 65535:
            raise ValueError
        port = candidate
    except (TypeError, ValueError, OverflowError):
        ui.print_warning(f"Invalid port '{port_input}', keeping {port}")
    tools = current.get("toolsets")
    if isinstance(tools, str):
        tools = [item.strip() for item in tools.split(",") if item.strip()]
    elif not isinstance(tools, list):
        tools = defaults["toolsets"]
    selected = ui.prompt(
        "Toolsets (comma-separated)", ", ".join(str(item) for item in tools)
    )
    toolsets = [item.strip() for item in selected.split(",") if item.strip()]
    current.update(host=host, port=port, toolsets=toolsets, **_model_overrides(current))
    _configure_token(host)
    config["agui"] = current
    ui.print_success("AG-UI settings updated.")
