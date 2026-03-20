"""Product-first setup flow for the hermes-core distribution."""

from __future__ import annotations

from typing import Any

from hermes_cli.config import ensure_hermes_home, get_hermes_home, load_config
from hermes_cli.product_config import initialize_product_config_file, load_product_config, save_product_config
from hermes_cli.product_stack import (
    bootstrap_first_admin_enrollment,
    ensure_product_stack_started,
    initialize_product_stack,
    resolve_product_urls,
)
from hermes_cli.setup import (
    PROJECT_ROOT,
    Colors,
    color,
    get_env_path,
    get_env_value,
    get_config_path,
    is_interactive_stdin,
    print_error,
    print_header,
    print_info,
    print_noninteractive_setup_guidance,
    print_success,
    print_warning,
    prompt,
    prompt_choice,
    save_config,
    setup_model_provider,
    setup_tools,
)


PRODUCT_SETUP_SECTIONS = [
    ("network", "Product Network"),
    ("model", "Model & Provider"),
    ("tools", "Tools"),
    ("bootstrap", "Pocket ID & First Admin"),
]


def setup_product_network() -> None:
    from hermes_cli.product_stack import _validate_public_host

    product_config = load_product_config()
    current_public_host = (
        str(product_config.get("network", {}).get("public_host", "")).strip() or "localhost"
    )

    print_header("Product Network")
    print_info("Choose the hostname users will use to reach this machine.")
    print_info("This hostname is used for local app URLs and Pocket ID OIDC origins.")
    print_info("Use a hostname like localhost, officebox.local, or a DNS name.")
    print_info("Raw IP addresses are not supported for the Pocket ID public host.")

    while True:
        public_host = (prompt("Public host", current_public_host) or current_public_host).strip()
        try:
            _validate_public_host(public_host)
        except ValueError as exc:
            print_warning(str(exc))
            continue
        product_config.setdefault("network", {})["public_host"] = public_host
        save_product_config(product_config)
        urls = resolve_product_urls(product_config)
        print_info(f"  App URL: {urls['app_base_url']}")
        print_info(f"  Pocket ID issuer: {urls['issuer_url']}")
        break


def _sync_model_route_from_hermes_config() -> None:
    hermes_config = load_config()
    model_cfg = hermes_config.get("model")
    if isinstance(model_cfg, str):
        model_cfg = {"default": model_cfg}
    if not isinstance(model_cfg, dict):
        return

    provider = str(model_cfg.get("provider") or "").strip()
    model_name = str(model_cfg.get("default") or "").strip()
    if not provider or not model_name:
        return

    product_config = load_product_config()
    route = product_config.setdefault("models", {}).setdefault("default_route", {})
    route["provider"] = provider
    route["model"] = model_name

    api_mode = str(model_cfg.get("api_mode") or "").strip()
    if api_mode:
        route["api_mode"] = api_mode
    else:
        route.pop("api_mode", None)

    base_url = str(model_cfg.get("base_url") or "").strip()
    if provider == "custom":
        base_url = base_url or str(get_env_value("OPENAI_BASE_URL") or "").strip()
    if base_url:
        route["base_url"] = base_url.rstrip("/")
    elif provider == "custom":
        print_warning("Hermes custom provider has no base URL; product route was not updated.")

    save_product_config(product_config)


def _sync_toolsets_from_hermes_config() -> None:
    hermes_config = load_config()
    platform_toolsets = hermes_config.get("platform_toolsets", {})
    cli_toolsets = platform_toolsets.get("cli") if isinstance(platform_toolsets, dict) else None
    if not isinstance(cli_toolsets, list):
        return

    normalized = [str(toolset).strip() for toolset in cli_toolsets if str(toolset).strip()]
    product_config = load_product_config()
    product_config.setdefault("tools", {})["hermes_toolsets"] = normalized
    save_product_config(product_config)


def _print_product_setup_summary() -> None:
    product_config = load_product_config()
    hermes_home = get_hermes_home()
    urls = resolve_product_urls(product_config)

    print()
    print_header("Product Setup Summary")
    print_info(f"Hermes config:  {get_config_path()}")
    print_info(f"Secrets file:   {get_env_path()}")
    print_info(f"Product config: {hermes_home / 'product.yaml'}")
    print_info(f"Data folder:    {hermes_home}")
    print_info(f"Install dir:    {PROJECT_ROOT}")
    print_info(f"App URL:        {urls['app_base_url']}")
    print_info(f"Pocket ID URL:  {urls['issuer_url']}")


def _start_product_stack_best_effort() -> None:
    try:
        ensure_product_stack_started()
        state = bootstrap_first_admin_enrollment()
        print_info("Bundled Pocket ID service is up.")
        print_info(f"  First admin: {state['username']}")
        if state["email"]:
            print_info(f"  First admin email: {state['email']}")
        print_info(f"  Auth mode: {state['auth_mode']}")
        print_info(f"  First admin setup URL: {state['setup_url']}")
        print_info(f"  OIDC client: {state['oidc_client_id']}")
    except FileNotFoundError:
        print_warning("Docker was not found. The bundled Pocket ID service was generated but not started.")
    except Exception as exc:
        print_warning(f"Could not start bundled Pocket ID automatically: {exc}")


def _run_model_section() -> None:
    config = load_config()
    setup_model_provider(config)
    save_config(config)
    _sync_model_route_from_hermes_config()


def _run_tools_section() -> None:
    config = load_config()
    platform_toolsets = config.get("platform_toolsets", {})
    first_install = not (isinstance(platform_toolsets, dict) and platform_toolsets.get("cli"))
    setup_tools(config, first_install=first_install)
    save_config(config)
    _sync_toolsets_from_hermes_config()


def _run_bootstrap_section() -> None:
    initialize_product_stack()
    _start_product_stack_best_effort()


def run_product_setup_wizard(args: Any) -> None:
    ensure_hermes_home()
    initialize_product_config_file()

    non_interactive = getattr(args, "non_interactive", False)
    if not non_interactive and not is_interactive_stdin():
        non_interactive = True
    if non_interactive:
        print_noninteractive_setup_guidance(
            "Running in a non-interactive environment (no TTY detected)."
        )
        return

    section = getattr(args, "section", None)
    if section:
        if section == "network":
            setup_product_network()
        elif section == "model":
            _run_model_section()
        elif section == "tools":
            _run_tools_section()
        elif section == "bootstrap":
            _run_bootstrap_section()
        else:
            print_error(f"Unknown product setup section: {section}")
            print_info(f"Available sections: {', '.join(key for key, _ in PRODUCT_SETUP_SECTIONS)}")
            return
        _print_product_setup_summary()
        return

    print()
    print(
        color(
            "┌─────────────────────────────────────────────────────────┐",
            Colors.MAGENTA,
        )
    )
    print(
        color(
            "│           ⚕ Hermes Core Product Setup Wizard            │",
            Colors.MAGENTA,
        )
    )
    print(
        color(
            "└─────────────────────────────────────────────────────────┘",
            Colors.MAGENTA,
        )
    )
    print()
    print_info("This configures the supplier-curated local product distribution.")
    print_info("It leaves the generic 'hermes setup' flow untouched.")
    print()

    setup_product_network()
    _run_model_section()
    _run_tools_section()
    _run_bootstrap_section()
    _print_product_setup_summary()
    print()
    print_success("Product setup complete!")
