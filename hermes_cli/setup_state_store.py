"""Shared setup/config flow for selecting the backend-neutral state store."""

from __future__ import annotations

from hermes_cli.postgresql_bootstrap import bootstrap_postgresql_state_store

POSTGRES_DSN_ENV = "HERMES_STATE_STORE_POSTGRES_DSN"
_CHOICES = [
    "Use an existing PostgreSQL DSN",
    "Start a local PostgreSQL Docker container (Compose)",
    "Keep default SQLite",
]


def _state_store_config(config: dict) -> dict:
    state_store = config.get("state_store")
    if not isinstance(state_store, dict):
        state_store = config["state_store"] = {}
    postgresql = state_store.get("postgresql")
    if not isinstance(postgresql, dict):
        postgresql = state_store["postgresql"] = {}
    postgresql.setdefault("dsn_env", POSTGRES_DSN_ENV)
    postgresql.setdefault("connect_timeout_seconds", 10)
    postgresql.setdefault("pool_max_size", 8)
    return state_store


def configure_state_store(config: dict) -> None:
    """Interactively select SQLite, an existing DSN, or the local Compose service."""
    from hermes_cli import setup

    setup.print_header("State Store")
    setup._info(
        "Choose where Hermes stores durable state.",
        "Choosing PostgreSQL starts with a new empty state store; existing SQLite history is not migrated.",
    )
    selection = setup.prompt_choice("State-store backend", _CHOICES, 2)
    state_store = _state_store_config(config)
    if selection == 2:
        state_store["backend"] = "sqlite"
        setup.print_info("Keeping SQLite as the state-store backend.")
        return
    if selection == 0:
        dsn = setup.prompt("PostgreSQL DSN", password=True)
        if not dsn:
            setup.print_warning("No DSN entered; state-store configuration was not changed.")
            return
    else:
        dsn = bootstrap_postgresql_state_store(setup.get_hermes_home(), existing_dsn=setup.get_env_value(POSTGRES_DSN_ENV))
    setup.save_env_value(POSTGRES_DSN_ENV, dsn)
    state_store["backend"] = "postgresql"
    state_store["postgresql"]["dsn_env"] = POSTGRES_DSN_ENV
    setup.print_success("PostgreSQL state store configured.")


def run_config_state_store() -> None:
    """Entry point for ``hermes config state-store`` using the same flow as setup."""
    from hermes_cli.config import load_config, save_config

    config = load_config()
    configure_state_store(config)
    save_config(config)
