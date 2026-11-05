"""User-facing warnings for config defaults that changed after a migration."""

from typing import Any


def persist_and_warn_default_drift(
    previous_version: int, latest_version: int, results: dict[str, Any], quiet: bool
) -> None:
    """Keep version persistence and its informational warning in the same migration step."""
    from hermes_cli.config import _persist_migration, read_raw_config

    config = read_raw_config()
    config["_config_version"] = latest_version
    _persist_migration(config)
    warn_default_drift(read_raw_config(), previous_version, results, quiet)


def warn_default_drift(
    config: dict[str, Any], previous_version: int, results: dict[str, Any], quiet: bool
) -> None:
    """Record and optionally display changed defaults without modifying user config."""
    from hermes_cli.config_migrations import collect_default_drift

    for drift in collect_default_drift(config, previous_version):
        msg = (
            f"  ⚠ {drift['key']} = {drift['value']} (user-set)\n"
            f"    Upstream default changed: {drift['change']}.\n"
            "    Your value may now behave differently than when you set it."
        )
        results["warnings"].append(msg.strip())
        if not quiet:
            print("\n⚠️  Config default changed since your last version:")
            print(msg)
            print("   Keep it if intentional; see docs for the new default.")
