"""Shared setup helpers for platform access policy prompts."""

from __future__ import annotations

from typing import Callable

from hermes_cli.config import save_env_value


_TRUE_VALUES = {"true", "1", "yes", "on"}


def is_open_access_enabled(value: str | None) -> bool:
    """Return True when an allow-all env value opts into open access."""
    return str(value or "").strip().lower() in _TRUE_VALUES


def configure_direct_message_access(
    *,
    platform_label: str,
    pairing_platform: str,
    allowed_users_env: str,
    allow_all_env: str,
    allowed_users_value: str,
    prompt_yes_no_fn: Callable[[str, bool], bool],
    print_info_fn: Callable[[str], None],
    print_success_fn: Callable[[str], None],
    print_warning_fn: Callable[[str], None],
    open_access_warning: str,
    allowlist_success: str | None = None,
    clean_allowlist: Callable[[str], str] | None = None,
) -> str:
    """Persist allowlist/open-access/pairing policy for DM-capable platforms."""
    raw_value = (allowed_users_value or "").strip()
    clean_allowlist = clean_allowlist or (lambda value: value.replace(" ", ""))

    if raw_value:
        cleaned = clean_allowlist(raw_value)
        save_env_value(allowed_users_env, cleaned)
        save_env_value(allow_all_env, "false")
        print_success_fn(allowlist_success or f"{platform_label} allowlist configured")
        return "allowlist"

    save_env_value(allowed_users_env, "")
    if prompt_yes_no_fn(
        f"Enable open access for {platform_label}? (otherwise DM pairing will be used)",
        False,
    ):
        save_env_value(allow_all_env, "true")
        print_warning_fn(open_access_warning)
        return "open"

    save_env_value(allow_all_env, "false")
    print_success_fn(f"{platform_label} DM pairing configured")
    print_info_fn(f"Approve with: hermes pairing approve {pairing_platform} <code>")
    return "pairing"
