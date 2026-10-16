"""Interactive ``hermes gateway setup`` wizard for Signal. Helpers resolve late through the
``hermes_cli.gateway`` facade so the plugin stays importable outside the CLI and the seams tests
patch there keep intercepting."""

import shutil


def _gw():
    from hermes_cli import gateway  # late: CLI-only dependency
    return gateway


def _signal_line_input(prompt_text: str) -> str | None:
    """``line_input`` for the Signal wizard; None (after printing the cancel line) on EOF/Ctrl+C."""
    try:
        return _gw().line_input(prompt_text).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Setup cancelled.")
        return None


def interactive_setup() -> None:
    """``hermes gateway setup`` flow for Signal (registered as the plugin's ``setup_fn``)."""
    _gw()._print_setup_header("📡 Signal")

    existing_url = _gw().get_env_value("SIGNAL_HTTP_URL")
    existing_account = _gw().get_env_value("SIGNAL_ACCOUNT")
    if not _gw()._confirm_reconfigure("Signal", "SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"):
        return

    print()
    if shutil.which("signal-cli"):
        _gw().print_success("signal-cli found on PATH.")
    else:
        _gw().print_warning("signal-cli not found on PATH.")
        _gw()._print_info_lines(
            "  Signal requires signal-cli running as an HTTP daemon.", "  Install options:",
            "    Linux:  download from https://github.com/AsamK/signal-cli/releases",
            "    macOS:  brew install signal-cli", "    Docker: bbernhard/signal-cli-rest-api",
        )
        print()
        _gw()._print_info_lines(
            "  After installing, link your account and start the daemon:",
            '    signal-cli link -n "HermesAgent"',
            "    signal-cli --account +YOURNUMBER daemon --http 127.0.0.1:8080",
        )
        print()

    print()
    _gw().print_info("  Enter the URL where signal-cli HTTP daemon is running.")
    default_url = existing_url or "http://127.0.0.1:8080"
    url = _signal_line_input(f"  HTTP URL [{default_url}]: ")
    if url is None:
        return
    url = url or default_url

    _gw().print_info("  Testing connection...")
    try:
        import httpx
        resp = httpx.get(f"{url.rstrip('/')}/api/v1/check", timeout=10.0)
        if resp.status_code == 200:
            _gw().print_success("  signal-cli daemon is reachable!")
        else:
            _gw().print_warning(f"  signal-cli responded with status {resp.status_code}.")
            if not _gw().prompt_yes_no("  Continue anyway?", False):
                return
    except Exception as e:  # health: allow BLE001 -- connectivity probe in a wizard: any failure is "unreachable", the user decides
        _gw().print_warning(f"  Could not reach signal-cli at {url}: {e}")
        if not _gw().prompt_yes_no("  Save this URL anyway? (you can start signal-cli later)", True):
            return

    _gw().save_env_value("SIGNAL_HTTP_URL", url)

    print()
    _gw()._print_info_lines("  Enter your Signal account phone number in E.164 format.", "  Example: +15551234567")
    default_account = existing_account or ""
    account = _signal_line_input(f"  Account number{f' [{default_account}]' if default_account else ''}: ")
    if account is None:
        return
    account = account or default_account
    if not account:
        _gw().print_error("  Account number is required.")
        return

    _gw().save_env_value("SIGNAL_ACCOUNT", account)

    print()
    _gw()._print_info_lines(
        "  The gateway DENIES all users by default for security.",
        "  Enter phone numbers or UUIDs of allowed users (comma-separated).",
    )
    default_allowed = _gw().get_env_value("SIGNAL_ALLOWED_USERS") or account
    allowed = _signal_line_input(f"  Allowed users [{default_allowed}]: ")
    if allowed is None:
        return
    _gw().save_env_value("SIGNAL_ALLOWED_USERS", allowed or default_allowed)

    print()
    if _gw().prompt_yes_no("  Enable group messaging? (disabled by default for security)", False):
        print()
        _gw().print_info("  Enter group IDs to allow, or * for all groups.")
        existing_groups = _gw().get_env_value("SIGNAL_GROUP_ALLOWED_USERS") or ""
        groups = _signal_line_input(f"  Group IDs [{existing_groups or '*'}]: ")
        if groups is None:
            return
        _gw().save_env_value("SIGNAL_GROUP_ALLOWED_USERS", groups or existing_groups or "*")

    print()
    _gw().print_success("Signal configured!")
    _gw()._print_info_lines(
        f"  URL: {url}", f"  Account: {account}", "  DM auth: via SIGNAL_ALLOWED_USERS + DM pairing",
        f"  Groups: {'enabled' if _gw().get_env_value('SIGNAL_GROUP_ALLOWED_USERS') else 'disabled'}",
    )
