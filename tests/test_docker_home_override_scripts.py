"""Regression tests for Docker HOME overrides under s6/with-contenv."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_RUN = REPO_ROOT / "docker" / "s6-rc.d" / "dashboard" / "run"
WINDOWS_COMPOSE = REPO_ROOT / "docker-compose.windows.yml"


def test_dashboard_run_resets_home_before_dropping_privileges() -> None:
    text = DASHBOARD_RUN.read_text(encoding="utf-8")

    assert "#!/command/with-contenv sh" in text
    assert "export HOME=/opt/data" in text
    assert "exec s6-setuidgid hermes hermes dashboard" in text


def test_dashboard_run_does_not_derive_insecure_from_bind_host() -> None:
    """The s6 dashboard run script MUST NOT auto-add ``--insecure`` based on
    ``HERMES_DASHBOARD_HOST``. Doing so disables the OAuth auth gate on
    every non-loopback bind even when an auth provider is registered —
    the exact regression that exposed every wildcard-subdomain agent
    dashboard publicly until early 2026.

    The opt-in is now explicit: ``HERMES_DASHBOARD_INSECURE=1`` (truthy).
    The auth gate is the authority on whether non-loopback binds are safe.
    """
    text = DASHBOARD_RUN.read_text(encoding="utf-8")

    # No legacy host-derived flip.
    assert '127.0.0.1|localhost' not in text, (
        "Run script still derives --insecure from the bind host. The gate "
        "is the authority now — opt in via HERMES_DASHBOARD_INSECURE instead."
    )
    assert 'case "$dash_host" in' not in text, (
        "Legacy host-derived --insecure case-statement is back."
    )

    # New opt-in env var present.
    assert "HERMES_DASHBOARD_INSECURE" in text, (
        "Explicit HERMES_DASHBOARD_INSECURE opt-in is missing."
    )
    # Truthy values aligned with the rest of the s6 scripts
    # (HERMES_DASHBOARD, HERMES_DASHBOARD_TUI).
    for truthy in ("1", "true", "TRUE", "True", "yes", "YES", "Yes"):
        assert truthy in text, (
            f"HERMES_DASHBOARD_INSECURE should accept truthy value {truthy!r}"
        )


def test_windows_compose_uses_dashboard_env_opt_in_for_insecure_mode() -> None:
    """Windows Docker Desktop needs a container-public bind for port mapping.

    Keep that local-only compose path on the s6 dashboard service so the
    insecure dashboard mode is still an explicit ``HERMES_DASHBOARD_INSECURE``
    opt-in, not a hard-coded CLI flag hidden in the command line.
    """
    text = WINDOWS_COMPOSE.read_text(encoding="utf-8")

    assert "HERMES_DASHBOARD=1" in text
    assert "HERMES_DASHBOARD_HOST=0.0.0.0" in text
    assert "HERMES_DASHBOARD_INSECURE=1" in text
    assert 'command: ["dashboard",' not in text
    assert '"--insecure"' not in text
