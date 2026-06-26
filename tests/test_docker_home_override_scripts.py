"""Regression tests for Docker HOME overrides under s6/with-contenv."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_RUN = REPO_ROOT / "docker" / "s6-rc.d" / "dashboard" / "run"
MAIN_WRAPPER = REPO_ROOT / "docker" / "main-wrapper.sh"
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"


def test_main_wrapper_preserves_docker_workdir() -> None:
    """The main-wrapper MUST save and restore the original working
    directory so the container starts in the Docker ``-w`` directory,
    not /opt/data.  Regression test for #35472.
    """
    text = MAIN_WRAPPER.read_text(encoding="utf-8")

    # Must save original cwd before cd /opt/data.
    assert "_hermes_orig_cwd" in text, (
        "main-wrapper.sh must save the original cwd before cd /opt/data"
    )
    assert 'HERMES_ORIG_CWD:-$PWD' in text, (
        "main-wrapper.sh must capture PWD as the fallback original cwd"
    )

    # Must cd to /opt/data for init (existing behaviour preserved).
    assert "cd /opt/data" in text

    # Must restore original cwd before exec'ing the user command.
    # The restore cd must appear AFTER venv activation but BEFORE the
    # first exec / if-block.
    activate_idx = text.index("/opt/hermes/.venv/bin/activate")
    restore_idx = text.index('cd "$_hermes_orig_cwd"')
    exec_idx = text.index("if [ $# -eq 0 ]")
    assert activate_idx < restore_idx < exec_idx, (
        "cd $_hermes_orig_cwd must appear after venv activation and "
        "before the exec routing block"
    )


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
    # (e.g. HERMES_DASHBOARD).
    for truthy in ("1", "true", "TRUE", "True", "yes", "YES", "Yes"):
        assert truthy in text, (
            f"HERMES_DASHBOARD_INSECURE should accept truthy value {truthy!r}"
        )


def test_stage2_hook_repairs_profiles_and_cron_ownership_on_every_boot() -> None:
    """profiles/ and cron/ must both be reclaimed after root-context writes."""
    text = STAGE2_HOOK.read_text(encoding="utf-8")

    assert 'if [ -d "$HERMES_HOME/profiles" ]; then' in text
    assert 'chown_hermes_tree "$HERMES_HOME/profiles"' in text
    assert 'chown -R hermes:hermes "$HERMES_HOME/profiles" 2>/dev/null || true' not in text

    assert 'if [ -d "$HERMES_HOME/cron" ]; then' in text
    assert 'chown_hermes_tree "$HERMES_HOME/cron"' in text
    assert 'chown -R hermes:hermes "$HERMES_HOME/cron" 2>/dev/null || true' not in text
