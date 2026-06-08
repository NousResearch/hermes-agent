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


def test_stage2_hook_repairs_cron_ownership_unconditionally() -> None:
   """cron/ must be in the unconditional chown block alongside profiles/.

   docker exec commands run as root, leaving jobs.json root-owned.
   The unprivileged hermes runtime then hits EACCES on next boot.
   Regression for issue #41966.
   """
   text = STAGE2_HOOK.read_text()

   # profiles/ unconditional chown must still be present (existing behaviour)
   assert 'chown -R hermes:hermes "$HERMES_HOME/profiles"' in text, (
       "Unconditional profiles/ chown was removed — restore it."
   )

   # cron/ unconditional chown must be present
   assert 'chown -R hermes:hermes "$HERMES_HOME/cron"' in text, (
       "Unconditional cron/ chown is missing. "
       "docker exec writes land as root and must be repaired on every boot."
   )

   # Both chowns must appear OUTSIDE the needs_chown conditional block
   # (i.e. they must not be indented inside an `if` that gates on ownership).
   lines = text.splitlines()
   cron_chown_line = next(
       (i for i, l in enumerate(lines)
        if 'chown -R hermes:hermes "$HERMES_HOME/cron"' in l),
       None,
   )
   assert cron_chown_line is not None, "cron/ chown line not found"
   # Unconditional chowns are at 4-space indent (inside if [ -d ... ]; then)
   # but NOT inside a deeper conditional like `if [ "$needs_chown" = true ]`
   assert lines[cron_chown_line].startswith("    chown"), (
       "cron/ chown must be at top-level indent (unconditional block)"
   )
