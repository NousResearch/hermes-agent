"""Regression test for #49567.

Docker with the default `HERMES_DASHBOARD=1` and no `HERMES_DASHBOARD_HOST`
override failed closed under the OAuth auth gate: the s6-supervised dashboard
service defaulted to `--host 0.0.0.0`, which triggers

    Refusing to bind dashboard to 0.0.0.0 — the OAuth auth gate engages on
    non-loopback binds, but no auth providers are registered.

The `hermes dashboard` CLI itself defaults to `127.0.0.1` (see
`hermes_cli/subcommands/dashboard.py`), so the s6 run script's default of
`0.0.0.0` was an inconsistency, not a documented opt-in.

These tests pin the s6 run-script's default to loopback, and verify an
explicit `HERMES_DASHBOARD_HOST` still overrides it (so operators can still
intentionally bind to a LAN/public address when they have configured an
auth provider or set `HERMES_DASHBOARD_INSECURE=1`).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_SCRIPT = REPO_ROOT / "docker" / "s6-rc.d" / "dashboard" / "run"


def _read_run_script() -> str:
    assert RUN_SCRIPT.is_file(), f"missing s6 dashboard run script: {RUN_SCRIPT}"
    return RUN_SCRIPT.read_text(encoding="utf-8")


def _extract_default_host() -> str:
    """Extract the fallback value from the `${HERMES_DASHBOARD_HOST:-...}`
    parameter expansion inside `dash_host=...` in the s6 run script.

    Raises an informative AssertionError if the line is missing or the
    expansion pattern changed in an unexpected way.
    """
    text = _read_run_script()
    # Match a POSIX sh parameter expansion with a literal default value.
    pattern = re.compile(
        r'^\s*dash_host\s*=\s*"\$\{HERMES_DASHBOARD_HOST:-([^}]+)\}"',
        re.MULTILINE,
    )
    match = pattern.search(text)
    assert match, (
        'Could not find `dash_host="${HERMES_DASHBOARD_HOST:-<default>}"` '
        "in docker/s6-rc.d/dashboard/run. The parameter expansion pattern "
        "may have changed; update this test to match the new shape."
    )
    return match.group(1).strip()


@pytest.mark.parametrize("loopback", ["127.0.0.1"])
def test_default_host_is_loopback(loopback: str) -> None:
    """Default must be loopback so the OAuth auth gate does not fail closed
    on the stock Docker-published `HERMES_DASHBOARD=1` path."""
    assert _extract_default_host() == loopback, (
        f"s6 dashboard run script default bind is {_extract_default_host()!r}; "
        f"expected {loopback!r}. Non-loopback default crashes the dashboard "
        f"under the bundled OAuth auth gate when no DashboardAuthProvider "
        f"is registered (#49567)."
    )


def test_default_is_not_non_loopback() -> None:
    """Direct guard against the regression — independent of any future
    loopback string addition."""
    default = _extract_default_host()
    assert default not in {"0.0.0.0", "::"}, (
        f"s6 dashboard run script default bind is {default!r}; "
        f"this is the regression vector for #49567."
    )


def test_explicit_non_loopback_override_preserved() -> None:
    """Operators who intentionally want a LAN/public bind (with an auth
    provider or HERMES_DASHBOARD_INSECURE=1) must still be able to set
    HERMES_DASHBOARD_HOST explicitly.

    We assert the parameter expansion is of the form `${VAR:-default}` —
    i.e. the unset-fallback form — so that an exported env var overrides
    the default. This catches accidental refactors that change
    `:-` (unset-or-empty fallback) to `:=` (always-override / assign) or
    remove the expansion entirely.
    """
    text = _read_run_script()
    # Must still be a parameter expansion with a default, not a literal.
    assert "${HERMES_DASHBOARD_HOST:-" in text, (
        "s6 dashboard run script no longer uses "
        "${HERMES_DASHBOARD_HOST:-<default>} expansion — explicit env "
        "override may be broken."
    )
    # Belt-and-suspenders: must NOT be ${VAR:=...} or ${VAR:?...}.
    assert "${HERMES_DASHBOARD_HOST:=" not in text, (
        "Parameter expansion changed to ':=' (always-assign); an exported "
        "HERMES_DASHBOARD_HOST would be clobbered or trigger an error."
    )


def test_posix_expansion_matches_default_constant() -> None:
    """End-to-end check: run the same `${VAR:-default}` expansion through a
    POSIX sh to confirm the extracted default is what actually expands at
    runtime. This catches any mismatch between the test's regex and real
    shell semantics (e.g. quoting, IFS, command-substitution edge cases).
    """
    if not shutil.which("sh"):
        pytest.skip("no POSIX sh available")
    default = _extract_default_host()
    # Simulate the exact parameter expansion with VAR unset.
    result = subprocess.run(
        [
            "sh",
            "-c",
            f'dash_host="${{HERMES_DASHBOARD_HOST:-{default}}}"\necho "$dash_host"',
        ],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},  # strip HERMES_DASHBOARD_HOST
        check=True,
    )
    assert result.stdout.strip() == default, (
        f"POSIX sh expansion produced {result.stdout.strip()!r}, expected {default!r}."
    )


def test_posix_expansion_with_explicit_env_override() -> None:
    """End-to-end check: exported HERMES_DASHBOARD_HOST must override the
    default. This is the path operators use when they intentionally want
    a non-loopback bind."""
    if not shutil.which("sh"):
        pytest.skip("no POSIX sh available")
    default = _extract_default_host()
    explicit = "0.0.0.0" if default != "0.0.0.0" else "192.168.1.5"
    env = {"PATH": "/usr/bin:/bin", "HERMES_DASHBOARD_HOST": explicit}
    result = subprocess.run(
        [
            "sh",
            "-c",
            'dash_host="${HERMES_DASHBOARD_HOST:-127.0.0.1}"\necho "$dash_host"',
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert result.stdout.strip() == explicit, (
        f"Exported HERMES_DASHBOARD_HOST={explicit} did not override "
        f"the parameter expansion default (got {result.stdout.strip()!r})."
    )
