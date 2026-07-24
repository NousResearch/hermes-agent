"""Tests for the boot-time git-credential provisioning wired into
``hermes_cli.container_boot.main`` behind the ``HERMES_GIT_CREDENTIALS_BOOT``
gate.

The provisioning primitive itself (``provision_all`` /
``provision_git_credentials``) is covered exhaustively by
``test_git_credentials_boot.py``. These tests cover the *boot lifecycle
decision*: that ``main()`` provisions ONLY when the gate is explicitly
enabled, never provisions in a dashboard container, and — the security
invariant — provisions nothing when the gate env var is absent (default OFF).

No real tokens: placeholders only (``ghp_test``, ``alpha_tok``).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import container_boot


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _inert_reconcile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``reconcile_profile_gateways`` a no-op returning no actions.

    The reconciler walks/writes s6 service slots under a real scandir; these
    tests only care about the credential-provisioning branch, so stub it out.
    """
    monkeypatch.setattr(
        container_boot,
        "reconcile_profile_gateways",
        lambda **_kwargs: [],
    )


@pytest.fixture(autouse=True)
def _gateway_container_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default the container role to a NON-dashboard (gateway) container.

    ``main()`` early-returns for dashboard containers before provisioning.
    Empty argv is treated as a gateway container (``_is_dashboard_container``
    returns False), so provisioning is reachable. The dashboard test overrides
    this.
    """
    monkeypatch.setattr(
        container_boot,
        "_read_container_argv",
        lambda: (),
    )


def _write_env(dir_path: Path, body: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / ".env").write_text(body, encoding="utf-8")


def _cred_path(hermes_home: Path) -> Path:
    return hermes_home / "home" / ".git-credentials"


# ---------------------------------------------------------------------------
# Gate ON
# ---------------------------------------------------------------------------


def test_boot_provisions_when_gate_enabled_against_isolated_hermes_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GIT_CREDENTIALS_BOOT", "1")
    # HERMES_AGENT_NAME unset — identity falls back to the home basename.
    monkeypatch.delenv("HERMES_AGENT_NAME", raising=False)
    _write_env(tmp_path, "GITHUB_PAT=ghp_test\n")

    rc = container_boot.main()

    assert rc == 0
    cred = _cred_path(tmp_path)
    assert cred.read_text() == "https://x-access-token:ghp_test@github.com\n"
    assert (cred.stat().st_mode & 0o777) == 0o600


def test_boot_provisions_named_profiles_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GIT_CREDENTIALS_BOOT", "true")
    monkeypatch.delenv("HERMES_AGENT_NAME", raising=False)
    # Root profile has no token; only the named profile does.
    _write_env(tmp_path / "profiles" / "alpha", "GITHUB_PAT=alpha_tok\n")

    rc = container_boot.main()

    assert rc == 0
    alpha_cred = tmp_path / "profiles" / "alpha" / "home" / ".git-credentials"
    assert alpha_cred.exists()
    assert "alpha_tok" in alpha_cred.read_text()
    assert (alpha_cred.stat().st_mode & 0o777) == 0o600


# ---------------------------------------------------------------------------
# Gate OFF / default OFF  (SECURITY INVARIANT)
# ---------------------------------------------------------------------------


def test_boot_does_not_provision_when_gate_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GIT_CREDENTIALS_BOOT", "0")
    _write_env(tmp_path, "GITHUB_PAT=ghp_test\n")

    rc = container_boot.main()

    assert rc == 0
    assert not _cred_path(tmp_path).exists()


def test_boot_gate_default_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SECURITY INVARIANT: with the gate env var entirely absent, boot must
    provision nothing — even when a token IS present in the profile's .env.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_GIT_CREDENTIALS_BOOT", raising=False)
    _write_env(tmp_path, "GITHUB_PAT=ghp_test\n")

    rc = container_boot.main()

    assert rc == 0
    assert not _cred_path(tmp_path).exists()


# ---------------------------------------------------------------------------
# Dashboard container never provisions, even with the gate ON
# ---------------------------------------------------------------------------


def test_dashboard_container_never_provisions_even_when_gate_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GIT_CREDENTIALS_BOOT", "1")
    _write_env(tmp_path, "GITHUB_PAT=ghp_test\n")
    # A dashboard-only container: main() early-returns before provisioning.
    monkeypatch.setattr(
        container_boot,
        "_read_container_argv",
        lambda: ("dashboard",),
    )

    rc = container_boot.main()

    assert rc == 0
    assert not _cred_path(tmp_path).exists()
    # And no named-profile creds either.
    assert not list(tmp_path.glob("profiles/*/home/.git-credentials"))
