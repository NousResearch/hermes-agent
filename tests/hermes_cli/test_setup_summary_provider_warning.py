"""Setup summary must warn loudly when no provider got configured.

Regression test for the "wizard silently succeeds with no model" dead end:
cancelling the API-key prompt mid-wizard printed "Cancelled." but the wizard
continued through the remaining sections and finished "successfully" with no
working model configured (consumer-onboarding audit finding #7, Aug 2026).
"""

from unittest.mock import patch

from hermes_cli.auth import AuthError
from hermes_cli import nous_subscription


def test_skills_hub_summary_accepts_gh_cli_auth(monkeypatch):
    """A keyring-backed ``gh auth login`` is enough for Skills Hub access."""
    from hermes_cli import setup
    from hermes_cli.setup_summary import _skills_hub_row

    monkeypatch.setattr(setup, "get_env_value", lambda key: "")
    monkeypatch.setattr("hermes_cli.doctor_state._gh_authenticated", lambda: True)

    assert _skills_hub_row({}, None) == ("Skills Hub (GitHub)", True, None)


def test_skills_hub_summary_accepts_github_token(monkeypatch):
    """A configured token remains the primary Skills Hub auth path."""
    from hermes_cli import setup
    from hermes_cli.setup_summary import _skills_hub_row

    monkeypatch.setattr(setup, "get_env_value", lambda key: "github-token")
    monkeypatch.setattr(
        "hermes_cli.doctor_state._gh_authenticated",
        lambda: (_ for _ in ()).throw(AssertionError("token auth must not invoke gh")),
    )

    assert _skills_hub_row({}, None) == ("Skills Hub (GitHub)", True, None)


def test_skills_hub_summary_reports_missing_auth(monkeypatch):
    """The summary remains unavailable when neither supported auth source exists."""
    from hermes_cli import setup
    from hermes_cli.setup_summary import _skills_hub_row

    monkeypatch.setattr(setup, "get_env_value", lambda key: "")
    monkeypatch.setattr("hermes_cli.doctor_state._gh_authenticated", lambda: False)

    assert _skills_hub_row({}, None) == ("Skills Hub (GitHub)", False, "GITHUB_TOKEN or gh auth login")


def _summary_output(capsys, provider_ready: bool):
    from hermes_cli import setup as setup_mod

    if provider_ready:
        resolver = lambda *a, **k: "openrouter"  # noqa: E731
    else:
        def resolver(*a, **k):
            raise AuthError(
                "No inference provider configured.",
                code="no_provider_configured",
            )

    # Keep the summary fast/hermetic: stub the heavier feature probes.
    with patch("hermes_cli.auth.resolve_provider", resolver), \
         patch.object(nous_subscription, "get_nous_subscription_features") as feats:
        feats.side_effect = Exception("stubbed")
        try:
            setup_mod._print_setup_summary({}, "/tmp/nowhere")
        except Exception:
            # Downstream summary sections may fail from the stubbed
            # features — the provider warning prints first and is what
            # this test asserts on.
            pass
    return capsys.readouterr().out


def test_summary_warns_when_no_provider(capsys):
    out = _summary_output(capsys, provider_ready=False)
    assert "No inference provider is configured" in out
    assert "hermes model" in out
    assert "hermes setup --portal" in out


def test_summary_quiet_when_provider_ready(capsys):
    out = _summary_output(capsys, provider_ready=True)
    assert "No inference provider is configured" not in out
