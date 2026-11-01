"""Regression tests for Skills Hub authentication in the setup summary (#121131)."""

import pytest

from hermes_cli.setup_summary import _skills_hub_row
from tools.skills_hub_github import GitHubAuth


@pytest.mark.parametrize("source", ("GITHUB_TOKEN", "GH_TOKEN", "gh-cli", "github-app"))
def test_skills_hub_summary_accepts_every_runtime_auth_source(source, monkeypatch):
    """The summary must report the same authenticated sources as Skills Hub itself."""
    for env_var in ("GITHUB_TOKEN", "GH_TOKEN"):
        monkeypatch.delenv(env_var, raising=False)

    if source in {"GITHUB_TOKEN", "GH_TOKEN"}:
        monkeypatch.setenv(source, f"token-from-{source}")
    elif source == "gh-cli":
        monkeypatch.setattr(GitHubAuth, "_try_gh_cli", lambda self: "token-from-gh-cli")
    else:
        monkeypatch.setattr(GitHubAuth, "_try_pat", staticmethod(lambda: None))
        monkeypatch.setattr(GitHubAuth, "_try_gh_cli", lambda self: None)
        monkeypatch.setattr(
            GitHubAuth, "_try_github_app", lambda self: "token-from-github-app"
        )

    assert GitHubAuth().is_authenticated() is True
    assert _skills_hub_row({}, object()) == ("Skills Hub (GitHub)", True, None)


def test_skills_hub_summary_stays_unavailable_without_runtime_auth(monkeypatch):
    for env_var in ("GITHUB_TOKEN", "GH_TOKEN"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(GitHubAuth, "_try_pat", staticmethod(lambda: None))
    monkeypatch.setattr(GitHubAuth, "_try_gh_cli", lambda self: None)
    monkeypatch.setattr(GitHubAuth, "_try_github_app", lambda self: None)

    assert GitHubAuth().is_authenticated() is False
    assert _skills_hub_row({}, object()) == (
        "Skills Hub (GitHub)",
        False,
        "GITHUB_TOKEN",
    )
