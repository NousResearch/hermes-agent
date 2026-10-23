"""Copilot CLI fallback requires trusted launch-profile provenance (#138553)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from agent import secret_scope
from hermes_cli.copilot_auth import COPILOT_ENV_VARS, resolve_copilot_token
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def launch_home(tmp_path, monkeypatch):
    home = tmp_path / "launch"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in COPILOT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    override = set_hermes_home_override(None)
    try:
        yield home
    finally:
        reset_hermes_home_override(override)


@pytest.mark.parametrize("home_form", ["exact", "parent", "symlink"])
def test_multiplex_launch_home_allows_gh_fallback(launch_home, home_form):
    scope_home = launch_home
    if home_form == "parent":
        child = launch_home / "child"
        child.mkdir()
        scope_home = child / ".."
    elif home_form == "symlink":
        scope_home = launch_home.parent / "alias"
        scope_home.symlink_to(launch_home, target_is_directory=True)
    secret_scope.set_multiplex_active(True)
    scope = secret_scope.set_secret_scope({}, profile_home=str(scope_home))
    try:
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value="gho_launch") as cli:
            assert resolve_copilot_token() == ("gho_launch", "gh auth token")
        cli.assert_called_once_with()
    finally:
        secret_scope.reset_secret_scope(scope)
        secret_scope.set_multiplex_active(False)


@pytest.mark.parametrize("multiplex, scope_kind", [
    (False, "foreign"), (True, "foreign"), (True, "unstamped"), (True, "unbound"),
])
def test_untrusted_scope_rejects_gh_without_home_override(launch_home, multiplex, scope_kind):
    secret_scope.set_multiplex_active(multiplex)
    scope = secret_scope.set_secret_scope(
        None if scope_kind == "unbound" else {},
        profile_home=str(launch_home.parent / "foreign") if scope_kind == "foreign" else None,
    )
    try:
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value="gho_launch") as cli:
            if scope_kind == "unbound":
                with pytest.raises(secret_scope.UnscopedSecretError):
                    resolve_copilot_token()
            else:
                assert resolve_copilot_token() == ("", "")
        cli.assert_not_called()
    finally:
        secret_scope.reset_secret_scope(scope)
        secret_scope.set_multiplex_active(False)


@pytest.mark.parametrize("explicit_token", [None, "ghp_invalid"])
def test_launch_home_honors_explicit_token_intent(launch_home, explicit_token):
    # Single-profile callers need no scope; multiplex callers must stamp theirs.
    secret_scope.set_multiplex_active(explicit_token is not None)
    scope = secret_scope.set_secret_scope(
        {"COPILOT_GITHUB_TOKEN": explicit_token} if explicit_token else None,
        profile_home=str(launch_home),
    )
    try:
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value="gho_launch") as cli:
            if explicit_token:
                assert resolve_copilot_token() == ("", "")
                cli.assert_not_called()
            else:
                assert resolve_copilot_token() == ("gho_launch", "gh auth token")
                cli.assert_called_once_with()
    finally:
        secret_scope.reset_secret_scope(scope)
        secret_scope.set_multiplex_active(False)
