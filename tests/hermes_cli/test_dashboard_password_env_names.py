"""``hermes dashboard password`` finds an override in a ``.env`` under the spelling the host's environment
gives it. On Windows environment names are case-insensitive: a ``hermes_dashboard_basic_auth_password``
line is loaded as HERMES_DASHBOARD_BASIC_AUTH_PASSWORD and overrides ``config.yaml``, so the command removes
it from the home's ``.env`` and refuses while the checkout's ``.env`` has it. On Linux and macOS it is
another variable, which the provider never reads, and stays where it is.
"""
import subprocess
import sys
from pathlib import Path

import pytest

PASSWORD_ENV = "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD"
USERNAME_ENV = "HERMES_DASHBOARD_BASIC_AUTH_USERNAME"


@pytest.mark.parametrize("is_windows, expected", [
    (True, {"hermes_dashboard_basic_auth_password": PASSWORD_ENV, "Hermes_Dashboard_Basic_Auth_Username": USERNAME_ENV,
            PASSWORD_ENV: PASSWORD_ENV}),
    (False, {PASSWORD_ENV: PASSWORD_ENV}),
])
def test_an_override_matches_in_any_case_on_windows_and_exactly_elsewhere(is_windows, expected):
    from hermes_cli.dashboard_password import _overrides_among
    names = ["hermes_dashboard_basic_auth_password", "Hermes_Dashboard_Basic_Auth_Username", PASSWORD_ENV,
             "HERMES_DASHBOARD_BASIC_AUTH_SECRET", "OTHER_KEY"]
    assert _overrides_among(names, is_windows=is_windows) == expected


def _lower_case_home(tmp_path):
    """This home's ``.env`` with lower-case override lines, as ``HERMES_HOME`` (set by the test fixtures)."""
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text("hermes_dashboard_basic_auth_password=throwaway-old\n"
                               "hermes_dashboard_basic_auth_username=throwaway-env-user\nOTHER_KEY=kept\n")
    project_env = tmp_path / "checkout.env"
    project_env.write_text("hermes_dashboard_basic_auth_password=throwaway-checkout\n")
    return home, project_env


@pytest.mark.platforms("windows")
def test_on_windows_a_lower_case_override_is_removed_from_home_and_refused_in_the_checkout(tmp_path, monkeypatch):
    from hermes_cli.dashboard_password import _checkout_refusal, _clear_dotenv_shadows, _current_username
    home, project_env = _lower_case_home(tmp_path)
    for name in (PASSWORD_ENV, USERNAME_ENV):
        monkeypatch.delenv(name, raising=False)
    # The loader, in a child so this process's environment stays as it was: the lower-case line is the
    # override the provider reads (``os.environ.get(PASSWORD_ENV)``).
    loaded = subprocess.run(
        [sys.executable, "-c", "import os, sys; from pathlib import Path; from hermes_cli import env_loader; "
         "env_loader._load_dotenv_with_fallback(Path(sys.argv[1]), override=True); "
         f"print(os.environ.get({PASSWORD_ENV!r}))", str(home / ".env")],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=120, check=True)
    assert loaded.stdout.strip() == "throwaway-old"
    assert _current_username() == "throwaway-env-user"
    refusal = _checkout_refusal(project_env, home)  # the home copy goes, so the checkout's fills the gap
    assert refusal is not None and PASSWORD_ENV in refusal and "throwaway-checkout" not in refusal
    assert _clear_dotenv_shadows() == ([USERNAME_ENV, PASSWORD_ENV], [])
    assert (home / ".env").read_text() == "OTHER_KEY=kept\n"
    assert _checkout_refusal(project_env, home) is not None
    project_env.write_text("OTHER_KEY=x\n")
    assert _checkout_refusal(project_env, home) is None


@pytest.mark.platforms("posix")
def test_on_posix_a_lower_case_name_is_another_variable_and_stays(tmp_path):
    from hermes_cli.dashboard_password import _checkout_refusal, _clear_dotenv_shadows, _current_username
    home, project_env = _lower_case_home(tmp_path)
    before = (home / ".env").read_text()
    assert _current_username() == ""
    assert _checkout_refusal(project_env, home) is None
    assert _clear_dotenv_shadows() == ([], [])
    assert (home / ".env").read_text() == before
