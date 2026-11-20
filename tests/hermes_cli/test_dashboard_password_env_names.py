"""``hermes dashboard password`` looks an override up in a ``.env`` by its exact name. On Linux and macOS a
``hermes_dashboard_basic_auth_password`` line is another variable, which the provider never reads, so it
stays where it is and nothing is refused for it.
"""
import pytest

PASSWORD_ENV = "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD"
USERNAME_ENV = "HERMES_DASHBOARD_BASIC_AUTH_USERNAME"


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


@pytest.mark.platforms("posix")
def test_on_posix_a_lower_case_name_is_another_variable_and_stays(tmp_path):
    from hermes_cli.dashboard_password import _checkout_refusal, _clear_dotenv_shadows, _current_username
    home, project_env = _lower_case_home(tmp_path)
    before = (home / ".env").read_text()
    assert _current_username() == ""
    assert _checkout_refusal(project_env, home) is None
    assert _clear_dotenv_shadows() == ([], [])
    assert (home / ".env").read_text() == before
