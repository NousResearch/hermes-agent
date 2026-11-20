"""``hermes dashboard password`` looks an override up in a ``.env`` by its exact name. On Linux and macOS a
``hermes_dashboard_basic_auth_password`` line is another variable, which the provider never reads, so it
stays where it is and nothing is refused for it. On Windows environment names ignore case, so the same line
is HERMES_DASHBOARD_BASIC_AUTH_PASSWORD there and would override the new password: the command refuses,
before writing anything, and names the file, line and key as spelled (never the value) so it can be fixed.
"""
import hashlib

import pytest

import hermes_yaml as yaml
from plugins.dashboard_auth.basic import _verify_password
from tests.hermes_cli.live_gateway import home_env, run_cli, write_config

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


def test_a_quoted_key_at_home_does_not_hide_the_checkouts_override(tmp_path):
    """The loader reads ``'KEY'=v`` as KEY, but this home's cleanup (exact names) does not remove that line,
    so the checkout's own KEY only fills a gap that is not there: still refused, as it always was."""
    from hermes_cli.dashboard_password import _checkout_refusal
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(f"'{PASSWORD_ENV}'=throwaway-old\n")
    project_env = tmp_path / "checkout.env"
    project_env.write_text(f"{PASSWORD_ENV}=throwaway-checkout\n")
    refusal = _checkout_refusal(project_env, home)
    assert refusal is not None and PASSWORD_ENV in refusal and "throwaway" not in refusal


def _env_files(tmp_path, home_text: str, checkout_text: str = "OTHER_KEY=x\n"):
    home_env_file, checkout_env = tmp_path / "home.env", tmp_path / "checkout.env"
    home_env_file.write_text(home_text)
    checkout_env.write_text(checkout_text)
    return home_env_file, checkout_env


@pytest.mark.parametrize("spelled", ["hermes_dashboard_basic_auth_password", "Hermes_Dashboard_Basic_Auth_Password",
                                     "hermes_dashboard_basic_auth_username", "HERMES_DASHBOARD_BASIC_AUTH_UserName"])
def test_on_windows_an_override_spelled_in_another_case_is_refused_by_file_line_and_spelling(tmp_path, spelled):
    from hermes_cli.dashboard_password import _case_variant_refusal
    home_env_file, checkout_env = _env_files(tmp_path, f"OTHER_KEY=a\n{spelled}=throwaway-secret-value\n")
    refusal = _case_variant_refusal([home_env_file, checkout_env], is_windows=True)
    assert refusal is not None
    assert f"{home_env_file} line 2 ({spelled})" in refusal
    assert spelled.upper() in refusal and "delete or rename" in refusal
    assert "throwaway" not in refusal
    assert str(checkout_env) not in refusal


def test_on_windows_the_checkouts_env_is_checked_too_and_every_line_is_named(tmp_path):
    from hermes_cli.dashboard_password import _case_variant_refusal
    home_env_file, checkout_env = _env_files(
        tmp_path, f"{PASSWORD_ENV}=throwaway-a\nhermes_dashboard_basic_auth_username=throwaway-b\n",
        "x=1\nexport hermes_dashboard_basic_auth_password=throwaway-c\nHermes_Dashboard_Basic_Auth_Username=\n")
    refusal = _case_variant_refusal([home_env_file, checkout_env], is_windows=True)
    assert f"{home_env_file} line 2 (hermes_dashboard_basic_auth_username)" in refusal
    assert (f"{checkout_env} line 2 (hermes_dashboard_basic_auth_password), "
            "line 3 (Hermes_Dashboard_Basic_Auth_Username)") in refusal
    assert "line 1" not in refusal and "throwaway" not in refusal


@pytest.mark.parametrize("text", [
    f"{PASSWORD_ENV}=throwaway-a\n{USERNAME_ENV}=throwaway-b\n",  # the canonical spelling (cleaned up as before)
    "X_hermes_dashboard_basic_auth_password=throwaway\nhermes_dashboard_basic_auth_password_old=throwaway\n"
    "# hermes_dashboard_basic_auth_password=throwaway\nhermes_dashboard_basic_auth_username\n",
])
def test_on_windows_the_canonical_spelling_and_other_names_are_not_refused(tmp_path, text):
    from hermes_cli.dashboard_password import _case_variant_refusal
    home_env_file, checkout_env = _env_files(tmp_path, text, text)
    assert _case_variant_refusal([home_env_file, checkout_env, tmp_path / "absent.env"], is_windows=True) is None


def test_on_windows_a_look_alike_line_inside_a_quoted_value_is_refused_too(tmp_path):
    """Lines are read one at a time, as this command reads its ``.env``: a ``p=...`` line inside a quoted
    multi-line value counts. A refusal is safe; renaming or deleting it is the user's call."""
    from hermes_cli.dashboard_password import _case_variant_refusal
    home_env_file, checkout_env = _env_files(
        tmp_path, 'NOTE="first\nhermes_dashboard_basic_auth_password=text\n"\n')
    assert f"{home_env_file} line 2 (hermes_dashboard_basic_auth_password)" in (
        _case_variant_refusal([home_env_file, checkout_env], is_windows=True) or "")


def test_off_windows_the_case_check_does_not_run(tmp_path):
    from hermes_cli.dashboard_password import _case_variant_refusal
    home_env_file, checkout_env = _env_files(tmp_path, "hermes_dashboard_basic_auth_password=throwaway\n",
                                             "Hermes_Dashboard_Basic_Auth_Username=throwaway\n")
    assert _case_variant_refusal([home_env_file, checkout_env], is_windows=False) is None


def _cli_home(tmp_path, dotenv: str):
    home, user = tmp_path / "state", tmp_path / "user"
    home.mkdir()
    user.mkdir()
    write_config(home)
    (home / ".env").write_text(dotenv)
    env = home_env(home, user)
    for name in (PASSWORD_ENV, USERNAME_ENV):
        env.pop(name, None)
    return home, env


def _digests(home) -> dict:
    return {name: hashlib.sha256((home / name).read_bytes()).hexdigest() for name in ("config.yaml", ".env")}


def _basic_auth(home) -> dict:
    return (yaml.safe_load((home / "config.yaml").read_text()).get("dashboard") or {}).get("basic_auth") or {}


def _set_password(env):
    return run_cli(env, "dashboard", "password", "--username", "spike-admin", "--password-stdin",
                   stdin="throwaway-new-Pw-1\n")


@pytest.mark.platforms("windows")
def test_on_windows_the_command_refuses_a_lower_case_override_and_succeeds_once_it_is_fixed(tmp_path):
    home, env = _cli_home(tmp_path, "OTHER_KEY=kept\nhermes_dashboard_basic_auth_password=throwaway-old-Pw\n")
    before = _digests(home)
    result = _set_password(env)
    assert result.returncode == 1, result.stdout
    assert f"{home / '.env'} line 2 (hermes_dashboard_basic_auth_password)" in result.stdout
    assert "delete or rename" in result.stdout and "nothing was changed" in result.stdout
    assert "throwaway" not in result.stdout
    assert _digests(home) == before
    (home / ".env").write_text("OTHER_KEY=kept\nOLD_hermes_dashboard_basic_auth_password=throwaway-old-Pw\n")
    fixed = _set_password(env)
    assert fixed.returncode == 0, fixed.stdout
    assert "throwaway" not in fixed.stdout
    assert (home / ".env").read_text() == "OTHER_KEY=kept\nOLD_hermes_dashboard_basic_auth_password=throwaway-old-Pw\n"


@pytest.mark.platforms("posix")
def test_on_posix_the_command_leaves_a_lower_case_line_alone_and_sets_the_password(tmp_path):
    dotenv = "OTHER_KEY=kept\nhermes_dashboard_basic_auth_password=throwaway-old-Pw\n"
    home, env = _cli_home(tmp_path, dotenv)
    result = _set_password(env)
    assert result.returncode == 0, result.stdout
    assert "line 2" not in result.stdout and "throwaway" not in result.stdout
    assert (home / ".env").read_text() == dotenv
    assert _verify_password("throwaway-new-Pw-1", _basic_auth(home)["password_hash"])
