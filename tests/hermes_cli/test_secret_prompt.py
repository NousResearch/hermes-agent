import time

import pytest

from hermes_cli.secret_prompt import (
    _collect_masked_input,
    capture_pre_dotenv_rotation_inputs,
    cli_secret_arg_warning,
    get_pre_dotenv_rotation_input,
    masked_secret_prompt,
    reset_pre_dotenv_rotation_inputs,
)


def _run_collect(chars: str):
    output: list[str] = []
    iterator = iter(chars)

    def read_char() -> str:
        return next(iterator, "")

    def write(text: str) -> None:
        output.append(text)

    value = _collect_masked_input(
        read_char,
        write,
        "API key: ",
    )
    return value, "".join(output)


def test_collect_masked_input_shows_feedback_without_echoing_secret():
    value, output = _run_collect("secret\n")

    assert value == "secret"
    assert output == "API key: ******\r\n"
    assert "secret" not in output




def test_collect_masked_input_raises_keyboard_interrupt():
    output: list[str] = []

    with pytest.raises(KeyboardInterrupt):
        _collect_masked_input(
            lambda: "\x03",
            output.append,
            "API key: ",
        )

    assert "".join(output) == "API key: \r\n"


def test_masked_secret_prompt_falls_back_to_getpass_for_non_tty(monkeypatch):
    class NonTty:
        def isatty(self):
            return False

    monkeypatch.setattr("sys.stdin", NonTty())
    monkeypatch.setattr("sys.stdout", NonTty())
    monkeypatch.setattr("getpass.getpass", lambda prompt: f"value from {prompt}")

    assert masked_secret_prompt("API key: ") == "value from API key: "


def test_cli_secret_arg_warning_does_not_include_secret_value():
    warning = cli_secret_arg_warning("--token", "OP_SERVICE_ACCOUNT_TOKEN")

    assert "process listings" in warning
    assert "shell history" in warning
    assert "CI logs" in warning
    assert "OP_SERVICE_ACCOUNT_TOKEN" in warning


def test_pre_dotenv_rotation_capture_keeps_only_configured_token(monkeypatch):
    monkeypatch.setenv("CUSTOM_BW_TOKEN", "injected-new")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-secret")

    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "bitwarden", "token"],
        config={
            "secrets": {
                "bitwarden": {"access_token_env": "CUSTOM_BW_TOKEN"},
            },
        },
    )

    assert get_pre_dotenv_rotation_input("CUSTOM_BW_TOKEN") == "injected-new"
    assert get_pre_dotenv_rotation_input("OPENAI_API_KEY") == ""
    reset_pre_dotenv_rotation_inputs()


@pytest.mark.parametrize(
    ("provider", "config_key", "config_section"),
    [
        ("bitwarden", "access_token_env", "bitwarden"),
        ("onepassword", "service_account_token_env", "onepassword"),
    ],
)
@pytest.mark.parametrize("template", ["${TOKEN_ENV_NAME}", "${env:TOKEN_ENV_NAME}"])
def test_pre_dotenv_capture_expands_provider_token_name(
    monkeypatch, provider, config_key, config_section, template
):
    monkeypatch.setenv("TOKEN_ENV_NAME", "CUSTOM_PROVIDER_TOKEN")
    monkeypatch.setenv("CUSTOM_PROVIDER_TOKEN", "injected-new")

    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", provider, "setup"],
        config={
            "secrets": {
                config_section: {config_key: template},
            },
        },
    )

    assert get_pre_dotenv_rotation_input("CUSTOM_PROVIDER_TOKEN") == "injected-new"
    assert get_pre_dotenv_rotation_input(template) == ""


@pytest.mark.parametrize(
    ("provider", "config_key", "config_section"),
    [
        ("bitwarden", "access_token_env", "bitwarden"),
        ("onepassword", "service_account_token_env", "onepassword"),
    ],
)
@pytest.mark.parametrize("template", ["${TOKEN_ENV_NAME}", "${env:TOKEN_ENV_NAME}"])
def test_pre_dotenv_capture_expands_managed_provider_token_name(
    monkeypatch, tmp_path, provider, config_key, config_section, template
):
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        "secrets:\n"
        f"  {config_section}:\n"
        f"    {config_key}: {template}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("TOKEN_ENV_NAME", "MANAGED_PROVIDER_TOKEN")
    monkeypatch.setenv("MANAGED_PROVIDER_TOKEN", "injected-new")

    from hermes_cli import managed_scope

    managed_scope.invalidate_managed_cache()
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", provider, "setup"], config={}
    )

    assert get_pre_dotenv_rotation_input("MANAGED_PROVIDER_TOKEN") == "injected-new"
    assert get_pre_dotenv_rotation_input(template) == ""


@pytest.mark.parametrize("provider", ["onepassword", "op", "1password"])
@pytest.mark.parametrize(
    "option",
    ["--token-env CUSTOM_OP_TOKEN", "--token-env=CUSTOM_OP_TOKEN"],
)
def test_pre_dotenv_capture_includes_cli_selected_onepassword_token_env(
    monkeypatch, provider, option
):
    monkeypatch.setenv("CUSTOM_OP_TOKEN", "injected-cli-custom")
    option_args = option.split(" ")

    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", provider, "setup", *option_args],
        config={"secrets": {"onepassword": {}}},
    )

    assert (
        get_pre_dotenv_rotation_input("CUSTOM_OP_TOKEN")
        == "injected-cli-custom"
    )


@pytest.mark.parametrize("template", ["${TOKEN_ENV_NAME}", "${env:TOKEN_ENV_NAME}"])
@pytest.mark.parametrize(
    ("provider", "config_key", "config_section"),
    [
        ("bitwarden", "access_token_env", "bitwarden"),
        ("onepassword", "service_account_token_env", "onepassword"),
    ],
)
def test_pre_dotenv_capture_resolves_user_dotenv_token_name(
    monkeypatch, tmp_path, provider, config_key, config_section, template
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TOKEN_ENV_NAME=CUSTOM_PROVIDER_TOKEN\n"
        "CUSTOM_PROVIDER_TOKEN=stale-dotenv-token\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TOKEN_ENV_NAME", raising=False)
    monkeypatch.setenv("CUSTOM_PROVIDER_TOKEN", "injected-before-dotenv")

    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", provider, "setup"],
        config={
            "secrets": {
                config_section: {config_key: template},
            },
        },
        dotenv_sources=[(env_file, True)],
    )

    assert (
        get_pre_dotenv_rotation_input("CUSTOM_PROVIDER_TOKEN")
        == "injected-before-dotenv"
    )
    assert get_pre_dotenv_rotation_input("stale-dotenv-token") == ""


def test_pre_dotenv_capture_rejects_overlong_dotenv_name_chain_quickly(
    monkeypatch, tmp_path
):
    """A persisted long reference chain cannot exhaust startup resolution."""
    env_file = tmp_path / ".env"
    assignments = [f"N{index}=${{N{index + 1}}}" for index in range(128)]
    assignments.extend(
        [
            "N128=CUSTOM_PROVIDER_TOKEN",
            "CUSTOM_PROVIDER_TOKEN=stale-dotenv-token",
            "UNRELATED_SECRET=must-not-be-snapshotted",
        ]
    )
    env_file.write_text("\n".join(assignments) + "\n", encoding="utf-8")
    monkeypatch.setenv("CUSTOM_PROVIDER_TOKEN", "injected-before-dotenv")
    monkeypatch.setenv("UNRELATED_SECRET", "unrelated-injected")

    started = time.perf_counter()
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "onepassword", "setup"],
        config={
            "secrets": {
                "onepassword": {
                    "service_account_token_env": "${N0}",
                },
            },
        },
        dotenv_sources=[(env_file, True)],
    )

    assert time.perf_counter() - started < 1.0
    assert get_pre_dotenv_rotation_input("CUSTOM_PROVIDER_TOKEN") == ""
    assert get_pre_dotenv_rotation_input("UNRELATED_SECRET") == ""


def test_pre_dotenv_capture_handles_dotenv_name_cycle_without_snapshotting(
    monkeypatch, tmp_path
):
    """A selected cyclic graph terminates and remains fail-closed."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FIRST_NAME=${SECOND_NAME}\n"
        "SECOND_NAME=${FIRST_NAME}\n"
        "CUSTOM_PROVIDER_TOKEN=stale-dotenv-token\n"
        "UNRELATED_SECRET=must-not-be-snapshotted\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CUSTOM_PROVIDER_TOKEN", "injected-before-dotenv")
    monkeypatch.setenv("UNRELATED_SECRET", "unrelated-injected")

    started = time.perf_counter()
    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "onepassword", "setup"],
        config={
            "secrets": {
                "onepassword": {
                    "service_account_token_env": "${FIRST_NAME}",
                },
            },
        },
        dotenv_sources=[(env_file, True)],
    )

    assert time.perf_counter() - started < 1.0
    assert get_pre_dotenv_rotation_input("CUSTOM_PROVIDER_TOKEN") == ""
    assert get_pre_dotenv_rotation_input("UNRELATED_SECRET") == ""


@pytest.mark.parametrize("command", ["setup", "token"])
@pytest.mark.parametrize(
    "op_assignments",
    [
        'TOKEN_ENV_NAME="CUSTOM_PROVIDER_TOKEN" # trailing comment\n',
        "TOKEN_ENV_NAME='CUSTOM_PROVIDER_TOKEN' # trailing comment\n",
        "SECOND_NAME=CUSTOM_PROVIDER_TOKEN\n"
        "TOKEN_ENV_NAME=${SECOND_NAME}\n",
    ],
)
@pytest.mark.parametrize("template", ["${TOKEN_ENV_NAME}", "${env:TOKEN_ENV_NAME}"])
def test_pre_dotenv_capture_resolves_op_env_binding_and_chain(
    monkeypatch, tmp_path, command, op_assignments, template
):
    user_env = tmp_path / ".env"
    user_env.write_text(
        "CUSTOM_PROVIDER_TOKEN=stale-user-dotenv-token\n",
        encoding="utf-8",
    )
    op_env = tmp_path / ".op.env"
    op_env.write_text(
        op_assignments + "CUSTOM_PROVIDER_TOKEN=stale-op-dotenv-token\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TOKEN_ENV_NAME", raising=False)
    monkeypatch.delenv("SECOND_NAME", raising=False)
    monkeypatch.setenv("CUSTOM_PROVIDER_TOKEN", "injected-before-dotenv")

    capture_pre_dotenv_rotation_inputs(
        ["hermes", "secrets", "onepassword", command],
        config={
            "secrets": {
                "onepassword": {
                    "service_account_token_env": template,
                },
            },
        },
        dotenv_sources=[(user_env, True), (op_env, False)],
    )

    assert (
        get_pre_dotenv_rotation_input("CUSTOM_PROVIDER_TOKEN")
        == "injected-before-dotenv"
    )
