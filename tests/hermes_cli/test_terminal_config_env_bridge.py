import os

from hermes_cli.config import apply_terminal_config_env_bridge


_TERMINAL_ENV_KEYS = (
    "TERMINAL_ENV",
    "TERMINAL_CWD",
    "TERMINAL_TIMEOUT",
    "TERMINAL_DOCKER_FORWARD_ENV",
    "TERMINAL_DOCKER_ENV",
)


def _clear_terminal_env(monkeypatch):
    for key in _TERMINAL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_terminal_backend_config_overrides_stale_env(monkeypatch):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("TERMINAL_ENV", "local")

    apply_terminal_config_env_bridge({"terminal": {"backend": "docker", "timeout": 7}})

    assert os.environ["TERMINAL_ENV"] == "docker"
    assert os.environ["TERMINAL_TIMEOUT"] == "7"


def test_nonlocal_placeholder_cwd_is_not_bridged(monkeypatch):
    _clear_terminal_env(monkeypatch)

    apply_terminal_config_env_bridge({"terminal": {"backend": "docker", "cwd": "."}})

    assert os.environ["TERMINAL_ENV"] == "docker"
    assert "TERMINAL_CWD" not in os.environ


def test_local_placeholder_cwd_resolves_to_process_cwd(monkeypatch, tmp_path):
    _clear_terminal_env(monkeypatch)
    monkeypatch.chdir(tmp_path)

    apply_terminal_config_env_bridge({"terminal": {"backend": "local", "cwd": "."}})

    assert os.environ["TERMINAL_ENV"] == "local"
    assert os.environ["TERMINAL_CWD"] == str(tmp_path)


def test_collection_values_are_json_encoded(monkeypatch):
    _clear_terminal_env(monkeypatch)

    apply_terminal_config_env_bridge(
        {
            "terminal": {
                "backend": "docker",
                "docker_forward_env": ["FOO"],
                "docker_env": {"BAR": "baz"},
            }
        }
    )

    assert os.environ["TERMINAL_DOCKER_FORWARD_ENV"] == '["FOO"]'
    assert os.environ["TERMINAL_DOCKER_ENV"] == '{"BAR": "baz"}'


def test_ignore_user_config_skips_implicit_raw_config(monkeypatch):
    _clear_terminal_env(monkeypatch)
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"terminal": {"backend": "docker"}},
    )

    apply_terminal_config_env_bridge()

    assert "TERMINAL_ENV" not in os.environ
