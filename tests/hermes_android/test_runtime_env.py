import os

from hermes_android.runtime_env import prepare_runtime_env


def test_prepare_runtime_env_sets_android_env_and_dirs(tmp_path, monkeypatch):
    files_dir = tmp_path / "files"
    runtime = prepare_runtime_env(
        files_dir,
        api_server_port=8765,
        api_server_key="secret-key",
    )

    assert runtime.files_dir == files_dir.resolve()
    assert runtime.hermes_home == files_dir.resolve() / "hermes-home"
    assert runtime.api_server_host == "127.0.0.1"
    assert runtime.api_server_port == 8765
    assert runtime.api_server_key == "secret-key"

    for child in ("logs", "sessions", "skills", "downloads", "workspace"):
        assert (runtime.hermes_home / child).is_dir()

    assert os.environ["HERMES_HOME"] == str(runtime.hermes_home)
    assert os.environ["HERMES_ANDROID_BOOTSTRAP"] == "1"
    assert os.environ["API_SERVER_HOST"] == "127.0.0.1"
    assert os.environ["API_SERVER_PORT"] == "8765"
    assert os.environ["API_SERVER_KEY"] == "secret-key"
    assert os.environ["API_SERVER_MODEL_NAME"] == "hermes-agent-android"


def test_prepare_runtime_env_generates_port_and_key(tmp_path):
    runtime = prepare_runtime_env(tmp_path / "files")

    assert runtime.api_server_port > 0
    assert runtime.api_server_key
