import importlib
import os
import sys

import pytest

from hermes_cli.env_loader import load_hermes_dotenv


def test_user_env_overrides_stale_shell_values(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("OPENAI_BASE_URL=https://new.example/v1\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("OPENAI_BASE_URL") == "https://new.example/v1"


def test_project_env_overrides_stale_shell_values_when_user_env_missing(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    project_env = tmp_path / ".env"
    project_env.write_text("OPENAI_BASE_URL=https://project.example/v1\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [project_env]
    assert os.getenv("OPENAI_BASE_URL") == "https://project.example/v1"


def test_project_env_is_sanitized_before_loading(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    project_env = tmp_path / ".env"
    project_env.write_text(
        "TELEGRAM_BOT_TOKEN=0123456789:test"
        "ANTHROPIC_API_KEY=sk-ant-test123\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [project_env]
    assert os.getenv("TELEGRAM_BOT_TOKEN") == "0123456789:test"
    assert os.getenv("ANTHROPIC_API_KEY") == "sk-ant-test123"


def test_user_env_takes_precedence_over_project_env(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    user_env = home / ".env"
    project_env = tmp_path / ".env"
    user_env.write_text("OPENAI_BASE_URL=https://user.example/v1\n", encoding="utf-8")
    project_env.write_text("OPENAI_BASE_URL=https://project.example/v1\nOPENAI_API_KEY=project-key\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [user_env, project_env]
    assert os.getenv("OPENAI_BASE_URL") == "https://user.example/v1"
    assert os.getenv("OPENAI_API_KEY") == "project-key"


def test_null_bytes_in_user_env_are_stripped(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    # Null bytes can be introduced when copy-pasting API keys.
    env_file.write_text("GLM_API_KEY=abc\x00\x00\nOPENAI_API_KEY=sk-123\n", encoding="utf-8")

    monkeypatch.delenv("GLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("GLM_API_KEY") == "abc"
    assert os.getenv("OPENAI_API_KEY") == "sk-123"


def test_main_import_applies_user_env_over_shell_values(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "OPENAI_BASE_URL=https://new.example/v1\nHERMES_INFERENCE_PROVIDER=custom\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "openrouter")

    sys.modules.pop("hermes_cli.main", None)
    importlib.import_module("hermes_cli.main")

    assert os.getenv("OPENAI_BASE_URL") == "https://new.example/v1"
    assert os.getenv("HERMES_INFERENCE_PROVIDER") == "custom"


def test_non_utf8_env_file_loads_via_latin1_fallback(tmp_path, monkeypatch):
    """Undecodable UTF-8 bytes reread permissively as latin-1 (real dotenv path)."""
    from hermes_cli import env_loader

    env_file = tmp_path / ".env"
    env_file.write_bytes(b"LATIN_VAR=caf\xe9\n")  # 0xE9 = é in latin-1

    monkeypatch.delenv("LATIN_VAR", raising=False)

    env_loader._load_dotenv_with_fallback(env_file, override=True)

    assert os.getenv("LATIN_VAR") == "caf\xe9"


class TestDotenvTransientKeyErrorRetry:
    """The dotenv reload retries a transient KeyError from another thread
    mutating os.environ mid-``env.update`` — on BOTH encoding paths.

    Regression for the review finding that the latin-1 fallback originally
    ran outside the retry handler, so a KeyError raised there escaped
    instead of retrying.
    """

    def test_keyerror_on_utf8_path_retries(self, tmp_path, monkeypatch):
        from hermes_cli import env_loader

        calls = []

        def fake_load_dotenv(*, dotenv_path, override, encoding):
            calls.append(encoding)
            if len(calls) == 1:
                raise KeyError("VAR_VANISHED_MID_UPDATE")
            return True

        monkeypatch.setattr(env_loader, "load_dotenv", fake_load_dotenv)

        env_loader._load_dotenv_with_fallback(tmp_path / ".env", override=True)

        assert calls == ["utf-8", "utf-8"]

    def test_keyerror_on_latin1_fallback_retries(self, tmp_path, monkeypatch):
        from hermes_cli import env_loader

        calls = []

        def fake_load_dotenv(*, dotenv_path, override, encoding):
            calls.append(encoding)
            if encoding == "utf-8":
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
            if calls.count("latin-1") == 1:
                raise KeyError("VAR_VANISHED_MID_UPDATE")
            return True

        monkeypatch.setattr(env_loader, "load_dotenv", fake_load_dotenv)

        env_loader._load_dotenv_with_fallback(tmp_path / ".env", override=True)

        # Attempt 1: utf-8 → UnicodeDecodeError → latin-1 → KeyError (transient).
        # Attempt 2: utf-8 → UnicodeDecodeError → latin-1 → success.
        assert calls == ["utf-8", "latin-1", "utf-8", "latin-1"]

    def test_persistent_keyerror_reraises_after_three_attempts(
        self, tmp_path, monkeypatch
    ):
        from hermes_cli import env_loader

        calls = []

        def fake_load_dotenv(*, dotenv_path, override, encoding):
            calls.append(encoding)
            raise KeyError("ALWAYS_MISSING")

        monkeypatch.setattr(env_loader, "load_dotenv", fake_load_dotenv)

        with pytest.raises(KeyError):
            env_loader._load_dotenv_with_fallback(tmp_path / ".env", override=True)

        assert calls == ["utf-8", "utf-8", "utf-8"]
