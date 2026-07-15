import importlib
import os
import sys

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


def test_utf8_bom_does_not_mangle_first_key(tmp_path, monkeypatch):
    """A leading UTF-8 BOM must not prefix the first key name in os.environ.

    PowerShell 5.1 ``Set-Content -Encoding UTF8`` and Windows Notepad write
    a BOM (EF BB BF). With encoding=utf-8, python-dotenv keeps U+FEFF on the
    first key so the canonical name is absent and callers see "not configured".
    """
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_bytes(
        b"\xef\xbb\xbfFIRST_KEY=first-value\nSECOND_KEY=second-value\n"
    )

    monkeypatch.delenv("FIRST_KEY", raising=False)
    monkeypatch.delenv("SECOND_KEY", raising=False)
    monkeypatch.delenv("\ufeffFIRST_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("FIRST_KEY") == "first-value"
    assert os.getenv("SECOND_KEY") == "second-value"
    assert os.environ.get("\ufeffFIRST_KEY") is None


def test_bomless_utf8_env_still_loads(tmp_path, monkeypatch):
    """BOM-less UTF-8 .env files must keep loading after utf-8-sig."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-plain\nSECOND_KEY=ok\n", encoding="utf-8")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SECOND_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("OPENAI_API_KEY") == "sk-plain"
    assert os.getenv("SECOND_KEY") == "ok"


def test_latin1_env_falls_back(tmp_path, monkeypatch):
    """Invalid UTF-8 bytes must still load via the latin-1 fallback."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    # 0xE9 is "é" in latin-1 and not a valid UTF-8 lead sequence alone.
    env_file.write_bytes(b"LATIN1_VALUE=caf\xe9\n")

    monkeypatch.delenv("LATIN1_VALUE", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("LATIN1_VALUE") == "café"


def test_utf8_bom_preserves_first_api_key_name(tmp_path, monkeypatch):
    """Real-world case: BOM + first line is a provider API key name."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    env_file.write_bytes(
        b"\xef\xbb\xbfANTHROPIC_API_KEY=sk-test-123\nSECOND_KEY=ok\n"
    )

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("SECOND_KEY", raising=False)
    monkeypatch.delenv("\ufeffANTHROPIC_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("ANTHROPIC_API_KEY") == "sk-test-123"
    assert os.getenv("SECOND_KEY") == "ok"
    assert os.environ.get("\ufeffANTHROPIC_API_KEY") is None


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
