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
        "ANTHROPIC_API_KEY=«redacted:sk-…\u00bb\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert loaded == [project_env]
    assert os.getenv("TELEGRAM_BOT_TOKEN") == "0123456789:test"
    assert os.getenv("ANTHROPIC_API_KEY") == "«redacted:sk-…»"


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


def test_utf16_env_file_is_reencoded(tmp_path):
    """UTF-16 .env file is silently re-encoded to UTF-8 without corruption."""
    from hermes_cli.env_loader import _detect_and_convert_utf16

    env_file = tmp_path / ".env"
    # UTF-16 LE with BOM — a real file saved by Windows Notepad or similar.
    payload = "OPENAI_API_KEY=sk-abc123\nANTHROPIC_API_KEY=sk-ant-xyz\n"
    # utf-16-le encoder does not add BOM; we prepend it manually.
    encoded = payload.encode("utf-16-le")
    env_file.write_bytes(b"\xff\xfe" + encoded)

    assert _detect_and_convert_utf16(env_file) is True

    text = env_file.read_text(encoding="utf-8")
    # Keys must NOT be prefixed with U+FFFD replacement characters.
    assert text.startswith("OPENAI_API_KEY"), (
        f"Expected 'OPENAI_API_KEY...', got: {text[:50]!r}"
    )
    assert "OPENAI_API_KEY=sk-abc123" in text
    assert "ANTHROPIC_API_KEY=sk-ant-xyz" in text


def test_utf16_be_env_file_is_reencoded(tmp_path):
    """UTF-16 BE .env file is silently re-encoded to UTF-8 without corruption."""
    from hermes_cli.env_loader import _detect_and_convert_utf16

    env_file = tmp_path / ".env"
    payload = "OPENAI_API_KEY=sk-abc123\n"
    # utf-16-be encoder does not add BOM; we prepend it manually.
    encoded = payload.encode("utf-16-be")
    env_file.write_bytes(b"\xfe\xff" + encoded)

    assert _detect_and_convert_utf16(env_file) is True

    text = env_file.read_text(encoding="utf-8")
    assert text.startswith("OPENAI_API_KEY"), (
        f"Expected 'OPENAI_API_KEY...', got: {text[:50]!r}"
    )


def test_utf8_env_file_is_not_touched(tmp_path):
    """UTF-8 .env file (no UTF-16 BOM) is left untouched by the detector."""
    from hermes_cli.env_loader import _detect_and_convert_utf16

    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-abc123\n", encoding="utf-8")

    assert _detect_and_convert_utf16(env_file) is False

    assert env_file.read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-abc123\n"


def test_utf16_env_via_load_hermes_dotenv(tmp_path, monkeypatch):
    """load_hermes_dotenv on a UTF-16 .env file does not corrupt the first key."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    # UTF-16 LE with BOM
    payload = "DEEPSEEK_API_KEY=sk-ds-123\nOPENAI_API_KEY=sk-oa-456\n"
    encoded = payload.encode("utf-16-le")
    env_file.write_bytes(b"\xff\xfe" + encoded)

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    # The first key must NOT have U+FFFD prefix corruption.
    assert os.getenv("DEEPSEEK_API_KEY") == "sk-ds-123", (
        f"Expected 'sk-ds-123', got: {os.getenv('DEEPSEEK_API_KEY')!r}"
    )
    assert os.getenv("OPENAI_API_KEY") == "sk-oa-456"


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
