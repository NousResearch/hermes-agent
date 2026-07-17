import codecs
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


# ---------------------------------------------------------------------------
# UTF-16 / UTF-32 .env sanitizer coverage
#
# Scope note: intentionally NO UTF-8-BOM assertions here. UTF-8 BOM handling
# for _load_dotenv_with_fallback is #65124's un-merged fix; a test here would
# couple the PRs. This suite covers only the sanitizer rewrite path for
# UTF-16/32 (and UTF-8 / cp1252 regression guards for that path).
# ---------------------------------------------------------------------------


def _assert_clean_utf8_env_on_disk(env_file, *, first_key: str) -> None:
    """On-disk file must be clean UTF-8: no BOM, no U+FFFD, canonical key."""
    after = env_file.read_bytes()
    assert not after.startswith(codecs.BOM_UTF8)
    assert not after.startswith(codecs.BOM_UTF16_LE)
    assert not after.startswith(codecs.BOM_UTF16_BE)
    text = after.decode("utf-8")  # strict — raises if not clean UTF-8
    assert "\ufffd" not in text
    assert text.startswith(f"{first_key}=") or f"\n{first_key}=" in text
    assert first_key.encode("ascii") in after


def test_utf16_le_bom_env_loads_and_rewrites_clean_utf8(tmp_path, monkeypatch):
    """Notepad 'Unicode' (UTF-16-LE + BOM): first key loads; file rewritten UTF-8."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    content = "HERMES_TEST_KEY=hello_utf16\nSECOND_KEY=world\n"
    env_file.write_bytes(codecs.BOM_UTF16_LE + content.encode("utf-16-le"))

    monkeypatch.delenv("HERMES_TEST_KEY", raising=False)
    monkeypatch.delenv("SECOND_KEY", raising=False)
    monkeypatch.delenv("\ufffd\ufffdHERMES_TEST_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("HERMES_TEST_KEY") == "hello_utf16"
    assert os.getenv("SECOND_KEY") == "world"
    assert os.environ.get("\ufffd\ufffdHERMES_TEST_KEY") is None
    _assert_clean_utf8_env_on_disk(env_file, first_key="HERMES_TEST_KEY")


def test_utf16_be_bom_env_loads_and_rewrites_clean_utf8(tmp_path, monkeypatch):
    """UTF-16-BE + BOM: first key loads; file rewritten as clean UTF-8."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    content = "HERMES_TEST_KEY=hello_utf16\nSECOND_KEY=world\n"
    env_file.write_bytes(codecs.BOM_UTF16_BE + content.encode("utf-16-be"))

    monkeypatch.delenv("HERMES_TEST_KEY", raising=False)
    monkeypatch.delenv("SECOND_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("HERMES_TEST_KEY") == "hello_utf16"
    assert os.getenv("SECOND_KEY") == "world"
    _assert_clean_utf8_env_on_disk(env_file, first_key="HERMES_TEST_KEY")


def test_utf16_le_no_bom_still_repairs_to_utf8(tmp_path, monkeypatch):
    """BOM-less UTF-16-LE: NUL-strip repair is now intentional; rewrites UTF-8."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    content = "HERMES_TEST_KEY=hello_utf16\nSECOND_KEY=world\n"
    env_file.write_bytes(content.encode("utf-16-le"))  # no BOM

    monkeypatch.delenv("HERMES_TEST_KEY", raising=False)
    monkeypatch.delenv("SECOND_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("HERMES_TEST_KEY") == "hello_utf16"
    assert os.getenv("SECOND_KEY") == "world"
    _assert_clean_utf8_env_on_disk(env_file, first_key="HERMES_TEST_KEY")


def test_utf16_be_no_bom_still_repairs_to_utf8(tmp_path, monkeypatch):
    """BOM-less UTF-16-BE: NULs are on the opposite side; still repairs."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    content = "HERMES_TEST_KEY=hello_utf16\nSECOND_KEY=world\n"
    env_file.write_bytes(content.encode("utf-16-be"))  # no BOM

    monkeypatch.delenv("HERMES_TEST_KEY", raising=False)
    monkeypatch.delenv("SECOND_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("HERMES_TEST_KEY") == "hello_utf16"
    assert os.getenv("SECOND_KEY") == "world"
    _assert_clean_utf8_env_on_disk(env_file, first_key="HERMES_TEST_KEY")


def test_utf16_le_bom_preserves_non_ascii_values(tmp_path, monkeypatch):
    """UTF-16-LE+BOM rewrite must preserve non-ASCII values (not just ASCII keys).

    Uses non-credential var names so _sanitize_loaded_credentials does not
    strip non-ASCII from values (that path only targets *_KEY/*_TOKEN/etc.).
    """
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    content = "GREETING=café\nCJK_LABEL=日本語\n"
    env_file.write_bytes(codecs.BOM_UTF16_LE + content.encode("utf-16-le"))

    monkeypatch.delenv("GREETING", raising=False)
    monkeypatch.delenv("CJK_LABEL", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("GREETING") == "café"
    assert os.getenv("CJK_LABEL") == "日本語"
    after = env_file.read_bytes()
    assert after.decode("utf-8")  # strict
    assert "café".encode("utf-8") in after
    assert "日本語".encode("utf-8") in after
    assert b"\xef\xbf\xbd" not in after


def test_utf32_le_bom_leaves_file_untouched(tmp_path, caplog):
    """UTF-32-LE BOM: refuse-to-mangle (leave bytes untouched + warning).

    UTF-32-LE's BOM starts with UTF-16-LE's FF FE; sniff order must check
    UTF-32 first so we never misdetect and corrupt.

    Exercises ``_sanitize_env_file_if_needed`` only: the dotenv load path
    is out of scope here (#65124's surface) and still cannot ingest UTF-32.
    """
    import logging

    from hermes_cli.env_loader import _sanitize_env_file_if_needed

    env_file = tmp_path / ".env"
    content = "HERMES_TEST_KEY=hello_utf32\nSECOND_KEY=world\n"
    raw = codecs.BOM_UTF32_LE + content.encode("utf-32-le")
    env_file.write_bytes(raw)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.env_loader"):
        _sanitize_env_file_if_needed(env_file)

    assert env_file.read_bytes() == raw  # untouched
    assert any("UTF-32" in r.message for r in caplog.records)


def test_utf32_be_bom_leaves_file_untouched(tmp_path, caplog):
    """UTF-32-BE BOM: same refuse-to-mangle path as LE (ordering independence)."""
    import logging

    from hermes_cli.env_loader import _sanitize_env_file_if_needed

    env_file = tmp_path / ".env"
    content = "HERMES_TEST_KEY=hello_utf32\nSECOND_KEY=world\n"
    raw = codecs.BOM_UTF32_BE + content.encode("utf-32-be")
    env_file.write_bytes(raw)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.env_loader"):
        _sanitize_env_file_if_needed(env_file)

    assert env_file.read_bytes() == raw
    assert any("UTF-32" in r.message for r in caplog.records)


def test_utf32_warning_fires_once_per_path(tmp_path, caplog, monkeypatch):
    """Three sanitize calls on the same UTF-32 file → exactly one warning.

    Matches house style for warn-once (module-level seen-set, same class as
    ``_WARNED_KEYS``): hot-reload / multi-entry load must not spam logs.
    """
    import logging

    import hermes_cli.env_loader as env_loader
    from hermes_cli.env_loader import _sanitize_env_file_if_needed

    # Isolate process-level seen-set so other tests' paths don't leak in.
    monkeypatch.setattr(env_loader, "_WARNED_UTF32_PATHS", set())

    env_file = tmp_path / ".env"
    content = "HERMES_TEST_KEY=hello_utf32\nSECOND_KEY=world\n"
    raw = codecs.BOM_UTF32_LE + content.encode("utf-32-le")
    env_file.write_bytes(raw)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.env_loader"):
        _sanitize_env_file_if_needed(env_file)
        _sanitize_env_file_if_needed(env_file)
        _sanitize_env_file_if_needed(env_file)

    utf32_warnings = [r for r in caplog.records if "UTF-32" in r.message]
    assert len(utf32_warnings) == 1
    assert env_file.read_bytes() == raw


def test_leading_replacement_char_does_not_rewrite(tmp_path):
    """errors=replace FFFD-on-first-line guard: do not persist mangling.

    Leading 0xFF is not a UTF-16/32 BOM (those need the second BOM byte) but
    is undecodable as UTF-8, so the replace path would glue U+FFFD onto the
    key. The guard must leave the on-disk bytes untouched.
    """
    from hermes_cli.env_loader import _sanitize_env_file_if_needed

    env_file = tmp_path / ".env"
    raw = b"\xffHERMES_TEST_KEY=should-not-rewrite\nSECOND_KEY=ok\n"
    env_file.write_bytes(raw)

    _sanitize_env_file_if_needed(env_file)

    assert env_file.read_bytes() == raw


def test_plain_utf8_env_regression(tmp_path, monkeypatch):
    """Plain UTF-8 .env must keep loading after the UTF-16 sanitize changes."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    before = b"OPENAI_API_KEY=sk-plain\nSECOND_KEY=ok\n"
    env_file.write_bytes(before)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SECOND_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("OPENAI_API_KEY") == "sk-plain"
    assert os.getenv("SECOND_KEY") == "ok"
    # No spurious rewrite of an already-clean file.
    assert env_file.read_bytes() == before


def test_cp1252_env_regression_does_not_crash(tmp_path, monkeypatch):
    """cp1252/latin-1 body must not crash sanitize; ASCII keys still usable.

    0xE9 is 'é' in cp1252 and incomplete as UTF-8. First line does not begin
    with U+FFFD, so the FFFD guard must not refuse the whole file.

    Sanitize leaves the file bytes alone when the only "change" is
    errors=replace on values (original already replace-decoded equals
    sanitized), so _load_dotenv_with_fallback's latin-1 path recovers café.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    before = b"ASCII_KEY=ok\nLATIN1_VALUE=caf\xe9\n"
    env_file.write_bytes(before)

    monkeypatch.delenv("ASCII_KEY", raising=False)
    monkeypatch.delenv("LATIN1_VALUE", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("ASCII_KEY") == "ok"
    assert os.getenv("LATIN1_VALUE") == "café"
    # Sanitize must not have rewritten (would have persisted U+FFFD).
    assert env_file.read_bytes() == before
