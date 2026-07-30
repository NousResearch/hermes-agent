import codecs
import importlib
import os
import sys

from hermes_cli.env_loader import load_hermes_dotenv










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


# ---------------------------------------------------------------------------
# Profile .env isolation: inherited known-key cleanup
# ---------------------------------------------------------------------------


def test_known_keys_absent_from_user_env_are_cleared(tmp_path, monkeypatch):
    """Known Hermes keys inherited from parent process are removed when absent
    from the profile's .env.

    This is the startup equivalent of ``reload_env()``'s known-key cleanup and
    fixes the isolation gap where one profile's ACP/provider settings silently
    leak into another profile's runtime via ``os.environ`` inheritance.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "OPENAI_BASE_URL=https://profile.example/v1\n", encoding="utf-8"
    )

    # Inherited known keys from parent process / other profile
    monkeypatch.setenv("OPENAI_BASE_URL", "https://stale.example/v1")
    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "cursor_login")
    monkeypatch.setenv("COPILOT_CLI_PATH", "/usr/bin/claude-code")
    # Unrelated shell var must NOT be touched
    monkeypatch.setenv("MY_SHELL_ONLY_VAR", "keep-me")

    load_hermes_dotenv(hermes_home=home)

    # OPENAI_BASE_URL is defined in the profile .env → overridden to the new value
    assert os.getenv("OPENAI_BASE_URL") == "https://profile.example/v1"
    # HERMES_ACP_AUTH_METHOD and COPILOT_CLI_PATH are NOT in the profile .env → cleared
    assert "HERMES_ACP_AUTH_METHOD" not in os.environ
    assert "COPILOT_CLI_PATH" not in os.environ
    # Unrelated shell vars must survive
    assert os.getenv("MY_SHELL_ONLY_VAR") == "keep-me"


def test_empty_assignment_in_user_env_is_preserved(tmp_path, monkeypatch):
    """An explicit ``KEY=`` (empty value) in the profile .env keeps the key
    in ``os.environ`` — distinct from a key absent from .env entirely.

    Empty ``HERMES_ACP_AUTH_METHOD=`` tells the ACP adapter to skip
    ``authenticate`` (the key exists, its value is just empty).  This is the
    documented workaround for the leak and must still work after the cleanup.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("HERMES_ACP_AUTH_METHOD=\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "cursor_login")
    monkeypatch.setenv("COPILOT_CLI_PATH", "/usr/bin/sneaky")  # NOT in .env → cleared

    load_hermes_dotenv(hermes_home=home)

    # KEY= in .env keeps the key (now empty string)
    assert "HERMES_ACP_AUTH_METHOD" in os.environ
    assert os.environ["HERMES_ACP_AUTH_METHOD"] == ""
    # COPILOT_CLI_PATH is absent from .env → cleared
    assert "COPILOT_CLI_PATH" not in os.environ


def test_no_user_env_does_not_clear_anything(tmp_path, monkeypatch):
    """When no profile .env exists (bare profile), load_hermes_dotenv must not
    wipe inherited known keys — the bare-profile case follows #66930 / #67027
    semantics and the user's shell environment should not be mutilated.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    # No .env in home — bare profile

    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "cursor_login")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("HERMES_ACP_AUTH_METHOD") == "cursor_login"
    assert os.getenv("PATH") == "/usr/bin:/bin"


def test_known_key_explicitly_set_in_user_env_is_kept(tmp_path, monkeypatch):
    """A known Hermes key that IS explicitly set in the profile .env survives
    the cleanup (overrides the inherited value).
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "HERMES_ACP_AUTH_METHOD=claude_code_cli\n", encoding="utf-8"
    )

    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "cursor_login")

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("HERMES_ACP_AUTH_METHOD") == "claude_code_cli"
