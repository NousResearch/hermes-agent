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


def test_export_prefixed_known_key_in_user_env_is_kept(tmp_path, monkeypatch):
    """A known Hermes key defined with the bash-compatible ``export KEY=value``
    form in the profile .env must be recognized as defined and survive the
    cleanup - mirrors the ``export `` stripping in config.py's load_env()
    (#6659).
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "export HERMES_ACP_AUTH_METHOD=claude_code_cli\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "cursor_login")
    load_hermes_dotenv(hermes_home=home)
    assert os.getenv("HERMES_ACP_AUTH_METHOD") == "claude_code_cli"


def test_shell_exported_credentials_survive_cleanup(tmp_path, monkeypatch):
    """User-shell-exported provider credentials must NOT be scrubbed.

    ``export OPENAI_API_KEY=…`` in the shell with a ``.env`` that doesn't
    contain the key is a documented, legitimate flow (see
    test_dump_env_visibility.py). The startup cleanup is scoped to
    _PROFILE_MANAGED_ENV_KEYS (ACP routing keys) precisely so it can never
    delete shell-supplied credentials — a process cannot distinguish a
    shell export from parent-process leakage, so credential isolation is
    owned by read-time secret scoping instead.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("SOME_OTHER_KEY=x\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345:token-from-shell")
    # A profile-managed routing key inherited alongside them IS cleared.
    monkeypatch.setenv("HERMES_ACP_AUTH_METHOD", "cursor_login")

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("OPENAI_API_KEY") == "sk-from-shell"
    assert os.getenv("ANTHROPIC_API_KEY") == "sk-ant-from-shell"
    assert os.getenv("TELEGRAM_BOT_TOKEN") == "12345:token-from-shell"
    assert "HERMES_ACP_AUTH_METHOD" not in os.environ


def test_cleanup_scope_is_the_profile_managed_set():
    """Lock the invariant: the startup scrub set contains only behavioral
    ACP/routing keys — never credential-shaped keys. If this fails, someone
    widened _PROFILE_MANAGED_ENV_KEYS toward the full known-key set, which
    re-introduces the shell-export deletion bug.
    """
    from hermes_cli.env_loader import _PROFILE_MANAGED_ENV_KEYS

    for key in _PROFILE_MANAGED_ENV_KEYS:
        assert not key.endswith(("_API_KEY", "_TOKEN", "_SECRET")), (
            f"{key} looks credential-shaped; startup scrub must not "
            "cover credentials — read-time secret scoping owns those"
        )


# ---------------------------------------------------------------------------
# config.yaml terminal.* re-apply after dotenv loads (#29186 / #67323)
#
# load_hermes_dotenv loads .env with override=True, so a stale
# TERMINAL_ENV=docker in .env used to silently beat config.yaml's
# terminal.backend on every reload (gateway per-turn reload, cron standalone
# runs). The bridge re-applies config.yaml's EXPLICIT terminal keys last via
# the shared hermes_cli.config.apply_terminal_config_to_env helper.
# ---------------------------------------------------------------------------


def _seed_terminal_home(tmp_path, monkeypatch, *, config_yaml=None, env_text=None):
    home = tmp_path / "hermes"
    home.mkdir()
    if config_yaml is not None:
        (home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    if env_text is not None:
        (home / ".env").write_text(env_text, encoding="utf-8")
    # The bridge is scoped to the process HERMES_HOME (a different profile's
    # load must not bridge this process's config), so point the process at
    # the seeded home like a real gateway/cron process would be.
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_config_yaml_terminal_backend_overrides_stale_env(tmp_path, monkeypatch):
    """Regression for #29186: a leftover TERMINAL_ENV=docker in ~/.hermes/.env
    must not silently override the user's choice in config.yaml. config.yaml
    is the documented source of truth, so its value must win after load."""
    home = _seed_terminal_home(
        tmp_path, monkeypatch,
        config_yaml="terminal:\n  backend: local\n",
        env_text="TERMINAL_ENV=docker\n",
    )

    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_ENV") == "local"


def test_config_yaml_terminal_backend_overrides_stale_shell(tmp_path, monkeypatch):
    """config.yaml must also beat a stale TERMINAL_ENV exported in the shell
    (e.g. set in ~/.zshrc when the user was experimenting with docker)."""
    home = _seed_terminal_home(
        tmp_path, monkeypatch,
        config_yaml="terminal:\n  backend: local\n",
    )

    monkeypatch.setenv("TERMINAL_ENV", "docker")

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_ENV") == "local"


def test_no_terminal_section_leaves_env_value_alone(tmp_path, monkeypatch):
    """When config.yaml has no terminal section, the .env value is still the
    user's active setting — the bridge must NOT clobber it with merged
    defaults."""
    home = _seed_terminal_home(
        tmp_path, monkeypatch,
        config_yaml="display:\n  streaming: true\n",
        env_text="TERMINAL_ENV=docker\n",
    )

    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_ENV") == "docker"


def test_config_yaml_terminal_omitted_key_does_not_clear_env(tmp_path, monkeypatch):
    """If config.yaml has a terminal block but no `backend`, the .env value
    must survive (only explicit config keys override env)."""
    home = _seed_terminal_home(
        tmp_path, monkeypatch,
        config_yaml="terminal:\n  timeout: 600\n",
        env_text="TERMINAL_ENV=docker\n",
    )

    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_ENV") == "docker"
    assert os.getenv("TERMINAL_TIMEOUT") == "600"


def test_other_profile_home_does_not_bridge_process_config(tmp_path, monkeypatch):
    """Loading a DIFFERENT profile's .env must not re-bridge this process's
    config.yaml — the shared bridge reads the process-global config, so
    applying it for another home would stamp the wrong profile's terminal
    settings into the env."""
    process_home = tmp_path / "process-home"
    process_home.mkdir()
    (process_home / "config.yaml").write_text(
        "terminal:\n  backend: local\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(process_home))

    other_home = tmp_path / "other-profile"
    other_home.mkdir()
    (other_home / ".env").write_text("TERMINAL_ENV=docker\n", encoding="utf-8")

    monkeypatch.delenv("TERMINAL_ENV", raising=False)

    load_hermes_dotenv(hermes_home=other_home)

    # The other profile's .env value stands; the process config was not applied.
    assert os.getenv("TERMINAL_ENV") == "docker"


# ---------------------------------------------------------------------------
# Desktop-injected session token preservation
#
# The desktop shell mints a fresh HERMES_DASHBOARD_SESSION_TOKEN and injects
# it into the spawned `hermes serve` process env (apps/desktop/electron/
# main.ts). A stale persisted value for the same key in the user .env would
# clobber it under override=True, breaking the desktop↔gateway WS auth
# handshake (web_server.py resolves _SESSION_TOKEN at module import).
# ---------------------------------------------------------------------------


def test_desktop_injected_session_token_survives_dotenv_load(tmp_path, monkeypatch):
    """An Electron-injected HERMES_DASHBOARD_SESSION_TOKEN must win over .env.

    The .env contains a stale persisted token from a previous update; the
    spawn-time injected token is fresh and per-launch, so it must survive
    the override=True .env load. Electron always pairs the token with
    HERMES_DESKTOP='1' (apps/desktop/electron/main.ts), which is the
    discriminator that authorizes the save/restore.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    injected = "injected-fresh-token-abc123"
    (home / ".env").write_text(
        "HERMES_DASHBOARD_SESSION_TOKEN=stale-persisted-token\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", injected)
    monkeypatch.setenv("HERMES_DESKTOP", "1")

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("HERMES_DASHBOARD_SESSION_TOKEN") == injected


def test_session_token_from_dotenv_when_not_injected(tmp_path, monkeypatch):
    """Without an injected token, the .env value keeps loading as before.

    Guards the other side of the save/restore: when the desktop did not
    inject a token (CLI / headless runs), the .env value must still reach
    os.environ so web_server._resolve_session_token can pick it up.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "HERMES_DASHBOARD_SESSION_TOKEN=persisted-token\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("HERMES_DASHBOARD_SESSION_TOKEN") == "persisted-token"


def test_unmarked_inherited_token_is_overridden_by_dotenv(tmp_path, monkeypatch):
    """A shell-exported token WITHOUT HERMES_DESKTOP keeps documented semantics.

    The loader rule is that a profile .env overrides stale shell exports.
    Only Electron-spawned processes (HERMES_DESKTOP='1') get the injected
    token preserved; an unmarked inherited token must still be overridden
    by the .env value so the documented rule is not silently changed for
    non-desktop runs.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "HERMES_DASHBOARD_SESSION_TOKEN=dotenv-wins\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "shell-exported-token")
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("HERMES_DASHBOARD_SESSION_TOKEN") == "dotenv-wins"


def test_desktop_token_preservation_does_not_affect_other_keys(tmp_path, monkeypatch):
    """Only the session token is save/restored; other keys keep override=True.

    Locks the minimal blast radius: OPENAI_BASE_URL in the .env still
    overrides a stale inherited value while the injected token survives.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "OPENAI_BASE_URL=https://profile.example/v1\n"
        "HERMES_DASHBOARD_SESSION_TOKEN=stale-persisted-token\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENAI_BASE_URL", "https://stale.example/v1")
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "injected-fresh-token")
    monkeypatch.setenv("HERMES_DESKTOP", "1")

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("HERMES_DASHBOARD_SESSION_TOKEN") == "injected-fresh-token"
    assert os.getenv("OPENAI_BASE_URL") == "https://profile.example/v1"


def test_desktop_token_survives_project_env_load(tmp_path, monkeypatch):
    """The guard must cover the project .env fallback path (override=True).

    When no user .env exists, the project .env loads with override=True and
    could clobber the desktop-injected token exactly like the user env does.
    With a user .env present the project env loads override=False, so this
    test pins the no-user-env case where the clobber was still possible.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    project_env = tmp_path / "project" / ".env"
    project_env.parent.mkdir()
    project_env.write_text(
        "HERMES_DASHBOARD_SESSION_TOKEN=stale-persisted-token\n",
        encoding="utf-8",
    )
    injected = "injected-fresh-token-xyz789"
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", injected)
    monkeypatch.setenv("HERMES_DESKTOP", "1")

    load_hermes_dotenv(hermes_home=home, project_env=project_env)

    assert os.getenv("HERMES_DASHBOARD_SESSION_TOKEN") == injected


def test_desktop_token_survives_managed_env_load(tmp_path, monkeypatch):
    """The guard must cover the managed-scope .env load (override=True).

    Managed scope is applied last with override=True and would replace the
    injected token with a stale pinned value just like the user env would.
    ``HERMES_MANAGED_DIR`` is the documented test hook for managed scope.
    """
    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    (managed_dir / ".env").write_text(
        "HERMES_DASHBOARD_SESSION_TOKEN=stale-managed-token\n",
        encoding="utf-8",
    )
    injected = "injected-fresh-token-managed"
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_dir))
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", injected)
    monkeypatch.setenv("HERMES_DESKTOP", "1")

    load_hermes_dotenv(hermes_home=tmp_path / "hermes")

    assert os.getenv("HERMES_DASHBOARD_SESSION_TOKEN") == injected


def test_desktop_token_survives_hot_reload_reentry(tmp_path, monkeypatch):
    """Re-running load_hermes_dotenv (gateway per-turn reload) must not clobber.

    gateway/run.py's _reload_runtime_env_preserving_config_authority calls
    load_hermes_dotenv() again on later turns to pick up rotated keys; that
    second user_env override=True load must not replace the injected token
    with the stale .env value.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text(
        "HERMES_DASHBOARD_SESSION_TOKEN=stale-persisted-token\n",
        encoding="utf-8",
    )
    injected = "injected-fresh-token-hotreload"
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", injected)
    monkeypatch.setenv("HERMES_DESKTOP", "1")

    load_hermes_dotenv(hermes_home=home)
    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("HERMES_DASHBOARD_SESSION_TOKEN") == injected
