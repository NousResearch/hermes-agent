"""Regression tests for setup access-policy prompts."""

from hermes_cli.access_setup import configure_direct_message_access
from hermes_cli import setup as setup_mod


_VALID_TELEGRAM_TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"


def _read_env(tmp_path):
    return (tmp_path / ".env").read_text(encoding="utf-8")


def test_access_helper_blank_allowlist_can_enable_open_access(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    configure_direct_message_access(
        platform_label="Example",
        pairing_platform="example",
        allowed_users_env="EXAMPLE_ALLOWED_USERS",
        allow_all_env="EXAMPLE_ALLOW_ALL_USERS",
        allowed_users_value="",
        prompt_yes_no_fn=lambda *_a, **_kw: True,
        print_info_fn=lambda *_a, **_kw: None,
        print_success_fn=lambda *_a, **_kw: None,
        print_warning_fn=lambda *_a, **_kw: None,
        open_access_warning="open",
    )

    env_text = _read_env(tmp_path)
    assert "EXAMPLE_ALLOWED_USERS=" in env_text
    assert "EXAMPLE_ALLOW_ALL_USERS=true" in env_text


def test_access_helper_blank_allowlist_can_enable_pairing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    configure_direct_message_access(
        platform_label="Example",
        pairing_platform="example",
        allowed_users_env="EXAMPLE_ALLOWED_USERS",
        allow_all_env="EXAMPLE_ALLOW_ALL_USERS",
        allowed_users_value="",
        prompt_yes_no_fn=lambda *_a, **_kw: False,
        print_info_fn=lambda *_a, **_kw: None,
        print_success_fn=lambda *_a, **_kw: None,
        print_warning_fn=lambda *_a, **_kw: None,
        open_access_warning="open",
    )

    env_text = _read_env(tmp_path)
    assert "EXAMPLE_ALLOWED_USERS=" in env_text
    assert "EXAMPLE_ALLOW_ALL_USERS=false" in env_text


def test_access_helper_allowlist_clears_allow_all(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    configure_direct_message_access(
        platform_label="Example",
        pairing_platform="example",
        allowed_users_env="EXAMPLE_ALLOWED_USERS",
        allow_all_env="EXAMPLE_ALLOW_ALL_USERS",
        allowed_users_value=" user1, user2 ",
        prompt_yes_no_fn=lambda *_a, **_kw: False,
        print_info_fn=lambda *_a, **_kw: None,
        print_success_fn=lambda *_a, **_kw: None,
        print_warning_fn=lambda *_a, **_kw: None,
        open_access_warning="open",
    )

    env_text = _read_env(tmp_path)
    assert "EXAMPLE_ALLOWED_USERS=user1,user2" in env_text
    assert "EXAMPLE_ALLOW_ALL_USERS=false" in env_text


def test_telegram_setup_blank_allowlist_can_enable_open_access(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    answers = iter([_VALID_TELEGRAM_TOKEN, "", ""])
    monkeypatch.setattr(setup_mod, "prompt", lambda *_a, **_kw: next(answers))
    monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *_a, **_kw: True)
    monkeypatch.setattr(setup_mod, "print_info", lambda *_a, **_kw: None)
    monkeypatch.setattr(setup_mod, "print_success", lambda *_a, **_kw: None)
    monkeypatch.setattr(setup_mod, "print_warning", lambda *_a, **_kw: None)

    setup_mod._setup_telegram()

    env_text = _read_env(tmp_path)
    assert f"TELEGRAM_BOT_TOKEN={_VALID_TELEGRAM_TOKEN}" in env_text
    assert "TELEGRAM_ALLOWED_USERS=" in env_text
    assert "TELEGRAM_ALLOW_ALL_USERS=true" in env_text


def test_telegram_setup_blank_allowlist_can_enable_pairing(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    answers = iter([_VALID_TELEGRAM_TOKEN, "", ""])
    monkeypatch.setattr(setup_mod, "prompt", lambda *_a, **_kw: next(answers))
    monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *_a, **_kw: False)
    monkeypatch.setattr(setup_mod, "print_info", lambda *_a, **_kw: None)
    monkeypatch.setattr(setup_mod, "print_success", lambda *_a, **_kw: None)
    monkeypatch.setattr(setup_mod, "print_warning", lambda *_a, **_kw: None)

    setup_mod._setup_telegram()

    env_text = _read_env(tmp_path)
    assert "TELEGRAM_ALLOWED_USERS=" in env_text
    assert "TELEGRAM_ALLOW_ALL_USERS=false" in env_text
