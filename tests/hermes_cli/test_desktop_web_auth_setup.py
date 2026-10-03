"""First-run authentication setup for the standalone Desktop Web host."""

from copy import deepcopy
import builtins
import types


def _args(host="127.0.0.1"):
    return types.SimpleNamespace(host=host)


def _clear_desktop_auth_env(monkeypatch):
    for name in (
        "HERMES_DESKTOP_WEB_PUBLIC_URL",
        "HERMES_DESKTOP_WEB_BASIC_AUTH_USERNAME",
        "HERMES_DESKTOP_WEB_BASIC_AUTH_PASSWORD",
        "HERMES_DESKTOP_WEB_BASIC_AUTH_PASSWORD_HASH",
    ):
        monkeypatch.delenv(name, raising=False)


def test_public_url_interactive_setup_persists_hashed_credentials(monkeypatch):
    import hermes_cli.config as config_module
    import hermes_cli.main as main_module

    _clear_desktop_auth_env(monkeypatch)
    config = {"desktop_web": {"public_url": "https://desktop.example.test"}}
    saved = []
    monkeypatch.setattr(config_module, "load_config", lambda: deepcopy(config))
    monkeypatch.setattr(config_module, "save_config", lambda value: saved.append(value))
    monkeypatch.setattr(main_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(main_module.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(builtins, "input", lambda _prompt: "")
    monkeypatch.setattr(main_module, "line_input", lambda _prompt: "alice")

    answers = iter(("correct horse battery staple", "correct horse battery staple"))
    monkeypatch.setattr("getpass.getpass", lambda _prompt: next(answers))

    main_module._maybe_setup_desktop_web_auth_interactively(_args())

    assert len(saved) == 1
    basic = saved[0]["desktop_web"]["basic_auth"]
    assert basic["username"] == "alice"
    assert basic["password"] == ""
    assert basic["password_hash"].startswith("scrypt$")
    assert len(basic["secret"]) >= 32


def test_non_tty_public_setup_does_not_prompt_or_write(monkeypatch):
    import hermes_cli.config as config_module
    import hermes_cli.main as main_module

    _clear_desktop_auth_env(monkeypatch)
    config = {"desktop_web": {"public_url": "https://desktop.example.test"}}
    saved = []
    monkeypatch.setattr(config_module, "load_config", lambda: deepcopy(config))
    monkeypatch.setattr(config_module, "save_config", lambda value: saved.append(value))
    monkeypatch.setattr(main_module.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(main_module.sys.stdout, "isatty", lambda: False)

    main_module._maybe_setup_desktop_web_auth_interactively(_args())

    assert saved == []


def test_loopback_without_public_url_skips_setup(monkeypatch):
    import hermes_cli.config as config_module
    import hermes_cli.main as main_module

    _clear_desktop_auth_env(monkeypatch)
    config = {"desktop_web": {}}
    saved = []
    monkeypatch.setattr(config_module, "load_config", lambda: deepcopy(config))
    monkeypatch.setattr(config_module, "save_config", lambda value: saved.append(value))
    monkeypatch.setattr(main_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(main_module.sys.stdout, "isatty", lambda: True)

    main_module._maybe_setup_desktop_web_auth_interactively(_args())

    assert saved == []