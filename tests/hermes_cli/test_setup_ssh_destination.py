"""Setup terminal SSH identity persistence (CWE-88 option smuggling)."""

from pathlib import Path

import hermes_cli.setup as setup_mod


def _select_ssh_backend(question, choices, default=0):
    for i, choice in enumerate(choices):
        if choice.startswith("SSH"):
            return i
    raise AssertionError(f"SSH choice missing from {choices!r}")


def test_ssh_setup_rejects_leading_dash_host_without_persisting(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {"terminal": {"backend": "local"}}
    prompts = iter(
        [
            "-oProxyCommand=evil",  # host
            "alice",  # user
            "22",  # port
            str(tmp_path / "id"),  # key
        ]
    )
    saved = {}
    monkeypatch.setattr(setup_mod, "prompt_choice", _select_ssh_backend)
    monkeypatch.setattr(setup_mod, "prompt", lambda *a, **k: next(prompts))
    monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *a, **k: False)
    monkeypatch.setattr(setup_mod, "save_config", lambda cfg: None)
    monkeypatch.setattr(setup_mod, "save_env_value", lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(setup_mod, "get_env_value", lambda k: "")

    setup_mod.setup_terminal_backend(config)

    assert config["terminal"]["backend"] == "local"
    assert "TERMINAL_SSH_HOST" not in saved
    assert "TERMINAL_SSH_USER" not in saved
    assert saved.get("TERMINAL_ENV") == "local"
    assert "Invalid SSH destination" in capsys.readouterr().out


def test_ssh_setup_rejects_leading_dash_user_even_when_host_blank(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {"terminal": {"backend": "local"}}
    prompts = iter(
        [
            "",  # host blank
            "-oProxyCommand=evil",  # user
            "22",
            str(tmp_path / "id"),
        ]
    )
    saved = {}
    monkeypatch.setattr(setup_mod, "prompt_choice", _select_ssh_backend)
    monkeypatch.setattr(setup_mod, "prompt", lambda *a, **k: next(prompts))
    monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *a, **k: False)
    monkeypatch.setattr(setup_mod, "save_config", lambda cfg: None)
    monkeypatch.setattr(setup_mod, "save_env_value", lambda k, v: saved.__setitem__(k, v))
    monkeypatch.setattr(setup_mod, "get_env_value", lambda k: "")

    setup_mod.setup_terminal_backend(config)

    assert config["terminal"]["backend"] == "local"
    assert "TERMINAL_SSH_USER" not in saved
    assert saved.get("TERMINAL_ENV") == "local"
    assert "Invalid SSH destination" in capsys.readouterr().out
