"""Gateway update prompts must not wait for a Windows gateway this run paused.

Regression for #137635.
"""

import threading
from types import SimpleNamespace


def test_windows_pause_token_makes_gateway_prompt_use_its_default(tmp_path, monkeypatch, capsys):
    """A Windows pause has no live gateway left to consume the prompt marker."""
    from hermes_cli import update_cmd

    token = {"resume_needed": True, "profiles": {"default": 1234}}
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(update_cmd, "_windows_gateways_paused_for_prompt", False, raising=False)
    monkeypatch.setattr(
        update_cmd,
        "_m",
        lambda: SimpleNamespace(_pause_windows_gateways_for_update=lambda: token),
    )

    assert update_cmd._pause_gateways_for_update(SimpleNamespace(no_gateway_restart=False)) == token
    answer: list[str] = []
    thread = threading.Thread(
        target=lambda: answer.append(update_cmd._gateway_prompt("Restore local changes now?", "n")), daemon=True)
    thread.start()
    thread.join(timeout=1)

    assert answer == ["n"], "a prompt waited for a Windows gateway this update had stopped"
    assert "gateways are paused for the update" in capsys.readouterr().out
    assert not (tmp_path / ".update_prompt.json").exists()
