"""The local prompt describes actual execution identity, not inferred privileges."""
import os

import pytest

from agent import prompt_builder


@pytest.mark.platforms("posix")
def test_local_prompt_names_actual_effective_uid_and_bounds_its_meaning():
    text = "\n".join(prompt_builder._local_host_hints())
    assert f"Effective user ID: {os.geteuid()}" in text
    assert "sudo availability" in text
    assert "remote privileges" in text
    assert "bash" in text


@pytest.mark.platforms("macos")
def test_macos_prompt_does_not_invite_linux_paths_or_service_commands():
    text = "\n".join(prompt_builder._local_host_hints())
    assert "macOS" in text
    assert "/proc" in text and "/etc/shadow" in text and "systemctl" in text
    assert "Linux" in text


def test_remote_backend_never_gets_control_host_identity(monkeypatch):
    monkeypatch.setattr(prompt_builder, "_tenv_read", lambda key, default="": "ssh" if key == "TERMINAL_ENV" else default)
    monkeypatch.setattr(prompt_builder, "_probe_remote_backend", lambda _backend: "  OS: Linux synthetic")
    monkeypatch.setattr(prompt_builder, "_embedder_environment_hint", lambda: "")
    text = prompt_builder.build_environment_hints()
    assert "Terminal backend: ssh" in text
    assert "Effective user ID:" not in text
    assert "User home directory:" not in text
