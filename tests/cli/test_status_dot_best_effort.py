"""Regression test for #50106.

A directory at ~/.hermes/agent_status must not cause _write_status_dot to
raise, which would skip self.chat() and silently drop the user's message.
The status indicator is cosmetic and must be truly best-effort.
"""

import cli


def test_status_dot_does_not_raise_when_path_is_directory(tmp_path, monkeypatch):
    # Redirect the agent_status path to a DIRECTORY (the #50106 condition).
    status_dir = tmp_path / "agent_status"
    status_dir.mkdir()

    monkeypatch.setattr(cli, "_AGENT_STATUS_PATH", str(status_dir), raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Must be best-effort: no exception, message handling can continue.
    cli._write_status_dot("thinking")
    cli._write_status_dot("idle")


def test_status_dot_writes_normally_to_a_file(tmp_path, monkeypatch):
    status_file = tmp_path / "agent_status"

    monkeypatch.setattr(cli, "_AGENT_STATUS_PATH", str(status_file), raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    cli._write_status_dot("thinking")

    # When the path is a normal file, the indicator should still be written.
    assert status_file.exists()
    assert status_file.read_text().strip() == "thinking"
