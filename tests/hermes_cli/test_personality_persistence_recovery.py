"""CLI /personality: persistence failure degrades to session-only, then recovers on retry.

Boundary: the real ``HermesCLI._handle_personality_command`` drives the real
``persist_personality`` against a real ``config.yaml``. When an external edit leaves the file
malformed the write is refused (bytes untouched) and the handler must still apply the choice to
the running session and say so; once the file is repaired the same CLI persists normally.
Only the agent's ``release_clients`` is spied.
"""

from pathlib import Path
from unittest.mock import MagicMock

import hermes_yaml as yaml

from agent.i18n import t

CUSTOM = "You are a custom helpful persona."
MANUAL = "manual forever"


def test_malformed_config_falls_back_to_session_only_then_recovers(tmp_path, monkeypatch, capsys):
    from cli import HermesCLI
    from hermes_cli.personality import available_personalities

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    config_path = home / "config.yaml"
    saved_marker = t("cli.commands.personality.scope_saved", lang="en")
    session_marker = t("cli.commands.personality.scope_session", lang="en")
    assert not saved_marker.startswith("cli.") and not session_marker.startswith("cli.")

    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {"agent": {"system_prompt": MANUAL, "personalities": {"helpful": CUSTOM}}}
    cli.personalities = available_personalities(cli.config)
    cli.system_prompt = MANUAL
    cli.console = MagicMock()

    def run(command):
        cli.agent = MagicMock()  # _retire_agent drops it, so arm a fresh spy per step
        agent = cli.agent
        cli._handle_personality_command(command)
        agent.release_clients.assert_called_once()
        assert cli.agent is None
        return capsys.readouterr().out

    # 1. External edit corrupts config.yaml: the choice applies in memory, nothing is written.
    malformed = b"display: [unterminated\n"
    config_path.write_bytes(malformed)
    out = run("/personality helpful")
    assert cli.system_prompt == CUSTOM
    assert session_marker in out and saved_marker not in out
    assert config_path.read_bytes() == malformed

    # 2. File repaired: retrying on the SAME CLI now persists and drops the session-only marker.
    config_path.write_text(yaml.safe_dump(
        {"agent": {"system_prompt": MANUAL, "personalities": {"helpful": CUSTOM}}}), encoding="utf-8")
    out = run("/personality helpful")
    assert saved_marker in out and session_marker not in out
    on_disk = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert on_disk["display"]["personality"] == "helpful"
    assert on_disk["agent"]["system_prompt"] == MANUAL

    # 3. /personality none restores the on-disk manual prompt and clears only the selection.
    out = run("/personality none")
    assert cli.system_prompt == MANUAL
    assert saved_marker in out and session_marker not in out
    on_disk = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert on_disk["display"]["personality"] == ""
    assert on_disk["agent"]["system_prompt"] == MANUAL
    assert on_disk["agent"]["personalities"] == {"helpful": CUSTOM}
