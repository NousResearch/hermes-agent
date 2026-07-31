"""Uninstall must remove every Hermes-managed PATH launcher."""

from pathlib import Path

import hermes_cli.uninstall as uninstall


def test_remove_wrapper_script_removes_hermes_agent_launcher(tmp_path, monkeypatch):
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    for name in ("hermes", "hermes-acp", "hermes-agent"):
        (local_bin / name).write_text(
            "#!/usr/bin/env bash\n# hermes-agent managed launcher\n",
            encoding="utf-8",
        )

    removed = uninstall.remove_wrapper_script()

    assert sorted(path.name for path in removed) == [
        "hermes",
        "hermes-acp",
        "hermes-agent",
    ]
    for name in ("hermes", "hermes-acp", "hermes-agent"):
        assert not (local_bin / name).exists()