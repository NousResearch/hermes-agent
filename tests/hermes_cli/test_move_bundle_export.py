"""Move bundle export: what must never leave the machine, and what must survive the trip."""
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest
import hermes_yaml as yaml

from hermes_cli.move_bundle import build_bundle

FAL = "fal-0123456789abcdef"
AGENT = "agt-fedcba9876543210-long"


def _home(root: Path) -> Path:
    home = root / "src"
    skill = home / "skills" / "custom" / "mine"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: mine\ndescription: x\n---\nuse it\n")
    (home / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": {
        "local": {"command": "/usr/local/bin/thing"}, "remote": {"url": "https://mcp.x"}}}))
    return home


def test_every_env_value_and_auth_credential_is_redacted_from_text(tmp_path):
    home = _home(tmp_path)
    (home / ".env").write_text(f"FAL_KEY={FAL}\nVERBOSE=true\n")
    (home / "auth.json").write_text(json.dumps({"providers": {"nous": {
        "agent_key": AGENT, "portal_base_url": "https://portal.example.com"}}}))
    quote = f"fal {FAL} agent {AGENT} verbose true https://portal.example.com"
    (home / "SOUL.md").write_text(quote)
    (home / "skills" / "custom" / "mine" / "notes.md").write_text(quote)
    from hermes_state import SessionDB
    db = SessionDB(db_path=home / "state.db")
    try:
        db.import_sessions([{"id": "s1", "source": "desktop", "messages": [
            {"role": "user", "content": quote}, {"role": "assistant", "content": "ok"}]}])
    finally:
        db.close()
    out = tmp_path / "move.zip"
    build_bundle(home, "default", out, include_history=True)
    with zipfile.ZipFile(out) as zf:
        texts = [zf.read(n).decode() for n in ("SOUL.md", "skills/custom/mine/notes.md", "sessions/0000.json")]
    for text in texts:
        assert FAL not in text and AGENT not in text
        assert "[redacted FAL_KEY]" in text and "[redacted auth.json:agent_key]" in text
        assert "verbose true https://portal.example.com" in text


def test_local_only_mcp_servers_stay_out_of_the_exported_config(tmp_path):
    home = _home(tmp_path)
    out = tmp_path / "move.zip"
    report = build_bundle(home, "default", out, include_history=False)
    with zipfile.ZipFile(out) as zf:
        config = yaml.safe_load(zf.read("config.yaml"))
    assert config["mcp_servers"] == {"remote": {"url": "https://mcp.x"}}
    assert {m["name"]: m["portable"] for m in report.mcp_servers} == {"local": False, "remote": True}


@pytest.mark.platforms("posix")
def test_bundle_is_private_from_creation_and_keeps_skill_exec_bits(tmp_path, monkeypatch):
    home = _home(tmp_path)
    script = home / "skills" / "custom" / "mine" / "run.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o755)
    seen: list[int] = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(stat.S_IMODE(os.stat(src).st_mode))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    old_umask = os.umask(0o022)
    try:
        out = tmp_path / "move.zip"
        build_bundle(home, "default", out, include_history=False)
    finally:
        os.umask(old_umask)
    assert seen == [0o600]
    with zipfile.ZipFile(out) as zf:
        assert stat.S_IMODE(zf.getinfo("skills/custom/mine/run.sh").external_attr >> 16) == 0o755
