"""``hermes import --keep-host``: the backup replaces the agent, the host's own setup stays.

Real ``hermes backup`` of a source home, real ``hermes import --keep-host`` into a destination home
that looks like a Hermes Cloud agent (Nous sign-in, platform config, NOUS_* keys), then the undo.
"""

import json
import sqlite3
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

import hermes_cli.gateway_setup_service as service_setup


@pytest.fixture(autouse=True)
def _no_real_gateway_service(monkeypatch):
    import hermes_cli.gateway as gateway_mod
    monkeypatch.setattr(service_setup, "ensure_gateway_service", lambda **kw: False)
    monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)


def _db(path: Path, sessions: list[str]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)")
        conn.executemany("INSERT INTO sessions VALUES (?)", [(s,) for s in sessions])


def _ids(path: Path) -> list[str]:
    with sqlite3.connect(path) as conn:
        return sorted(r[0] for r in conn.execute("SELECT id FROM sessions"))


def _write(root: Path, rel: str, text: str) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text)


@pytest.fixture
def homes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write(src, "config.yaml", "model:\n  provider: openrouter\n  default: x/y\ndisplay:\n  skin: ares\n"
                               "platforms:\n  telegram:\n    enabled: true\n")
    _write(src, ".env", "OPENROUTER_API_KEY=SRC-OR\nTELEGRAM_BOT_TOKEN=SRC-TG\nNOUS_API_KEY=SRC-NOUS\n")
    _write(src, "auth.json", json.dumps({"version": 1, "active_provider": "openrouter", "providers": {
        "nous": {"access_token": "SRC-NOUS", "refresh_token": "SRC-NOUS-RT"},
        "anthropic": {"access_token": "SRC-ANT"}}}))
    _write(src, "SOUL.md", "source persona\n")
    _write(src, "memories/MEMORY.md", "source memory\n")
    _write(src, "skills/mine/SKILL.md", "# mine\n")
    _write(src, "cron/jobs.json", json.dumps({"jobs": [{"id": "srcjob"}]}))
    _db(src / "state.db", ["src-chat"])

    _write(dst, "config.yaml", "model:\n  provider: nous\n  default: Hermes-4\nnous:\n  guest: false\n"
                               "cron:\n  preflight: true\ndashboard:\n  oauth:\n    client_id: agent:abc\n"
                               "display:\n  skin: mono\n")
    _write(dst, ".env", "NOUS_API_KEY=DST-NOUS\nAPI_SERVER_KEY=DST-API\nOPENROUTER_API_KEY=DST-OR\n")
    _write(dst, "auth.json", json.dumps({"version": 1, "active_provider": "nous", "providers": {
        "nous": {"access_token": "DST-NOUS", "refresh_token": "DST-NOUS-RT"}}}))
    _write(dst, "SOUL.md", "cloud persona\n")
    _write(dst, "memories/MEMORY.md", "cloud memory\n")
    _write(dst, "skills/cloud-only/SKILL.md", "# cloud only\n")
    _db(dst / "state.db", ["cloud-chat"])
    return src, dst, tmp_path


def _backup(monkeypatch, home: Path, out: Path) -> Path:
    from hermes_cli.backup import run_backup
    monkeypatch.setenv("HERMES_HOME", str(home))
    run_backup(Namespace(output=str(out)))
    return out


def _import(monkeypatch, home: Path, archive: Path, **flags) -> None:
    from hermes_cli.backup import run_import
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert run_import(Namespace(zipfile=str(archive), force=True, **flags)) is None


def _yaml(path: Path) -> dict:
    import hermes_yaml as yaml
    return yaml.safe_load(path.read_text())


def test_keep_host_replaces_the_agent_but_keeps_its_nous_signin_and_platform_settings(homes, monkeypatch):
    src, dst, tmp = homes
    archive = _backup(monkeypatch, src, tmp / "src.zip")
    host_cfg = _yaml(dst / "config.yaml")

    _import(monkeypatch, dst, archive, keep_host=True, undo_point=str(tmp / "undo.zip"))

    # The source's agent arrived: persona, memory, skills (and only its skills), jobs, chats.
    assert (dst / "SOUL.md").read_text() == "source persona\n"
    assert (dst / "memories/MEMORY.md").read_text() == "source memory\n"
    assert (dst / "skills/mine/SKILL.md").exists() and not (dst / "skills/cloud-only").exists()
    assert json.loads((dst / "cron/jobs.json").read_text())["jobs"][0]["id"] == "srcjob"
    assert _ids(dst / "state.db") == ["src-chat"]
    # Config: the source's own settings, the host's platform sections.
    cfg = _yaml(dst / "config.yaml")
    assert cfg["display"] == {"skin": "ares"} and cfg["platforms"] == {"telegram": {"enabled": True}}
    for key in ("model", "nous", "cron", "dashboard"):
        assert cfg[key] == host_cfg[key]
    # .env: the source's other keys (incl. its bot token) come over; NOUS_*/API_SERVER_KEY stay.
    env = dict(line.split("=", 1) for line in (dst / ".env").read_text().split())
    assert env == {"OPENROUTER_API_KEY": "SRC-OR", "TELEGRAM_BOT_TOKEN": "SRC-TG",
                   "NOUS_API_KEY": "DST-NOUS", "API_SERVER_KEY": "DST-API"}
    # auth.json: the source's other logins, the destination's Nous sign-in (never the source's).
    auth = json.loads((dst / "auth.json").read_text())
    assert auth["providers"]["anthropic"] == {"access_token": "SRC-ANT"}
    assert auth["providers"]["nous"] == {"access_token": "DST-NOUS", "refresh_token": "DST-NOUS-RT"}
    assert auth["active_provider"] == "nous"
    assert "SRC-NOUS" not in (dst / "auth.json").read_text() + (dst / ".env").read_text()


def test_the_undo_point_puts_the_agent_back(homes, monkeypatch):
    src, dst, tmp = homes
    archive = _backup(monkeypatch, src, tmp / "src.zip")
    before = {rel: (dst / rel).read_bytes() for rel in ("SOUL.md", "memories/MEMORY.md", ".env",
                                                        "skills/cloud-only/SKILL.md")}
    before_cfg, before_auth = _yaml(dst / "config.yaml"), json.loads((dst / "auth.json").read_text())
    undo = tmp / "undo.zip"

    _import(monkeypatch, dst, archive, keep_host=True, undo_point=str(undo))
    assert zipfile.is_zipfile(undo)
    _import(monkeypatch, dst, undo, keep_host=True, undo_point=str(tmp / "undo2.zip"))

    assert {rel: (dst / rel).read_bytes() for rel in before} == before
    assert not (dst / "skills/mine").exists()
    assert _ids(dst / "state.db") == ["cloud-chat"]
    assert _yaml(dst / "config.yaml") == before_cfg
    assert json.loads((dst / "auth.json").read_text())["providers"] == before_auth["providers"]


def test_undo_restores_existing_external_memory_provider_file(homes, monkeypatch):
    """A source's external memory file must not irreversibly replace the cloud copy."""
    src, dst, tmp = homes
    archive = _backup(monkeypatch, src, tmp / "src.zip")
    member = "_external/memory-provider/store.json"
    # The source and cloud homes are on different hosts; emulate the source's
    # real ``hermes backup`` external member in this shared-filesystem fixture.
    with zipfile.ZipFile(archive, "a") as zf:
        zf.writestr(member, '{"memory": "source"}\n')
    _write(tmp, "memory-provider/store.json", '{"memory": "cloud"}\n')
    external_file = tmp / "memory-provider/store.json"
    undo = tmp / "undo.zip"

    _import(monkeypatch, dst, archive, keep_host=True, undo_point=str(undo))
    assert external_file.read_text() == '{"memory": "source"}\n'
    with zipfile.ZipFile(undo) as zf:
        assert zf.read(member) == b'{"memory": "cloud"}\n'

    _import(monkeypatch, dst, undo, keep_host=True, undo_point=str(tmp / "undo2.zip"))
    assert external_file.read_text() == '{"memory": "cloud"}\n'


def test_plain_import_is_unchanged(homes, monkeypatch):
    """Without --keep-host the archive's own Nous sign-in and config win, as before."""
    src, dst, tmp = homes
    _import(monkeypatch, dst, _backup(monkeypatch, src, tmp / "src.zip"))
    assert _yaml(dst / "config.yaml")["model"]["provider"] == "openrouter"
    assert "NOUS_API_KEY=SRC-NOUS" in (dst / ".env").read_text()
    assert (dst / "skills/cloud-only/SKILL.md").exists()  # a plain import never removes files
