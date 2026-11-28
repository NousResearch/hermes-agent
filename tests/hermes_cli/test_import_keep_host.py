"""``hermes import --keep-host``: the backup replaces the agent, the host's own setup stays.

Real ``hermes backup`` of a source home, real ``hermes import --keep-host`` into a destination home
that looks like a Hermes Cloud agent (Nous sign-in, platform config, NOUS_* keys), then the undo.
"""

import hashlib
import json
import os
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


DST_CONFIG = ("# Provisioned by Hermes Cloud\nmodel:\n  provider: nous  # host routing\n  default: Hermes-4\n"
              "nous:\n  guest: false\ncron:\n  preflight: true\ndashboard:\n  oauth:\n    client_id: agent:abc\n"
              "display:\n  skin: mono  # my pick\n")


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

    _write(dst, "config.yaml", DST_CONFIG)
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


def _run(monkeypatch, home: Path, archive: Path, **flags):
    from hermes_cli.backup import run_import
    monkeypatch.setenv("HERMES_HOME", str(home))
    return run_import(Namespace(zipfile=str(archive), force=True, **flags))


def _import(monkeypatch, home: Path, archive: Path, **flags) -> None:
    assert _run(monkeypatch, home, archive, **flags) is None


def _undo_point(dst: Path) -> Path:
    return max((dst / "backups").glob("pre-transfer-*.zip"))


def _yaml(path: Path) -> dict:
    import hermes_yaml as yaml
    return yaml.safe_load(path.read_text())


def _tree(root: Path) -> dict[str, str]:
    """Every file the agent owns, by hash (backups and the host's own logs aside)."""
    out = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        # profiles/.deleted/ holds `hermes profile delete`'s tombstones (an undo removes a profile that way).
        if (path.is_file() and rel.parts[0] not in {"backups", "logs"} and not path.name.endswith(".lock")
                and rel.parts[:2] != ("profiles", ".deleted")):
            out[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def test_keep_host_replaces_the_agent_but_keeps_its_nous_signin_and_platform_settings(homes, monkeypatch):
    src, dst, tmp = homes
    archive = _backup(monkeypatch, src, tmp / "src.zip")
    host_cfg = _yaml(dst / "config.yaml")

    _import(monkeypatch, dst, archive, keep_host=True)

    assert (dst / "SOUL.md").read_text() == "source persona\n"
    assert (dst / "memories/MEMORY.md").read_text() == "source memory\n"
    assert (dst / "skills/mine/SKILL.md").exists() and not (dst / "skills/cloud-only").exists()
    assert json.loads((dst / "cron/jobs.json").read_text())["jobs"][0]["id"] == "srcjob"
    assert _ids(dst / "state.db") == ["src-chat"]
    cfg = _yaml(dst / "config.yaml")
    assert cfg["display"] == {"skin": "ares"} and cfg["platforms"] == {"telegram": {"enabled": True}}
    for key in ("model", "nous", "cron", "dashboard"):
        assert cfg[key] == host_cfg[key]
    env = dict(line.split("=", 1) for line in (dst / ".env").read_text().split())
    assert env == {"OPENROUTER_API_KEY": "SRC-OR", "TELEGRAM_BOT_TOKEN": "SRC-TG",
                   "NOUS_API_KEY": "DST-NOUS", "API_SERVER_KEY": "DST-API"}
    auth = json.loads((dst / "auth.json").read_text())
    assert auth["providers"]["anthropic"] == {"access_token": "SRC-ANT"}
    assert auth["providers"]["nous"] == {"access_token": "DST-NOUS", "refresh_token": "DST-NOUS-RT"}
    assert auth["active_provider"] == "nous"
    assert "SRC-NOUS" not in (dst / "auth.json").read_text() + (dst / ".env").read_text()


def test_keep_host_only_merges_the_root_home(homes, monkeypatch):
    """The host set up the root home only: a profile it never had and a skill's own config.yaml
    arrive whole, while a profile's Nous tokens still never land."""
    src, dst, tmp = homes
    _write(src, "profiles/writer/config.yaml", "model:\n  provider: anthropic\ntts:\n  provider: elevenlabs\n")
    _write(src, "profiles/writer/.env", "ANTHROPIC_API_KEY=SRC-ANT\nHERMES_MAX_ITERATIONS=200\n")
    _write(src, "profiles/writer/auth.json", json.dumps({"version": 1, "providers": {
        "nous": {"access_token": "SRC-NOUS-P"}, "anthropic": {"access_token": "SRC-ANT-P"}}}))
    _write(src, "skills/tool/templates/config.yaml", "model: gpt-x\nweb: true\n")
    _write(src, "skills/tool/auth.json", '{"providers": {"nous": {}}}')

    _import(monkeypatch, dst, _backup(monkeypatch, src, tmp / "src.zip"), keep_host=True)

    assert (dst / "profiles/writer/config.yaml").read_bytes() == (src / "profiles/writer/config.yaml").read_bytes()
    assert (dst / "profiles/writer/.env").read_text() == (src / "profiles/writer/.env").read_text()
    assert (dst / "skills/tool/templates/config.yaml").read_text() == "model: gpt-x\nweb: true\n"
    assert (dst / "skills/tool/auth.json").read_text() == '{"providers": {"nous": {}}}'
    profile_auth = json.loads((dst / "profiles/writer/auth.json").read_text())["providers"]
    assert profile_auth == {"anthropic": {"access_token": "SRC-ANT-P"}}


@pytest.mark.parametrize("member", ["../escaped/auth.json", "/abs/auth.json", "profiles/../auth.json"])
def test_keep_host_refuses_an_auth_member_outside_the_home(homes, monkeypatch, member):
    _src, dst, tmp = homes
    member = str(tmp) + member if member.startswith("/") else member
    archive = tmp / "crafted.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("config.yaml", "display:\n  skin: ares\n")
        zf.writestr(member, json.dumps({"version": 1, "providers": {"x": {"access_token": "CRAFTED"}}}))

    result = _run(monkeypatch, dst, archive, keep_host=True)

    target = (dst / member).resolve()
    assert result == 1 and not (target.exists() and "CRAFTED" in target.read_text())


@pytest.mark.platforms("posix")  # a symlink needs a privilege on Windows
def test_keep_host_refuses_a_linked_folder_and_deletes_nothing(homes, monkeypatch, capsys):
    """A linked skills/ (or profile) folder: refused before any write, nothing behind it touched."""
    src, dst, tmp = homes
    archive = _backup(monkeypatch, src, tmp / "src.zip")
    outside = tmp / "outside-skills"
    _write(outside, "precious/SKILL.md", "# precious\n")
    (dst / "skills").rename(tmp / "old-skills")
    (dst / "skills").symlink_to(outside, target_is_directory=True)
    before = _tree(dst)

    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, dst, archive, keep_host=True)

    assert exc.value.code == 1 and "skills is a linked folder" in capsys.readouterr().out
    assert (outside / "precious/SKILL.md").read_text() == "# precious\n"
    assert _tree(dst) == before


def test_keep_host_clears_logins_the_backup_does_not_carry(homes, monkeypatch):
    """Replace: a backup without auth.json/.env leaves only the host's Nous sign-in and keys."""
    src, dst, tmp = homes
    (src / "auth.json").unlink()
    (src / ".env").unlink()
    store = json.loads((dst / "auth.json").read_text())
    store["providers"]["openai-codex"] = {"tokens": {"access_token": "DST-CODEX"}}
    (dst / "auth.json").write_text(json.dumps(store))

    _import(monkeypatch, dst, _backup(monkeypatch, src, tmp / "src.zip"), keep_host=True)

    assert json.loads((dst / "auth.json").read_text())["providers"] == {
        "nous": {"access_token": "DST-NOUS", "refresh_token": "DST-NOUS-RT"}}
    assert (dst / ".env").read_text() == "NOUS_API_KEY=DST-NOUS\nAPI_SERVER_KEY=DST-API\n"


def test_keep_host_keeps_the_backups_config_comments(homes, monkeypatch):
    """The merged config.yaml is the backup's own document, comments included, with the host's
    sections written into it."""
    src, dst, tmp = homes
    _write(src, "config.yaml", "# my agent\nmodel:\n  provider: openrouter\ndisplay:\n  skin: ares  # dark\n")

    _import(monkeypatch, dst, _backup(monkeypatch, src, tmp / "src.zip"), keep_host=True)

    text = (dst / "config.yaml").read_text()
    assert text.startswith("# my agent\n") and "skin: ares  # dark" in text
    assert _yaml(dst / "config.yaml")["model"] == {"provider": "nous", "default": "Hermes-4"}


def test_transfer_then_undo_returns_the_agent_byte_for_byte(homes, monkeypatch):
    """Undo restores the undo point with replace semantics: files the transfer brought (the source's
    cron/jobs.json, its skills, a new profile) go, and config.yaml keeps its comments."""
    src, dst, tmp = homes
    _write(src, "profiles/writer/config.yaml", "display:\n  skin: ares\n")
    _write(dst, "profiles/solo/SOUL.md", "a profile only the cloud agent has\n")
    before = _tree(dst)
    archive = _backup(monkeypatch, src, tmp / "src.zip")

    _import(monkeypatch, dst, archive, keep_host=True)
    assert (dst / "cron/jobs.json").exists() and (dst / "profiles/writer").is_dir()
    assert (dst / "profiles/solo/SOUL.md").exists()  # not in the backup: left as it is
    _import(monkeypatch, dst, _undo_point(dst), keep_host=True)

    after = _tree(dst)
    after.pop("state.db"), before.pop("state.db")  # restored page by page: compared by content
    assert after == before
    assert not (dst / "profiles/writer").exists()  # the profile the transfer brought is gone again
    assert (dst / "config.yaml").read_text() == DST_CONFIG
    assert _ids(dst / "state.db") == ["cloud-chat"]


@pytest.mark.parametrize("spelling", [
    "platforms:\n  telegram:\n    enabled: true\n",
    "gateway:\n  telegram:\n    enabled: true\n",
    "telegram:\n  enabled: true\n",
])
def test_a_bot_switched_off_by_the_transfer_comes_back_on_from_the_messaging_page(homes, monkeypatch, spelling):
    """Every spelling of ``enabled`` the gateway reads: after ``--messaging-off`` the Messaging page's
    switch (``platforms.<id>.enabled``) alone turns the bot back on."""
    from gateway.config import Platform, load_gateway_config
    from hermes_cli.web_server_messaging import _write_platform_enabled
    src, dst, tmp = homes
    _write(src, "config.yaml", spelling)
    _write(src, ".env", "TELEGRAM_BOT_TOKEN=SRC-TG\n")
    _import(monkeypatch, dst, _backup(monkeypatch, src, tmp / "src.zip"), keep_host=True, messaging_off=True)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "SRC-TG")
    assert not load_gateway_config().platforms[Platform.TELEGRAM].enabled

    _write_platform_enabled("telegram", True)

    assert load_gateway_config().platforms[Platform.TELEGRAM].enabled


def test_a_transfer_brings_the_bots_switched_off_and_its_undo_brings_the_cloud_bot_back(homes, monkeypatch, capsys):
    """``--messaging-off``: every bot the backup runs (an ``enabled`` flag or just a token), in the
    root and in a profile, arrives off with its token and settings; the undo (no flag) puts the
    cloud agent's own enabled bot back byte for byte."""
    src, dst, tmp = homes
    _write(src, "config.yaml", "display:\n  skin: ares\nplatforms:\n  telegram:\n    enabled: true\n"
                               "    reply_to_mode: all\ndiscord:\n  require_mention: true\n")
    _write(src, ".env", "TELEGRAM_BOT_TOKEN=SRC-TG\nDISCORD_BOT_TOKEN=SRC-DC\nSLACK_BOT_TOKEN=\n")
    _write(src, "profiles/writer/config.yaml", "platforms:\n  telegram:\n    enabled: true\n")
    _write(src, "profiles/writer/.env", "TELEGRAM_BOT_TOKEN=WRITER-TG\n")
    _write(dst, "config.yaml", DST_CONFIG + "platforms:\n  discord:\n    enabled: true\n")
    _write(dst, ".env", "NOUS_API_KEY=DST-NOUS\nDISCORD_BOT_TOKEN=DST-DC\n")
    before = _tree(dst)
    archive = _backup(monkeypatch, src, tmp / "src.zip")

    _import(monkeypatch, dst, archive, keep_host=True, messaging_off=True)

    cfg, writer = _yaml(dst / "config.yaml"), _yaml(dst / "profiles/writer/config.yaml")
    assert cfg["platforms"] == {"telegram": {"enabled": False, "reply_to_mode": "all"},
                                "discord": {"enabled": False}}
    assert cfg["discord"] == {"require_mention": True} and cfg["display"] == {"skin": "ares"}
    assert "slack" not in cfg["platforms"]  # a blank token runs nothing
    assert writer == {"platforms": {"telegram": {"enabled": False}}}
    assert "TELEGRAM_BOT_TOKEN=SRC-TG" in (dst / ".env").read_text()
    assert "DISCORD_BOT_TOKEN=SRC-DC" in (dst / ".env").read_text()
    assert (dst / "profiles/writer/.env").read_text() == "TELEGRAM_BOT_TOKEN=WRITER-TG\n"
    assert "Messaging switched off: Discord, Telegram\n" in capsys.readouterr().out

    from gateway.config import Platform, load_gateway_config
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "SRC-TG")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "SRC-DC")
    gateway = load_gateway_config()
    assert not gateway.platforms[Platform.TELEGRAM].enabled and not gateway.platforms[Platform.DISCORD].enabled
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN"), monkeypatch.delenv("DISCORD_BOT_TOKEN")

    _import(monkeypatch, dst, _undo_point(dst), keep_host=True)
    after = _tree(dst)
    after.pop("state.db"), before.pop("state.db")
    assert after == before and _yaml(dst / "config.yaml")["platforms"] == {"discord": {"enabled": True}}


def test_a_transfer_leaves_what_no_backup_carries_when_home_is_the_hermes_home(homes, monkeypatch):
    """Hermes Cloud runs with HOME set to the Hermes home: the agent's workspace and the container's
    dotfiles sit beside the agent's files, and a backup from another machine never carries them."""
    src, dst, tmp = homes
    monkeypatch.setattr(Path, "home", lambda: dst)
    kept = {"workspace/app/main.py": "print('cloud')\n", ".ssh/id_ed25519": "KEY\n",
            ".config/gh/hosts.yml": "github.com: {}\n", "home/notes.txt": "notes\n"}
    for rel, text in kept.items():
        _write(dst, rel, text)
    _write(dst, "sessions/cloud.json", "{}\n")
    _write(dst, "skills/cloud-only/.hidden", "x\n")  # only the root's dot-entries are HOME's
    for hermes_dotfile in (".op.env", ".anthropic_oauth.json"):  # Hermes' own: replaced as usual
        _write(dst, hermes_dotfile, "stale login\n")
    _write(src, "plugins/mine/plugin.yaml", "name: mine\n")

    _import(monkeypatch, dst, _backup(monkeypatch, src, tmp / "src.zip"), keep_host=True)

    assert {rel: (dst / rel).read_text() for rel in kept} == kept
    assert not (dst / "skills/cloud-only").exists() and not (dst / "sessions/cloud.json").exists()
    assert not (dst / ".op.env").exists() and not (dst / ".anthropic_oauth.json").exists()
    _import(monkeypatch, dst, _undo_point(dst), keep_host=True)
    assert not (dst / "plugins/mine").exists() and {rel: (dst / rel).read_text() for rel in kept} == kept


def test_undo_removes_the_profile_the_transfer_brought_only_after_releasing_its_lock(homes, monkeypatch):
    """Removing a home whose gateway.lock the import still holds fails on Windows and frees the
    lock on POSIX: the undo tombstones it under the reservation, then deletes it like
    ``hermes profile delete`` once the reservation is released."""
    from hermes_cli import profiles
    from gateway.runtime_ownership import held_by_this_process
    from hermes_constants import named_profile_is_deleted
    src, dst, tmp = homes
    _write(src, "profiles/writer/config.yaml", "display:\n  skin: ares\n")
    _import(monkeypatch, dst, _backup(monkeypatch, src, tmp / "src.zip"), keep_host=True)
    seen, real_delete = [], profiles.delete_profile

    def delete(name, yes=False):
        home = dst / "profiles" / name
        seen.append((name, held_by_this_process(home), named_profile_is_deleted(home)))
        return real_delete(name, yes=yes)

    monkeypatch.setattr(profiles, "delete_profile", delete)
    _import(monkeypatch, dst, _undo_point(dst), keep_host=True)

    assert seen == [("writer", False, True)] and not (dst / "profiles/writer").exists()


def test_undo_point_holds_the_memory_provider_files_outside_the_home(homes, monkeypatch):
    """A source's external memory file must not irreversibly replace the cloud copy."""
    src, dst, tmp = homes
    archive = _backup(monkeypatch, src, tmp / "src.zip")
    member = "_external/memory-provider/store.json"
    with zipfile.ZipFile(archive, "a") as zf:
        zf.writestr(member, '{"memory": "source"}\n')
    _write(tmp, "memory-provider/store.json", '{"memory": "cloud"}\n')
    external_file = tmp / "memory-provider/store.json"

    _import(monkeypatch, dst, archive, keep_host=True)
    assert external_file.read_text() == '{"memory": "source"}\n'
    with zipfile.ZipFile(_undo_point(dst)) as zf:
        assert zf.read(member) == b'{"memory": "cloud"}\n'

    _import(monkeypatch, dst, _undo_point(dst), keep_host=True)
    assert external_file.read_text() == '{"memory": "cloud"}\n'


def test_a_retry_after_a_failed_restore_keeps_the_first_undo_point(homes, monkeypatch, capsys):
    """A restore that wrote files and failed names its undo point; the retry reuses it instead of
    backing up the half-restored agent, and the refusal message is never claimed."""
    from hermes_cli import backup
    src, dst, tmp = homes
    archive = _backup(monkeypatch, src, tmp / "src.zip")
    real = backup._extract_member_atomically

    def failing(zf, member, target, mode=None):
        if member == "SOUL.md":
            raise OSError("disk full")
        return real(zf, member, target, mode)

    monkeypatch.setattr(backup, "_extract_member_atomically", failing)
    assert _run(monkeypatch, dst, archive, keep_host=True) == 1
    out = capsys.readouterr().out
    first = _undo_point(dst)
    assert f"Undo point: {first}" in out and "no files restored" not in out
    assert (dst / "skills/cloud-only/SKILL.md").exists()  # nothing pruned after a failed restore

    monkeypatch.setattr(backup, "_extract_member_atomically", real)
    _import(monkeypatch, dst, archive, keep_host=True)
    assert f"Undo point: {first}" in capsys.readouterr().out
    assert sorted((dst / "backups").glob("pre-transfer-*.zip")) == [first]
    _import(monkeypatch, dst, first, keep_host=True)
    assert (dst / "SOUL.md").read_text() == "cloud persona\n"


def test_plain_import_is_unchanged(homes, monkeypatch):
    """Without --keep-host the archive's own Nous sign-in and config win, as before."""
    src, dst, tmp = homes
    _import(monkeypatch, dst, _backup(monkeypatch, src, tmp / "src.zip"))
    assert _yaml(dst / "config.yaml")["model"]["provider"] == "openrouter"
    assert "NOUS_API_KEY=SRC-NOUS" in (dst / ".env").read_text()
    assert (dst / "skills/cloud-only/SKILL.md").exists()  # a plain import never removes files
    assert not (dst / "backups").exists() or not list((dst / "backups").glob("pre-transfer-*"))


def test_a_repeated_host_key_in_the_backup_never_replaces_the_cloud_key():
    """.env loading takes the last assignment: a backup whose first one matches still gets merged."""
    from hermes_cli.backup_keep_host import merge_env
    merged = merge_env("API_SERVER_KEY=cloud\nOPENAI_API_KEY=mine\nAPI_SERVER_KEY=laptop\n", "API_SERVER_KEY=cloud\n")
    assert [line for line in merged.splitlines() if line.startswith("API_SERVER_KEY=")][-1] == "API_SERVER_KEY=cloud"
    assert "OPENAI_API_KEY=mine" in merged


def test_a_gateway_that_does_not_start_again_after_the_restore_reaches_the_action_status(tmp_path, monkeypatch, capsys):
    """The files arrived but the agent's bots and jobs are down: the status says so, not plain success."""
    from gateway import control_socket
    from hermes_cli import backup_keep_host as keep_host
    from hermes_cli import profiles_remove
    from hermes_cli import service_manager
    from hermes_cli.web_routers.actions import _import_report
    from hermes_cli.web_server_gateway import _ACTION_LOG_STARTS

    class Manager:
        def _run_svc(self, *args):
            pass

        def start(self, slot):
            raise service_manager.S6Error(f"s6-svc -u {slot}: timed out")

    monkeypatch.setattr(keep_host, "_s6_slot", lambda home: "gateway-default")
    monkeypatch.setattr(service_manager, "S6ServiceManager", Manager)
    monkeypatch.setattr(control_socket, "pause_gateway_for_update", lambda home, timeout: {"pausing": True})
    monkeypatch.setattr(profiles_remove, "wait_profile_unowned", lambda home, timeout: True)
    with keep_host.gateway_paused(tmp_path):
        print("Import complete: 3 files restored")
    log = tmp_path / "import.log"
    log.write_text(capsys.readouterr().out)
    monkeypatch.setitem(_ACTION_LOG_STARTS, "import", (4242, 0))

    assert _import_report(log, 4242)["gateway_down"] == "gateway-default: s6-svc -u gateway-default: timed out"


def _deny_reading(monkeypatch, blocked: Path) -> None:
    """Make the archive writer fail on *blocked* the way a root-owned file fails for the agent."""
    real_write = zipfile.ZipFile.write

    def write(self, filename, arcname=None, *args, **kwargs):
        if Path(filename).resolve() == blocked.resolve():
            raise PermissionError(13, "Permission denied", str(filename))
        return real_write(self, filename, arcname, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "write", write)


def test_keep_host_transfer_skips_unreadable_portal_recovery_and_preserves_host_dirs(homes, monkeypatch):
    """Regression for #158231: root-owned portal-recovery/ and lost+found/ on a Cloud destination
    must neither break the pre-transfer undo point nor be overwritten/pruned by --keep-host."""
    src, dst, tmp = homes
    archive = _backup(monkeypatch, src, tmp / "src.zip")
    with zipfile.ZipFile(archive, "a") as zf:
        zf.writestr("portal-recovery/snapshots/last-ready/config.yaml", "from-archive: true\n")
        zf.writestr("lost+found/orphan.bin", "from-archive\n")
        zf.writestr("lost+found/from-archive.bin", "from-archive\n")

    _write(dst, "portal-recovery/snapshots/last-ready/config.yaml", "host-recovery: true\n")
    _write(dst, "lost+found/orphan.bin", "host-orphan\n")
    host_snapshot = dst / "portal-recovery/snapshots/last-ready/config.yaml"
    _deny_reading(monkeypatch, host_snapshot)

    _import(monkeypatch, dst, archive, keep_host=True)

    with zipfile.ZipFile(_undo_point(dst)) as zf:
        assert not any(n.startswith(("portal-recovery/", "lost+found/")) for n in zf.namelist())
    assert not list((dst / "backups").glob("pre-transfer-*.incomplete.zip"))
    assert host_snapshot.read_text() == "host-recovery: true\n"
    assert (dst / "lost+found/orphan.bin").read_text() == "host-orphan\n"
    assert not (dst / "lost+found/from-archive.bin").exists()


@pytest.mark.platforms("posix")
@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root bypasses chmod 0o000")
def test_keep_host_transfer_succeeds_with_chmod_0000_portal_recovery_on_posix(homes, monkeypatch):
    """The same transfer with a host file the agent really cannot read (mode 0o000)."""
    src, dst, tmp = homes
    archive = _backup(monkeypatch, src, tmp / "src.zip")
    _write(dst, "portal-recovery/snapshots/last-ready/config.yaml", "host-recovery: true\n")
    host_snapshot = dst / "portal-recovery/snapshots/last-ready/config.yaml"
    host_snapshot.chmod(0o000)
    try:
        _import(monkeypatch, dst, archive, keep_host=True)
        with zipfile.ZipFile(_undo_point(dst)) as zf:
            assert not any(n.startswith("portal-recovery/") for n in zf.namelist())
        assert host_snapshot.exists()
    finally:
        host_snapshot.chmod(0o600)


def test_full_backup_excludes_portal_recovery_and_lost_and_found_at_the_hermes_root_only(tmp_path, monkeypatch):
    """Only the Hermes root's portal-recovery/ and lost+found/ belong to the host. Same-named folders
    in a skill or a named profile are user data: they stay in backups (and so in undo points)."""
    from hermes_cli.backup import _write_full_zip_backup

    home = tmp_path / ".hermes"
    _write(home, "config.yaml", "model: test\n")
    _write(home, "portal-recovery/snapshots/last-ready/config.yaml", "root-only\n")
    _write(home, "lost+found/orphan.bin", "orphan\n")
    kept = ("skills/demo/portal-recovery/notes.md", "profiles/work/portal-recovery/notes.md",
            "profiles/work/lost+found/notes.md")
    for rel in kept:
        _write(home, rel, "user content\n")
    _deny_reading(monkeypatch, home / "portal-recovery/snapshots/last-ready/config.yaml")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    out_zip = home / "backups" / "pre-transfer-test.zip"
    out_zip.parent.mkdir(parents=True)
    assert _write_full_zip_backup(out_zip, home) == out_zip
    with zipfile.ZipFile(out_zip) as zf:
        names = set(zf.namelist())
    assert {"config.yaml", *kept} <= names
    assert not any(n.startswith(("portal-recovery/", "lost+found/")) for n in names)


def test_plain_import_of_an_older_archive_leaves_the_hosts_portal_recovery_alone(homes, monkeypatch):
    """Archives made before the backup exclusion still carry portal-recovery/: a plain import must not
    write it (on Hermes Cloud the folder is root-owned, so the write fails and the import reports incomplete)."""
    src, dst, tmp = homes
    archive = _backup(monkeypatch, src, tmp / "src.zip")
    with zipfile.ZipFile(archive, "a") as zf:
        zf.writestr("portal-recovery/portal-selected-model", "from-archive\n")
        zf.writestr("lost+found/from-archive.bin", "from-archive\n")
        zf.writestr("profiles/work/config.yaml", "model: test\n")
        zf.writestr("profiles/work/portal-recovery/notes.md", "user content\n")
    _write(dst, "portal-recovery/portal-selected-model", "host\n")

    _import(monkeypatch, dst, archive)

    assert (dst / "portal-recovery/portal-selected-model").read_text() == "host\n"
    assert not (dst / "lost+found").exists()
    assert (dst / "profiles/work/portal-recovery/notes.md").read_text() == "user content\n"
