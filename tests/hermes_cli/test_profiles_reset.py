"""``hermes profile reset`` / ``POST /api/profiles/{name}/reset``: the default profile is reset, not removed.

The default home is the install root: it holds ``profiles/`` (every other profile), the source
checkout, and a ``state.db`` that also carries other profiles' rows. A reset must wipe only the
default agent's own state (memories, skills, settings, persona, sessions, cron jobs) and keep
credentials, other profiles and the install.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import hermes_yaml as yaml
from hermes_cli import profiles as profiles_mod
from hermes_cli.default_soul import DEFAULT_SOUL_MD
from hermes_cli.profiles_reset import reset_profile


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Seeding runs ``sync_skills`` in a subprocess against the real bundled tree; the contract here
    # is that the reset asks for it, so stand in a seeder that installs one "bundled" skill.
    seeded: list[Path] = []

    def fake_seed(profile_dir, quiet=False):
        seeded.append(Path(profile_dir))
        bundled = Path(profile_dir) / "skills" / "bundled-skill"
        bundled.mkdir(parents=True, exist_ok=True)
        (bundled / "SKILL.md").write_text("---\nname: bundled-skill\n---\n", encoding="utf-8")
        return {"copied": ["bundled-skill"]}

    monkeypatch.setattr(profiles_mod, "seed_profile_skills", fake_seed)
    return home, seeded


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _populate_default(home: Path) -> dict:
    """A lived-in default profile next to a named profile and the install."""
    from hermes_state import SessionDB
    from cron.jobs import create_job, use_cron_store

    _write(home / "config.yaml", yaml.safe_dump({
        "model": {"default": "anthropic/claude-sonnet-4", "provider": "openrouter"},
        "display": {"skin": "ares"},
        "agent": {"max_turns": 7},
        "gateway": {"multiplex_profiles": False},
    }))
    _write(home / ".env", "OPENROUTER_API_KEY=sk-test\n")
    _write(home / "auth.json", json.dumps({"providers": {"nous": {"token": "t"}}}))
    _write(home / "SOUL.md", "You are a pirate.")
    _write(home / "profile.yaml", "display_name: Captain\n")
    _write(home / "memories" / "MEMORY.md", "user likes tea")
    _write(home / "memories" / "USER.md", "name: Ada")
    _write(home / "skills" / "my-skill" / "SKILL.md", "---\nname: my-skill\n---\n")
    _write(home / ".no-bundled-skills", "opted out\n")
    _write(home / "hermes-agent" / "run_agent.py", "# install checkout\n")
    _write(home / "logs" / "agent.log", "log line\n")
    _write(home / "workspace" / "notes.txt", "user file\n")

    coder = profiles_mod.create_profile("coder", no_alias=True)
    _write(coder / "memories" / "MEMORY.md", "coder memory")

    db = SessionDB(home / "state.db")
    db.create_session("default-cli", "cli")
    db.create_session("default-gw", "telegram", session_key="agent:main:telegram:dm:1")
    # Another profile's history that landed in the default store: never the default's to delete.
    db.create_session("crossed-coder", "telegram", session_key="agent:coder:telegram:dm:2",
                      profile_name="coder")
    db.close()
    _write(home / "sessions" / "default-cli.jsonl", "{}\n")
    _write(home / "sessions" / "crossed-coder.jsonl", "{}\n")

    coder_db = SessionDB(coder / "state.db")
    coder_db.create_session("coder-own", "cli")
    coder_db.close()

    with use_cron_store(home):
        create_job("say hi", "every 1h", name="hourly")
    with use_cron_store(coder):
        create_job("coder job", "every 2h", name="coder-hourly")
    return {"coder": coder}


def _session_ids(db_path: Path) -> set[str]:
    from hermes_state import SessionDB
    db = SessionDB(db_path)
    try:
        return {row["id"] for row in db._read_all("SELECT id FROM sessions")}
    finally:
        db.close()


def _cron_job_names(home: Path) -> list[str]:
    from cron.jobs import load_jobs, use_cron_store
    with use_cron_store(home):
        return [job["name"] for job in load_jobs()]


def test_reset_default_wipes_agent_state_and_keeps_credentials_profiles_and_install(profile_env):
    home, seeded = profile_env
    coder = _populate_default(home)["coder"]

    result = reset_profile("default", yes=True)

    # Wiped: memories, user skills (bundled re-seeded), persona, settings, sessions, cron.
    assert list((home / "memories").iterdir()) == []
    assert not (home / "skills" / "my-skill").exists()
    assert (home / "skills" / "bundled-skill" / "SKILL.md").is_file()
    assert seeded == [home]
    assert not (home / ".no-bundled-skills").exists()
    assert not (home / "profile.yaml").exists()
    assert (home / "SOUL.md").read_text(encoding="utf-8") == DEFAULT_SOUL_MD
    config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert config["model"] == {"default": "anthropic/claude-sonnet-4", "provider": "openrouter"}
    assert config["gateway"] == {"multiplex_profiles": False}
    assert "display" not in config and "agent" not in config
    assert _session_ids(home / "state.db") == {"crossed-coder"}
    assert not (home / "sessions" / "default-cli.jsonl").exists()
    assert (home / "sessions" / "crossed-coder.jsonl").exists()
    assert _cron_job_names(home) == []
    assert result.sessions_deleted == 2
    assert result.cron_jobs_removed == 1
    assert result.failures == []

    # Kept: credentials, the install, logs, user files, and the other profile untouched.
    assert (home / ".env").read_text(encoding="utf-8") == "OPENROUTER_API_KEY=sk-test\n"
    assert json.loads((home / "auth.json").read_text(encoding="utf-8"))["providers"]["nous"]["token"] == "t"
    assert (home / "hermes-agent" / "run_agent.py").is_file()
    assert (home / "logs" / "agent.log").is_file()
    assert (home / "workspace" / "notes.txt").is_file()
    assert (coder / "memories" / "MEMORY.md").read_text(encoding="utf-8") == "coder memory"
    assert _session_ids(coder / "state.db") == {"coder-own"}
    assert _cron_job_names(coder) == ["coder-hourly"]
    assert "coder" in profiles_mod.list_profile_names()


def test_typed_confirmation_mismatch_changes_nothing(profile_env, monkeypatch):
    home, seeded = profile_env
    _populate_default(home)
    monkeypatch.setattr("builtins.input", lambda _prompt: "nope")

    assert reset_profile("default") is None

    assert (home / "memories" / "MEMORY.md").exists()
    assert (home / "skills" / "my-skill" / "SKILL.md").exists()
    assert (home / "SOUL.md").read_text(encoding="utf-8") == "You are a pirate."
    assert _session_ids(home / "state.db") == {"default-cli", "default-gw", "crossed-coder"}
    assert seeded == []


def test_a_locked_path_fails_the_reset_after_every_other_step_ran(profile_env, monkeypatch):
    """Windows: a file another process holds open (WinError 32) must not abort the remaining steps,
    and the partial reset must be reported, never presented as success."""
    home, _seeded = profile_env
    _populate_default(home)
    from hermes_cli import profiles_remove

    real_rmtree = profiles_remove._rmtree_with_retry

    def locked_memories(path, handler):
        if Path(path).name == "memories":
            raise PermissionError(32, "The process cannot access the file because it is being used")
        real_rmtree(path, handler)

    monkeypatch.setattr(profiles_remove, "_rmtree_with_retry", locked_memories)

    with pytest.raises(RuntimeError, match="only partly reset") as excinfo:
        reset_profile("default", yes=True)

    assert "memories" in str(excinfo.value)
    assert (home / "memories" / "MEMORY.md").exists()
    assert not (home / "skills" / "my-skill").exists()
    assert _session_ids(home / "state.db") == {"crossed-coder"}
    assert (home / "SOUL.md").read_text(encoding="utf-8") == DEFAULT_SOUL_MD


def test_cli_delete_default_resets_instead_of_refusing(profile_env, capsys):
    home, _seeded = profile_env
    _populate_default(home)
    from hermes_cli.profile_cmd import cmd_profile

    cmd_profile(argparse.Namespace(profile_action="delete", profile_name="default", yes=True))

    out = capsys.readouterr().out
    assert "reset instead" in out
    assert "Profile 'default' reset." in out
    assert not (home / "memories" / "MEMORY.md").exists()
    assert (home / "profiles" / "coder").is_dir()


def test_core_delete_of_default_still_refuses_and_names_the_reset(profile_env):
    with pytest.raises(ValueError, match="hermes profile reset default"):
        profiles_mod.delete_profile("default", yes=True)


@pytest.fixture()
def client(profile_env):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from hermes_cli import web_server

    with TestClient(web_server.app, raise_server_exceptions=False) as c:
        c.headers["Authorization"] = f"Bearer {web_server._SESSION_TOKEN}"
        yield c


def test_api_reset_default_resets_and_delete_default_points_to_it(profile_env, client):
    home, _seeded = profile_env
    _populate_default(home)

    refused = client.delete("/api/profiles/default")
    assert refused.status_code == 400
    assert "reset" in refused.json()["detail"]
    assert (home / "memories" / "MEMORY.md").exists()

    resp = client.post("/api/profiles/default/reset")

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["profile"] == "default"
    assert body["sessions_deleted"] == 2
    assert body["cron_jobs_removed"] == 1
    assert not (home / "memories" / "MEMORY.md").exists()
    assert (home / ".env").exists()


def test_api_reset_requires_the_session_token(profile_env, client):
    home, _seeded = profile_env
    _populate_default(home)
    client.headers.pop("Authorization")

    resp = client.post("/api/profiles/default/reset")

    assert resp.status_code in (401, 403)
    assert (home / "memories" / "MEMORY.md").exists()
