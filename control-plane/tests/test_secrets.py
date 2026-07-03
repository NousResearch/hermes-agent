import stat
from pathlib import Path

from orchard.config import Settings
from orchard.secrets import LinkStore, LocalStore
from orchard.skills import required_secrets, secret_status


def _settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.paths.root = tmp_path / "data"
    s.paths.runtime = tmp_path / "run"
    (s.paths.home_for("alice")).mkdir(parents=True, exist_ok=True)
    return s


def test_localstore_roundtrip_and_perms(tmp_path: Path):
    s = _settings(tmp_path)
    store = LocalStore(s)
    assert store.names("alice") == []
    store.set("alice", "JIRA_TOKEN", "sekret")
    assert store.get("alice", "JIRA_TOKEN") == "sekret"
    assert store.all("alice") == {"JIRA_TOKEN": "sekret"}
    assert store.names("alice") == ["JIRA_TOKEN"]
    # file is 0600
    f = s.paths.home_for("alice") / "secrets.json"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert store.delete("alice", "JIRA_TOKEN") is True
    assert store.get("alice", "JIRA_TOKEN") is None


def test_localstore_is_per_tenant(tmp_path: Path):
    s = _settings(tmp_path)
    (s.paths.home_for("bob")).mkdir(parents=True, exist_ok=True)
    store = LocalStore(s)
    store.set("alice", "T", "a")
    store.set("bob", "T", "b")
    assert store.get("alice", "T") == "a"
    assert store.get("bob", "T") == "b"  # no cross-tenant bleed


def test_linkstore_one_time_and_expiry(tmp_path: Path):
    ls = LinkStore(tmp_path / "links.db")
    tok = ls.mint("alice", "integration:jira", "Jira", ttl=100.0, now=1000.0)
    assert ls.peek(tok, now=1050.0)["target"] == "integration:jira"
    # consume once
    assert ls.consume(tok, now=1050.0)["tenant_id"] == "alice"
    # not reusable
    assert ls.consume(tok, now=1060.0) is None
    assert ls.peek(tok, now=1060.0) is None
    # expiry
    tok2 = ls.mint("alice", "secret:X", "X", ttl=100.0, now=2000.0)
    assert ls.peek(tok2, now=2200.0) is None  # expired
    assert ls.consume(tok2, now=2200.0) is None


def test_skill_secret_requirements(tmp_path: Path):
    s = _settings(tmp_path)
    # a tenant-owned skill
    mine = s.paths.home_for("alice") / "skills" / "mine"
    mine.mkdir(parents=True)
    (mine / "SKILL.md").write_text(
        "---\nname: mine\nmetadata:\n  orchard:\n    secrets:\n"
        "      - env: MY_TOKEN\n        label: My token\n        required: true\n---\nbody"
    )
    # a shared base skill
    shared = tmp_path / "base" / "weather"
    shared.mkdir(parents=True)
    (shared / "SKILL.md").write_text(
        "---\nname: weather\nmetadata:\n  orchard:\n    secrets:\n"
        "      - env: WEATHER_TOKEN\n        label: Weather\n---\nbody"
    )
    s.skills.shared_dir = str(tmp_path / "base")

    envs = {r["env"] for r in required_secrets(s, "alice")}
    assert envs == {"MY_TOKEN", "WEATHER_TOKEN"}

    store = LocalStore(s)
    store.set("alice", "MY_TOKEN", "x")
    status = {r["env"]: r["set"] for r in secret_status(s, store, "alice")}
    assert status == {"MY_TOKEN": True, "WEATHER_TOKEN": False}
