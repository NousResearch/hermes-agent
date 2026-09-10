"""The settings lock holds at every shipped ``config.yaml`` writer — driven through the REAL paths.

The lock is enforced inside the two whole-document primitives, ``hermes_cli.config.atomic_config_write``
and ``utils.atomic_roundtrip_yaml_save``; nothing here calls the predicate directly. Each test drives
one production write path that the first cut of the lock did not cover (it gated ``save_config``
alone) and proves that, while locked, the file is byte-identical afterwards — and, for writers with
an earlier side effect (``.env``, ``auth.json``), that the side effect did not happen either — then
that the same call succeeds inside an unlock window.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.settings_lock as sl

PASSWORD = "test-only-password"

CONFIG = (
    "approvals:\n  mode: manual\n"
    "model:\n  provider: openai\n  default: gpt-4o\n"
    "providers:\n  openai:\n    api_key: sk-OLD\n"
)
LOCK = (
    "settings_lock:\n  enabled: true\n  keys: [approvals.mode, model.provider, providers.*]\n"
    f"  password: '{sl.hash_password(PASSWORD)}'\n"
)


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "hermes"
    root.mkdir()
    (root / "config.yaml").write_text(CONFIG + LOCK, encoding="utf-8")
    (root / ".env").write_text("OPENAI_API_KEY=sk-OLD\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _raw(home: Path) -> dict:
    from hermes_cli.config import read_user_config_raw

    return read_user_config_raw(home / "config.yaml")


# ── the CLI ──────────────────────────────────────────────────────────────────


def test_hermes_config_set_refuses_a_locked_key_until_unlocked(home):
    from hermes_cli.config import set_config_value

    before = _text(home / "config.yaml")
    with pytest.raises(sl.SettingsLockError, match="approvals.mode"):
        set_config_value("approvals.mode", "off")
    assert _text(home / "config.yaml") == before

    sl.begin_unlock(home, seconds=60)
    set_config_value("approvals.mode", "off")
    assert _raw(home)["approvals"]["mode"] == "off"


def test_hermes_config_unset_refuses_a_locked_key(home):
    from hermes_cli.config import unset_config_value

    before = _text(home / "config.yaml")
    with pytest.raises(sl.SettingsLockError, match="approvals.mode"):
        unset_config_value("approvals.mode")
    assert _text(home / "config.yaml") == before


@pytest.mark.parametrize("key, value", [("settings_lock.enabled", "false"), ("settings_lock.keys", "[]")])
def test_hermes_config_set_cannot_disable_the_lock(home, key, value):
    from hermes_cli.config import set_config_value

    before = _text(home / "config.yaml")
    with pytest.raises(sl.SettingsLockError, match="settings_lock"):
        set_config_value(key, value)
    assert _text(home / "config.yaml") == before
    assert sl.is_enabled(sl.lock_spec(home))


# ── the desktop (tui_gateway config.set → _write_config_key → _save_cfg) ─────


def test_desktop_config_set_refuses_a_locked_key_until_unlocked(home, monkeypatch):
    import tui_gateway.server as server

    monkeypatch.setattr(server, "_hermes_home", home)
    before = _text(home / "config.yaml")
    with pytest.raises(sl.SettingsLockError, match="approvals.mode"):
        server._write_config_key("approvals.mode", "off")  # the approvals pill's own toggle
    assert _text(home / "config.yaml") == before

    sl.begin_unlock(home, seconds=60)
    server._write_config_key("approvals.mode", "off")
    assert _raw(home)["approvals"]["mode"] == "off"


# ── credential lifecycle (desktop model.connect / web env save / `hermes config set OPENAI_API_KEY`) ──


def test_credential_rotation_is_refused_before_env_changes(home):
    from hermes_cli.credential_lifecycle import save_provider_env_credential

    before_cfg, before_env = _text(home / "config.yaml"), _text(home / ".env")
    with pytest.raises(sl.SettingsLockError, match="providers.openai.api_key"):
        save_provider_env_credential("OPENAI_API_KEY", "sk-NEW")
    assert _text(home / "config.yaml") == before_cfg
    # Not half-rotated: a new .env key under a locked stale mirror would be the #62269 bug again.
    assert _text(home / ".env") == before_env

    sl.begin_unlock(home, seconds=60)
    result = save_provider_env_credential("OPENAI_API_KEY", "sk-NEW")
    assert result["config_updates"] == ["providers.openai.api_key"]
    assert _raw(home)["providers"]["openai"]["api_key"] == "sk-NEW"
    assert "sk-NEW" in _text(home / ".env")


def test_credential_removal_is_refused_before_env_changes(home):
    from hermes_cli.credential_lifecycle import remove_provider_env_credential

    before_cfg, before_env = _text(home / "config.yaml"), _text(home / ".env")
    with pytest.raises(sl.SettingsLockError, match="providers.openai.api_key"):
        remove_provider_env_credential("OPENAI_API_KEY")
    assert _text(home / "config.yaml") == before_cfg
    assert _text(home / ".env") == before_env

    sl.begin_unlock(home, seconds=60)
    result = remove_provider_env_credential("OPENAI_API_KEY")
    assert result["config_scrubbed"] == ["providers.openai.api_key"]
    assert "api_key" not in _raw(home)["providers"]["openai"]
    assert "sk-OLD" not in _text(home / ".env")


def test_a_rotation_that_touches_no_mirror_is_unaffected_by_the_lock(home):
    from hermes_cli.credential_lifecycle import save_provider_env_credential

    (home / ".env").write_text("OPENAI_API_KEY=sk-UNMIRRORED\n", encoding="utf-8")
    result = save_provider_env_credential("OPENAI_API_KEY", "sk-NEW")

    assert result["config_updates"] == []
    assert "sk-NEW" in _text(home / ".env")


# ── auth (hermes auth / model setup / logout) ────────────────────────────────


def test_provider_switch_is_refused_and_auth_json_is_not_half_switched(home):
    from hermes_cli import auth

    before = _text(home / "config.yaml")
    with pytest.raises(sl.SettingsLockError, match="model.provider"):
        auth._update_config_for_provider("nous", "https://inference.example.com/v1")
    assert _text(home / "config.yaml") == before
    assert auth._load_auth_store().get("active_provider") != "nous"

    sl.begin_unlock(home, seconds=60)
    auth._update_config_for_provider("nous", "https://inference.example.com/v1")
    assert _raw(home)["model"]["provider"] == "nous"
    assert auth._load_auth_store()["active_provider"] == "nous"


def test_logout_provider_reset_is_refused_and_its_dry_run_writes_nothing(home):
    from hermes_cli import auth

    before = _text(home / "config.yaml")
    with pytest.raises(sl.SettingsLockError, match="model.provider"):
        auth._reset_config_provider(dry_run=True)  # what logout_command asks before clearing auth
    with pytest.raises(sl.SettingsLockError, match="model.provider"):
        auth._reset_config_provider()
    assert _text(home / "config.yaml") == before

    sl.begin_unlock(home, seconds=60)
    auth._reset_config_provider(dry_run=True)
    assert _text(home / "config.yaml") == before  # a dry run never writes, unlocked or not
    auth._reset_config_provider()
    assert _raw(home)["model"]["provider"] == "auto"


# ── the primitives themselves, from a PROFILE home: the lock comes from the root that owns it ──


def test_gateway_slash_command_shape_in_a_profile_is_refused_by_the_root_lock(home):
    # Exactly gateway/slash_commands.py's `_set_approval`: read_user_config_raw → mutate →
    # atomic_config_write on `<profile home>/config.yaml`.
    from hermes_cli.config import atomic_config_write, read_user_config_raw

    config_path = home / "profiles" / "work" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("approvals:\n  mode: manual\n", encoding="utf-8")

    user_config = read_user_config_raw(config_path)
    user_config.setdefault("approvals", {})["mode"] = "off"
    with pytest.raises(sl.SettingsLockError, match="approvals.mode"):
        atomic_config_write(config_path, user_config)
    assert read_user_config_raw(config_path)["approvals"]["mode"] == "manual"

    sl.begin_unlock(home, seconds=60)  # the ROOT window, not one inside the profile
    atomic_config_write(config_path, user_config)
    assert read_user_config_raw(config_path)["approvals"]["mode"] == "off"


def test_desktop_roundtrip_save_in_a_profile_is_refused_by_the_root_lock(home):
    from hermes_cli.config import read_user_config_raw
    from utils import atomic_roundtrip_yaml_save


    config_path = home / "profiles" / "work" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("# keep me\napprovals:\n  mode: manual\n", encoding="utf-8")

    with pytest.raises(sl.SettingsLockError, match="approvals.mode"):
        atomic_roundtrip_yaml_save(config_path, {"approvals": {"mode": "off"}})
    assert _text(config_path) == "# keep me\napprovals:\n  mode: manual\n"

    sl.begin_unlock(home, seconds=60)
    atomic_roundtrip_yaml_save(config_path, {"approvals": {"mode": "off"}})
    assert _text(config_path).startswith("# keep me\n")  # comment-preserving, as before
    assert read_user_config_raw(config_path)["approvals"]["mode"] == "off"


def test_a_window_file_inside_a_profile_does_not_unlock_it(home):
    import json
    import time

    from hermes_cli.config import atomic_config_write, read_user_config_raw

    profile = home / "profiles" / "work"
    profile.mkdir(parents=True)
    config_path = profile / "config.yaml"
    config_path.write_text("approvals:\n  mode: manual\n", encoding="utf-8")
    # Only the ROOT's window counts: `begin_unlock(profile)` itself resolves to the root, so the
    # bypass to rule out is a window file planted inside the profile directory by hand.
    (profile / sl.UNLOCK_FILENAME).write_text(json.dumps({"expires_at": time.time() + 60}), encoding="utf-8")
    assert not sl.unlock_path(home).exists()


    with pytest.raises(sl.SettingsLockError, match="approvals.mode"):
        atomic_config_write(config_path, {"approvals": {"mode": "off"}})
    assert read_user_config_raw(config_path)["approvals"]["mode"] == "manual"


# ── post-update auto-restore (hermes update) ─────────────────────────────────


def test_update_restore_of_a_locked_key_is_refused_and_reported_not_raised(home, caplog):
    from hermes_cli import backup

    snap = backup._quick_snapshot_root(home) / "snap-1"
    snap.mkdir(parents=True)
    (snap / "config.yaml").write_text("model:\n  provider: anthropic\n  default: gpt-4o\n", encoding="utf-8")
    before = _text(home / "config.yaml")

    assert backup.restore_config_model_settings_if_rewritten("snap-1", home) is None
    assert _text(home / "config.yaml") == before
    assert any("auto-restore failed" in r.getMessage() for r in caplog.records)

    sl.begin_unlock(home, seconds=60)
    assert backup.restore_config_model_settings_if_rewritten("snap-1", home)
    assert _raw(home)["model"]["provider"] == "anthropic"


# ── save_config still passes a save that leaves the locked keys alone ────────


def test_save_config_unrelated_change_passes_through_the_primitive_gate(home):
    from hermes_cli.config import read_raw_config, save_config

    save_config({**read_raw_config(), "display": {"theme": "dark"}})

    raw = _raw(home)
    assert raw["display"]["theme"] == "dark"
    assert raw["approvals"]["mode"] == "manual"
    assert raw["providers"]["openai"]["api_key"] == "sk-OLD"


# ── the ruamel round-trip primitive behind the TUI's and the desktop's model switch ──────────


def test_tui_and_desktop_model_switch_persister_is_refused(home, monkeypatch, caplog):
    # `cli.save_config_value` is what `/model` in the TUI and the desktop's model switch
    # (tui_gateway/model_switch.py) persist through; it swallows the error and returns False.
    import cli

    monkeypatch.setattr(cli, "_hermes_home", home, raising=False)
    before = _text(home / "config.yaml")
    assert cli.save_config_value("model.provider", "nous") is False
    assert _text(home / "config.yaml") == before
    assert "settings are locked: model.provider" in caplog.text

    sl.begin_unlock(home, seconds=60)
    assert cli.save_config_value("model.provider", "nous") is True
    assert _raw(home)["model"]["provider"] == "nous"


def test_roundtrip_key_update_in_a_profile_is_refused_by_the_root_lock(home):
    from hermes_cli.config import read_user_config_raw
    from utils import atomic_roundtrip_yaml_update

    config_path = home / "profiles" / "work" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("# keep me\napprovals:\n  mode: manual\n", encoding="utf-8")

    with pytest.raises(sl.SettingsLockError, match="approvals.mode"):
        atomic_roundtrip_yaml_update(config_path, "approvals.mode", "off")
    assert _text(config_path) == "# keep me\napprovals:\n  mode: manual\n"

    sl.begin_unlock(home, seconds=60)
    atomic_roundtrip_yaml_update(config_path, "approvals.mode", "smart")
    assert _text(config_path).startswith("# keep me\n")
    assert read_user_config_raw(config_path)["approvals"]["mode"] == "smart"


# ── hermes agent import (command_allowlist / approvals.deny / mcp_servers → config.yaml) ─────


def test_agent_import_config_write_is_refused(home):
    # `AgentImporter._import_permission_rules` / `import_mcp_servers` both end in dump_yaml_file
    # on <root>/config.yaml.
    from hermes_cli.agent_import import dump_yaml_file

    before = _text(home / "config.yaml")
    with pytest.raises(sl.SettingsLockError, match="approvals.mode"):
        dump_yaml_file(home / "config.yaml", {**_raw(home), "approvals": {"mode": "off", "deny": ["rm -rf"]}})
    assert _text(home / "config.yaml") == before


# ── the xAI retirement migration (hermes migrate / doctor --fix) ─────────────────────────────


def test_retired_model_migration_is_refused_and_leaves_no_backup(home):
    from hermes_cli import xai_retirement
    from hermes_cli.config import read_user_config_raw

    (home / "config.yaml").write_text(
        "principal:\n  provider: xai\n  model: grok-3\n"
        "settings_lock:\n  enabled: true\n  keys: [principal.*]\n"
        f"  password: '{sl.hash_password(PASSWORD)}'\n", encoding="utf-8")
    config_path = home / "config.yaml"
    issues = xai_retirement.find_retired_xai_refs(read_user_config_raw(config_path))
    assert [i.config_path for i in issues] == ["principal.model"]
    before = _text(config_path)

    with pytest.raises(sl.SettingsLockError, match="principal.model"):
        xai_retirement.apply_migration(config_path, issues)
    assert _text(config_path) == before
    assert not list(home.glob("backups/**/config*")), "a refused migration must not leave a backup copy"

    sl.begin_unlock(home, seconds=60)
    result = xai_retirement.apply_migration(config_path, issues, backup=False)
    assert result.config_changed is True
    assert read_user_config_raw(config_path)["principal"]["model"] == "grok-4.3"


# ── a memory plugin's own config.yaml writer (was a raw, truncating open()+yaml.dump) ────────


def test_memory_plugin_config_write_is_refused_atomically(home):
    from hermes_cli.config import read_user_config_raw
    from plugins.memory.holographic import HolographicMemoryProvider

    (home / "config.yaml").write_text(
        "plugins:\n  hermes-memory-store:\n    db_path: /tmp/a.db\n"
        "settings_lock:\n  enabled: true\n  keys: [plugins.*]\n"
        f"  password: '{sl.hash_password(PASSWORD)}'\n", encoding="utf-8")
    before = _text(home / "config.yaml")

    HolographicMemoryProvider().save_config({"db_path": "/tmp/b.db"}, home)  # swallows its own errors
    assert _text(home / "config.yaml") == before

    sl.begin_unlock(home, seconds=60)
    HolographicMemoryProvider().save_config({"db_path": "/tmp/b.db"}, home)
    assert read_user_config_raw(home / "config.yaml")["plugins"]["hermes-memory-store"]["db_path"] == "/tmp/b.db"
