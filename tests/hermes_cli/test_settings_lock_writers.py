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


def test_a_literal_dotted_key_cannot_mask_a_change_to_the_locked_nested_path(home):
    # The reviewer's sequence: a literal `approvals.mode` sibling, then a --force section rewrite.
    # With a dot-joined diff the literal key shadowed the nested one and both writes passed.
    from hermes_cli.config import set_config_value

    before = _text(home / "config.yaml")
    with pytest.raises(sl.SettingsLockError, match="approvals.mode"):
        set_config_value("approvals\\.mode", "manual", force=True)
    assert _text(home / "config.yaml") == before

    # Even if the literal sibling is already there (hand-written), the nested node stays locked.
    (home / "config.yaml").write_text(CONFIG + "approvals.mode: manual\n" + LOCK, encoding="utf-8")
    before = _text(home / "config.yaml")
    with pytest.raises(sl.SettingsLockError, match="approvals.mode"):
        set_config_value("approvals", '{"mode": "off"}', force=True)
    assert _text(home / "config.yaml") == before
    assert _raw(home)["approvals"] == {"mode": "manual"}

    # Control: an unlocked path through a legitimately dotted model ID still writes.
    set_config_value("models.grok-4\\.6.supports_vision", "true")
    assert _raw(home)["models"]["grok-4.6"]["supports_vision"] is True


@pytest.mark.parametrize("retire", ["disable", "clear"])
def test_an_unlock_does_not_revive_when_the_same_lock_is_recreated(home, retire):
    # Unlock A (password verified), retire A through a supported writer, then bring back a
    # byte-identical A — same keys, same stored hash, so the same content fingerprint. The old
    # receipt must stay dead: A's new incarnation never had its password presented.
    import types

    from hermes_cli.config import _cmd_config_lock, atomic_config_write, read_user_config_raw, set_config_value

    state = sl.lock_state(home)
    assert sl.verify_password(PASSWORD, state.spec["password"])
    sl.begin_unlock(home, seconds=600, spec=state.spec)
    lock_a = _raw(home)["settings_lock"]

    if retire == "disable":
        set_config_value("settings_lock.enabled", "false")
        set_config_value("settings_lock.enabled", "true")
    else:
        _cmd_config_lock(types.SimpleNamespace(clear=True, keys=[], no_password=False))
        assert "settings_lock" not in _raw(home)
        atomic_config_write(home / "config.yaml", {**read_user_config_raw(home / "config.yaml"),
                                                   "settings_lock": lock_a})
    assert _raw(home)["settings_lock"] == lock_a
    assert sl.spec_fingerprint(sl.lock_state(home).spec) == sl.spec_fingerprint(state.spec)

    assert sl.is_unlocked(home) is False
    before = _text(home / "config.yaml")
    with pytest.raises(sl.SettingsLockError, match="approvals.mode"):
        set_config_value("approvals.mode", "off")
    assert _text(home / "config.yaml") == before


@pytest.mark.parametrize("spelled", ["absolute", "tilde"])
def test_lock_administration_from_a_profile_targets_the_root_it_is_enforced_from(tmp_path, monkeypatch, spelled):
    # `hermes -p work config lock …` / `config lock --clear` / `config relock`: the status, the file
    # written, the receipt closed and the enforcing owner must all be the shared root, however
    # HERMES_HOME is spelled; the profile's own config stays untouched.
    import types

    from hermes_cli.config import _cmd_config_lock

    root = tmp_path / "hermes"
    profile = root / "profiles" / "work"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text(CONFIG, encoding="utf-8")
    (profile / "config.yaml").write_text("approvals:\n  mode: manual\ndisplay:\n  theme: dark\n", encoding="utf-8")
    profile_before = _text(profile / "config.yaml")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(profile) if spelled == "absolute" else "~/hermes/profiles/work")

    _cmd_config_lock(types.SimpleNamespace(clear=False, keys=["approvals.mode"], no_password=True))
    assert _raw(root)["settings_lock"]["keys"] == ["approvals.mode"]
    assert _text(profile / "config.yaml") == profile_before
    assert sl.lock_state(profile).status == "valid"
    assert sl.lock_state().status == "valid"  # the no-argument form every CLI/RPC door uses

    sl.begin_unlock(profile, seconds=60)
    sl.end_unlock()  # `hermes config relock`
    assert not sl.is_unlocked(profile)

    sl.begin_unlock(profile, seconds=60)
    _cmd_config_lock(types.SimpleNamespace(clear=True, keys=[], no_password=False))
    assert "settings_lock" not in _raw(root)
    assert _text(profile / "config.yaml") == profile_before
    assert sl.lock_state(profile).status == "off"


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


def _relock_at(monkeypatch, home: Path, env_writer: str) -> None:
    """Close the unlock window at the instant *env_writer* runs: the legal relock interleaving."""
    from hermes_cli import config

    real = getattr(config, env_writer)

    def relock_then_write(*args, **kwargs):
        sl.end_unlock(home)
        return real(*args, **kwargs)

    monkeypatch.setattr(config, env_writer, relock_then_write)


def test_credential_rotation_is_refused_before_env_changes(home, monkeypatch):
    from hermes_cli.credential_lifecycle import save_provider_env_credential

    before_cfg, before_env = _text(home / "config.yaml"), _text(home / ".env")
    with pytest.raises(sl.SettingsLockError, match="providers.openai.api_key"):
        save_provider_env_credential("OPENAI_API_KEY", "sk-NEW")
    assert _text(home / "config.yaml") == before_cfg
    # Not half-rotated: a new .env key under a locked stale mirror would be the #62269 bug again.
    assert _text(home / ".env") == before_env

    # Relock lands at the .env effect: the guarded mirror write must already be on disk, so the
    # operation completes instead of leaving the new key under the old mirror.
    _relock_at(monkeypatch, home, "save_env_value")
    sl.begin_unlock(home, seconds=60)
    result = save_provider_env_credential("OPENAI_API_KEY", "sk-NEW")
    assert result["config_updates"] == ["providers.openai.api_key"]
    assert _raw(home)["providers"]["openai"]["api_key"] == "sk-NEW"
    assert "sk-NEW" in _text(home / ".env")
    assert not sl.is_unlocked(home)


def test_credential_removal_is_refused_before_env_changes(home, monkeypatch):
    from hermes_cli.credential_lifecycle import remove_provider_env_credential

    before_cfg, before_env = _text(home / "config.yaml"), _text(home / ".env")
    with pytest.raises(sl.SettingsLockError, match="providers.openai.api_key"):
        remove_provider_env_credential("OPENAI_API_KEY")
    assert _text(home / "config.yaml") == before_cfg
    assert _text(home / ".env") == before_env

    _relock_at(monkeypatch, home, "remove_env_value")
    sl.begin_unlock(home, seconds=60)
    result = remove_provider_env_credential("OPENAI_API_KEY")
    assert result["config_scrubbed"] == ["providers.openai.api_key"]
    assert "api_key" not in _raw(home)["providers"]["openai"]
    assert "sk-OLD" not in _text(home / ".env")
    assert not sl.is_unlocked(home)


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


def test_logout_is_refused_before_auth_state_is_cleared_and_relock_cannot_split_it(home, monkeypatch):
    import types

    from hermes_cli import auth

    cleared = []
    monkeypatch.setattr(auth, "is_known_auth_provider", lambda provider: True)
    monkeypatch.setattr(auth, "_should_reset_config_provider_on_logout", lambda target: True)
    monkeypatch.setattr(auth, "clear_provider_auth", lambda target: cleared.append(target) or True)
    logout = types.SimpleNamespace(provider="openai")

    before = _text(home / "config.yaml")
    with pytest.raises(sl.SettingsLockError, match="model.provider"):
        auth.logout_command(logout)
    assert _text(home / "config.yaml") == before
    assert cleared == []

    # Relock lands the moment auth state is cleared: the guarded reset must already be on disk.
    def clear_then_relock(target):
        sl.end_unlock(home)
        cleared.append(target)
        return True

    monkeypatch.setattr(auth, "clear_provider_auth", clear_then_relock)
    sl.begin_unlock(home, seconds=60)
    auth.logout_command(logout)
    assert cleared == ["openai"]
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


@pytest.mark.parametrize("spelling", ["direct", "dot-dot", "alias-symlink", "external-profile"])
def test_desktop_roundtrip_save_in_a_profile_is_refused_by_the_root_lock(home, tmp_path, spelling):
    # Every spelling of the same profile directory is owned by the same root: read through any
    # other identity, the profile's own config.yaml would be taken for the policy (and say "off").
    from hermes_cli.config import read_user_config_raw
    from utils import atomic_roundtrip_yaml_save

    profile = home / "profiles" / "work"
    if spelling == "external-profile":  # stored elsewhere, linked in as <root>/profiles/work
        real = tmp_path / "elsewhere" / "work"
        real.mkdir(parents=True)
        profile.parent.mkdir(parents=True)
        profile.symlink_to(real, target_is_directory=True)
    else:
        profile.mkdir(parents=True)
    if spelling == "alias-symlink":
        (tmp_path / "alias").symlink_to(profile, target_is_directory=True)
    written = {"direct": profile, "dot-dot": profile / ".." / "work",
               "alias-symlink": tmp_path / "alias", "external-profile": tmp_path / "elsewhere" / "work"}[spelling]
    config_path = written / "config.yaml"
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


@pytest.mark.parametrize("err", ["EACCES", "EIO"])
def test_an_unreadable_root_refuses_a_profile_write_and_only_a_missing_root_is_off(home, monkeypatch, err):
    # The profile's own file stays readable, so its readability guard passes; only the ROOT, where
    # the policy lives, cannot be read. Unknown policy is not "no lock".
    import errno
    import os

    from utils import atomic_roundtrip_yaml_save

    config_path = home / "profiles" / "work" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("approvals:\n  mode: manual\n", encoding="utf-8")
    root_config = home / "config.yaml"
    real_read_text = Path.read_text
    code = getattr(errno, err)

    def read_text(self, *args, **kwargs):
        if self == root_config:
            raise OSError(code, os.strerror(code), str(self))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(sl.SettingsLockError, match="cannot be read"):
        atomic_roundtrip_yaml_save(config_path, {"approvals": {"mode": "off"}})
    assert _text(config_path) == "approvals:\n  mode: manual\n"

    monkeypatch.setattr(Path, "read_text", real_read_text)
    root_config.unlink()  # proven absence: no policy exists, so the same write goes through
    atomic_roundtrip_yaml_save(config_path, {"approvals": {"mode": "off"}})
    from hermes_cli.config import read_user_config_raw

    assert read_user_config_raw(config_path)["approvals"]["mode"] == "off"


@pytest.mark.parametrize("spelling", [
    '"settings\\u005flock"', '"settings\\x5flock"',
    # An escaped YAML 1.1 line break (NEL, LS, PS) splitting an explicit double-quoted key.
    *(pytest.param('? "settings\\' + chr(cp) + '  _lock"\n', id=f"escaped-U+{cp:04X}")
      for cp in (0x85, 0x2028, 0x2029)),
])
def test_an_escaped_spelling_of_the_root_stanza_still_governs_a_profile(home, spelling):
    # YAML resolves both spellings to the key `settings_lock`; neither contains it literally.
    from hermes_cli.config import read_user_config_raw
    from utils import atomic_roundtrip_yaml_save

    config_path = home / "profiles" / "work" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("approvals:\n  mode: manual\n", encoding="utf-8")
    root_config = home / "config.yaml"
    root_config.write_text(spelling + ": {enabled: true, keys: [approvals.mode]}\n", encoding="utf-8")
    assert "settings_lock" not in _text(root_config)

    with pytest.raises(sl.SettingsLockError, match="approvals.mode"):
        atomic_roundtrip_yaml_save(config_path, {"approvals": {"mode": "off"}})
    assert _text(config_path) == "approvals:\n  mode: manual\n"

    # Control: a root that spells no lock at all stays "no lock", even with an escape in it and
    # even when it does not parse (a broken root must not block every profile of a lock-less install).
    root_config.write_text('path: "C:\\\\u005fdir"\nbroken: [unclosed\n', encoding="utf-8")
    atomic_roundtrip_yaml_save(config_path, {"approvals": {"mode": "off"}})
    assert read_user_config_raw(config_path)["approvals"]["mode"] == "off"


_ALIAS_LOCK = "settings_lock: {enabled: true, keys: [approvals.mode, allowed]}\n"


_BOTH = ("replace", "update")
# (id, document, dotted path the proposal changes, new value, writers the case applies to). The
# key-update writer REPLACES a binding rather than writing into the node it names, so it carries a
# change through an alias only when the path reaches inside the shared node.
_JUDGED = [
    ("mapping", "approvals: &a {mode: manual}\nunlocked_copy: *a\n" + _ALIAS_LOCK, "unlocked_copy.mode", "off", _BOTH),
    ("sequence", "approvals: {mode: manual}\nallowed: &s [one]\nunlocked_copy: *s\n" + _ALIAS_LOCK,
     "unlocked_copy", ["two"], ("replace",)),
    ("policy", "approvals: {mode: manual}\nsettings_lock: &p {enabled: true, keys: [approvals.mode, allowed]}\n"
     "unlocked_copy: *p\n", "unlocked_copy.enabled", False, _BOTH),
    # A `<<:` merge: the round-trip object keeps the inherited value cached, the emitted bytes do not.
    ("merge-key", "defaults: &d {mode: manual}\napprovals: {<<: *d}\n" + _ALIAS_LOCK, "defaults.mode", "off", _BOTH),
    ("merge-key-policy", "approvals: {mode: manual}\nlock_template: &p {enabled: true, keys: [approvals.mode, allowed]}\n"
     "settings_lock: {<<: *p}\n", "lock_template.enabled", False, _BOTH),
    # Python's 1 == True: `max_turns: 1` becoming `true` is unlimited to its reader.
    ("int-to-bool", "approvals: {mode: manual}\nallowed: 1\n" + _ALIAS_LOCK, "allowed", True, _BOTH),
    ("kind-in-list", "approvals: {mode: manual}\nallowed: [0]\n" + _ALIAS_LOCK, "allowed", [False], _BOTH),
]


@pytest.mark.parametrize("writer, target, document, key, value", [
    pytest.param(writer, target, document, key, value, id=f"{writer}-{target}-{case}")
    for case, document, key, value, writers in _JUDGED for writer in writers for target in ("root", "profile")])
def test_what_is_judged_is_the_written_document_as_a_reader_sees_it(home, writer, target, document, key, value):
    # The proposal may change only an UNLOCKED node, so its own diff names no locked path; aliases
    # and merge keys then carry it onto a locked one in the bytes that are written, and Python
    # equality calls a changed kind unchanged. What is judged must be the canonical reading of the
    # bytes that land.
    import hermes_yaml
    from hermes_cli.config import atomic_config_replace
    from utils import atomic_roundtrip_yaml_update

    root_config = home / "config.yaml"
    config_path = root_config if target == "root" else home / "profiles" / "work" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    root_config.write_text(document, encoding="utf-8")
    config_path.write_text(document, encoding="utf-8")
    before = _text(config_path)

    proposal = hermes_yaml.safe_load(before)
    *parents, leaf = key.split(".")
    node = proposal
    for segment in parents:
        node = node[segment]
    node[leaf] = value
    with pytest.raises(sl.SettingsLockError):
        if writer == "replace":
            atomic_config_replace(config_path, proposal)
        else:
            atomic_roundtrip_yaml_update(config_path, key, value)
    assert _text(config_path) == before
    assert sl.lock_state(home).status == "valid"

    # Control: on the same aliased document, a change that reaches no locked node still writes.
    unrelated = hermes_yaml.safe_load(before)
    unrelated["display"] = {"theme": "dark"}
    atomic_config_replace(config_path, unrelated)
    assert hermes_yaml.safe_load(_text(config_path))["display"] == {"theme": "dark"}


@pytest.mark.parametrize("minutes", [float("inf"), float("nan"), 1e300, 24 * 60 + 1])
def test_an_unlock_window_is_always_finite_and_bounded(home, monkeypatch, minutes):
    # `hermes config unlock --minutes inf` (argparse's float() accepts it, and `1e309` overflows to
    # it) must fail BEFORE a receipt exists; a stored non-finite or far-future expiry is not a window.
    import json
    import time
    import types

    import tui_gateway.server as server
    from hermes_cli.config import _cmd_config_unlock

    monkeypatch.setattr("getpass.getpass", lambda *_a, **_k: PASSWORD)
    with pytest.raises(SystemExit) as exit_info:
        _cmd_config_unlock(types.SimpleNamespace(minutes=minutes))
    assert exit_info.value.code == 1
    assert not sl.unlock_path(home).exists()

    reply = server._methods["config.unlock"](1, {"password": PASSWORD, "minutes": minutes})
    assert "error" in reply and not sl.unlock_path(home).exists()

    state = sl.lock_state(home)
    stored = minutes if minutes != minutes or minutes == float("inf") else time.time() + minutes * 60
    sl.unlock_path(home).write_text(json.dumps(
        {"expires_at": stored, "lock": sl.spec_fingerprint(state.spec), "epoch": state.epoch}), encoding="utf-8")
    assert sl.is_unlocked(home) is False

    # Control: the same doors with a finite duration open a window that a locked write can use.
    _cmd_config_unlock(types.SimpleNamespace(minutes=5.0))
    assert sl.is_unlocked(home) is True


@pytest.mark.parametrize("door", ["cli", "rpc"])
def test_a_relock_that_cannot_remove_the_receipt_reports_the_window_as_still_open(home, monkeypatch, door, capsys):
    import types

    import tui_gateway.server as server
    from hermes_cli.config import _cmd_config_relock, set_config_value

    def relock():
        if door == "rpc":
            return "error" not in server._methods["config.relock"](1, {})
        try:
            _cmd_config_relock(types.SimpleNamespace())
        except SystemExit as exc:
            return exc.code in (0, None)
        return True

    assert relock() is True  # no receipt at all: closing a closed window is a success

    sl.begin_unlock(home, seconds=600)
    receipt = sl.unlock_path(home)
    real_unlink = Path.unlink

    def unlink(self, *args, **kwargs):
        if self == receipt:
            raise PermissionError(13, "Permission denied", str(self))
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    capsys.readouterr()
    assert relock() is False
    assert "Unlock window closed" not in capsys.readouterr().out
    assert sl.is_unlocked(home) is True

    monkeypatch.setattr(Path, "unlink", real_unlink)
    assert relock() is True
    before = _text(home / "config.yaml")
    with pytest.raises(sl.SettingsLockError, match="approvals.mode"):
        set_config_value("approvals.mode", "off")
    assert _text(home / "config.yaml") == before


@pytest.mark.parametrize("paused", ["after_the_check", "at_the_dump"])
def test_no_locked_write_lands_after_relock_has_returned(home, monkeypatch, paused):
    # A writer the open window authorised is held mid-flight while another thread relocks.
    # Whichever side wins, a locked value may only reach disk BEFORE `end_unlock` returns.
    import threading

    import utils
    from hermes_cli.config import read_user_config_raw

    config_path = home / "profiles" / "work" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("approvals:\n  mode: manual\n", encoding="utf-8")
    sl.begin_unlock(home, seconds=600)

    reached, resume, events, outcome = threading.Event(), threading.Event(), [], []
    real_authorize, real_dump = sl._authorize, utils._roundtrip_dump

    def authorize(*args, **kwargs):
        used_window = real_authorize(*args, **kwargs)
        if paused == "after_the_check" and not reached.is_set():
            reached.set()
            assert resume.wait(30)
        return used_window

    def dump(*args, **kwargs):
        if paused == "at_the_dump":
            reached.set()
            assert resume.wait(30)
        real_dump(*args, **kwargs)
        events.append("landed")

    monkeypatch.setattr(sl, "_authorize", authorize)
    monkeypatch.setattr(utils, "_roundtrip_dump", dump)

    def write():
        try:
            if paused == "at_the_dump":
                utils.atomic_roundtrip_yaml_save(config_path, {"approvals": {"mode": "off"}})
            else:
                utils.atomic_roundtrip_yaml_update(config_path, "approvals.mode", "off")
        except sl.SettingsLockError:
            outcome.append("refused")

    def relock():
        sl.end_unlock(home)
        events.append("relocked")

    writer, relocker = threading.Thread(target=write), threading.Thread(target=relock)
    writer.start()
    assert reached.wait(30)
    relocker.start()
    relocker.join(2.0)  # returns at once unless the writer holds the fence; either way, let it go on
    resume.set()
    writer.join(30)
    relocker.join(30)
    assert not writer.is_alive() and not relocker.is_alive()

    assert sl.is_unlocked(home) is False
    if paused == "after_the_check":
        assert outcome == ["refused"] and events == ["relocked"]
        assert read_user_config_raw(config_path)["approvals"]["mode"] == "manual"
    else:
        assert events == ["landed", "relocked"]


# ── hermes agent import (command_allowlist / approvals.deny / mcp_servers → config.yaml) ─────


def test_profile_clone_channel_strip_is_refused_by_the_root_lock(home):
    # `hermes profile create --clone`: strip_channel_config rewrites the clone's config.yaml whole.
    from hermes_cli.config import read_user_config_raw
    from hermes_cli.profile_channels import strip_channel_config

    (home / "config.yaml").write_text(
        CONFIG + LOCK.replace("providers.*", "platforms.*"), encoding="utf-8")
    config_path = home / "profiles" / "clone" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("model:\n  default: gpt-4o\nplatforms:\n  discord:\n    token: t\n", encoding="utf-8")
    before = _text(config_path)

    with pytest.raises(sl.SettingsLockError, match="platforms"):
        strip_channel_config(config_path)
    assert _text(config_path) == before

    sl.begin_unlock(home, seconds=60)
    assert strip_channel_config(config_path) == ["platforms"]
    assert "platforms" not in read_user_config_raw(config_path)


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


def test_a_refused_always_approval_is_not_published_to_other_sessions(home):
    # "Always" persists to command_allowlist; when the lock refuses that write, the approval must
    # not stay live in memory for every other session. This session's own approval stands.
    from tools import approval

    (home / "config.yaml").write_text(
        CONFIG + "settings_lock: {enabled: true, keys: [command_allowlist]}\n", encoding="utf-8")
    before = _text(home / "config.yaml")
    pattern = "git reset --hard (destroys uncommitted changes)"
    approval.load_permanent(set())

    approval._persist_choice("first", "always", [(pattern, "fixture warning", False)])
    assert _text(home / "config.yaml") == before
    assert approval.is_approved("first", pattern)
    assert not approval.is_approved("second", pattern)

    sl.begin_unlock(home, seconds=60)
    approval._persist_choice("first", "always", [(pattern, "fixture warning", False)])
    assert pattern in _raw(home)["command_allowlist"]
    assert approval.is_approved("second", pattern)


def test_memory_plugin_config_write_is_refused_atomically(home):
    from hermes_cli.config import read_user_config_raw
    from plugins.memory.holographic import HolographicMemoryProvider

    (home / "config.yaml").write_text(
        "plugins:\n  hermes-memory-store:\n    db_path: /tmp/a.db\n"
        "settings_lock:\n  enabled: true\n  keys: [plugins.*]\n"
        f"  password: '{sl.hash_password(PASSWORD)}'\n", encoding="utf-8")
    before = _text(home / "config.yaml")

    # The plugin routes through the canonical hermes_cli.config.save_config, which lets the refusal
    # propagate instead of swallowing it — the dashboard shows "settings are locked", not a silent no-op.
    with pytest.raises(sl.SettingsLockError):
        HolographicMemoryProvider().save_config({"db_path": "/tmp/b.db"}, home)
    assert _text(home / "config.yaml") == before

    sl.begin_unlock(home, seconds=60)
    HolographicMemoryProvider().save_config({"db_path": "/tmp/b.db"}, home)
    assert read_user_config_raw(home / "config.yaml")["plugins"]["hermes-memory-store"]["db_path"] == "/tmp/b.db"
