"""The operator settings lock.

Named config paths that no writer may change until an unlock window is opened. The gate lives in
the two whole-document write primitives (``atomic_config_write`` / ``atomic_roundtrip_yaml_save``);
these tests exercise the predicate layer and a real ``save_config`` round trip, and
``test_settings_lock_writers.py`` drives every other shipped writer through its real entry point.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import settings_lock as sl


def _root(tmp_path: Path, body: str = "") -> Path:
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(body, encoding="utf-8")
    return home


LOCKED = "settings_lock:\n  enabled: true\n  keys: ['approvals.mode', 'providers.*']\n"


# ── path patterns ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path,pattern,expected",
    [
        ("approvals.mode", "approvals.mode", True),
        ("approvals.modes", "approvals.mode", False),
        ("approvals", "approvals.mode", False),
        ("providers", "providers.*", True),
        ("providers.openai.api_key", "providers.*", True),
        ("providers_extra.x", "providers.*", False),
        ("model.default", "providers.*", False),
    ],
)
def test_path_matches(path, pattern, expected):
    assert sl.path_matches(path, pattern) is expected


def test_changed_paths_reports_adds_removes_and_alterations():
    before = {"a": 1, "b": {"c": 2}, "gone": True}
    after = {"a": 1, "b": {"c": 3}, "added": "x"}

    assert sl.changed_paths(before, after) == ("added", "b.c", "gone")


def test_a_list_value_is_one_leaf_so_reordering_counts_as_a_change():
    """Order and length are part of the value: a locked list must not be quietly rewritten."""
    assert sl.changed_paths({"k": [1, 2]}, {"k": [2, 1]}) == ("k",)
    assert sl.changed_paths({"k": [1, 2]}, {"k": [1, 2]}) == ()


# ── what the lock refuses ────────────────────────────────────────────────────


def test_no_lock_configured_refuses_nothing(tmp_path):
    home = _root(tmp_path, "approvals:\n  mode: manual\n")

    sl.check_write({"approvals": {"mode": "manual"}}, {"approvals": {"mode": "off"}}, home)


def test_a_locked_path_cannot_change(tmp_path):
    home = _root(tmp_path, LOCKED)

    with pytest.raises(sl.SettingsLockError) as exc:
        sl.check_write({"approvals": {"mode": "manual"}}, {"approvals": {"mode": "off"}}, home)

    assert exc.value.paths == ("approvals.mode",)


def test_an_unrelated_change_still_goes_through(tmp_path):
    home = _root(tmp_path, LOCKED)

    sl.check_write({"approvals": {"mode": "manual"}},
                   {"approvals": {"mode": "manual"}, "display": {"theme": "dark"}}, home)


def test_a_prefix_pattern_covers_the_whole_subtree(tmp_path):
    home = _root(tmp_path, LOCKED)

    with pytest.raises(sl.SettingsLockError) as exc:
        sl.check_write({"providers": {"openai": {"api_key": "a"}}},
                       {"providers": {"openai": {"api_key": "b"}}}, home)

    assert exc.value.paths == ("providers.openai.api_key",)


def test_the_lock_protects_itself(tmp_path):
    """Otherwise every front door is one `config set settings_lock.enabled false` from off."""
    home = _root(tmp_path, LOCKED)

    with pytest.raises(sl.SettingsLockError):
        sl.check_write({"settings_lock": {"enabled": True}}, {"settings_lock": {"enabled": False}},
                       home)


def test_removing_a_locked_value_is_also_a_change(tmp_path):
    home = _root(tmp_path, LOCKED)

    with pytest.raises(sl.SettingsLockError):
        sl.check_write({"approvals": {"mode": "manual"}}, {}, home)


def test_an_enabled_but_keyless_lock_fails_CLOSED(tmp_path):
    """Opposite of the fail-OPEN Bot Mode flags: a broken lock must not silently stop protecting.
    Recoverable by editing the file, which already needs the access the lock does not claim."""
    home = _root(tmp_path, "settings_lock:\n  enabled: true\n  keys: 'approvals.mode'\n")

    with pytest.raises(sl.SettingsLockError, match="cannot be applied"):
        sl.check_write({}, {"anything": 1}, home)


@pytest.mark.parametrize("value,enabled", [
    (True, True), (1, True), ("true", True), ("YES", True), (" on ", True),
    (False, False), (0, False), ("no", False), ("off", False), ("FALSE", False),
    (None, False), ("", False),
])
def test_enabled_accepts_the_yaml_spellings(value, enabled):
    assert sl.is_enabled({"enabled": value}) is enabled


@pytest.mark.parametrize("value", ["maybe", "ture", "enabled", "1.0", 2, -1, 0.5, ["true"], {"a": 1}])
def test_an_unrecognised_enabled_value_is_unusable_not_off(tmp_path, value):
    """The fail-open this feature exists to avoid: `enabled: maybe` used to be indistinguishable
    from an explicit disable, so a typo in the one field that arms the lock silently unarmed it."""
    assert sl._enabled_state(value) == "invalid"
    home = _root(tmp_path, "")
    (home / "config.yaml").write_text(
        f"settings_lock:\n  enabled: {value!r}\n  keys: ['approvals.mode']\n", encoding="utf-8")

    assert sl.lock_state(home).status == "unusable"
    with pytest.raises(sl.SettingsLockError, match="cannot be applied"):
        sl.check_write({"approvals": {"mode": "manual"}}, {"approvals": {"mode": "off"}}, home)


def test_a_settings_lock_that_is_not_a_mapping_is_unusable(tmp_path):
    home = _root(tmp_path, "settings_lock: enabled\napprovals:\n  mode: manual\n")

    state = sl.lock_state(home)
    assert state.status == "unusable"
    assert "not a mapping" in state.reason
    with pytest.raises(sl.SettingsLockError):
        sl.check_write({"approvals": {"mode": "manual"}}, {"approvals": {"mode": "off"}}, home)


def test_a_root_config_that_no_longer_parses_does_not_switch_the_lock_off(tmp_path):
    """A typo anywhere in the root file must not unarm the lock for the profiles, whose own
    config.yaml still parses and writes normally."""
    home = _root(tmp_path, LOCKED + "model: [unclosed\n")
    profile = home / "profiles" / "lucky"
    profile.mkdir(parents=True)

    assert sl.lock_state(profile).status == "unusable"
    with pytest.raises(sl.SettingsLockError, match="cannot be applied"):
        sl.check_config_write(profile / "config.yaml", {"approvals": {"mode": "manual"}},
                              {"approvals": {"mode": "off"}})

    # A broken root that never named a lock has nothing to fail closed on.
    (home / "config.yaml").write_text("model: [unclosed\n", encoding="utf-8")
    assert sl.lock_state(profile).status == "off"


def test_an_unusable_lock_is_refused_even_inside_an_unlock_window(tmp_path):
    """A window cannot authorise writes against a spec that cannot be normalised: it has no
    generation to be bound to, and the password meant to gate it may be the malformed part."""
    home = _root(tmp_path, "settings_lock:\n  enabled: maybe\n  keys: ['approvals.mode']\n")
    # A receipt that DOES match this spec's fingerprint — otherwise the window is stale for an
    # unrelated reason and the test would pass even if the window were consulted first.
    state = sl.lock_state(home)
    assert state.status == "unusable"
    sl.unlock_path(home).write_text(
        '{"expires_at": %d, "lock": "%s"}' % (int(time.time() + 600), sl.spec_fingerprint(state.spec)),
        encoding="utf-8")
    assert sl.unlock_expiry(home, spec=state.spec) is not None  # the window really is live

    with pytest.raises(sl.SettingsLockError, match="cannot be applied"):
        sl.check_write({"approvals": {"mode": "manual"}}, {"approvals": {"mode": "off"}}, home)


def test_the_lock_is_read_from_the_ROOT_not_the_profile(tmp_path):
    """A per-profile copy must not be able to lock — or unlock — its own profile."""
    home = _root(tmp_path, "")
    profile = home / "profiles" / "lucky"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(LOCKED, encoding="utf-8")

    # Root has no lock, so a write scoped to the profile is unaffected by the profile's own stanza.
    sl.check_write({"approvals": {"mode": "manual"}}, {"approvals": {"mode": "off"}}, profile)

    # With the lock in the ROOT, the same profile-scoped write is refused.
    (home / "config.yaml").write_text(LOCKED, encoding="utf-8")
    with pytest.raises(sl.SettingsLockError):
        sl.check_write({"approvals": {"mode": "manual"}}, {"approvals": {"mode": "off"}}, profile)


# ── password ─────────────────────────────────────────────────────────────────


def test_password_round_trip_and_rejections():
    stored = sl.hash_password("correct horse")

    assert stored.startswith("scrypt$")
    assert "correct horse" not in stored  # never the plaintext
    assert sl.verify_password("correct horse", stored) is True
    assert sl.verify_password("Correct Horse", stored) is False
    assert sl.verify_password("", stored) is False


@pytest.mark.parametrize("stored", [None, "", 42, "plaintext", "scrypt$bad", "bcrypt$1$1$1$a$b",
                                    "scrypt$x$8$1$YQ==$Yg=="])
def test_a_malformed_stored_hash_never_verifies(stored):
    assert sl.verify_password("anything", stored) is False


def test_two_hashes_of_one_password_differ_by_salt():
    assert sl.hash_password("same") != sl.hash_password("same")


@pytest.mark.parametrize("stored", [123456, True, ["hash"], {"h": 1}, "plaintext", "scrypt$bad",
                                    "bcrypt$16384$8$1$YQ==$Yg==", "scrypt$x$8$1$YQ==$Yg==",
                                    "scrypt$16384$8$1$not-base64!$Yg==", "scrypt$16384$8$1$$Yg=="])
def test_a_malformed_password_makes_the_lock_unusable_never_passwordless(tmp_path, stored):
    """`has_password` gates whether the unlock doors ask for anything at all, so a hash this build
    cannot verify must not read as "no password configured" — that opened the window to anyone."""
    home = _root(tmp_path, "")
    (home / "config.yaml").write_text(
        "settings_lock:\n  enabled: true\n  keys: ['approvals.mode']\n"
        f"  password: {stored!r}\n", encoding="utf-8")

    state = sl.lock_state(home)
    assert state.status == "unusable"
    assert "password" in state.reason
    assert sl.has_password(state.spec) is False  # and the doors refuse before ever consulting it
    with pytest.raises(sl.SettingsLockError, match="cannot be applied"):
        sl.check_write({"approvals": {"mode": "manual"}}, {"approvals": {"mode": "off"}}, home)


def test_a_verifiable_password_keeps_the_lock_valid_and_required(tmp_path):
    stored = sl.hash_password("operator")
    home = _root(tmp_path, "")
    (home / "config.yaml").write_text(
        "settings_lock:\n  enabled: true\n  keys: ['approvals.mode']\n"
        f"  password: {stored}\n", encoding="utf-8")

    state = sl.lock_state(home)
    assert state.status == "valid"
    assert sl.has_password(state.spec) is True
    assert sl.describe(home)["password_required"] is True


# ── the unlock window ────────────────────────────────────────────────────────


def test_the_unlock_window_opens_expires_and_closes(tmp_path):
    home = _root(tmp_path, LOCKED)

    assert sl.is_unlocked(home) is False
    sl.begin_unlock(home, seconds=60)
    assert sl.is_unlocked(home) is True
    sl.check_write({"approvals": {"mode": "manual"}}, {"approvals": {"mode": "off"}}, home)

    sl.end_unlock(home)
    assert sl.is_unlocked(home) is False
    with pytest.raises(sl.SettingsLockError):
        sl.check_write({"approvals": {"mode": "manual"}}, {"approvals": {"mode": "off"}}, home)


def test_an_expired_window_does_not_unlock(tmp_path):
    home = _root(tmp_path, LOCKED)
    # Correctly bound to this lock, but lapsed — so this still tests expiry, not the binding.
    fingerprint = sl.spec_fingerprint(sl.lock_state(home).spec)
    sl.unlock_path(home).write_text(
        '{"expires_at": %d, "lock": "%s"}' % (int(time.time() - 5), fingerprint), encoding="utf-8")

    assert sl.is_unlocked(home) is False
    with pytest.raises(sl.SettingsLockError):
        sl.check_write({"approvals": {"mode": "manual"}}, {"approvals": {"mode": "off"}}, home)


@pytest.mark.parametrize("body", ["", "not json", '{"expires_at": "soon"}', "{}"])
def test_an_unreadable_window_file_is_not_an_unlock(tmp_path, body):
    home = _root(tmp_path, LOCKED)
    sl.unlock_path(home).write_text(body, encoding="utf-8")

    assert sl.is_unlocked(home) is False


def test_an_unlock_does_not_survive_the_lock_it_authorised_being_replaced(tmp_path):
    """Stale authority: unlock A, swap in lock B while the window is live, and B used to be born
    unlocked with B's password never verified. The receipt is bound to one lock generation."""
    stored_a, stored_b = sl.hash_password("operator-A"), sl.hash_password("attacker-B")
    lock_a = ("settings_lock:\n  enabled: true\n  keys: ['approvals.mode']\n"
              f"  password: {stored_a}\n")
    lock_b = ("settings_lock:\n  enabled: true\n  keys: ['yolo']\n"
              f"  password: {stored_b}\n")
    home = _root(tmp_path, lock_a)

    spec_a = sl.lock_state(home).spec
    assert sl.verify_password("operator-A", spec_a.get("password")) is True
    sl.begin_unlock(home, seconds=600, spec=spec_a)
    assert sl.is_unlocked(home) is True

    (home / "config.yaml").write_text(lock_b, encoding="utf-8")
    assert sl.is_unlocked(home) is False
    with pytest.raises(sl.SettingsLockError):
        sl.check_write({"yolo": False}, {"yolo": True}, home)

    # Restoring the exact lock the window was opened against keeps that window live.
    (home / "config.yaml").write_text(lock_a, encoding="utf-8")
    assert sl.is_unlocked(home) is True


def test_changing_only_the_password_also_lapses_the_window(tmp_path):
    """Same keys, re-hashed password: a new generation the old authority never proved."""
    home = _root(tmp_path, "")
    (home / "config.yaml").write_text(
        "settings_lock:\n  enabled: true\n  keys: ['approvals.mode']\n"
        f"  password: {sl.hash_password('one')}\n", encoding="utf-8")
    sl.begin_unlock(home, seconds=600, spec=sl.lock_state(home).spec)
    assert sl.is_unlocked(home) is True

    (home / "config.yaml").write_text(
        "settings_lock:\n  enabled: true\n  keys: ['approvals.mode']\n"
        f"  password: {sl.hash_password('two')}\n", encoding="utf-8")

    assert sl.is_unlocked(home) is False


def test_a_receipt_without_a_generation_is_never_honoured(tmp_path):
    """A window file that names no lock cannot be shown to belong to this one."""
    home = _root(tmp_path, LOCKED)
    sl.unlock_path(home).write_text(
        '{"expires_at": %d}' % int(time.time() + 600), encoding="utf-8")

    assert sl.is_unlocked(home) is False
    with pytest.raises(sl.SettingsLockError):
        sl.check_write({"approvals": {"mode": "manual"}}, {"approvals": {"mode": "off"}}, home)


def test_the_receipt_never_carries_the_stored_hash(tmp_path):
    home = _root(tmp_path, "")
    stored = sl.hash_password("operator")
    (home / "config.yaml").write_text(
        "settings_lock:\n  enabled: true\n  keys: ['approvals.mode']\n"
        f"  password: {stored}\n", encoding="utf-8")
    sl.begin_unlock(home, seconds=60, spec=sl.lock_state(home).spec)

    body = sl.unlock_path(home).read_text(encoding="utf-8")
    assert stored not in body
    assert stored.split("$")[-1] not in body


def test_the_window_file_is_not_world_readable(tmp_path):
    home = _root(tmp_path, LOCKED)
    sl.begin_unlock(home, seconds=30)

    assert sl.unlock_path(home).stat().st_mode & 0o077 == 0


# ── through the real save_config ─────────────────────────────────────────────


def test_save_config_refuses_a_locked_change_and_leaves_the_file_alone(tmp_path, monkeypatch):
    home = _root(tmp_path, "approvals:\n  mode: manual\n" + LOCKED)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.config import read_raw_config, save_config

    before = read_raw_config()
    with pytest.raises(sl.SettingsLockError):
        save_config({**before, "approvals": {"mode": "off"}})

    assert read_raw_config()["approvals"]["mode"] == "manual"


def test_save_config_allows_an_unrelated_change_while_locked(tmp_path, monkeypatch):
    home = _root(tmp_path, "approvals:\n  mode: manual\n" + LOCKED)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.config import read_raw_config, save_config

    save_config({**read_raw_config(), "display": {"theme": "dark"}})

    raw = read_raw_config()
    assert raw["display"]["theme"] == "dark"
    assert raw["approvals"]["mode"] == "manual"
