"""The operator settings lock.

Named config paths that no writer may change until an unlock window is opened. The gate lives in
``save_config`` — the one function every config writer in the tree funnels through — so these
tests exercise both the predicate layer and a real ``save_config`` round trip.
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
    (False, False), (0, False), ("no", False), ("maybe", False), (None, False), ([], False),
])
def test_enabled_accepts_the_yaml_spellings(value, enabled):
    assert sl.is_enabled({"enabled": value}) is enabled


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
    sl.unlock_path(home).write_text('{"expires_at": %d}' % int(time.time() - 5), encoding="utf-8")

    assert sl.is_unlocked(home) is False
    with pytest.raises(sl.SettingsLockError):
        sl.check_write({"approvals": {"mode": "manual"}}, {"approvals": {"mode": "off"}}, home)


@pytest.mark.parametrize("body", ["", "not json", '{"expires_at": "soon"}', "{}"])
def test_an_unreadable_window_file_is_not_an_unlock(tmp_path, body):
    home = _root(tmp_path, LOCKED)
    sl.unlock_path(home).write_text(body, encoding="utf-8")

    assert sl.is_unlocked(home) is False


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
