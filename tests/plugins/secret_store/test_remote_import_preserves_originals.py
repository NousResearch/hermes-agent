"""The one-time import never destroys what the user wrote, and never stages a name the plane refuses.

Before anything is staged the import keeps the exact bytes it read of ``.env`` and ``auth.json`` in
``.env.pre-import`` / ``auth.pre-import.json`` (0600, beside them). The first original wins: an
existing backup is never overwritten, and cleanup never removes one. So a value the import could not
carry whole (a satellite's quoted multiline PEM, a U+2028 / NEL inside a value: the satellite scope
serves only the first line, and that is what the plane stores) stays recoverable byte for byte.

Every name to be sent is checked against the plane's name rules (contract §2) first: an invalid one
(a UTF-16 file read as text, a quoted key, a name with a space, an upper-case auth provider id) fails
the boot closed before anything is staged, backed up, refreshed or sent, naming it (never a value).
A leftover staging file holding an invalid name fails closed the same way, naming the file.
"""
from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

import hermes_secret_routing as routing
from plugins.secret_store.remote import importer
from tests.plugins.secret_store.stub_secrets_plane import CLIENT_ID, CLIENT_SECRET
from tests.plugins.secret_store.test_remote_import_refusal import _files, _fresh, _write
from tests.plugins.secret_store.test_remote_load_at_start import _boot

pytest_plugins = ["tests.plugins.secret_store.test_remote_secret_store"]

CID = "GATEWAY_RELAY_IDP_CLIENT_ID"
SECRET = "GATEWAY_RELAY_IDP_CLIENT_SECRET"
PEM = "-----BEGIN RV30 KEY-----\nMIIrv30secretbodyline\n-----END RV30 KEY-----"
SECRETS = ("rv30-secretbody", "MIIrv30secretbodyline", "tail-secret", "rv30-utf16-secret", "rv30-quoted-key-secret",
           "rv30-space-key-secret", "codex-local", "stale-secret", "rv-plain")


@pytest.fixture(autouse=True)
def _plane_token(monkeypatch):
    from plugins.secret_store.remote import client
    from tests.plugins.secret_store.stub_secrets_plane import TOKEN
    monkeypatch.setattr(client, "bearer", lambda home: TOKEN)


def _start(monkeypatch):
    _fresh(monkeypatch, {CID: CLIENT_ID, SECRET: CLIENT_SECRET})


def _sat(plane):
    home = plane.home.parent / "sat"
    home.mkdir(exist_ok=True)
    return home


def _boot_sat(home):
    from agent.secret_scope import build_profile_secret_scope
    return build_profile_secret_scope(home)


def _stored(plane, profile):
    return {name: value for name, (value, _v) in plane.profile(profile)["env"].items()}


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _backups(home):
    return Path(home) / ".env.pre-import", Path(home) / "auth.pre-import.json"


def _no_value_shown(err):
    shown = str(err.value.code)
    for value in SECRETS:
        assert value not in shown, shown
    assert "no fallback to the local store" in shown
    return shown


# --- P1: bytes the import cannot carry whole stay recoverable -------------------------------------

LOSSY = {
    "pem": (f'RV_PEM="{PEM}"\nRV_A=plain\n'.encode(), "RV_PEM", PEM.encode()),
    "u2028": ("RV_API_KEY=rv30-head\u2028tail-secret\nRV_A=plain\n".encode(), "RV_API_KEY", b"tail-secret"),
    "nel": (b"RV_API_KEY=rv30-head\x85tail-secret\nRV_A=plain\n", "RV_API_KEY", b"tail-secret"),
}


@pytest.mark.parametrize("case", LOSSY)
def test_a_satellite_value_imported_in_part_is_kept_whole_in_the_pre_import_backup(plane, monkeypatch, case):
    raw, name, body = LOSSY[case]
    _start(monkeypatch)
    _boot(plane.home)
    home = _sat(plane)
    (home / ".env").write_bytes(raw)
    (home / ".env").chmod(0o644)
    auth_raw = json.dumps({"providers": {"openai-codex": {"t": "codex-local"}}}).encode()
    (home / "auth.json").write_bytes(auth_raw)
    _boot_sat(home)
    stored = _stored(plane, "sat")
    assert stored["RV_A"] == "plain" and name in stored             # the import ran (base's served value)
    assert not (home / ".secrets-import.json").exists()
    assert body not in (home / ".env").read_bytes()                  # cleanup removed the binding...
    env_backup, auth_backup = _backups(home)
    assert env_backup.read_bytes() == raw                            # ...but the original is kept, exactly
    assert auth_backup.read_bytes() == auth_raw
    assert _mode(env_backup) == 0o600 and _mode(auth_backup) == 0o600
    routing.routed_provider()._states.clear()
    _boot_sat(home)                                                  # a later start keeps them
    assert env_backup.read_bytes() == raw and auth_backup.read_bytes() == auth_raw


def test_the_process_home_backup_is_the_bytes_the_import_read(plane, monkeypatch):
    """The process home's loader sanitiser runs first, as on every load; the backup is what was read."""
    _start(monkeypatch)
    _write(plane.home, f'RV_PEM="{PEM}"\nRV_A=rv-plain\n')
    raw = (plane.home / ".env").read_bytes()
    _boot(plane.home)
    assert _stored(plane, "default") == {"RV_PEM": PEM, "RV_A": "rv-plain"}
    assert _backups(plane.home)[0].read_bytes() == raw
    assert not _backups(plane.home)[1].exists()                       # no auth.json, no backup


def test_nothing_to_import_writes_no_backup(plane, monkeypatch):
    _start(monkeypatch)
    _write(plane.home, f"{CID}={CLIENT_ID}\n")
    _boot(plane.home)
    assert not any(p.exists() for p in _backups(plane.home))


def test_an_earlier_backup_is_never_overwritten_nor_removed(plane, monkeypatch):
    _start(monkeypatch)
    env_backup, auth_backup = _backups(plane.home)
    env_backup.write_bytes(b"EARLIER=original\n")
    auth_backup.write_bytes(b'{"earlier": true}')
    _write(plane.home, "RV_A=rv-plain\n")
    (plane.home / "auth.json").write_text(json.dumps({"providers": {"openai-codex": {"t": "codex-local"}}}),
                                          encoding="utf-8")
    _boot(plane.home)
    assert _stored(plane, "default") == {"RV_A": "rv-plain"}
    assert env_backup.read_bytes() == b"EARLIER=original\n"
    assert auth_backup.read_bytes() == b'{"earlier": true}'


def test_backups_survive_the_plane_already_having_the_profile(plane, monkeypatch):
    _start(monkeypatch)
    plane.set_level(plane.profile("default"), env={"RV_OTHER": "x"})
    _write(plane.home, "RV_A=rv-plain\n")
    raw = (plane.home / ".env").read_bytes()
    _boot(plane.home)
    assert (plane.home / ".env.unimported").exists()
    assert _backups(plane.home)[0].read_bytes() == raw


# --- crash at each step: the first original wins ---------------------------------------------------

def _crash_once(monkeypatch, target, attr, when):
    real = getattr(target, attr)
    state = {"left": 1}

    def wrapped(*args, **kwargs):
        if state["left"] and when(*args, **kwargs):
            state["left"] = 0
            raise OSError("simulated crash")
        return real(*args, **kwargs)
    monkeypatch.setattr(target, attr, wrapped)


_REAL = {name: getattr(importer, name) for name in ("_write_bytes", "_write_private_json", "_remove")}


def _named(path_part):
    return lambda path, *a, **k: Path(path).name == path_part


CRASHES = {
    "env_backup": (importer, "_write_bytes", _named(".env.pre-import")),
    "auth_backup": (importer, "_write_bytes", _named("auth.pre-import.json")),
    "staging": (importer, "_write_private_json", _named(".secrets-import.json")),
    "env_cleanup": (importer, "_write_bytes", _named(".env")),
    "auth_cleanup": (importer, "_write_bytes", _named("auth.json")),
    "staging_removal": (importer, "_remove", _named(".secrets-import.json")),
}


@pytest.mark.parametrize("step", CRASHES)
def test_a_crash_at_any_step_keeps_the_original_bytes(plane, monkeypatch, step):
    _start(monkeypatch)
    _write(plane.home, f'{CID}={CLIENT_ID}\nRV_PEM="{PEM}"\n')
    raw = (plane.home / ".env").read_bytes()
    auth_raw = json.dumps({"providers": {"openai-codex": {"t": "codex-local"}}}).encode()
    (plane.home / "auth.json").write_bytes(auth_raw)
    target, attr, when = CRASHES[step]
    _crash_once(monkeypatch, target, attr, when)
    with pytest.raises((OSError, routing.SecretStoreUnavailable)):
        _boot(plane.home)
    monkeypatch.setattr(target, attr, _REAL[attr])   # the next start runs the real step
    _start(monkeypatch)
    _boot(plane.home)
    assert _stored(plane, "default") == {"RV_PEM": PEM}
    assert (plane.home / ".env").read_bytes() == f"{CID}={CLIENT_ID}\n".encode()
    assert not (plane.home / ".secrets-import.json").exists()
    env_backup, auth_backup = _backups(plane.home)
    assert env_backup.read_bytes() == raw and auth_backup.read_bytes() == auth_raw
    assert _mode(env_backup) == 0o600 and _mode(auth_backup) == 0o600
    assert not [p.name for p in plane.home.iterdir() if p.name.endswith(".tmp")]


# --- P2: names the plane refuses never reach the staging file --------------------------------------

INVALID_SAT = {
    "utf16": ("RV_API_KEY=rv30-utf16-secret\nRV_A=plain\n".encode("utf-16"), None),
    "quoted_key": (b"'RV_Q'=rv30-quoted-key-secret\nRV_A=plain\n", "'RV_Q'"),
}


def _refused_before_staging(plane, err, home, before, shown_name):
    shown = _no_value_shown(err)
    assert "invalid" in shown and f"'{Path(home).name}'" in shown, shown
    if shown_name is not None:
        assert shown_name in shown, shown
    assert plane.patches() == []
    assert _files(home) == before
    assert not (Path(home) / ".secrets-import.json").exists()
    assert not any(p.exists() for p in _backups(home))
    return shown


@pytest.mark.parametrize("case", INVALID_SAT)
def test_a_satellite_name_the_plane_refuses_fails_closed_before_staging(plane, monkeypatch, case):
    raw, shown_name = INVALID_SAT[case]
    _start(monkeypatch)
    _boot(plane.home)
    home = _sat(plane)
    (home / ".env").write_bytes(raw)
    before = _files(home)
    for _attempt in range(2):
        with pytest.raises(routing.SecretStoreUnavailable) as err:
            _boot_sat(home)
        _refused_before_staging(plane, err, home, before, shown_name)
    (home / ".env").write_bytes(b"RV_Q=rv30-quoted-key-secret\nRV_A=plain\n")   # fixed: imports
    _boot_sat(home)
    assert _stored(plane, "sat") == {"RV_Q": "rv30-quoted-key-secret", "RV_A": "plain"}


def test_a_process_home_name_with_a_space_fails_closed_before_staging(plane, monkeypatch):
    _start(monkeypatch)
    _write(plane.home, "'RV BAD'=rv30-space-key-secret\nRV_A=plain\n")
    before = _files(plane.home)
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        _boot(plane.home)
    shown = _refused_before_staging(plane, err, plane.home, before, "'RV BAD'")
    assert ".env" in shown
    _start(monkeypatch)
    _write(plane.home, "RV_A=plain\n")
    _boot(plane.home)
    assert _stored(plane, "default") == {"RV_A": "plain"}


def test_an_auth_provider_id_the_plane_refuses_fails_closed_before_staging(plane, monkeypatch):
    _start(monkeypatch)
    _write(plane.home, "RV_A=plain\n")
    (plane.home / "auth.json").write_text(json.dumps({"providers": {"My Provider": {"t": "codex-local"}}}),
                                          encoding="utf-8")
    before = _files(plane.home)
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        _boot(plane.home)
    _refused_before_staging(plane, err, plane.home, before, "'My Provider'")


def test_a_control_character_in_a_name_is_escaped_in_the_message(plane, monkeypatch):
    _start(monkeypatch)
    _boot(plane.home)
    home = _sat(plane)
    (home / ".env").write_bytes(b"RV\x1b[2JX=rv-plain\n")
    with pytest.raises(routing.SecretStoreUnavailable) as err:
        _boot_sat(home)
    shown = _no_value_shown(err)
    assert "\x1b" not in shown and "\\x1b" in shown


@pytest.mark.parametrize("staged", [{"env": {"RV BAD": "stale-secret"}, "auth": {}},
                                    {"env": {}, "auth": {"Bad Id": {"t": "stale-secret"}}}])
def test_a_leftover_staging_file_with_an_invalid_name_fails_closed_naming_it(plane, monkeypatch, staged):
    """The staging an earlier version left: every boot stops, naming the file and the way out,
    until the user deletes it (after checking the backups); nothing is merged, sent or rewritten."""
    _start(monkeypatch)
    _write(plane.home, "RV_A=plain\n")
    staging = plane.home / ".secrets-import.json"
    staging.write_text(json.dumps(staged), encoding="utf-8")
    before = _files(plane.home)
    for _attempt in range(2):
        _start(monkeypatch)
        with pytest.raises(routing.SecretStoreUnavailable) as err:
            _boot(plane.home)
        shown = _no_value_shown(err)
        assert str(staging) in shown and ".env.pre-import" in shown and "delete" in shown, shown
        assert plane.patches() == [] and _files(plane.home) == before
    staging.unlink()
    _start(monkeypatch)
    _boot(plane.home)
    assert _stored(plane, "default") == {"RV_A": "plain"}


def test_a_valid_leftover_staging_still_imports(plane, monkeypatch):
    """Control: an unfinished staging of valid names is merged and sent, as before."""
    _start(monkeypatch)
    _write(plane.home, "RV_A=plain\n")
    (plane.home / ".secrets-import.json").write_text(json.dumps({"env": {"RV_STAGED": "s"}, "auth": {}}),
                                                     encoding="utf-8")
    _boot(plane.home)
    assert _stored(plane, "default") == {"RV_A": "plain", "RV_STAGED": "s"}


def test_doctor_names_the_backups_without_their_contents(plane, monkeypatch):
    _start(monkeypatch)
    _write(plane.home, "RV_A=rv-plain\n")
    _boot(plane.home)
    described = routing.routed_provider().describe(plane.home)
    assert ".env.pre-import" in described and "rv-plain" not in described
    assert routing.routed_provider().doctor_warnings(plane.home) == ()
