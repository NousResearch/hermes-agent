"""Every version of ``.env`` / ``auth.json`` the import replaces stays recoverable (contract §6).

The backups are append-only: before any rewrite of either file (the cleanup after a PATCH, the
plane-wins path) the import appends the exact bytes it replaces as one timestamped entry, so a
later edit (a satellite PEM or U+2028 value the satellite tokenizer carries only in part, an edit
the cleanup's compare-and-swap left whole) is in the backup before its binding is removed. A crash
mid-append leaves a torn last entry that hides no earlier one. And ``.env`` is read once per
import: what the PATCH carries is what the checks read.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import hermes_secret_routing as routing
from plugins.secret_store.remote import importer, pre_import
from tests.plugins.secret_store.stub_secrets_plane import CLIENT_ID, CLIENT_SECRET, TOKEN, remote_env
from tests.plugins.secret_store.test_remote_import_refusal import _fresh, _write
from tests.plugins.secret_store.test_remote_load_at_start import _boot

pytest_plugins = ["tests.plugins.secret_store.test_remote_secret_store"]

CID = "GATEWAY_RELAY_IDP_CLIENT_ID"
SECRET = "GATEWAY_RELAY_IDP_CLIENT_SECRET"
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _plane_token(monkeypatch):
    from plugins.secret_store.remote import client
    monkeypatch.setattr(client, "bearer", lambda home: TOKEN)


def _start(monkeypatch):
    _fresh(monkeypatch, {CID: CLIENT_ID, SECRET: CLIENT_SECRET})


def _sat(plane):
    home = plane.home.parent / "sat"
    home.mkdir(exist_ok=True)
    return home


def _boot_sat(home):
    from agent.secret_scope import build_profile_secret_scope
    routing.routed_provider()._states.clear()
    return build_profile_secret_scope(home)


def _stored(plane, profile):
    return {name: value for name, (value, _v) in plane.profile(profile)["env"].items()}


def _versions(path):
    return [(e.reason, e.data) for e in pre_import.read_entries(path) if e.whole]


def _append_to(path, data):
    with open(path, "ab") as f:
        f.write(data)


LATER = {
    "pem": b'RV_PEM2="L1-head\nRV31TAILSECRET\nL3"\n',
    "u2028": "RV_K2=rv31-head\u2028RV31TAILSECRET\n".encode(),
}


@pytest.mark.parametrize("case", LATER)
def test_a_later_satellite_edit_is_backed_up_before_its_binding_is_removed(plane, monkeypatch, case):
    """First import, then a value only partly served (multiline / U+2028), then another edit: the
    plane already has the profile, the tail reaches neither the plane nor ``.env.unimported``; each
    version the import replaced is an entry of the backup, byte for byte."""
    _start(monkeypatch)
    _boot(plane.home)
    home = _sat(plane)
    v1 = b"RV_A=plain\n"
    (home / ".env").write_bytes(v1)
    _boot_sat(home)
    assert _stored(plane, "sat") == {"RV_A": "plain"}
    _append_to(home / ".env", LATER[case])
    v2 = (home / ".env").read_bytes()
    _boot_sat(home)                                         # plane wins: set aside, binding removed
    assert b"RV31TAILSECRET" not in (home / ".env").read_bytes()
    assert b"RV31TAILSECRET" not in (home / ".env.unimported").read_bytes()
    _append_to(home / ".env", b"RV_C=third\n")
    v3 = (home / ".env").read_bytes()
    _boot_sat(home)
    backup = home / ".env.pre-import"
    assert _versions(backup) == [("staged", v1), ("cleanup", v1), ("staged", v2), ("plane-wins", v2),
                                 ("staged", v3), ("plane-wins", v3)]
    assert all(e.at.endswith("Z") for e in pre_import.read_entries(backup))
    assert (backup.stat().st_mode & 0o777) == 0o600


def test_the_409_path_backs_up_auth_json_too(plane, monkeypatch):
    """Another writer stores the profile first (409 on the import's PATCH): the plane wins and
    each file's replaced bytes are appended."""
    _start(monkeypatch)
    _write(plane.home, f'{CID}={CLIENT_ID}\nRV_PEM="a\nb\nc"\n')
    raw = (plane.home / ".env").read_bytes()
    auth_raw = json.dumps({"providers": {"openai-codex": {"t": "codex-local"}}}).encode()
    (plane.home / "auth.json").write_bytes(auth_raw)
    plane.conflict_next_patch = 1
    _boot(plane.home)
    assert (plane.home / ".env").read_bytes() == f"{CID}={CLIENT_ID}\n".encode()
    assert _versions(plane.home / ".env.pre-import") == [("staged", raw), ("plane-wins", raw)]
    assert _versions(plane.home / "auth.pre-import.json") == [("staged", auth_raw), ("plane-wins", auth_raw)]


def test_an_edit_the_cleanup_left_whole_is_backed_up_on_the_next_boot(plane, monkeypatch):
    """The cleanup's compare-and-swap leaves a file edited during the import whole; the next
    boot appends what it then holds before removing the bindings."""
    _start(monkeypatch)
    _boot(plane.home)
    home = _sat(plane)
    (home / ".env").write_bytes(b"RV_A=plain\n")
    real = importer._keep_bootstrap_only
    edit = b'RV_PEM3="L1-head\nRV31CASTAIL\nL3"\n'

    def edit_then_cleanup(h, provider, sources, reason):
        _append_to(Path(h) / ".env", edit)
        return real(h, provider, sources, reason)
    monkeypatch.setattr(importer, "_keep_bootstrap_only", edit_then_cleanup)
    _boot_sat(home)
    monkeypatch.setattr(importer, "_keep_bootstrap_only", real)
    edited = (home / ".env").read_bytes()
    assert edited == b"RV_A=plain\n" + edit                         # left whole, as designed
    _boot_sat(home)
    assert b"RV31CASTAIL" not in (home / ".env").read_bytes()
    assert ("plane-wins", edited) in _versions(home / ".env.pre-import")


def test_a_torn_last_entry_hides_no_earlier_entry(tmp_path):
    path = tmp_path / ".env.pre-import"
    two = b'TWO="a\n#hermes-pre-import v1 at=x reason=staged bytes=1 sha256=' + b"0" * 64 + b'\n"\n'
    pre_import.append(path, b"ONE=1\n", "staged")
    pre_import.append(path, two, "cleanup")                          # a header-like line inside a value
    whole = path.read_bytes()
    pre_import.append(path, b"THREE=3\n" * 50, "staged")
    torn = path.read_bytes()[:len(whole) + 200]                      # a crash mid-append
    path.write_bytes(torn)
    found = pre_import.read_entries(path)
    assert [e.data for e in found if e.whole] == [b"ONE=1\n", two]
    assert found[-1].reason == "torn" and not found[-1].whole
    pre_import.append(path, b"FOUR=4\n", "cleanup")                  # the next append starts after it
    found = pre_import.read_entries(path)
    assert [(e.reason, e.whole) for e in found] == [("staged", True), ("cleanup", True), ("torn", False),
                                                     ("cleanup", True)]
    assert found[-1].data == b"FOUR=4\n" and path.read_bytes().startswith(torn)


def test_an_unchanged_staging_snapshot_is_not_repeated(tmp_path):
    path = tmp_path / "auth.pre-import.json"
    pre_import.append(path, b"{}", "staged", unless_last=True)
    pre_import.append(path, b"{}", "staged", unless_last=True)
    pre_import.append(path, b"{}", "cleanup")
    assert _versions(path) == [("staged", b"{}"), ("cleanup", b"{}")]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/fd")
def test_a_real_crash_mid_append_then_restart_keeps_every_version(plane, monkeypatch, tmp_path):
    """A process killed half-way through appending a later version: the restart finishes the
    import, and the backup still holds every version whole (the torn one beside them)."""
    _start(monkeypatch)
    _write(plane.home, "RV_A=one\n")
    v1 = (plane.home / ".env").read_bytes()
    _boot(plane.home)
    _append_to(plane.home / ".env", b'RV_PEM="a\nb-RV31TORN\nc"\n')
    v2 = (plane.home / ".env").read_bytes()
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent('''
        import os, sys
        from pathlib import Path
        sys.path.insert(0, sys.argv[1])
        from plugins.secret_store.remote import client
        client.bearer = lambda h: sys.argv[3]
        real = os.write
        def torn(fd, data):
            if os.readlink(f"/proc/self/fd/{fd}").endswith(".env.pre-import"):
                real(fd, bytes(data)[: len(data) // 2])
                os._exit(9)
            return real(fd, data)
        os.write = torn
        from hermes_cli.env_loader import load_hermes_dotenv
        load_hermes_dotenv(hermes_home=Path(sys.argv[2]), project_env=None, load_external_secrets=False)
    '''), encoding="utf-8")
    env = {**os.environ, **remote_env(plane)}
    done = subprocess.run([sys.executable, "-I", str(script), str(ROOT), str(plane.home), TOKEN],
                          env=env, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 9, done.stderr[-2000:]
    assert (plane.home / ".env").read_bytes() == v2                   # nothing replaced yet
    _start(monkeypatch)
    _boot(plane.home)
    found = pre_import.read_entries(plane.home / ".env.pre-import")
    assert [(e.reason, e.data) for e in found if e.whole] == [("staged", v1), ("cleanup", v1), ("staged", v2),
                                                              ("plane-wins", v2)]
    assert [e.reason for e in found if not e.whole] == ["torn"]
    assert b"RV31TORN" not in (plane.home / ".env").read_bytes()


# --- .env is read once per import (ABA) ------------------------------------------------------------

ORIG = b"RV_A=checked\nRV_B=${RV_A}-ref\n"
SNEAK = b"RV_A=sneaky\nRV_B=${RV_A}-ref\nRV_C=unchecked\n"


@pytest.mark.parametrize("satellite", [False, True])
def test_the_patch_carries_the_env_bytes_the_checks_read(plane, monkeypatch, satellite):
    """Another writer replaces ``.env`` right after the checks and restores it right after
    staging: the PATCH carries what the checks read, for a satellite too."""
    _start(monkeypatch)
    for name in ("RV_A", "RV_B", "RV_C"):
        monkeypatch.delenv(name, raising=False)
    home = _sat(plane) if satellite else plane.home
    if satellite:
        _boot(plane.home)
    env = home / ".env"
    env.write_bytes(ORIG)
    real_check, real_write = importer.non_literal_kept_names, importer._write_private_json
    edits = []

    def check_then_edit(*args):
        if not edits:
            env.write_bytes(SNEAK)
            edits.append("edit")
        return real_check(*args)

    def write_then_undo(path, data):
        real_write(path, data)
        if Path(path).name == importer.STAGING and Path(path).parent == home:
            env.write_bytes(ORIG)
            edits.append("undo")
    monkeypatch.setattr(importer, "non_literal_kept_names", check_then_edit)
    monkeypatch.setattr(importer, "_write_private_json", write_then_undo)
    if satellite:
        _boot_sat(home)
    else:
        _boot(plane.home)
    assert edits == ["edit", "undo"]
    assert _stored(plane, "sat" if satellite else "default") == {"RV_A": "checked", "RV_B": "checked-ref"}
