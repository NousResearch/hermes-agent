"""The one-time import stores what the local store SERVES for the profile, by its own readers.

Process home: what its dotenv load puts in ``os.environ`` (the launch environment, the loader's
order and credential sanitising, the sanitiser's rewrite of the file). Any other profile: its secret
scope's ``.env`` layer (lookups against process-global names only, no credential sanitising, the
file read as it is). Every expected value below was executed on the original base (06189b9508, local
store, fresh process, same launch environment).
"""
from __future__ import annotations

import os

import pytest

import hermes_secret_routing as routing
from tests.plugins.secret_store.stub_secrets_plane import CLIENT_ID, CLIENT_SECRET
from tests.plugins.secret_store.test_remote_import_refusal import _fresh
from tests.plugins.secret_store.test_remote_load_at_start import _boot, _readers

pytest_plugins = ["tests.plugins.secret_store.test_remote_secret_store"]

CID = "GATEWAY_RELAY_IDP_CLIENT_ID"
SECRET = "GATEWAY_RELAY_IDP_CLIENT_SECRET"
SCOPE = "GATEWAY_RELAY_IDP_SCOPE"


@pytest.fixture(autouse=True)
def _plane_token(monkeypatch):
    """The plane credential comes from the stub directly: these files reassign the IdP names."""
    from plugins.secret_store.remote import client
    from tests.plugins.secret_store.stub_secrets_plane import TOKEN
    monkeypatch.setattr(client, "bearer", lambda home: TOKEN)


def _stored(plane, profile):
    return {name: value for name, (value, _version) in plane.profile(profile)["env"].items()}


def _import(plane, monkeypatch, raw: bytes, *, satellite: bool):
    """A first boot of the process home (and, for *satellite*, of a sibling profile) on *raw*."""
    _fresh(monkeypatch, {CID: CLIENT_ID, SECRET: CLIENT_SECRET})
    monkeypatch.delenv(SCOPE, raising=False)
    home = plane.home.parent / "sat" if satellite else plane.home
    home.mkdir(exist_ok=True)
    (home / ".env").write_bytes(raw)
    _boot(plane.home)
    if satellite:
        from agent.secret_scope import build_profile_secret_scope
        build_profile_secret_scope(home)
    return home


# --- an imported line referencing a kept line written after it ------------------------------------

FORWARD = {
    # the loader resolves a forward reference against the launch environment, never the later line
    "client_id": (f"RV_A=${{{CID}}}\n{CID}=cid-file\n", "RV_A", CLIENT_ID),
    "scope_default": (f"RV_A=${{{SCOPE}:-none}}\n{SCOPE}=s-file\n", "RV_A", "none"),
    "client_secret": (f"RV_API_KEY=${{{SECRET}}}x\n{SECRET}=sec-file\n", "RV_API_KEY", CLIENT_SECRET + "x"),
    # control: a backward reference sees the earlier line
    "backward": (f"{CID}=cid-file\nRV_A=${{{CID}}}\n", "RV_A", "cid-file"),
}


@pytest.mark.parametrize("case", FORWARD)
def test_a_reference_to_a_later_kept_line_resolves_as_the_local_store_does(plane, monkeypatch, case):
    text, name, expected = FORWARD[case]
    home = _import(plane, monkeypatch, text.encode(), satellite=False)
    assert _stored(plane, "default")[name] == expected
    assert set(_readers(home, name).values()) == {expected}
    _fresh(monkeypatch, {CID: CLIENT_ID, SECRET: CLIENT_SECRET})
    _boot(home)  # a later start serves the stored value
    assert set(_readers(home, name).values()) == {expected}


# --- the process home's credential and file sanitising are the process home's only ----------------

@pytest.mark.parametrize(("satellite", "expected"), [(True, "k\u00e9y"), (False, "ky")])
def test_only_the_process_home_strips_non_ascii_from_a_credential(plane, monkeypatch, satellite, expected):
    home = _import(plane, monkeypatch, "RV_API_KEY=k\u00e9y\nRV_A=normal\n".encode(), satellite=satellite)
    assert _stored(plane, home.name) == {"RV_API_KEY": expected, "RV_A": "normal"}
    assert set(_readers(home, "RV_API_KEY", ambient=not satellite).values()) == {expected}


@pytest.mark.parametrize(("satellite", "expected"), [(True, "\u00ff\u00fe-bytes"), (False, "\ufffd\ufffd-bytes")])
def test_only_the_process_home_file_is_rewritten_by_the_loader_sanitiser(plane, monkeypatch, satellite, expected):
    """Latin-1 bytes: a satellite reads them as Latin-1, as the local store does; the process home's
    load rewrites its file first (lossy, as it always was)."""
    home = _import(plane, monkeypatch, b"RV_A=\xff\xfe-bytes  \n", satellite=satellite)
    assert _stored(plane, home.name) == {"RV_A": expected}
    assert set(_readers(home, "RV_A", ambient=not satellite).values()) == {expected}


def test_a_refused_satellite_file_is_left_byte_for_byte(plane, monkeypatch):
    """The satellite's loader never sanitises its file, so a refusal leaves even NULs and Latin-1."""
    raw = b"TERMINAL_RV_K=${RV_A}\nRV_A=\x00\xff\xfe-bytes  \r\n"
    _fresh(monkeypatch, {CID: CLIENT_ID, SECRET: CLIENT_SECRET})
    _boot(plane.home)
    home = plane.home.parent / "sat"
    home.mkdir()
    (home / ".env").write_bytes(raw)
    from agent.secret_scope import build_profile_secret_scope
    with pytest.raises(routing.SecretStoreUnavailable):
        build_profile_secret_scope(home)
    assert (home / ".env").read_bytes() == raw
    assert plane.patches() == []


def test_a_satellite_stores_its_own_layer_under_the_managed_env(plane, monkeypatch, tmp_path):
    """The managed ``.env`` still wins over the stored value, as it won over the file; the store
    holds the profile's own value, never the administrator's."""
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".env").write_text("RV_A=managed\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    home = _import(plane, monkeypatch, b"RV_A=own\nRV_B=${RV_A}\n", satellite=True)
    assert _stored(plane, "sat") == {"RV_A": "own", "RV_B": "own"}
    from agent.secret_scope import build_profile_secret_scope
    scope = build_profile_secret_scope(home)
    assert (scope["RV_A"], scope["RV_B"]) == ("managed", "own")


def test_a_kept_store_credential_is_served_as_the_dotenv_load_leaves_it(plane, monkeypatch):
    """The store's early publication of its own credential is in the dotenv load's form, so it never
    re-publishes a value the load sanitised (non-ASCII stripped, as with the local store)."""
    home = _import(plane, monkeypatch, f"{SECRET}=s\u00e9c\nRV_A=a\n".encode(), satellite=False)
    assert os.environ[SECRET] == "sc"
    _boot(home)
    assert os.environ[SECRET] == "sc"
