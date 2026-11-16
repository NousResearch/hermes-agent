"""Kept-local ``.env`` lines are published once, by the ordinary dotenv loader, exactly as the local
store publishes them; the remote store's boot adds only its stored values.

The remote store keeps its own credential (the IdP names) and every process-global name in the
profile's ``.env``. A kept line may reference another kept line before or after it. The loader
resolves such a reference in file order, so a forward reference sees the launch environment, not the
later line. A second publication of the store's bootstrap names after the profile layer would
resolve that line again against the environment the layer just wrote, and serve the later line's
value instead (rv27 P1). Expected values were produced by executing the base loader (06189b9508,
local store) on each text in a fresh process with the same launch environment, at the first load
and at a reload in the same process.
"""
from __future__ import annotations

import pytest

import hermes_secret_routing as routing
from tests.plugins.secret_store.stub_secrets_plane import CLIENT_ID, CLIENT_SECRET
from tests.plugins.secret_store.test_remote_load_at_start import _boot, _fresh_start, _readers

pytest_plugins = ["tests.plugins.secret_store.test_remote_secret_store"]

NAME = "GATEWAY_RELAY_IDP_CLIENT_SECRET"
CID = "GATEWAY_RELAY_IDP_CLIENT_ID"
SHELL = {NAME: CLIENT_SECRET, CID: CLIENT_ID}   # the launch environment of every run below
_CLEARED = ("TERMINAL_RV_K", "RV_MARK", "RV_LATER", "RV_DIR", "RV_X")

CASES = {
    "forward_kept": (f"{NAME}=${{TERMINAL_RV_K}}\nTERMINAL_RV_K=v\nRV_MARK=m\n", {NAME: "", "TERMINAL_RV_K": "v"}),
    "forward_kept_no_import": (f"{NAME}=${{TERMINAL_RV_K}}\nTERMINAL_RV_K=v\n", {NAME: "", "TERMINAL_RV_K": "v"}),
    "backward_kept": (f"TERMINAL_RV_K=v\n{NAME}=${{TERMINAL_RV_K}}\nRV_MARK=m\n", {NAME: "v", "TERMINAL_RV_K": "v"}),
    "forward_idp_only": (f"{NAME}=${{{CID}}}x\n{CID}=cid\nRV_MARK=m\n", {NAME: f"{CLIENT_ID}x", CID: "cid"}),
    "backward_idp_only": (f"{CID}=cid\n{NAME}=${{{CID}}}x\nRV_MARK=m\n", {NAME: "cidx", CID: "cid"}),
    "forward_kept_default": (f"{NAME}=${{TERMINAL_RV_K:-dflt}}\nTERMINAL_RV_K=v\nRV_MARK=m\n",
                             {NAME: "dflt", "TERMINAL_RV_K": "v"}),
    "forward_imported": (f"{NAME}=${{RV_LATER}}\nRV_LATER=v\nRV_MARK=m\n", {NAME: ""}),
    "chain_forward": (f"{NAME}=${{TERMINAL_RV_K}}\nRV_DIR=last\nTERMINAL_RV_K=${{RV_X:-dflt}}\nRV_X=${{RV_DIR}}\nRV_MARK=m\n",
                      {NAME: "", "TERMINAL_RV_K": "dflt"}),
}


@pytest.fixture(autouse=True)
def _plane_token(monkeypatch):
    """The plane credential comes from the stub directly: these files reassign the IdP names."""
    from plugins.secret_store.remote import client
    from tests.plugins.secret_store.stub_secrets_plane import TOKEN
    monkeypatch.setattr(client, "bearer", lambda home: TOKEN)


def _fresh(monkeypatch):
    """A new process with only the launch environment (no name an earlier boot published)."""
    _fresh_start(monkeypatch, SHELL)
    for name in _CLEARED:
        monkeypatch.delenv(name, raising=False)


def _served(home, expected):
    return {name: set(_readers(home, name).values()) for name in expected}


@pytest.mark.parametrize("store", ["local", "remote"])
@pytest.mark.parametrize("case", CASES)
def test_kept_lines_are_served_as_the_base_loader_serves_them(plane, monkeypatch, store, case):
    """Every reader (ambient included) serves the base value for each kept name: at the import's
    boot, at a reload in the same process, and at a later fresh start from what the import left."""
    text, expected = CASES[case]
    want = {name: {value} for name, value in expected.items()}
    _fresh(monkeypatch)
    if store == "local":
        monkeypatch.setenv("HERMES_SECRET_STORE", "local")
        routing.reset_for_tests()
    (plane.home / ".env").write_text(text, encoding="utf-8", newline="")
    _boot(plane.home)
    assert _served(plane.home, expected) == want, "first boot"
    _boot(plane.home)
    assert _served(plane.home, expected) == want, "reload"
    if store == "remote":
        assert len(plane.patches()) == (0 if case == "forward_kept_no_import" else 1)
    _fresh(monkeypatch)
    _boot(plane.home)
    assert _served(plane.home, expected) == want, "later start"

