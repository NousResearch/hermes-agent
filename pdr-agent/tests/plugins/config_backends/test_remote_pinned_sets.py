"""A keyed ``config set`` stores its value at the profile level even when it equals the inherited
one (the file backend writes every keyed set into config.yaml), and a rotated IdP client secret
takes effect on the next request instead of after the cached token expires."""
from __future__ import annotations

from pathlib import Path

import pytest

from plugins.config_backends import remote as remote_pkg
from plugins.config_backends.remote import backend as backend_mod
from plugins.config_backends.remote import credentials as cred_mod

from .stub_plane import StubPlane, remote_env


@pytest.fixture
def plane(tmp_path, monkeypatch):
    home = Path(tmp_path) / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with StubPlane() as p:
        for k, v in remote_env(p).items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(backend_mod, "BOOT_RETRY_DELAYS", (0.0, 0.0))
        remote_pkg._reset_for_tests()
        cred_mod._IDP_CACHE.clear()
        p.home = home
        yield p
        remote_pkg._reset_for_tests()
        cred_mod._IDP_CACHE.clear()


def test_set_equal_to_inherited_value_is_stored_and_survives_an_upper_change(plane):
    from hermes_cli.config import set_config_value
    from hermes_cli.config_backend import read_config_doc

    plane.upper = {"display": {"skin": "ares"}}

    set_config_value("display.skin", "ares")

    assert plane.profile("default")["values"] == {"display": {"skin": "ares"}}
    plane.upper = {"display": {"skin": "slate"}}
    remote_pkg._reset_for_tests()
    assert read_config_doc(plane.home / "config.yaml")["display"]["skin"] == "ares"


def test_rotated_idp_client_secret_fetches_a_new_token(plane, monkeypatch):
    cred_mod.plane_token(plane.home)
    first = plane.token_requests
    cred_mod.plane_token(plane.home)
    assert plane.token_requests == first  # same secret: served from the cache

    monkeypatch.setenv("GATEWAY_RELAY_IDP_CLIENT_SECRET", "rotated-secret")
    with pytest.raises(cred_mod.PlaneCredentialError):
        cred_mod.plane_token(plane.home)  # the stub only accepts the original secret
    assert plane.token_requests == first + 1
