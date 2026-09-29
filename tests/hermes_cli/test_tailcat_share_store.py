"""Connection codes and paired devices for sharing a backend over tailcat."""

from __future__ import annotations

import json

import pytest

from hermes_cli import tailcat_share_store as store

ADDRESS = "tc" + "A" * 150


@pytest.fixture
def home(tmp_path):
    return tmp_path


def test_code_round_trips_and_redeems_exactly_once(home):
    code = store.mint_code(ADDRESS, 41234, home=home, now=1000.0)
    parsed = store.parse_code(code.render())

    assert parsed == code
    assert store.consume_code(parsed.secret, home=home, now=1001.0) is True
    assert store.consume_code(parsed.secret, home=home, now=1002.0) is False


def test_expired_code_is_refused_and_pruned(home):
    code = store.mint_code(ADDRESS, 41234, home=home, now=1000.0)
    later = 1000.0 + store.CODE_TTL_S + 1

    assert store.consume_code(code.secret, home=home, now=later) is False
    assert json.loads((store.share_dir(home) / "codes.json").read_text())["codes"] == []


@pytest.mark.parametrize("text", [
    "", "hermes-tailcat:", f"hermes-tailcat:{ADDRESS}:0:s", f"hermes-tailcat:{ADDRESS}:x:s",
    f"hermes-tailcat:nottc:80:s", f"other:{ADDRESS}:80:s", f"hermes-tailcat:{ADDRESS}:80",
])
def test_malformed_codes_do_not_parse(text):
    assert store.parse_code(text) is None


def test_secrets_are_stored_only_as_digests(home):
    code = store.mint_code(ADDRESS, 41234, home=home)
    device, token = store.pair_device("laptop", home=home)

    on_disk = "".join(p.read_text() for p in store.share_dir(home).glob("*.json"))
    assert code.secret not in on_disk
    assert token not in on_disk
    assert store.device_for_token(token, home=home)["id"] == device["id"]
    assert "token_sha256" not in device


def test_revoked_device_token_stops_authenticating(home):
    device, token = store.pair_device("laptop", home=home)
    _, other_token = store.pair_device("phone", home=home)

    assert store.revoke_device(device["id"], home=home) is True
    assert store.device_for_token(token, home=home) is None
    assert store.device_for_token(other_token, home=home) is not None
    assert store.revoke_device(device["id"], home=home) is False


def test_reset_forgets_identity_devices_and_codes(home):
    base = store.share_dir(home)
    base.mkdir(parents=True)
    store.server_key_path(home).write_text("{}")
    store.save_share_state({"address": ADDRESS, "port": 41234}, home=home)
    code = store.mint_code(ADDRESS, 41234, home=home)
    _, token = store.pair_device("laptop", home=home)

    store.forget_identity(home)

    assert not store.server_key_path(home).exists()
    assert store.device_for_token(token, home=home) is None
    assert store.consume_code(code.secret, home=home) is False
    # The listener port survives so the next address still pairs with the same port.
    assert store.load_share_state(home) == {"port": 41234}
