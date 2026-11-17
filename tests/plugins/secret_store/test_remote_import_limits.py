"""The import checks the plane's limits before it stages anything (contract §6).

gateway-gateway ``src/secretsPlane`` refuses an env value over 64 KiB (UTF-8), an auth record whose
JSON is over 64 KiB, a profile left holding over 1000 entries, and a body over 1 MiB. Each is
checked before anything is backed up, staged, refreshed or sent: the boot fails closed naming the
entry (never its value), and once the user fixes the file it imports. A leftover staging file
holding such an entry (an earlier version staged it) fails closed naming that file, even after the
entry is gone from ``.env``.
"""
from __future__ import annotations

import json
import math

import pytest

import hermes_secret_routing as routing
from plugins.secret_store.remote import limits, plane_json
from tests.plugins.secret_store import stub_secrets_plane as stub
from tests.plugins.secret_store.stub_secrets_plane import CLIENT_ID, CLIENT_SECRET, TOKEN
from tests.plugins.secret_store.test_remote_import_refusal import _files, _fresh, _write
from tests.plugins.secret_store.test_remote_load_at_start import _boot

pytest_plugins = ["tests.plugins.secret_store.test_remote_secret_store"]

CID = "GATEWAY_RELAY_IDP_CLIENT_ID"
SECRET = "GATEWAY_RELAY_IDP_CLIENT_SECRET"
KIB64 = 64 * 1024


@pytest.fixture(autouse=True)
def _plane_token(monkeypatch):
    from plugins.secret_store.remote import client
    monkeypatch.setattr(client, "bearer", lambda home: TOKEN)


def _start(monkeypatch):
    _fresh(monkeypatch, {CID: CLIENT_ID, SECRET: CLIENT_SECRET})


def _stored(plane):
    return {name: value for name, (value, _v) in plane.profile("default")["env"].items()}


def _refused(plane, monkeypatch, *, shows, hides=()):
    """Boot twice: each fails closed before any request and writes nothing."""
    before = _files(plane.home)
    for _attempt in range(2):
        _start(monkeypatch)
        with pytest.raises(routing.SecretStoreUnavailable) as err:
            _boot(plane.home)
        shown = str(err.value.code)
        assert all(s in shown for s in shows), shown
        assert not any(h in shown for h in hides), shown
        assert "no fallback to the local store" in shown
        assert plane.requests == []                                    # not even a GET
        assert _files(plane.home) == before
    return shown


def test_the_limits_are_the_planes():
    """gateway-gateway ``src/secretsPlane/types.ts``: MAX_ENTRY_BYTES = 64 * 1024,
    MAX_ENTRIES_PER_LEVEL = 1000, MAX_BODY_BYTES = 1024 * 1024 (the stub enforces the same)."""
    assert (limits.MAX_ENTRY_BYTES, limits.MAX_ENTRIES_PER_LEVEL, limits.MAX_BODY_BYTES) == \
        (stub.MAX_ENTRY_BYTES, stub.MAX_ENTRIES_PER_LEVEL, stub.MAX_BODY_BYTES) == (KIB64, 1000, 1024 * 1024)


@pytest.mark.parametrize("value", ["x" * (KIB64 + 1), "\u00e9" * (KIB64 // 2 + 1)], ids=["ascii", "two_byte"])
def test_an_env_value_over_64_kib_fails_closed_before_any_request(plane, monkeypatch, value):
    _write(plane.home, f"RV_BIG={value}\nRV_A=plain\n")
    shown = _refused(plane, monkeypatch, shows=["'RV_BIG'", str(KIB64)], hides=[value[:40]])
    assert len(shown) < 1000
    assert not (plane.home / ".secrets-import.json").exists() and not (plane.home / ".env.pre-import").exists()
    _write(plane.home, "RV_A=plain\n")                                 # fixed: imports
    _start(monkeypatch)
    _boot(plane.home)
    assert _stored(plane) == {"RV_A": "plain"}


def test_an_env_value_of_exactly_64_kib_imports(plane, monkeypatch):
    _start(monkeypatch)
    _write(plane.home, "RV_EDGE=" + "\u00e9" * (KIB64 // 2) + "\n")
    _boot(plane.home)
    assert len(_stored(plane)["RV_EDGE"].encode()) == KIB64


def _auth(record):
    return json.dumps({"providers": {"openai-codex": record}})


def _record_of(size):
    """An auth record whose ``JSON.stringify`` is exactly *size* bytes (number formats: below)."""
    return {"t": "y" * (size - len('{"t":"","n":1}')), "n": 1}


def test_an_auth_record_over_64_kib_fails_closed_and_one_at_the_limit_imports(plane, monkeypatch):
    _write(plane.home, "RV_A=plain\n")
    (plane.home / "auth.json").write_text(_auth(_record_of(KIB64 + 1)), encoding="utf-8")
    _refused(plane, monkeypatch, shows=["'openai-codex'", str(KIB64)], hides=["yyyyyyyyyy"])
    (plane.home / "auth.json").write_text(_auth(_record_of(KIB64)), encoding="utf-8")
    _start(monkeypatch)
    _boot(plane.home)
    stored = plane.profile("default")["auth"]["openai-codex"][0]
    assert len(plane_json.stringify(stored).encode()) == KIB64


def test_more_entries_than_a_profile_holds_fail_closed_and_the_limit_imports(plane, monkeypatch):
    _write(plane.home, "".join(f"RV_N{i}=v\n" for i in range(1001)))
    _refused(plane, monkeypatch, shows=["1001 entries", "1000"])
    _write(plane.home, "".join(f"RV_N{i}=v\n" for i in range(1000)))
    _start(monkeypatch)
    _boot(plane.home)
    assert len(_stored(plane)) == 1000


def test_a_request_over_1_mib_fails_closed(plane, monkeypatch):
    _write(plane.home, "".join(f"RV_V{i}={'z' * (KIB64 - 8)}\n" for i in range(17)))
    _refused(plane, monkeypatch, shows=["1048576", "too much at once"])
    _write(plane.home, "".join(f"RV_V{i}={'z' * (KIB64 - 8)}\n" for i in range(15)))
    _start(monkeypatch)
    _boot(plane.home)
    assert len(_stored(plane)) == 15


def test_a_leftover_staging_with_an_oversized_value_fails_closed_naming_it(plane, monkeypatch):
    """An earlier version staged a value the plane refuses; the user removed it from .env:
    the staging still holds it, so every boot stops naming the staging file."""
    staging = plane.home / ".secrets-import.json"
    staging.write_text(json.dumps({"env": {"RV_BIG": "q" * (KIB64 + 1)}, "auth": {}}), encoding="utf-8")
    _write(plane.home, "RV_A=plain\n")
    _refused(plane, monkeypatch, shows=[str(staging), "'RV_BIG'", "delete", "only copy"], hides=["qqqqqqqqqq"])
    staging.unlink()
    _start(monkeypatch)
    _boot(plane.home)
    assert _stored(plane) == {"RV_A": "plain"}


def test_a_leftover_staging_that_overflows_together_with_the_files_fails_closed_naming_it(plane, monkeypatch):
    staging = plane.home / ".secrets-import.json"
    staging.write_text(json.dumps({"env": {f"RV_S{i}": "v" for i in range(600)}, "auth": {}}), encoding="utf-8")
    _write(plane.home, "".join(f"RV_F{i}=v\n" for i in range(401)))
    _refused(plane, monkeypatch, shows=[str(staging), "1001 entries"])


def test_a_value_that_cannot_be_sent_fails_closed(plane, monkeypatch):
    (plane.home / "auth.json").write_text('{"providers": {"openai-codex": {"t": "\\ud800"}}}', encoding="utf-8")
    _write(plane.home, "RV_A=plain\n")
    _refused(plane, monkeypatch, shows=["'openai-codex'", "UTF-8"])


# --- the record size is JSON.stringify's, after the plane parsed the numbers ---------------------

@pytest.mark.parametrize(("number", "js"), [
    (1e21, "1e+21"), (1e-7, "1e-7"), (123.0, "123"), (0.000001, "0.000001"), (1.5e-7, "1.5e-7"),
    (9007199254740993, "9007199254740992"), (1e300, "1e+300"), (-0.0, "0"), (100, "100"), (1.25, "1.25"),
    (123456789012345680000, "123456789012345680000"), (1731000000.123, "1731000000.123"),
    (-2.5e-10, "-2.5e-10"), (5e-324, "5e-324"), (1.7976931348623157e308, "1.7976931348623157e+308"),
    (0.1, "0.1"), (1e20, "100000000000000000000"), (12345678901234567890123, "1.2345678901234568e+22")])
def test_numbers_measure_as_javascript_writes_them(number, js):
    """Expected strings are node's ``JSON.stringify`` of the same literals."""
    assert plane_json.stringify(number) == js


def test_strings_and_nesting_measure_as_javascript_writes_them():
    record = {"a": ["\u2028\n\"\\\x01", True, None, {"b": 2.0}], "\u00e9": ""}
    assert plane_json.stringify(record) == '{"a":["\u2028\\n\\"\\\\\\u0001",true,null,{"b":2}],"\u00e9":""}'


def _staging(plane, payload):
    path = plane.home / ".secrets-import.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


# --- numbers the plane cannot hold as written ------------------------------------------------------
# Valid JSON in auth.json, but a JavaScript number cannot hold it: Python loads it (inf / nan / an
# exact int) and either writes bare Infinity/NaN (not JSON: the plane refuses the request after the
# staging is written, and every later boot resends it) or digits the plane turns into null.

NONFINITE = {"overflow": "1e400", "negative_overflow": "-1e400", "nan_literal": "NaN",
             "infinity_literal": "-Infinity", "int_beyond_double": "1" + "0" * 400}


@pytest.mark.parametrize("number", NONFINITE.values(), ids=NONFINITE)
def test_a_number_the_plane_cannot_hold_fails_closed_before_staging(plane, monkeypatch, number):
    _write(plane.home, "RV_A=plain\n")
    (plane.home / "auth.json").write_text('{"providers":{"openai-codex":{"t":"q","n":' + number + "}}}",
                                          encoding="utf-8")
    shown = _refused(plane, monkeypatch, shows=["'openai-codex'", "auth.json", "Infinity"])
    assert ".secrets-import.json" not in shown
    assert not (plane.home / ".secrets-import.json").exists()


@pytest.mark.parametrize("number", ["1e400", "Infinity", "NaN"])
def test_a_leftover_staging_with_a_number_the_plane_cannot_hold_fails_closed_naming_it(plane, monkeypatch, number):
    staging = _staging(plane, '{"env":{},"auth":{"openai-codex":{"n":' + number + "}}}")
    _write(plane.home, "RV_A=plain\n")
    _refused(plane, monkeypatch, shows=[str(staging), "'openai-codex'", "delete"])


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_the_transport_never_writes_a_number_json_cannot_carry(value):
    from plugins.config_backends.remote.client import encode_body
    with pytest.raises(ValueError):
        encode_body({"auth": {"openai-codex": {"record": {"n": value}}}})


# --- reserved names: the plane's secret_name_reserved, from whichever source ------------------------

@pytest.mark.parametrize("payload", [
    {"env": {"PATH": "stale-path"}, "auth": {}}, {"env": {"hermes_home": "/x"}, "auth": {}},
    {"env": {"GATEWAY_RELAY_IDP_EXTRA": "x"}, "auth": {}}, {"env": {"HERMES_SECRET_STORE": "x"}, "auth": {}},
    {"env": {"TERMINAL_CWD": "x"}, "auth": {}}, {"env": {}, "auth": {"nous": {"access_token": "stale-tok"}}},
], ids=["PATH", "any_case_launch_only", "idp_prefix", "secret_store", "global_prefix", "auth_nous"])
def test_a_leftover_staging_with_a_reserved_name_fails_closed_naming_it(plane, monkeypatch, payload):
    """An earlier version staged a name the plane never stores: every boot stops naming the staging
    file and the name (a reserved name is a known constant), before any request or write."""
    staging = _staging(plane, payload)
    _write(plane.home, "RV_A=plain\n")
    name = next(iter({**payload["env"], **payload["auth"]}))
    _refused(plane, monkeypatch, shows=[str(staging), repr(name), "never stores", "delete"],
             hides=["stale-tok", "stale-path"])
    staging.unlink()
    _start(monkeypatch)
    _boot(plane.home)
    assert _stored(plane) == {"RV_A": "plain"}


def test_the_body_measured_is_the_body_sent(plane, monkeypatch):
    """The request-size check and the transport encode the import with the same function, and
    the import sends the body the check measured."""
    from plugins.config_backends.remote import client as plane_client
    encoded = []
    real = plane_client.encode_body

    def spy(body):
        encoded.append(body)
        return real(body)
    monkeypatch.setattr(plane_client, "encode_body", spy)
    monkeypatch.setattr(limits, "encode_body", spy)
    _write(plane.home, "RV_A=pl\u00e9in\u2028x\n")
    (plane.home / "auth.json").write_text(_auth({"t": "c", "n": 1.0}), encoding="utf-8")
    _start(monkeypatch)
    _boot(plane.home)
    measured, sent = encoded[0], encoded[-1]
    assert len(encoded) == 2 and measured == sent
    assert sent == limits.import_body({"RV_A": "pl\u00e9in\u2028x"}, {"openai-codex": {"t": "c", "n": 1.0}})
