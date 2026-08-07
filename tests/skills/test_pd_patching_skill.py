"""Regression tests for the `pd-patching` optional skill.

Standard library + pytest only; no Pure Data, no network. Exercises the patch
builder / safety linter (`pd_patch.py`) and the TCP client's message formatting
(`pd_client.py`). Socket I/O is mocked.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "creative"
    / "pd-patching"
    / "scripts"
)


@pytest.fixture(scope="module", autouse=True)
def _add_scripts_to_path():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield
    finally:
        sys.path.remove(str(SCRIPTS))


# --- pd_patch: builder + safety linter ------------------------------------

def test_simple_patch_is_safe():
    import pd_patch as m

    p = m.build_simple_synth()
    assert p.is_safe()
    assert p.safety_violations() == []


def test_raw_osc_into_dac_is_flagged():
    import pd_patch as m

    p = m.PdPatch()
    p.clear()
    osc = p.obj("osc~", 440)
    dac = p.obj("dac~")
    p.connect(osc, 0, dac, 0)          # no clip~/[*~]/hip~ master chain
    assert not p.is_safe()
    violations = p.safety_violations()
    assert any("clip~" in v for v in violations)


def test_master_gain_over_limit_is_flagged():
    import pd_patch as m

    p = m.PdPatch()
    p.clear()
    osc = p.obj("osc~", 440)
    p.to_dac(osc, 0.9)                 # 0.9 > MAX_MASTER_GAIN (0.1)
    assert not p.is_safe()
    assert any("exceeds" in v for v in p.safety_violations())


def test_obj_indices_increment_in_creation_order():
    import pd_patch as m

    p = m.PdPatch()
    p.clear()
    a = p.obj("osc~", 440)
    b = p.obj("lop~", 800)
    assert (a, b) == (0, 1)


def test_to_pd_file_writes_a_valid_patch(tmp_path):
    import pd_patch as m

    out = tmp_path / "synth.pd"
    m.build_simple_synth().to_pd_file(str(out))
    text = out.read_text(encoding="utf-8")
    assert text.startswith("#N canvas")     # Pd file header
    assert "#X obj" in text
    assert "osc~" in text
    assert "#X connect" in text


# --- pd_client: TCP-only transport + FUDI formatting ----------------------

def test_client_is_tcp_only():
    import pd_client as c

    # UDP was removed to match the TCP receiver stub; no proto knob remains.
    params = inspect.signature(c.PureDataClient.__init__).parameters
    assert "proto" not in params
    assert not hasattr(c, "DEFAULT_PROTO")


def test_format_message_strips_leading_semicolon_and_terminates():
    import pd_client as c

    fmt = c.PureDataClient.format_message
    assert fmt("; pd dsp 1") == "pd dsp 1;\n"     # leading ';' dropped
    assert fmt("pd dsp 1") == "pd dsp 1;\n"       # trailing ';' + newline added
    assert fmt("pd-target_canvas obj 1 2 osc~;") == "pd-target_canvas obj 1 2 osc~;\n"


def test_send_opens_tcp_and_writes_formatted_payload(monkeypatch):
    import socket as real_socket

    import pd_client as c

    captured = {}

    class FakeSock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def settimeout(self, t):
            pass

        def connect(self, addr):
            captured["addr"] = addr

        def sendall(self, payload):
            captured["payload"] = payload

    def fake_socket(family, kind):
        captured["kind"] = kind
        return FakeSock()

    monkeypatch.setattr(c.socket, "socket", fake_socket)
    ok = c.PureDataClient().send("; pd dsp 1")

    assert ok is True
    assert captured["kind"] == real_socket.SOCK_STREAM   # TCP
    assert captured["addr"] == ("127.0.0.1", 9999)
    assert captured["payload"] == b"pd dsp 1;\n"


def test_send_reports_failure_on_oserror(monkeypatch):
    import pd_client as c

    def boom(family, kind):
        raise OSError("connection refused")

    monkeypatch.setattr(c.socket, "socket", boom)
    assert c.PureDataClient().send("pd dsp 1") is False   # error surfaced, not raised
