"""Regression tests for the `supercollider` optional skill.

Standard library + pytest only; no scsynth, no network. Exercises the two pure
units: the SynthDef v2 compiler / safety linter (`sc_synthdef.py`) and the OSC
encoder (`sc_client.py`). Socket I/O in `sc_client` is mocked.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "creative"
    / "supercollider"
    / "scripts"
)


@pytest.fixture(scope="module", autouse=True)
def _add_scripts_to_path():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield
    finally:
        sys.path.remove(str(SCRIPTS))


# --- sc_synthdef: compiler ------------------------------------------------

def test_simple_synth_compiles_to_valid_synthdef_v2():
    import sc_synthdef as m

    sd = m.build_simple_synth(name="t_sine", freq=440, gain=0.1)
    blob = sd.compile()

    assert blob[:4] == b"SCgf"                       # file type id
    assert struct.unpack(">i", blob[4:8])[0] == 2    # file version 2
    assert struct.unpack(">h", blob[8:10])[0] == 1   # one synthdef
    # pstring name follows: length byte then that many bytes
    name_len = blob[10]
    assert blob[11:11 + name_len] == b"t_sine"


def test_control_becomes_a_parameter():
    import sc_synthdef as m

    sd = m.SynthDef("t_ctl")
    f = sd.control("freq", 330)
    sd.to_out(sd.sinosc(f), 0.1)

    assert sd.params == [("freq", 330.0)]
    # exactly one Control ugen carries the single parameter
    controls = [u for u in sd.ugens if u.name == "Control"]
    assert len(controls) == 1 and controls[0].num_outputs == 1


def test_constants_are_deduplicated():
    import sc_synthdef as m

    sd = m.SynthDef("t_const")
    # two SinOscs share phase 0.0; the constant pool must not duplicate it
    a = sd.sinosc(200)
    b = sd.sinosc(300)
    sd.to_out(a + b, 0.1)
    blob = sd.compile()

    # walk to the constant count: 4 (SCgf) + 4 (ver) + 2 (ndefs) + pstring name
    off = 10
    off += 1 + blob[10]
    n_consts = struct.unpack(">i", blob[off:off + 4])[0]
    consts = struct.unpack(f">{n_consts}f", blob[off + 4:off + 4 + n_consts * 4])
    assert consts.count(0.0) == 1     # phase 0.0 registered once, not twice


# --- sc_synthdef: safety linter -------------------------------------------

def test_to_out_chain_is_safe():
    import sc_synthdef as m

    sd = m.build_simple_synth(gain=0.1)
    assert sd.is_safe()
    assert sd.safety_violations() == []


def test_raw_signal_into_out_is_flagged():
    import sc_synthdef as m

    sd = m.SynthDef("t_raw")
    sd.out(0, sd.sinosc(440))          # no LeakDC / gain / Limiter
    violations = sd.safety_violations()
    assert not sd.is_safe()
    assert any("Limiter" in v for v in violations)
    assert any("LeakDC" in v for v in violations)


def test_master_gain_over_limit_is_flagged():
    import sc_synthdef as m

    sd = m.SynthDef("t_loud")
    sd.to_out(sd.sinosc(440), gain=0.9)   # 0.9 > MAX_MASTER_GAIN (0.2)
    assert not sd.is_safe()
    assert any("outside 0" in v for v in sd.safety_violations())


def test_negative_master_gain_is_flagged():
    import sc_synthdef as m

    # A negative gain has magnitude too: -10.0 must NOT pass just because
    # -10.0 <= 0.2. The linter requires 0 <= gain <= MAX_MASTER_GAIN.
    sd = m.SynthDef("t_neg")
    sd.to_out(sd.sinosc(440), gain=-10.0)
    assert not sd.is_safe()
    assert any("outside 0" in v for v in sd.safety_violations())


# --- sc_client: OSC encoding ----------------------------------------------

def test_osc_message_no_args_is_padded():
    import sc_client as c

    msg = c.encode_message("/status")
    # "/status\0" -> 8 bytes (already aligned); typetag ",\0\0\0" -> 4 bytes
    assert msg == b"/status\x00" + b",\x00\x00\x00"
    assert len(msg) % 4 == 0


def test_osc_message_typetags_and_args():
    import sc_client as c

    msg = c.encode_message("/n_set", 1000, "freq", 220.0)
    # address "/n_set\0\0" (8 bytes) then typetag ",isf\0\0\0\0" (8 bytes),
    # so the first arg (int 1000) begins at offset 16.
    assert msg[8:12] == b",isf"
    assert struct.unpack(">i", msg[16:20])[0] == 1000
    assert len(msg) % 4 == 0


def test_osc_blob_length_prefixed_and_padded():
    import sc_client as c

    blob = c.encode_message("/d_recv", b"abc")   # 3-byte blob -> len prefix + pad to 4
    # find the blob: address "/d_recv\0" (8) + typetag ",b\0\0" (4) = 12
    length = struct.unpack(">i", blob[12:16])[0]
    assert length == 3
    assert blob[16:20] == b"abc\x00"             # padded to 4 bytes


def test_synthdef_name_extracted_from_bytes():
    import sc_client as c
    import sc_synthdef as m

    blob = m.build_simple_synth(name="named_def").compile()
    assert c._synthdef_name(blob) == "named_def"


def test_decode_message_roundtrips_args():
    import sc_client as c

    addr, args = c.decode_message(c.encode_message("/fail", "/d_recv", "boom", 3))
    assert addr == "/fail"
    assert args == ["/d_recv", "boom", 3]


class _FakeSock:
    """Stand-in socket that replays a fixed list of reply datagrams."""

    def __init__(self, replies):
        self._replies = list(replies)
        self._timeout = None

    def gettimeout(self):
        return self._timeout

    def settimeout(self, t):
        self._timeout = t

    def sendto(self, payload, addr):
        pass

    def recvfrom(self, n):
        import socket as _s
        if self._replies:
            return self._replies.pop(0), ("127.0.0.1", 57110)
        raise _s.timeout()


def test_verify_succeeds_on_done_then_synced():
    import sc_client as c

    client = c.SCClient()
    client._sock = _FakeSock([
        c.encode_message("/done", "/d_recv"),
        c.encode_message("/synced", 1),
    ])
    ok, err = client.verify(timeout=0.5)
    assert ok is True and err is None


def test_verify_fails_when_synthdef_rejected():
    import sc_client as c

    client = c.SCClient()
    client._sock = _FakeSock([
        c.encode_message("/fail", "/d_recv", "SynthDef not found"),
        c.encode_message("/synced", 1),
    ])
    ok, err = client.verify(timeout=0.5)
    assert ok is False
    assert "SynthDef not found" in err


def test_verify_fails_on_timeout():
    import sc_client as c

    client = c.SCClient()
    client._sock = _FakeSock([])          # no replies -> timeout
    ok, err = client.verify(timeout=0.2)
    assert ok is False and "synced" in err


def test_play_sends_d_recv_with_completion(monkeypatch):
    import sc_client as c
    import sc_synthdef as m

    sent = []

    def fake_sendto(self, payload, addr):
        sent.append(payload)

    monkeypatch.setattr(c.socket.socket, "sendto", fake_sendto, raising=True)
    client = c.SCClient()
    node = client.play(m.build_simple_synth(name="p_sine"), node_id=1234)

    assert node == 1234
    assert len(sent) == 1                     # one datagram: /d_recv + completion
    assert sent[0].startswith(b"/d_recv\x00")
    assert b"/s_new" in sent[0]               # completion message embedded
    assert b"p_sine" in sent[0]
