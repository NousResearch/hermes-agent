#!/usr/bin/env python3
"""Lightweight OSC-over-UDP bridge to a running SuperCollider server (scsynth).

scsynth speaks Open Sound Control. This module encodes OSC 1.0 packets in pure
Python and sends them to ``scsynth`` on UDP :57110 -- no ``sclang``, no
``python-osc``, no ``supriya``; **standard library only.**

A synth is created, controlled and freed with plain server commands:

    /d_recv  <synthdef-bytes> [completion]   load a compiled SynthDef
    /s_new   <name> <id> <add> <target> ...  start a synth node
    /n_set   <id> <ctrl> <val> ...           change a control live
    /n_free  <id>                            free a node
    /g_freeAll 0                             free everything (panic / stop)
    /status                                  query the server

The killer combination is ``/d_recv``'s *completion message*: pack a ``/s_new``
into the same packet and the synth starts the instant its definition finishes
loading -- one UDP datagram, no round-trip. That's what :meth:`SCClient.play`
does.

CLI examples
------------
    python3 sc_client.py /status
    python3 sc_client.py /s_new hermes_sine 1000 0 0 freq 300
    python3 sc_client.py /n_set 1000 freq 220
    python3 sc_client.py --free-all          # stop all synths
    python3 sc_client.py --recv-file foo.scsyndef   # load a compiled def
    python3 sc_client.py --quit              # shut the server down
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from typing import Any, List, Optional

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 57110          # scsynth's default UDP port
DEFAULT_TIMEOUT = 1.0


# -- OSC 1.0 encoding (big-endian) ------------------------------------------
def _pad4(b: bytes) -> bytes:
    """Pad to the next 4-byte boundary with NULs (OSC alignment rule)."""
    return b + b"\x00" * (-len(b) % 4)


def _osc_string(s: str) -> bytes:
    return _pad4(s.encode("utf-8") + b"\x00")   # NUL-terminated, then padded


def _osc_blob(b: bytes) -> bytes:
    return struct.pack(">i", len(b)) + _pad4(b)


def encode_message(address: str, *args: Any) -> bytes:
    """Encode one OSC message. Types are inferred: int->i, float->f, str->s, bytes->b."""
    tags = ","
    payload = bytearray()
    for a in args:
        if isinstance(a, bool):
            raise TypeError("OSC has no bool type; use 0/1")
        if isinstance(a, int):
            tags += "i"
            payload += struct.pack(">i", a)
        elif isinstance(a, float):
            tags += "f"
            payload += struct.pack(">f", a)
        elif isinstance(a, (bytes, bytearray)):
            tags += "b"
            payload += _osc_blob(bytes(a))
        elif isinstance(a, str):
            tags += "s"
            payload += _osc_string(a)
        else:
            raise TypeError(f"unsupported OSC arg type: {type(a).__name__}")
    return _osc_string(address) + _osc_string(tags) + bytes(payload)


def decode_message(data: bytes):
    """Minimal OSC message decoder -> (address, [args]). Best-effort.

    Used to read scsynth replies (``/done``, ``/fail``, ``/synced``, ...). Only
    the common tags (i/f/s/b) are decoded; anything unexpected stops parsing and
    returns what was read so far.
    """
    def read_string(buf: bytes, i: int):
        end = buf.index(b"\x00", i)
        s = buf[i:end].decode("utf-8", "replace")
        return s, (end // 4 + 1) * 4          # advance past NUL, 4-byte aligned

    try:
        address, i = read_string(data, 0)
        if i >= len(data) or data[i:i + 1] != b",":
            return address, []
        tags, i = read_string(data, i)
        args: List[Any] = []
        for t in tags[1:]:
            if t == "i":
                args.append(struct.unpack(">i", data[i:i + 4])[0]); i += 4
            elif t == "f":
                args.append(struct.unpack(">f", data[i:i + 4])[0]); i += 4
            elif t == "s":
                s, i = read_string(data, i); args.append(s)
            elif t == "b":
                n = struct.unpack(">i", data[i:i + 4])[0]; i += 4
                args.append(data[i:i + n]); i += (n + 3) // 4 * 4
            else:
                break
        return address, args
    except Exception:
        return "", []


def _coerce_cli_arg(tok: str) -> Any:
    """Turn a CLI token into an OSC arg: int if it looks like one, else float, else str."""
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        return tok


class SCClient:
    """Sends server commands to scsynth over UDP; keeps one socket for replies."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout: float = DEFAULT_TIMEOUT):
        self.host = host
        self.port = int(port)
        self.timeout = timeout
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(timeout)
        self._next_node = 1000            # node-id allocator for auto-assigned synths

    # -- transport ----------------------------------------------------------
    def send(self, address: str, *args: Any) -> None:
        self._sock.sendto(encode_message(address, *args), (self.host, self.port))

    def send_recv(self, address: str, *args: Any, match: Optional[str] = None,
                  bufsize: int = 65536) -> Optional[bytes]:
        """Send, then wait for a reply datagram (or None on timeout).

        If ``match`` is given, skip datagrams whose OSC address does not start
        with it -- so a stale ``/done`` left over from an earlier ``/d_recv``
        doesn't get mistaken for the reply we actually want.
        """
        self.send(address, *args)
        while True:
            try:
                data, _ = self._sock.recvfrom(bufsize)
            except socket.timeout:
                return None
            if match is None or data.startswith(match.encode("ascii")):
                return data

    def close(self) -> None:
        self._sock.close()

    # -- node ids -----------------------------------------------------------
    def alloc_node(self) -> int:
        nid = self._next_node
        self._next_node += 1
        return nid

    # -- high-level server commands ----------------------------------------
    def load(self, synthdef: Any) -> None:
        """Load a compiled SynthDef (a ``SynthDef`` object or raw bytes) via /d_recv."""
        self.send("/d_recv", _synthdef_bytes(synthdef))

    def play(self, synthdef: Any, node_id: Optional[int] = None,
             add_action: int = 0, target: int = 0, **controls: float) -> int:
        """Load ``synthdef`` and start it in one packet; return the node id.

        Uses ``/d_recv``'s completion message so the ``/s_new`` fires only after
        the definition is registered. ``controls`` become initial ``/s_new``
        control pairs. ``add_action`` 0=head, 1=tail, 2=before, 3=after; ``target``
        is the group/node it is added relative to (0 == the root group).
        """
        name = _synthdef_name(synthdef)
        nid = self.alloc_node() if node_id is None else node_id
        pairs: List[Any] = []
        for k, v in controls.items():
            pairs.append(k)
            pairs.append(float(v))
        completion = encode_message("/s_new", name, nid, add_action, target, *pairs)
        self.send("/d_recv", _synthdef_bytes(synthdef), completion)
        return nid

    def new_synth(self, name: str, node_id: Optional[int] = None,
                  add_action: int = 0, target: int = 0, **controls: float) -> int:
        """Start an already-loaded SynthDef by name; return the node id."""
        nid = self.alloc_node() if node_id is None else node_id
        pairs: List[Any] = []
        for k, v in controls.items():
            pairs.append(k)
            pairs.append(float(v))
        self.send("/s_new", name, nid, add_action, target, *pairs)
        return nid

    def set(self, node_id: int, **controls: float) -> None:
        """Change controls on a running node live (``/n_set``)."""
        pairs: List[Any] = []
        for k, v in controls.items():
            pairs.append(k)
            pairs.append(float(v))
        self.send("/n_set", node_id, *pairs)

    def verify(self, sync_id: int = 1, timeout: float = 2.0):
        """Confirm the preceding commands landed *and* were accepted.

        Sends ``/sync`` and drains replies until the matching ``/synced``. A
        ``/d_recv`` reports success as ``/done`` and rejection as ``/fail``; a
        bare ``/synced`` only proves the async queue drained, not that the
        SynthDef loaded. So this watches for ``/fail`` arriving before
        ``/synced`` and reports it. Returns ``(ok, error)``: ``ok`` is ``False``
        on a ``/fail`` or on timeout, and ``error`` carries the ``/fail`` text.
        """
        self.send("/sync", sync_id)
        error: Optional[str] = None
        old = self._sock.gettimeout()
        self._sock.settimeout(timeout)
        try:
            while True:
                try:
                    data, _ = self._sock.recvfrom(65536)
                except socket.timeout:
                    return (False, error or "no /synced reply from scsynth")
                address, args = decode_message(data)
                if address == "/fail":
                    error = " ".join(str(a) for a in args) or "/fail"
                elif address == "/synced":
                    return (error is None, error)
        finally:
            self._sock.settimeout(old)

    def sync(self, sync_id: int = 1, timeout: float = 2.0) -> bool:
        """Confirm the preceding commands landed and were accepted (bool form).

        Thin wrapper over :meth:`verify` -- ``False`` on a ``/fail`` or timeout.
        """
        return self.verify(sync_id, timeout)[0]

    def free(self, node_id: int) -> None:
        self.send("/n_free", node_id)

    def free_all(self, group: int = 0) -> None:
        """Free every synth in a group (default the root group) -- the stop/panic button."""
        self.send("/g_freeAll", group)

    def quit(self) -> None:
        """Ask the server to shut down."""
        self.send("/quit")

    def status(self) -> Optional[bytes]:
        """Query the server; returns the raw ``/status.reply`` datagram or None."""
        return self.send_recv("/status", match="/status.reply")

    # -- recording (needs the server booted with audio) --------------------
    def record(self, path: str, node_id: int = 1999, bufnum: int = 99,
               frames: int = 65536, channels: int = 2) -> int:
        """Start recording the output bus to ``path`` (a WAV). Returns the recorder node.

        Allocates a streaming buffer, opens the soundfile for writing, then plays
        a DiskOut recorder synth at the **tail** of the node tree so it captures
        the summed, post-limiter output. Stop with :meth:`record_stop`.
        """
        from sc_synthdef import build_recorder
        self.send("/b_alloc", bufnum, frames, channels)
        time.sleep(0.05)  # let the buffer allocate before opening the file
        # /b_write: path, header, sample-format, numFrames(-1=all), start, leaveOpen=1
        self.send("/b_write", bufnum, path, "wav", "float", -1, 0, 1)
        time.sleep(0.05)
        rec = build_recorder(channels=channels)
        # addAction 1 == addToTail: recorder runs after the sound-producing synths
        self.play(rec, node_id=node_id, add_action=1, target=0, buf=float(bufnum))
        return node_id

    def record_stop(self, node_id: int = 1999, bufnum: int = 99) -> None:
        """Stop recording: free the recorder, close and free the soundfile buffer."""
        self.free(node_id)
        time.sleep(0.05)
        self.send("/b_close", bufnum)
        self.send("/b_free", bufnum)


def _synthdef_bytes(synthdef: Any) -> bytes:
    if isinstance(synthdef, (bytes, bytearray)):
        return bytes(synthdef)
    return synthdef.compile()


def _synthdef_name(synthdef: Any) -> str:
    if isinstance(synthdef, (bytes, bytearray)):
        # SynthDef v2: "SCgf" + int32 version + int16 count + pstring name
        length = synthdef[10]
        return synthdef[11:11 + length].decode("ascii")
    return synthdef.name


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Send OSC server commands to a running scsynth.",
    )
    p.add_argument("message", nargs="*",
                   help="An OSC address then args, e.g. /s_new hermes_sine 1000 0 0 freq 300")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--recv-file", metavar="PATH",
                   help="Load a compiled .scsyndef file via /d_recv")
    p.add_argument("--free-all", action="store_true", help="Free all synths (/g_freeAll 0)")
    p.add_argument("--status", action="store_true", help="Print the server /status reply")
    p.add_argument("--quit", action="store_true", help="Shut the server down (/quit)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    client = SCClient(host=args.host, port=args.port)

    if args.recv_file:
        with open(args.recv_file, "rb") as fh:
            client.load(fh.read())
        return 0
    if args.free_all:
        client.free_all()
        return 0
    if args.quit:
        client.quit()
        return 0
    if args.status:
        reply = client.status()
        if reply is None:
            print("no reply (is scsynth running on "
                  f"{args.host}:{args.port}?)", file=sys.stderr)
            return 1
        print(_format_status(reply))
        return 0

    if not args.message:
        _build_parser().print_usage(sys.stderr)
        print("error: no OSC message to send", file=sys.stderr)
        return 2

    address = args.message[0]
    osc_args = [_coerce_cli_arg(t) for t in args.message[1:]]
    client.send(address, *osc_args)
    return 0


def _format_status(reply: bytes) -> str:
    """Decode a /status.reply datagram into a short human summary."""
    # /status.reply ,iiiiiffdd : cmds, ugens, synths, groups, defs, avgCPU, peakCPU, sr, actualSr
    try:
        # skip address + typetag, both OSC strings
        idx = reply.index(b"\x00")
        idx = (idx + 4) & ~3            # past address padding
        tt_end = reply.index(b"\x00", idx)
        tt_end = (tt_end + 4) & ~3      # past typetag padding
        body = reply[tt_end:]
        vals = struct.unpack(">iiiiiffdd", body[:4 * 5 + 4 * 2 + 8 * 2])
        return (f"ugens={vals[1]} synths={vals[2]} groups={vals[3]} "
                f"defs={vals[4]} avgCPU={vals[5]:.1f}% peakCPU={vals[6]:.1f}% "
                f"sr={vals[8]:.0f}")
    except Exception:
        return f"<status reply, {len(reply)} bytes>"


if __name__ == "__main__":
    sys.exit(main())
