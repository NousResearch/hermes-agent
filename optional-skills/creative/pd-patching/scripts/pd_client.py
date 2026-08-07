#!/usr/bin/env python3
"""Lightweight socket bridge for Pure Data dynamic patching (Approach A).

Sends FUDI / [netreceive] messages to a running Pure Data instance over TCP
(matching the bundled receiver stub). Standard library only -- no external
dependencies.

A Pd dynamic-patching command is a receiver-addressed FUDI message: the first
atom names the receiver, the rest is the message, e.g.:

    pd-target_canvas obj 100 100 osc~ 440

The stub patch (templates/receiver_stub.pd) wires
``[netreceive] -> [route pd-target_canvas pd] -> [send ...]`` so that the first
atom is stripped and the remainder is dispatched to the canvas editor (or the
global ``pd`` receiver for ``dsp``). NOTE: a leading ';' is NOT used here -- over
[netreceive] it does not route to a receiver, it only injects an empty message.

CLI examples
------------
    # single command
    python pd_client.py "pd-target_canvas obj 100 100 osc~ 440"

    # several commands in one connection (order preserved)
    python pd_client.py "pd-target_canvas clear" \
                        "pd-target_canvas obj 100 50 osc~ 440"

    # read newline separated commands from a file
    python pd_client.py --file patch.txt

    # read from stdin
    cat patch.txt | python pd_client.py -

    # target a different host/port
    python pd_client.py --host 127.0.0.1 --port 9999 "pd dsp 1"
"""
from __future__ import annotations

import argparse
import socket
import sys
from typing import Iterable, List

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9999
DEFAULT_TIMEOUT = 2.0


class PureDataClient:
    """Sends canvas-manipulation messages to a Pd [netreceive] socket over TCP.

    The bundled ``templates/receiver_stub.pd`` opens a default (TCP)
    ``[netreceive 9999]``, so this client speaks TCP. (Pd's UDP mode needs
    ``[netreceive -u 9999]``; ship that receiver if you want a UDP transport.)
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.host = host
        self.port = int(port)
        self.timeout = timeout

    @staticmethod
    def format_message(message: str) -> str:
        """Normalize one Pd command into a single FUDI message.

        A command reaches Pd as ``<receiver> <selector> <args...>;`` -- the
        receiver name (e.g. ``pd-target_canvas`` or ``pd``) is the first atom,
        which the stub's ``[route]`` dispatches. This method:

        * strips any leading ``;`` -- over ``[netreceive]`` a leading semicolon
          does NOT route to a receiver (that's a message-box-only feature); it
          just injects a spurious empty message, so we drop it, and
        * ensures exactly one trailing ``;`` terminator plus a newline to flush
          the packet.
        """
        message = message.strip().lstrip(";").strip()
        if not message.endswith(";"):
            message += ";"
        return message + "\n"

    def _sendall(self, payload: bytes) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(self.timeout)
                s.connect((self.host, self.port))
                s.sendall(payload)
            return True
        except OSError as e:
            print(
                f"[pd_client] error sending to {self.host}:{self.port}: {e}",
                file=sys.stderr,
            )
            return False

    def send(self, message: str) -> bool:
        """Send a single command (semicolon auto-appended if missing)."""
        return self._sendall(self.format_message(message).encode("utf-8"))

    def send_batch(self, messages: Iterable[str]) -> bool:
        """Send several commands over a single connection, order preserved."""
        payload = "".join(self.format_message(m) for m in messages).encode("utf-8")
        return self._sendall(payload)

    # Backwards-compatible alias for the name used in the original handoff.
    def send_message(self, message: str) -> bool:
        return self.send(message)


def _read_commands(text: str) -> List[str]:
    """Split a blob of text into individual Pd commands.

    Commands may be separated by newlines. Blank lines and lines beginning
    with '#' (comments) are ignored.
    """
    out: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(stripped)
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Send Pure Data dynamic-patching messages over a socket.",
    )
    p.add_argument(
        "commands",
        nargs="*",
        help="One or more Pd commands. Use '-' to read from stdin.",
    )
    p.add_argument("--host", default=DEFAULT_HOST, help="Pd host (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Pd port (default 9999)")
    p.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Socket timeout in seconds (default 2.0)",
    )
    p.add_argument("--file", help="Read commands from a file (newline separated)")
    return p


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    messages: List[str] = []
    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            messages.extend(_read_commands(fh.read()))
    if args.commands == ["-"]:
        messages.extend(_read_commands(sys.stdin.read()))
    else:
        messages.extend(args.commands)

    if not messages:
        parser.print_usage(sys.stderr)
        print("error: no commands to send", file=sys.stderr)
        return 2

    client = PureDataClient(
        host=args.host, port=args.port, timeout=args.timeout
    )
    ok = client.send_batch(messages)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
