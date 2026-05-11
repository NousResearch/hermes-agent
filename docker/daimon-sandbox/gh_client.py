#!/usr/bin/env python3
"""Client shim installed as `gh` inside the untrusted Daimon sandbox."""
from __future__ import annotations

import json
import os
import socket
import sys

BROKER_HOST = os.environ.get("DAIMON_GH_BROKER_HOST", "daimon-github-broker")
BROKER_PORT = int(os.environ.get("DAIMON_GH_BROKER_PORT", "7842"))


def _request(argv: list[str]) -> dict:
    payload = json.dumps(
        {
            "argv": argv,
            "cwd": os.getcwd(),
            "timeout_sec": int(os.environ.get("DAIMON_GH_TIMEOUT_SEC", "60")),
        }
    ).encode()
    with socket.create_connection((BROKER_HOST, BROKER_PORT), timeout=5) as sock:
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)
        response = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            response += chunk
    return json.loads(response.decode("utf-8"))


def main() -> int:
    try:
        response = _request(sys.argv[1:])
    except (ConnectionRefusedError, socket.gaierror, TimeoutError):
        print("Error: GitHub broker is not accepting connections.", file=sys.stderr)
        return 1
    except Exception:
        print("Error: GitHub broker request failed.", file=sys.stderr)
        return 1

    stdout = response.get("stdout") or ""
    stderr = response.get("stderr") or ""
    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
    return int(response.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
