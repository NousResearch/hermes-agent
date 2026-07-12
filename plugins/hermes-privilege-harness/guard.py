"""
VIP Guard — passive privilege harness.

Philosophy:
  - Hermes handles: dangerous command detection, approval cards, blocking sudo
  - VIP handles: execution via daemon (only after proven approval)

Security: vip_sudo handler refuses any command it hasn't stamped in check().
          A command must pass through the approval gate before it executes.
"""

import json
import logging
import os
import socket
import struct
import time

logger = logging.getLogger("hermes-vip.guard")

REQUEST_SOCK = os.environ.get("VIP_REQUEST_SOCK", "/var/run/hermes-vip/request.sock")

# ── Defense-in-depth: commands must be stamped by check() before execution ──
# check() stores a stamp → handler verifies it → handler clears it.
# A direct call to vip_sudo (bypassing the approval card) will be rejected.
_STAMP_TTL = 30  # seconds — generous: handler runs immediately after approval
_stamps: dict[str, float] = {}


def _stamp(command: str):
    """Mark a command as having passed through the approval gate."""
    key = command[:120]  # use prefix as key (rejects command-truncation attacks)
    _stamps[key] = time.time()
    # Clean expired stamps
    now = time.time()
    for k in list(_stamps):
        if now - _stamps[k] > _STAMP_TTL * 2:
            del _stamps[k]


def _verify(command: str) -> bool:
    """Verify the command was stamped by check(). Returns True and clears stamp."""
    key = command[:120]
    ts = _stamps.pop(key, None)
    if ts is None:
        return False
    if time.time() - ts > _STAMP_TTL:
        return False
    return True


# ── pre_tool_call ──

def check(tool_name: str, args: dict):
    """Stamp vip_sudo commands before Hermes shows the approval card."""
    if tool_name == "vip_sudo":
        command = args.get("command", "") if isinstance(args, dict) else ""
        _stamp(command)
        return {
            "action": "approve",
            "message": f"Execute with root: {command[:80]}",
        }
    return None


# ── vip_sudo handler ──

def vip_sudo(command: str, reason: str = "") -> str:
    """
    Execute via daemon. REFUSES to execute unless check() stamped this command first.
    Called only after Hermes native card approval.
    """
    if not command:
        return json.dumps({"error": "command required", "exit_code": -1})

    if not _verify(command):
        return json.dumps({
            "error": "REJECTED: command was not approved through the privilege gate",
            "exit_code": -1,
        })

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(600)

    try:
        sock.connect(REQUEST_SOCK)
    except OSError as exc:
        logger.error("daemon unreachable: %s", exc)
        return json.dumps({"error": "VIP daemon not running", "exit_code": -1})

    req = {
        "type": "sudo_execute",
        "command": command,
        "reason": reason or "privilege request",
        "origin": {"channel": "vip_sudo", "timestamp": time.time()},
    }
    payload = json.dumps(req).encode()

    try:
        sock.sendall(struct.pack("!I", len(payload)) + payload)
    except OSError as exc:
        sock.close()
        return json.dumps({"error": f"submit failed: {exc}", "exit_code": -1})

    try:
        raw = _recv_all(sock, 4)
        if not raw or len(raw) < 4:
            sock.close()
            return json.dumps({"error": "daemon closed", "exit_code": -1})
        mlen = struct.unpack("!I", raw)[0]
        data = _recv_all(sock, mlen)
        if len(data) != mlen:
            sock.close()
            return json.dumps({"error": "incomplete response", "exit_code": -1})
        result = json.loads(data.decode())
        sock.close()
    except Exception as exc:
        sock.close()
        return json.dumps({"error": f"read failed: {exc}", "exit_code": -1})

    status = result.get("status", "")
    if status == "approved":
        r = result.get("result", {})
        stdout = r.get("stdout", "")
        stderr = r.get("stderr", "")
        ec = r.get("exit_code", -1)
        if ec == 0:
            return stdout or json.dumps({"status": "ok", "exit_code": 0})
        return json.dumps({"error": stderr or f"exit {ec}", "exit_code": ec})
    return json.dumps({"error": result.get("error", "unknown"), "exit_code": -1})


def _recv_all(sock: socket.socket, size: int) -> bytes:
    if size <= 0:
        return b""
    chunks, remaining = [], size
    while remaining > 0:
        c = sock.recv(remaining)
        if not c:
            break
        chunks.append(c)
        remaining -= len(c)
    return b"".join(chunks)
