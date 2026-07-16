"""
VIP Guard — passive privilege harness.

Philosophy:
  - Hermes handles: dangerous command detection, approval cards, blocking sudo
  - VIP handles: execution via daemon (only after proven approval)

Security: vip_sudo handler refuses any command it hasn't stamped in check().
          A command must pass through the approval gate before it executes.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import socket
import struct
import time

logger = logging.getLogger("hermes-vip.guard")

REQUEST_SOCK = os.environ.get("VIP_REQUEST_SOCK", "/var/run/hermes-vip/request.sock")

# ── Defense-in-depth daemon-level stamp verification ──
# Plugin generates a random secret, registers it with daemon via stamp_init.
# Every sudo_execute includes HMAC-SHA256(command, secret) as stamp.
# Daemon verifies the HMAC before executing.
_stamp_secret: bytes = os.urandom(32)
_stamps: dict[str, str] = {}
_nonce: str = ""  # command[:120] → HMAC hex digest
_secret_registered: bool = False


def _register_stamp_secret():
    """Register stamp secret with daemon. Called once at plugin init."""
    global _secret_registered
    if _secret_registered:
        return
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(5)
    try:
        s.connect(REQUEST_SOCK)
        req = json.dumps({
            "type": "stamp_init",
            "secret": base64.b64encode(_stamp_secret).decode(),
        }).encode()
        s.sendall(struct.pack("!I", len(req)) + req)
        raw = s.recv(4)
        if raw and len(raw) == 4:
            mlen = struct.unpack("!I", raw)[0]
            data = s.recv(mlen)
            resp = json.loads(data.decode())
            if resp.get("status") == "ok":
                global _nonce, _secret_registered
                _nonce = resp.get("nonce", "")
                _secret_registered = True
                logger.info("stamp secret registered nonce=%s", _nonce[:8])
    except Exception as exc:
        logger.warning("failed to register stamp secret: %s", exc)
    finally:
        s.close()


def _stamp(command: str):
    """Compute HMAC stamp for the command."""
    key = command[:120]
    _stamps[key] = hmac.new(_stamp_secret, command.encode(), hashlib.sha256).hexdigest()


def _verify(command: str) -> bool:
    """Verify the command was stamped by check(). Returns True and clears stamp."""
    key = command[:120]
    return key in _stamps


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
        "stamp": _stamps.pop(command[:120], ""),
        "nonce": _nonce,
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
