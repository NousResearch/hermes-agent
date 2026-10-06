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

import threading
_lock = threading.Lock()

# ── Defense-in-depth daemon-level stamp verification ──
# The capability is ISSUED BY THE DAEMON (stamp_init) and bound to the
# plugin's peer uid. The plugin never self-mints secrets. Every
# sudo_execute includes HMAC-SHA256(command, cap) as stamp; the daemon
# verifies cap ownership + HMAC before executing.
_stamp_cap: bytes = b""          # daemon-issued capability
_stamps: dict[str, tuple[str, float]] = {}   # sha256(command) -> (hmac, ts)
_cap_registered: bool = False


def _register_stamp_cap():
    """Ask the daemon to issue a capability. Called once at plugin init.

    The daemon mints the random cap and binds it to our peer uid — a local
    process cannot self-mint a credential, so the old self-authentication
    bypass is closed.
    """
    global _cap_registered, _stamp_cap
    if _cap_registered:
        return
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(5)
    try:
        s.connect(REQUEST_SOCK)
        req = json.dumps({"type": "stamp_init"}).encode()
        s.sendall(struct.pack("!I", len(req)) + req)
        raw = _recv_all(s, 4)
        if raw and len(raw) == 4:
            mlen = struct.unpack("!I", raw)[0]
            data = _recv_all(s, mlen)
            resp = json.loads(data.decode())
            if resp.get("status") == "ok" and resp.get("cap"):
                _stamp_cap = base64.b64decode(resp["cap"])
                _cap_registered = True
                logger.info("stamp capability issued (%d bytes)",
                            len(_stamp_cap))
    except Exception as exc:
        logger.warning("failed to register stamp capability: %s", exc)
    finally:
        s.close()


_STAMP_TTL = 15.0


def _stamp(command: str):
    """Record that this exact command passed check() (full-sha256 key)."""
    key = hashlib.sha256(command.encode()).hexdigest()
    digest = hmac.new(_stamp_cap, command.encode(), hashlib.sha256).hexdigest()
    now = time.time()
    with _lock:
        _stamps[key] = (digest, now)
        for k in [k for k, (_, ts) in _stamps.items()
                  if now - ts > _STAMP_TTL * 2]:
            del _stamps[k]


def _verify(command: str) -> bool:
    """Verify the command was stamped by check() and matches exactly.

    Full-sha256 key (not a 120-char prefix), HMAC value comparison, and a
    TTL check — a same-prefix command cannot pass.
    """
    key = hashlib.sha256(command.encode()).hexdigest()
    with _lock:
        entry = _stamps.pop(key, None)
    if entry is None:
        return False
    digest, ts = entry
    if time.time() - ts > _STAMP_TTL:
        return False
    expected = hmac.new(_stamp_cap, command.encode(),
                        hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, expected)


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

    if not _cap_registered or not _stamp_cap:
        return json.dumps({
            "error": "REJECTED: no stamp capability (daemon unreachable at init?)",
            "exit_code": -1,
        })

    req = {
        "type": "sudo_execute",
        "command": command,
        "reason": reason or "privilege request",
        "origin": {"channel": "vip_sudo", "timestamp": time.time()},
        "cap": base64.b64encode(_stamp_cap).decode(),
        "stamp": hmac.new(_stamp_cap, command.encode(),
                          hashlib.sha256).hexdigest(),
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
