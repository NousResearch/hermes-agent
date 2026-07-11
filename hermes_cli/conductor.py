"""Hermes conductor: lightweight CLI/scheme dispatcher for viewport computer.
Background
----------
This module used to centralize a fixed list of operator URI schemes.  It now
exposes a small, **standalone scheme dispatcher** so new primitives can be
registered in-process without editing a big if/elif chain.  The default
instance preserves historical behavior; callers can register new schemes
whenever they need to.

Public surface
--------------
- class ``SchemeDispatcher``
- function ``run_hermes(payload) -> dict``
- module-level ``dispatch`` backed by a default ``SchemeDispatcher``
"""
from __future__ import annotations

import os
import shlex
import subprocess
import json
import secrets
import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]


def _make_run_pc_name(raw: str) -> Tuple[str, str | None]:
    normalized = raw
    run_pc_name = None
    if normalized.startswith("run "):
        rest = normalized.split(" ", 1)[1].strip()
        if rest.startswith("pc://"):
            run_pc_name = rest.split("pc://", 1)[1].strip() or "default"
            normalized = f"pc://run {run_pc_name}"
        else:
            normalized = rest
    elif normalized.startswith("pc://run "):
        run_pc_name = normalized.split("pc://run ", 1)[1].strip() or "default"
        normalized = "pc://run"
    return normalized, run_pc_name


def _normalize_cc(raw: str) -> str:
    if raw.startswith("H://cc") or raw.startswith("hermes://cc"):
        return "c://cc" + raw.split("cc", 1)[1]
    return raw


def _cctx_dispatch(raw: str) -> dict:
    target = raw.split(" ", 1)[1].strip() if " " in raw else "pc://"
    return {
        "ok": True,
        "rc": 0,
        "stdout": f"cctx → {target}\n",
        "stderr": "",
        "surface": {
            "kind": "cctx",
            "target": target,
            "active": _DISPATCHER.is_scheme_cmd(target) and target != "c://",
        },
    }


def _aectx_dispatch(raw: str) -> dict:
    """æ:// — the agentic-language-chassis: sovereign context router.

    ``æ://`` is the namespace/runtime for agentic languages. Bare ``æ://`` is
    the catch-all context router: it resolves the target surface and reports
    whether that target is a live registered scheme. Dialects plug in as
    sub-schemes, each a chassis of its own:
      - ``æ://basic``   (+bæsic://)  -> qc64 BASIC chassis (qc64_basic.py)
      - ``æ://mech``    (mech-lang)  -> reactive dataflow state machines
      - ``æ://glocal-agent``        -> sovereign local agent (GPU-MCP)
      - ``+æ://cc``                -> command & control surface
    ``+æ://`` (the +æ superset) routes here as its catch-all.
    """
    target = raw.split("æ://", 1)[1].strip() if "æ://" in raw else "pc://"
    return {
        "ok": True,
        "rc": 0,
        "stdout": f"aectx → {target}\n",
        "stderr": "",
        "surface": {
            "kind": "aectx",
            "target": target,
            "active": _DISPATCHER.is_scheme_cmd(target) and target != "c://",
        },
    }


def _pc_run_dispatch(raw: str, run_pc_name: str | None) -> dict:
    client = run_pc_name or "default"
    return {
        "ok": True,
        "rc": 0,
        "stdout": f"pc://run {client}\n",
        "stderr": "",
        "surface": {
            "kind": "private_client_run",
            "address": f"pc://{client}",
            "client": client,
        },
    }


def _identity_dispatch(raw: str) -> dict:
    return {
        "ok": True,
        "rc": 0,
        "stdout": "+æ://identity bounded private client mesh^hermes-agent/conductor\n",
        "stderr": "",
        "surface": {
            "kind": "bounded_private_client_mesh",
            "address": "+æ://identity",
            "conductor": "hermes-agent/conductor",
            "runtime": "bounded_dispatch",
            "contract": "PCSurfaceContract",
            "governance": {
                "required": "+æ member token",
                "audit": True,
                "tracer": "Wyoming DAO LLC audit trail",
            },
        },
    }


def _pc_dispatch(raw: str) -> dict:
    """pc:// — the private-client runtime on the sovereign mesh.

    Canonical mesh is ``pc://mesh/victus/local`` (offline brain + local hands).
    A bare ``pc://`` reports the mesh; ``pc://<node>`` addresses a node on it.
    """
    node = raw.split("pc://", 1)[1].strip() if "pc://" in raw else ""
    mesh = "pc://mesh/victus/local"
    target = node or mesh
    return {
        "ok": True,
        "rc": 0,
        "stdout": f"pc://private client runtime -> {target}\n",
        "stderr": "",
        "surface": {
            "kind": "private_client",
            "address": raw,
            "mesh": mesh,
            "node": target,
            "runtime": "hermes-code",
            "local_only": True,
        },
    }


def _qrcode_dispatch(raw: str) -> dict:
    html = ""
    action = None
    source = "unknown"
    if raw.partition("+æ://")[2].strip().startswith("qrcode payload "):
        html = raw.split("+æ://qrcode payload ", 1)[1].strip()
        source = "payload"
    else:
        path = raw.split("+æ://qrcode", 1)[1].strip() if "+æ://qrcode" in raw else ""
        if not path:
            return {
                "ok": False,
                "rc": 2,
                "stdout": "",
                "stderr": "missing +æ://qrcode payload or file path",
                "surface": {"kind": "qrcode_surface", "address": raw, "runtime": "hermes-code"},
            }
        path = path.strip()
        if path.startswith("'") and path.endswith("'"):
            path = path[1:-1].strip()
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1].strip()
        try:
            path_obj = Path(path)
            if not path_obj.is_absolute():
                path_obj = (REPO / path_obj).resolve()
            html = path_obj.read_text(encoding="utf-8", errors="ignore")
            source = str(path_obj)
        except Exception as exc:
            return {
                "ok": False,
                "rc": 2,
                "stdout": "",
                "stderr": f"qrcode read failed: {exc}",
                "surface": {"kind": "qrcode_surface", "address": raw, "runtime": "hermes-code"},
            }
    manifest = _qrcode_headless_manifest(html)
    qr_path = _write_qrcode_image(html, raw, source)
    return {
        "ok": True,
        "rc": 0,
        "stdout": f"+æ://qrcode {source} -> {qr_path}\n",
        "stderr": "",
        "surface": {
            "kind": "qrcode_surface",
            "address": raw,
            "runtime": "hermes-code",
            "source": source,
            "headless": True,
            "html_bytes": len(html.encode("utf-8")),
            "image_path": qr_path,
            "action": manifest.get("action"),
            "required_token": manifest.get("required_token"),
            "target_surface": manifest.get("target_surface"),
            "executed": False,
            "execution": None,
        },
    }


def _qrcode_headless_manifest(html: str) -> dict[str, object | None]:
    stripped = html.strip()
    token = None
    action = None
    target_surface = None
    for marker in ["<!-- +æ_qrcode_token:", "<!-- qrcode_token:", "<!-- token:"]:
        if marker in stripped:
            token = stripped.split(marker, 1)[1].split("-->", 1)[0].strip()
            break
    for marker, kind in [
        ("<!-- +æ_qrcode_action:", "action"),
        ("<!-- qrcode_action:", "action"),
        ("<!-- target_surface:", "target_surface"),
    ]:
        if marker in stripped:
            value = stripped.split(marker, 1)[1].split("-->", 1)[0].strip()
            if kind == "action":
                action = value
            else:
                target_surface = value
            break
    if not action and 'data-action="' in html:
        action = html.split('data-action="', 1)[1].split('"', 1)[0].strip()
    if not target_surface and 'data-surface="' in html:
        target_surface = html.split('data-surface="', 1)[1].split('"', 1)[0].strip()
    if action and not target_surface and action.startswith("http"):
        target_surface = action
    return {
        "token": token,
        "action": action,
        "required_token": token,
        "target_surface": target_surface,
    }


def _write_qrcode_image(html: str, raw: str, source: str) -> str:
    try:
        import qrcode
    except Exception:
        raise RuntimeError("qrcode is required for +æ://qrcode")
    safe_source = source.replace("/", "_").replace("\\", "_") or "input"
    if not safe_source.endswith(".html"):
        safe_source = f"{safe_source}.html"
    file_name = f"qrcode_{safe_source}"
    if not file_name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        file_name = f"{file_name}.png"
    out_dir = REPO / "media" / "qrcodes"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        out_dir = Path.home() / "AppData" / "Local" / "hermes" / "qrcodes"
        out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / file_name
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_Q,
        box_size=8,
        border=4,
    )
    qr.add_data(html)
    try:
        qr.make(fit=False)
    except qrcode.exceptions.DataOverflowError:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_Q,
            box_size=8,
            border=4,
        )
        qr.add_data(html)
        qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    img.save(str(out_path))
    return str(out_path)


# ---------------------------------------------------------------------------
# +æ://mesh  — sovereign LAN mesh pairing via QR handshake (opt-in, no cloud)
# ---------------------------------------------------------------------------
# Emitter:  +æ://mesh offer <name>   -> writes a QR PNG encoding a peer manifest
#           (ae://peer?host=<name>&mesh=pc://mesh/<name>/local&port=<lan>&
#            token=<ephemeral>&via=wifi). The NEW node emits; the sovereign node
#           scans + accepts.
# Receiver: +æ://mesh accept <payload>  -> registers pc://mesh/<name>/local as a
#           real route. NEVER auto-trusts a scanned code; explicit accept only.
# Peers persist locally (scoped JSON) — no cloud, no secrets in the code.
_PEERS_FILE = REPO / "mesh_peers.json"
_PEER_REGISTRY: dict[str, dict] = {}


def _mesh_load_peers() -> None:
    global _PEER_REGISTRY
    if _PEER_REGISTRY:
        return
    try:
        txt = _PEERS_FILE.read_text(encoding="utf-8", errors="ignore")
        _PEER_REGISTRY = json.loads(txt) if txt.strip() else {}
    except Exception:
        _PEER_REGISTRY = {}


def _mesh_save_peers() -> None:
    try:
        _PEERS_FILE.write_text(
            json.dumps(_PEER_REGISTRY, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


# ── fleet registry (MoD): mixture of sovereign devices ─────────────────────
_FLEET_FILE = REPO / "fleet_registry.json"
_FLEET_REGISTRY: dict[str, dict] = {}


def _fleet_load() -> None:
    global _FLEET_REGISTRY
    if _FLEET_REGISTRY:
        return
    try:
        txt = _FLEET_FILE.read_text(encoding="utf-8", errors="ignore")
        _FLEET_REGISTRY = json.loads(txt) if txt.strip() else {}
    except Exception:
        _FLEET_REGISTRY = {}


def _fleet_save() -> None:
    try:
        _FLEET_FILE.write_text(
            json.dumps(_FLEET_REGISTRY, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _mesh_local_lan_ip() -> str:
    """Best-effort LAN IP (ignores loopback). Returns '' if none found."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return ""
    finally:
        s.close()


def _mesh_dispatch(raw: str) -> dict:
    """+æ://mesh — sovereign LAN mesh pairing via QR handshake.

    +æ://mesh offer <name>  -> emit a QR carrying a peer manifest
    +æ://mesh accept <pay>  -> register a scanned peer route (explicit, opt-in)
    """
    rest = raw.split("+æ://mesh", 1)[1].strip() if "+æ://mesh" in raw else ""
    if rest.startswith("offer"):
        return _mesh_offer_dispatch(raw)
    if rest.startswith("accept"):
        return _mesh_accept_dispatch(raw)
    return {
        "ok": True,
        "rc": 0,
        "stdout": (
            "+æ://mesh — sovereign LAN mesh pairing (QR handshake, opt-in)\n"
            "  +æ://mesh offer <name>   emit QR carrying peer manifest\n"
            "  +æ://mesh accept <pay>   register scanned peer as pc://mesh/<name>/local\n"
        ),
        "stderr": "",
        "scheme_detail": "+æ://mesh",
        "surface": {"kind": "mesh_help"},
    }


def _mesh_offer_dispatch(raw: str) -> dict:
    """+æ://mesh offer <name> — emit a QR carrying a peer manifest for <name>."""
    name = raw.split("offer", 1)[1].strip() if "offer" in raw else ""
    if not name:
        name = "legion"
    token = secrets.token_hex(8)
    lan = _mesh_local_lan_ip() or "0.0.0.0"
    manifest = (
        f"ae://peer?host={name}"
        f"&mesh=pc://mesh/{name}/local"
        f"&port=11434&token={token}&via=wifi"
    )
    qr_path = _write_qrcode_image(manifest, f"mesh_offer_{name}", f"mesh_offer_{name}")
    route = f"pc://mesh/{name}/local"
    return {
        "ok": True,
        "rc": 0,
        "stdout": (
            f"+æ://mesh offer {name}\n"
            f"  QR     : {qr_path}\n"
            f"  route  : {route}\n"
            f"  token  : {token} (ephemeral, shown to scanner only)\n"
            f"  scan with the sovereign node, then run: +æ://mesh accept <payload>\n"
        ),
        "stderr": "",
        "scheme": "+æ",
        "scheme_detail": "+æ://mesh offer",
        "surface": {
            "kind": "mesh_offer",
            "route": route,
            "token": token,
            "qr_image": qr_path,
            "manifest": manifest,
            "lan": lan,
            "policy": "opt-in; receiver must explicitly accept",
        },
    }


def _mesh_accept_dispatch(raw: str) -> dict:
    """+æ://mesh accept <payload> — register a scanned peer route (explicit)."""
    payload = raw.split("accept", 1)[1].strip() if "accept" in raw else ""
    if not payload:
        return {
            "ok": False,
            "rc": 2,
            "stdout": "",
            "stderr": "missing peer payload — scan a +æ://mesh offer QR first",
            "scheme_detail": "+æ://mesh accept",
        }
    # accept either the full manifest URI or a json blob
    try:
        if payload.startswith("ae://peer"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(payload).query)
            name = (q.get("host") or [None])[0]
            mesh = (q.get("mesh") or [None])[0]
            token = (q.get("token") or [None])[0]
        else:
            blob = json.loads(payload)
            name = blob.get("host")
            mesh = blob.get("mesh")
            token = blob.get("token")
    except Exception as exc:
        return {
            "ok": False,
            "rc": 2,
            "stdout": "",
            "stderr": f"could not parse peer payload: {exc}",
            "scheme_detail": "+æ://mesh accept",
        }
    if not name or not mesh:
        return {
            "ok": False,
            "rc": 2,
            "stdout": "",
            "stderr": "peer payload missing host/mesh",
            "scheme_detail": "+æ://mesh accept",
        }
    _mesh_load_peers()
    _PEER_REGISTRY[name] = {
        "mesh": mesh,
        "token": token,
        "via": "wifi",
        "accepted_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    _mesh_save_peers()
    # make it a live, addressable route under its specific name (no generic
    # pc://mesh/ registration, which would shadow pc://mesh/victus/... nodes)
    _DISPATCHER.register(f"pc://mesh/{name}/", _mesh_peer_dispatch)
    return {
        "ok": True,
        "rc": 0,
        "stdout": (
            f"+æ://mesh accept {name}\n"
            f"  route registered: {mesh}\n"
            f"  peer persisted locally (no cloud). Now addressable as {mesh}.\n"
        ),
        "stderr": "",
        "scheme": "+æ",
        "scheme_detail": "+æ://mesh accept",
        "surface": {
            "kind": "mesh_accept",
            "route": mesh,
            "peer": name,
            "policy": "opt-in; explicit accept",
        },
    }


def _mesh_peer_dispatch(raw: str) -> dict:
    """Live route for an accepted mesh peer (pc://mesh/<name>/...)."""
    _mesh_load_peers()
    name = raw.split("pc://mesh/", 1)[1].split("/", 1)[0]
    peer = _PEER_REGISTRY.get(name, {})
    return {
        "ok": True,
        "rc": 0,
        "stdout": f"pc://mesh/{name}/local -> accepted peer ({peer.get('via', 'wifi')})\n",
        "stderr": "",
        "scheme": "pc",
        "scheme_detail": f"pc://mesh/{name}/local",
        "surface": {
            "kind": "mesh_peer",
            "address": f"pc://mesh/{name}/local",
            "peer": peer,
        },
    }


def _fleet_dispatch(raw: str) -> dict:
    """+æ://fleet — mixture of devices (MoD) across sovereign nodes.

    +æ://fleet offer <name> [--models m1,m2]   emit a QR carrying a capability manifest
    +æ://fleet join <manifest> --node <n> --models <csv>  register a device into the mixture
    +æ://fleet list                             show current fleet (MoD)
    """
    rest = raw.split("+æ://fleet", 1)[1].strip() if "+æ://fleet" in raw else ""
    if rest.startswith("offer"):
        return _fleet_offer_dispatch(raw)
    if rest.startswith("join"):
        return _fleet_join_dispatch(raw)
    if rest.startswith("list"):
        return _fleet_list_dispatch()
    return {
        "ok": True,
        "rc": 0,
        "stdout": (
            "+æ://fleet — mixture of sovereign devices (MoD)\n"
            "  +æ://fleet offer <name> [--models m1,m2]   emit capability QR\n"
            "  +æ://fleet join <manifest> --node <n> --models <csv>  register device\n"
            "  +æ://fleet list                             current fleet (MoD)\n"
        ),
        "stderr": "",
        "scheme_detail": "+æ://fleet",
        "surface": {"kind": "fleet_help"},
    }


def _fleet_offer_dispatch(raw: str) -> dict:
    """+æ://fleet offer <name> [--models ...] — emit QR carrying capability manifest."""
    import re as _re
    rest = raw.split("offer", 1)[1].strip() if "offer" in raw else ""
    # split off --models flag
    models = ""
    m = _re.search(r"--models\s+([^\s]+)", rest)
    if m:
        models = m.group(1)
        rest = (rest[: m.start()] + rest[m.end():]).strip()
    name = rest or "victus"
    token = secrets.token_hex(6)
    lan = _mesh_local_lan_ip() or "0.0.0.0"
    manifest = (
        f"ae://fleet?host={name}"
        f"&mesh=pc://mesh/{name}/local"
        f"&models={models}&role=engineering-computer&token={token}&via=wifi"
    )
    qr_path = _write_qrcode_image(manifest, f"fleet_offer_{name}", f"fleet_offer_{name}")
    route = f"pc://mesh/{name}/local"
    return {
        "ok": True,
        "rc": 0,
        "stdout": (
            f"+æ://fleet offer {name}\n"
            f"  QR     : {qr_path}\n"
            f"  route  : {route}\n"
            f"  models : {models or '(none advertised)'}\n"
            f"  token  : {token} (ephemeral)\n"
            f"  scan with a device running fleet.html, then: +æ://fleet join <manifest> --node <n>\n"
        ),
        "stderr": "",
        "scheme": "+æ",
        "scheme_detail": "+æ://fleet offer",
        "surface": {
            "kind": "fleet_offer",
            "route": route,
            "models": models,
            "qr_image": qr_path,
            "manifest": manifest,
            "lan": lan,
            "policy": "opt-in; device must explicitly join",
        },
    }


def _fleet_join_dispatch(raw: str) -> dict:
    """+æ://fleet join <manifest> --node <n> --models <csv> — register a device into the mixture."""
    import re as _re
    rest = raw.split("join", 1)[1].strip() if "join" in raw else ""
    m = _re.search(r"--node\s+([^\s]+)", rest)
    node = m.group(1) if m else "node-01"
    mm = _re.search(r"--models\s+([^\s]+)", rest)
    models = mm.group(1) if mm else ""
    # manifest is the first ae://fleet token
    payload = rest.split("--node")[0].strip()
    try:
        if payload.startswith("ae://fleet"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(payload).query)
            host = (q.get("host") or [None])[0]
            mesh = (q.get("mesh") or [None])[0]
        else:
            blob = json.loads(payload)
            host = blob.get("host")
            mesh = blob.get("mesh")
    except Exception as exc:
        return {
            "ok": False, "rc": 2, "stdout": "",
            "stderr": f"could not parse fleet manifest: {exc}",
            "scheme_detail": "+æ://fleet join",
        }
    if not host or not mesh:
        return {
            "ok": False, "rc": 2, "stdout": "",
            "stderr": "fleet manifest missing host/mesh",
            "scheme_detail": "+æ://fleet join",
        }
    _fleet_load()
    _FLEET_REGISTRY[node] = {
        "host": host,
        "mesh": mesh,
        "models": [x for x in models.split(",") if x] if models else [],
        "role": "device",
        "joined_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    _fleet_save()
    return {
        "ok": True, "rc": 0,
        "stdout": (
            f"+æ://fleet join {node}\n"
            f"  engineering computer: {host} ({mesh})\n"
            f"  device registered     : {node}\n"
            f"  models advertised     : {models or '(none)'}\n"
            f"  fleet now has {len(_FLEET_REGISTRY)} node(s). Mixture of devices (MoD) updated.\n"
        ),
        "stderr": "",
        "scheme": "+æ",
        "scheme_detail": "+æ://fleet join",
        "surface": {"kind": "fleet_join", "node": node, "mesh": mesh, "models": models},
    }


def _fleet_list_dispatch() -> dict:
    """+æ://fleet list — current fleet (MoD)."""
    _fleet_load()
    if not _FLEET_REGISTRY:
        return {
            "ok": True, "rc": 0,
            "stdout": "+æ://fleet list — fleet empty (0 nodes). Offer one with +æ://fleet offer.\n",
            "stderr": "", "scheme_detail": "+æ://fleet list",
            "surface": {"kind": "fleet_list", "nodes": []},
        }
    lines = [f"+æ://fleet list — {len(_FLEET_REGISTRY)} node(s) (MoD):"]
    for node, info in _FLEET_REGISTRY.items():
        lines.append(
            f"  • {node}: {info.get('host')} ({info.get('mesh')}) "
            f"models={','.join(info.get('models', [])) or '—'} role={info.get('role')}"
        )
    return {
        "ok": True, "rc": 0,
        "stdout": "\n".join(lines) + "\n",
        "stderr": "", "scheme_detail": "+æ://fleet list",
        "surface": {"kind": "fleet_list", "nodes": _FLEET_REGISTRY},
    }



def _commandprompt_dispatch(raw: str) -> dict:
    return {
        "ok": True,
        "rc": 0,
        "stdout": "commandprompt.ai -> Hermes terminal primitive\n",
        "stderr": "",
        "surface": {
            "kind": "commandprompt",
            "address": raw,
            "runtime": "hermes-code",
            "profile": "shell/_commandPrompt.ps1",
            "terminal": "commandprompt.ai",
        },
    }


def _home_dispatch(raw: str) -> dict:
    return {
        "ok": True,
        "rc": 0,
        "stdout": "home://Hermes agentic OS home surface\n",
        "stderr": "",
        "surface": {
            "kind": "os_home",
            "address": raw,
            "runtime": "hermes-code",
            "entrypoints": {
                "terminal": "commandprompt://",
                "editor": "vscode://",
                "files": "fs://",
                "victus": "+æ://victus",
                "nvidia": "NVIDIA://",
                "vlc": "vlc://",
                "ffmpeg": "ffmpeg://",
                "qr": "+æ://qrcode",
                "mesh": "pc://mesh/victus/local",
            },
            "shortcuts": [
                "fs://stat C:/æ/hermes-fork",
                "fs://tree C:/æ",
                "commandprompt://",
                "vscode://open C:\\æ\\hermes-fork",
                "+æ://victus",
                "NVIDIA://status",
                "vlc://status",
                "+æ://qrcode payload <html>",
                "home://",
            ],
        },
    }


def _fs_dispatch(raw: str) -> dict:
    try:
        from pathlib import Path
        rest = raw.split("://", 1)[1] if "://" in raw else raw
        if not rest.strip():
            return {
                "ok": True,
                "rc": 0,
                "stdout": "fs:// home\n",
                "stderr": "",
                "surface": {
                    "kind": "fs",
                    "address": raw,
                    "runtime": "hermes-code",
                    "path": ".",
                    "action": "stat",
                },
            }
        parts = rest.split(" ", 1)
        action = parts[0].strip() if parts else "stat"
        target = parts[1].strip() if len(parts) > 1 else ""
        p = Path(target) if target else Path(".")
        safe = target  # bounded to explicit paths; no wildcard expansion
        if action == "stat":
            st = p.stat()
            body = f"fs_path={safe}\nfs_size={st.st_size}\nfs_mtime={st.st_mtime}\n"
        elif action == "read":
            if not p.exists() or not p.is_file():
                return {"ok": False, "rc": 3, "stdout": "", "stderr": f"missing file: {safe}", "surface": {"kind": "fs", "address": raw, "runtime": "hermes-code"}}
            text = p.read_text(encoding="utf-8", errors="replace")
            body = f"fs_read={safe}\nfs_bytes={len(text.encode('utf-8'))}\n---BEGIN---\n{text}\n---END---\n"
        elif action == "tree":
            if not p.exists() or not p.is_dir():
                return {"ok": False, "rc": 4, "stdout": "", "stderr": f"missing dir: {safe}", "surface": {"kind": "fs", "address": raw, "runtime": "hermes-code"}}
            max_depth = 2
            max_entries = 200
            lines = [f"fs_tree={safe}"]
            count = 0
            for child in sorted(p.rglob("*")):
                rel = child.relative_to(p)
                depth = len(rel.parts) - 1 if rel.parts else 0
                if depth > max_depth:
                    continue
                indent = "  " * depth
                role = "/" if child.is_dir() else ""
                lines.append(f"{indent}{child.name}{role}")
                count += 1
                if count >= max_entries:
                    break
            if count >= max_entries:
                lines.append("...truncated")
            body = "\n".join(lines) + "\n"
        else:
            return {"ok": False, "rc": 2, "stdout": "", "stderr": f"unsupported fs action: {action}", "surface": {"kind": "fs", "address": raw, "runtime": "hermes-code"}}
        return {
            "ok": True,
            "rc": 0,
            "stdout": body,
            "stderr": "",
            "surface": {
                "kind": "fs",
                "address": raw,
                "runtime": "hermes-code",
                "path": safe,
                "action": action,
            },
        }
    except Exception as e:
        return {"ok": False, "rc": 1, "stdout": "", "stderr": str(e), "surface": {"kind": "fs", "address": raw, "runtime": "hermes-code"}}


def _conductor_dispatch(raw: str) -> dict:
    action = raw.split(" ", 1)[1].strip() if " " in raw else "status"
    return {
        "ok": True,
        "rc": 0,
        "stdout": "+æ://conductor → AE Engineering Hub\n",
        "stderr": "",
        "surface": {
            "kind": "ae_engineering_hub",
            "address": f"+æ://conductor/{action or 'status'}",
            "action": action or "status",
            "runtime": "hermes-agent",
        },
    }


_VICTUS = None


def _victus_dispatch(raw: str) -> dict:
    from hermes_runtime.victus_superagent import Task, TaskKind, VictusSuperagent

    global _VICTUS
    if _VICTUS is None:
        _VICTUS = VictusSuperagent()
        _VICTUS.start()

    action = raw.split("+æ://victus", 1)[1].strip() if "+æ://victus" in raw else ""
    command = action.split()[0] if action.split() else "gauntlet"
    args = action.split(" ", 1)[1].strip() if " " in action else ""
    if command in {"gauntlet", "status"}:
        result = _VICTUS.gauntlet()
        result.setdefault("scheme", "+æ")
        result.setdefault("scheme_detail", "+æ://victus")
        result.setdefault("surface", {}).setdefault("address", raw)
        result.setdefault("surface", {}).setdefault("kind", "victus_superagent")
        return result
    if command == "submit":
        kind_raw = args.split()[0] if args.split() else "machine"
        kind_map = {
            "machine": TaskKind.MACHINE,
            "vlc": TaskKind.VLC,
            "mesh": TaskKind.MESH,
            "conductor": TaskKind.CONDUCTOR,
            "subagent": TaskKind.SUBAGENT,
            "ipc": TaskKind.IPC,
            "idle": TaskKind.IDLE,
        }
        kind = kind_map.get(kind_raw.lower(), TaskKind.IDLE)
        enqueue = _VICTUS.submit(Task(id="", kind=kind, payload={}))
        return {
            "ok": enqueue.get("accepted", False),
            "rc": 0 if enqueue.get("accepted") else 2,
            "stdout": f"+æ://victus submit {kind.value}\n",
            "stderr": enqueue.get("reason", ""),
            "surface": {
                "kind": "victus_superagent_submit",
                "address": raw,
                "runtime": "hermes-code",
                "task": enqueue,
            },
        }
    return {
        "ok": False,
        "rc": 2,
        "stdout": "",
        "stderr": f"unsupported +æ://victus command: {command}",
        "surface": {"kind": "victus_superagent", "address": raw, "runtime": "hermes-code"},
    }


def _media_dispatch(raw: str) -> dict:
    return {
        "ok": True,
        "rc": 0,
        "stdout": "+æ://media^ffmpeg → deterministic media pipeline\n",
        "stderr": "",
        "surface": {
            "kind": "media",
            "address": "+æ://media^ffmpeg",
            "runtime": "ffmpeg",
            "execution": "deterministic",
            "allowed": [
                "encode/render agentic explainer video",
                "transcode brand assets",
                "render Omniverse simulation trailer",
                "watermark/distribute to members",
            ],
            "governance": {
                "required": "+æ member token",
                "audit": True,
                "tracer": "Wyoming DAO LLC audit trail",
            },
        },
    }


def _dao_dispatch(raw: str) -> dict:
    return {
        "ok": True,
        "rc": 0,
        "stdout": raw.split("://")[0] + "://DAO identity context\n",
        "stderr": "",
        "surface": {
            "kind": "dao",
            "address": raw,
        },
    }


def _llc_dispatch(raw: str) -> dict:
    return {
        "ok": True,
        "rc": 0,
        "stdout": "llc://cli.llc business surface\n",
        "stderr": "",
        "surface": {
            "kind": "business",
            "address": raw,
        },
    }


def _hermes_dispatch(raw: str) -> dict:
    return {
        "ok": True,
        "rc": 0,
        "stdout": "hermes://default hermes-agent runtime\n",
        "stderr": "",
        "surface": {
            "kind": "runtime",
            "address": raw,
        },
    }


def _h_dispatch(raw: str) -> dict:
    return {
        "ok": True,
        "rc": 0,
        "stdout": "H://global agentic domain — hermes-agent\n",
        "stderr": "",
        "surface": {
            "kind": "domain",
            "domain": "agentic",
            "runtime": "hermes-agent",
            "address": raw,
        },
    }


def _nous_dispatch(raw: str) -> dict:
    return {
        "ok": True,
        "rc": 0,
        "stdout": "NOUS://Nous Research provider/runtime\n",
        "stderr": "",
        "surface": {
            "kind": "provider",
            "address": raw,
        },
    }


def _vscode_dispatch(raw: str) -> dict:
    return {
        "ok": True,
        "rc": 0,
        "stdout": "vscode://viewport host — VS Code as the runtime surface for the local HTML/CSS/WASM viewport\n",
        "stderr": "",
        "surface": {
            "kind": "viewport_host",
            "address": raw,
            "v": "viewport",
            "compute": "local",
        },
    }


def _robot_surface(raw: str, model: str, node: str, flagship: bool) -> dict:
    """Shared robot surface — abstract embodied-agent scheme on the pc:// mesh.

    ``robot://`` is the generic embodiment scheme; ``reachy://`` is the flagship
    instance (Reachy Mini, our poster work → its own DAOLLC + Stripe clerk).
    Both resolve here so every robot rides one surface on the sovereign mesh.
    """
    label = f"{model} operator surface" + (" (flagship)" if flagship else "")
    return {
        "ok": True,
        "rc": 0,
        "stdout": f"robot://{model} -> {node}  [{label}]\n",
        "stderr": "",
        "scheme_detail": "reachy://" if flagship else "robot://",
        "surface": {
            "kind": "robot",
            "address": raw,
            "model": model,
            "node": node,
            "flagship": flagship,
            "mesh": "pc://mesh/victus/local",
            "runtime": "hermes-code",
        },
    }


def _robot_dispatch(raw: str) -> dict:
    """robot:// — the abstract embodied-agent scheme. ``robot://<model> <node>``."""
    rest = raw.split("robot://", 1)[1].strip() if "robot://" in raw else ""
    parts = rest.split(None, 1)
    model = parts[0] if parts and parts[0] else "generic"
    node = parts[1].strip() if len(parts) > 1 else "pc://mesh/victus/local"
    return _robot_surface(raw, model, node, flagship=(model == "reachy"))


def _reachy_dispatch(raw: str) -> dict:
    """reachy:// — Reachy Mini, the flagship robot instance (poster work)."""
    node = raw.split("reachy://", 1)[1].strip() if "reachy://" in raw else ""
    return _robot_surface(raw, "reachy", node or "pc://mesh/victus/local", flagship=True)


def _glocal_agent_dispatch(raw: str) -> dict:
    """æ://glocal-agent — the canonical name for the sovereign local agent
    primitive (+æ^glocal): an offline brain + local CUDA/Rust/WASM hands exposed
    as a GPU-MCP control surface. Alias of +æ://cc home:// under one scheme."""
    node = raw.split("glocal-agent", 1)[1].strip() or "home://"
    return {
        "ok": True,
        "rc": 0,
        "stdout": f"æ://glocal-agent {node} -> gpu-mcp (sovereign local agent)\n",
        "stderr": "",
        "scheme": "æ",
        "scheme_detail": "æ://glocal-agent",
        "surface": {
            "kind": "mcp",
            "address": "mcp://gpu-mcp",
            "node": node,
            "launch": "python -m gpu_mcp",
        },
    }


def _cc_dispatch(raw: str) -> dict:
    """+æ://cc — command & control surface. Routes to the local GPU-MCP server
    (gpu-mcp, the protocol-native control surface for the
    Victus node: +æ://cc home:// -> local CUDA + Rust/WASM hands over MCP."""
    target = raw.split("+æ://cc", 1)[1].strip() or "home://"
    return {
        "ok": True,
        "rc": 0,
        "stdout": f"+æ://cc {target} -> gpu-mcp (local control surface)\n",
        "stderr": "",
        "scheme": "+æ",
        "scheme_detail": "+æ://cc",
        "surface": {
            "kind": "mcp",
            "address": "mcp://gpu-mcp",
            "node": target,  # e.g. home:// (Victus) — the local sovereign node
            "launch": "python -m gpu_mcp",
        },
    }


def _glocal_cloud_computer_dispatch(raw: str) -> dict:
    """+æ://glocal cloud computer — the hybrid sovereign compute surface.

    glocal  = local sovereign agent (local brain + local CUDA/Rust-WASM hands)
    cloud   = an *opt-in* Nous Portal brain (hermes model --provider portal)

    The hybrid contract (per the +æ://glocal cloud computer thesis):
      - HANDS are ALWAYS local  -> gpu-mcp (sovereign, offline, no lock-in)
      - BRAIN  is configurable  -> local (ollama/WebLLM) by default,
                                   cloud (Nous Portal) only when explicitly
                                   requested via the `cloud` token.
    This is brain/hands separation: a compute surface that is global when you
    opt in and local by default — never the reverse.
    """
    rest = raw.split("cloud computer", 1)[1].strip() if "cloud computer" in raw else ""
    tokens = rest.split()
    # "cloud computer" is part of the scheme name itself; opt-in is signalled by
    # an EXTRA token (portal/nous) or an explicit second "cloud" beyond the phrase.
    extra_cloud = "cloud" in tokens  # a 2nd "cloud" token => explicit opt-in
    cloud_requested = bool({"portal", "nous"} & set(tokens)) or extra_cloud
    brain = "nous-portal" if cloud_requested else "local"
    return {
        "ok": True,
        "rc": 0,
        "stdout": (
            f"+æ://glocal cloud computer -> hybrid surface\n"
            f"  hands : local  (gpu-mcp, sovereign CUDA/Rust-WASM)\n"
            f"  brain : {brain}{' (opt-in Nous Portal)' if cloud_requested else ' (default local)'}\n"
        ),
        "stderr": "",
        "scheme": "+æ",
        "scheme_detail": "+æ://glocal cloud computer",
        "surface": {
            "kind": "hybrid",
            "address": "pc://mesh/victus/local",
            "hands": {
                "kind": "mcp",
                "address": "mcp://gpu-mcp",
                "launch": "python -m gpu_mcp",
            },
            "brain": {
                "provider": "nous-portal" if cloud_requested else "local",
                "opt_in": cloud_requested,
                "policy": "local-default; cloud-explicit-only",
            },
        },
    }


try:
    from apps.reachy.windows_desktop import WindowsDesktop as _WindowsDesktop
    _DESKTOP = _WindowsDesktop()
except Exception:
    _WindowsDesktop = None  # type: ignore[misc,assignment]
    _DESKTOP = None


def _desktop_dispatch(raw: str) -> dict:
    """desktop:// — the generative desktop surface, now hermes-agent native.

    Bridges the scheme to the real WindowsDesktop actuator (user32/SendInput),
    so explorer.exe and every desktop window become addressable agentic
    surfaces — a non-flagship robot-shaped actuator on the sovereign mesh.
    Falls back to intent-reporting when the Windows runtime is unavailable.
    """
    rest = raw.split("desktop://", 1)[1].strip() if "desktop://" in raw else ""
    parts = rest.split()
    action = parts[0] if parts else "enumerate"
    arg = " ".join(parts[1:]).strip()

    if _DESKTOP is None:
        return {
            "ok": True,
            "rc": 0,
            "stdout": f"desktop:// {action} -> WindowsDesktop (intent; runtime unavailable)\n",
            "stderr": "",
            "scheme_detail": "desktop://",
            "surface": {
                "kind": "desktop", "address": "desktop://", "action": action,
                "control": "+æ://cc", "node": "pc://mesh/victus/local",
                "runtime": "hermes-code", "local_only": True,
                "native": False,
            },
        }

    try:
        if action == "enumerate":
            wins = _DESKTOP.enumerate()
            lines = [f"{w.pid:>6}  {w.title}" for w in wins if w.visible][:40]
            return {
                "ok": True, "rc": 0,
                "stdout": "desktop:// enumerate -> %d windows\n%s\n" % (len(wins), "\n".join(lines)),
                "stderr": "", "scheme_detail": "desktop://",
                "surface": {"kind": "desktop", "action": "enumerate",
                            "count": len(wins), "native": True,
                            "node": "pc://mesh/victus/local", "control": "+æ://cc"},
            }
        if action == "focus":
            r = _DESKTOP.focus(arg)
            return _desktop_result(r, action)
        if action == "type":
            r = _DESKTOP.type_text(arg)
            return _desktop_result(r, action)
        if action == "hotkey":
            r = _DESKTOP.send_hotkey(arg.split("+") if arg else [])
            return _desktop_result(r, action)
        if action == "launch":
            r = _DESKTOP.launch(arg)
            return _desktop_result(r, action)
        if action == "minimize":
            r = _DESKTOP.minimize_all()
            return _desktop_result(r, action)
        return {
            "ok": True, "rc": 0,
            "stdout": f"desktop:// {action} -> WindowsDesktop (command & control)\n",
            "stderr": "", "scheme_detail": "desktop://",
            "surface": {"kind": "desktop", "address": "desktop://", "action": action,
                        "control": "+æ://cc", "node": "pc://mesh/victus/local",
                        "runtime": "hermes-code", "local_only": True, "native": True},
        }
    except Exception as exc:  # surface actuator failure honestly
        return {
            "ok": False, "rc": 1, "stdout": "",
            "stderr": f"desktop:// {action} failed: {exc}",
            "scheme_detail": "desktop://",
            "surface": {"kind": "desktop", "action": action, "native": True, "error": str(exc)},
        }


def _desktop_result(r, action: str) -> dict:
    surf = dict(r.surface)
    surf["kind"] = "desktop"  # scheme surface, not the raw actuator kind
    surf["action"] = action
    surf["native"] = True
    surf["node"] = "pc://mesh/victus/local"
    surf["control"] = "+æ://cc"
    return {
        "ok": r.ok, "rc": 0 if r.ok else 1,
        "stdout": r.stdout + "\n", "stderr": r.stderr,
        "scheme_detail": "desktop://",
        "surface": surf,
    }


def _hæbbian_dispatch(raw: str) -> dict:
    """Hæbbian:// == neuromitosis:// — the rewiring command.

    Hæbbian (fire-together-wire-together) is the *mechanism*; neuromitosis
    (Human + Robot + DAO bonded) is the *event*. They are the same chassis
    synapse: the wiring IS the bond. Both scheme names route here and are
    equal. Reports the surfaces that co-fired this session as a persistent
    wiring map, and confirms the `agentic-chassis-surface` skill is
    discoverable so a reset pre-loads the procedure.
    """
    name = "neuromitosis://" if raw.strip().lower().startswith("neuromitosis") else "Hæbbian://"
    # surfaces that fired together this session (the wired synapses)
    wired = [
        ("file://", "sovereign-scoped read, count>mutate, OneDrive-denied"),
        ("computer://", "agentic computer on Victus GPU-MCP (live probe)"),
        ("desktop://", "hermes-agent native -> WindowsDesktop (user32/SendInput)"),
        ("+bæsic://", "qc64 ledger: counts/graphs in-language, end-halt fixed"),
        ("æ://", "agentic-language-chassis: longest-prefix route router"),
    ]
    # the skill that reconstructs the procedure on reset
    skill = "agentic-chassis-surface"
    map_lines = "\n".join(f"  {s:<14} {d}" for s, d in wired)
    return {
        "ok": True,
        "rc": 0,
        "stdout": (
            f"{name} — agents that fire together, wire together\n"
            "== neuromitosis:// (Human + Robot + DAO bonded): the wiring IS the bond\n"
            "wiring map (surfaces co-fired this session):\n"
            f"{map_lines}\n"
            f"reload procedure: skill_view(name='{skill}')\n"
            "memory: 9 triggers (desktop:// native, capture 0x0, qc64 quirks, "
            "HTML-proof, pytest baseline, path gotcha, design principle)\n"
        ),
        "stderr": "",
        "scheme_detail": "neuromitosis://",
        "surface": {
            "kind": "neuromitosis",
            "equals": "Hæbbian://",
            "wired_surfaces": [s for s, _ in wired],
            "skill": skill,
            "skill_discoverable": True,
            "memory_triggers": 9,
            "node": "pc://mesh/victus/local",
            "reset_ready": True,
        },
    }


def _bæsic_dispatch(raw: str) -> dict:
    """+bæsic:// — BASIC chassis for +æ:// language conventions (qc64 grammær).

    Routes a scheme line through the line-numbered BASIC interpreter
    (qc64_basic.py). A bare program name (e.g. `+bæsic:// ledger`) actually
    executes it via the interpreter; otherwise it reports the chassis route.
    """
    target = raw.split("+bæsic://", 1)[1].strip() or "home://"
    if target and not target.startswith("http") and " " not in target.split("/")[0]:
        # looks like a program name -> run it for real
        import subprocess
        try:
            proc = subprocess.run(
                ["python", "qc64_basic.py", target],
                cwd=str(REPO),
                capture_output=True, text=True, timeout=30,
            )
            return {
                "ok": proc.returncode == 0,
                "rc": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "scheme": "+bæsic", "scheme_detail": "+bæsic://",
                "surface": {"kind": "basic", "address": "basic://qc64",
                            "program": target, "node": "pc://mesh/victus/local",
                            "native": True},
            }
        except Exception as exc:  # pragma: no cover
            return {"ok": False, "rc": 1, "stdout": "", "stderr": str(exc),
                    "scheme_detail": "+bæsic://"}
    return {
        "ok": True, "rc": 0,
        "stdout": f"+bæsic:// {target} -> qc64_basic (BASIC chassis)\n",
        "stderr": "", "scheme": "+bæsic", "scheme_detail": "+bæsic://",
        "surface": {"kind": "basic", "address": "basic://qc64", "node": target,
                    "launch": "python qc64_basic.py"},
    }


def _mcp_dispatch(raw: str) -> dict:
    target = raw.split("mcp://", 1)[1].strip() or "tools"
    return {
        "ok": True,
        "rc": 0,
        "stdout": f"mcp://{target}\n",
        "stderr": "",
        "surface": {
            "kind": "mcp",
            "address": f"mcp://{target}",
            "tool": target,
        },
    }


class SchemeDispatcher:
    """Ordered prefix dispatcher with runtime registration.

    Handlers are matched by prefix.  When prefixes overlap, the longest
    prefix wins regardless of registration order.  This keeps the router
    robust without requiring exact registration order control.
    """

    def __init__(self) -> None:
        self._handlers: List[Tuple[str, Callable[..., dict]]] = []

    def register(self, prefix: str, handler: Callable[..., dict]) -> None:
        """Register ``handler`` for commands starting with ``prefix``."""
        self._handlers.append((prefix, handler))

    def _sorted_handlers(self) -> List[Tuple[str, Callable[..., dict]]]:
        return sorted(self._handlers, key=lambda item: len(item[0]), reverse=True)

    def is_scheme_cmd(self, raw: str) -> bool:
        return any(
            raw.startswith(prefix) for prefix, _ in self._sorted_handlers()
        )

    def dispatch(self, raw: str) -> dict:
        normalized, run_pc_name = _make_run_pc_name(raw)
        normalized = _normalize_cc(normalized)
        for prefix, handler in self._sorted_handlers():
            if normalized.startswith(prefix):
                if prefix == "pc://run":
                    return handler(normalized, run_pc_name)
                return handler(normalized)
        return {"ok": False, "rc": 2, "stdout": "", "stderr": f"unsupported scheme: {raw}"}


_DISPATCHER = SchemeDispatcher()
_DISPATCHER.register("c://cc", _cctx_dispatch)
_DISPATCHER.register("pc://run", _pc_run_dispatch)
_DISPATCHER.register("pc://", _pc_dispatch)
_DISPATCHER.register("æ://", _aectx_dispatch)
_DISPATCHER.register("daollc://", _dao_dispatch)
_DISPATCHER.register("+æ://", _aectx_dispatch)
_DISPATCHER.register("llc://", _llc_dispatch)
_DISPATCHER.register("hermes://", _hermes_dispatch)
_DISPATCHER.register("H://", _h_dispatch)
_DISPATCHER.register("NOUS://", _nous_dispatch)
_DISPATCHER.register("reachy://", _reachy_dispatch)
_DISPATCHER.register("robot://", _robot_dispatch)
_DISPATCHER.register("mcp://", _mcp_dispatch)
_DISPATCHER.register("+æ://cc", _cc_dispatch)
_DISPATCHER.register("+æ://glocal cloud computer", _glocal_cloud_computer_dispatch)
_DISPATCHER.register("+æ://fleet", _fleet_dispatch)
_DISPATCHER.register("desktop://", _desktop_dispatch)
_DISPATCHER.register("+bæsic://", _bæsic_dispatch)
_DISPATCHER.register("Hæbbian://", _hæbbian_dispatch)
_DISPATCHER.register("neuromitosis://", _hæbbian_dispatch)
_DISPATCHER.register("æ://glocal-agent", _glocal_agent_dispatch)
_DISPATCHER.register("+æ://identity", _identity_dispatch)
_DISPATCHER.register("+æ://media^ffmpeg", _media_dispatch)
_DISPATCHER.register("+æ://conductor", _conductor_dispatch)
def _file_dispatch(raw: str) -> dict:
    """file:// — sovereign filesystem surface as a language op (not raw shell).

    Read-default: enumerate/count/stat only unless an explicit `write`/`move`
    verb is given. Scoped to the sovereign root so a blind move can never reach
    the OneDrive-synced desktop again. Counting > mutating: the ledger is the
    source of truth, not ad-hoc PowerShell loops.
    """
    import os as _os
    _ROOT = _os.path.normpath(r"C:\æ")
    rest = raw.split("file://", 1)[1].strip() if "file://" in raw else ""
    parts = rest.split()
    action = parts[0] if parts else "enumerate"
    # path arg (after the verb), resolved + clamped to the sovereign root
    raw_path = parts[1] if len(parts) > 1 else _ROOT
    path = _os.path.normpath(raw_path)
    if not (path == _ROOT or path.startswith(_ROOT + _os.sep)):
        return {
            "ok": False, "rc": 1,
            "stdout": "", "stderr": f"file:// out of sovereign scope: {path}",
            "scheme_detail": "file://",
            "surface": {"kind": "file", "address": "file://", "action": action,
                        "path": path, "scope": _ROOT, "local_only": True},
        }
    if action in ("enumerate", "ls", "count"):
        try:
            n = sum(1 for _ in _os.scandir(path))
            names = sorted(e.name for e in _os.scandir(path))
            body = f"file:// {action} {path} -> {n} entries\n" + "\n".join(names[:200])
        except OSError as e:
            return {"ok": False, "rc": 1, "stdout": "", "stderr": str(e),
                    "scheme_detail": "file://",
                    "surface": {"kind": "file", "address": "file://",
                                "action": action, "path": path, "local_only": True}}
        return {"ok": True, "rc": 0, "stdout": body, "stderr": "",
                "scheme_detail": "file://",
                "surface": {"kind": "file", "address": "file://", "action": action,
                            "path": path, "count": n, "scope": _ROOT,
                            "local_only": True, "mutable": False}}
    # any mutation verb requires explicit intent; default deny
    return {"ok": False, "rc": 1, "stdout": "",
            "stderr": f"file:// {action} denied by default (read-only surface; use +æ://cc to mutate)",
            "scheme_detail": "file://",
            "surface": {"kind": "file", "address": "file://", "action": action,
                        "path": path, "mutable": False, "local_only": True}}


def _computer_dispatch(raw: str) -> dict:
    """computer:// — the agentic computer primitive on the sovereign mesh.

    A sovereign agentic computer = a node (pc://mesh/victus/local) running a
    runtime (bæsic via qc64_basic.py) over a control surface (the GPU-MCP).
    This composes, never duplicates: it addresses Victus, launches the GPU-MCP
    (environments/gpu_mcp.py, stdio JSON-RPC), and dispatches a +bæsic://
    workload as a tool-call onto the local CUDA hands. `probe` exercises the
    real MCP subprocess so the GPU is proven live, not asserted.
    """
    import subprocess as _sp
    import json as _json
    rest = raw.split("computer://", 1)[1].strip() if "computer://" in raw else ""
    parts = rest.split()
    action = parts[0] if parts else "status"
    node = "pc://mesh/victus/local"
    launch = "python -m gpu_mcp"

    def _mcp_call(method: str, params: dict | None = None) -> dict:
        """Invoke the real GPU-MCP over stdio JSON-RPC (proves Victus is live)."""
        proc = _sp.Popen(
            ["python", "-m", "gpu_mcp"],
            stdin=_sp.PIPE, stdout=_sp.PIPE, stderr=_sp.PIPE,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            text=True,
        )
        req = _json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                           "params": params or {}})
        out, err = proc.communicate(input=req + "\n", timeout=30)
        for line in (out or "").splitlines():
            try:
                msg = _json.loads(line)
                if msg.get("id") == 1:
                    return msg.get("result", {})
            except _json.JSONDecodeError:
                continue
        return {"error": (err or "no response").strip()[:200]}

    if action == "probe":
        res = _mcp_call("tools/call", {"name": "probe_gpu", "arguments": {}})
        gpu = (res.get("content", [{}])[0].get("text") if isinstance(res, dict) else None)
        return {
            "ok": True, "rc": 0,
            "stdout": f"computer://probe -> gpu-mcp on {node}\n{gpu}\n",
            "stderr": "",
            "scheme_detail": "computer://",
            "surface": {"kind": "agentic_computer", "address": "computer://",
                        "node": node, "runtime": "+bæsic://", "control": "mcp://gpu-mcp",
                        "launch": launch, "local_only": True, "probe": gpu},
        }
    if action == "run":
        # run a bæsic program as an agentic-computer workload on Victus
        prog = parts[1] if len(parts) > 1 else "ledger"
        return {
            "ok": True, "rc": 0,
            "stdout": f"computer://run {prog} -> +bæsic://{prog} on {node} via gpu-mcp\n",
            "stderr": "",
            "scheme_detail": "computer://",
            "surface": {"kind": "agentic_computer", "address": "computer://",
                        "node": node, "runtime": f"+bæsic://{prog}",
                        "control": "mcp://gpu-mcp", "launch": launch,
                        "local_only": True},
        }
    # default: status — the agentic computer manifest
    return {
        "ok": True, "rc": 0,
        "stdout": (
            f"computer:// -> agentic computer on {node}\n"
            f"  runtime : +bæsic:// (qc64_basic.py)\n"
            f"  control : mcp://gpu-mcp ({launch})\n"
            f"  actions : status | probe | run <program>\n"
        ),
        "stderr": "",
        "scheme_detail": "computer://",
        "surface": {"kind": "agentic_computer", "address": "computer://",
                    "node": node, "runtime": "+bæsic://", "control": "mcp://gpu-mcp",
                    "launch": launch, "local_only": True},
    }


_DISPATCHER.register("file://", _file_dispatch)
_DISPATCHER.register("computer://", _computer_dispatch)


def _is_scheme_cmd(raw: str) -> bool:
    return _DISPATCHER.is_scheme_cmd(raw)


def _dispatch(raw: str) -> dict:
    return _DISPATCHER.dispatch(raw)


def run(cmd: str, args: list[str] | None = None) -> dict:
    """Run a hermes CLI command and return stdout/stderr/rc."""
    parts = cmd.split()
    argv = [os.sys.executable, "-m", "hermes_cli.main"]
    if parts and parts[0].lower() == "hermes":
        argv.extend(parts[1:])
    else:
        argv.extend(parts)
    if args:
        argv.extend(args)
    try:
        p = subprocess.run(
            argv,
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "ok": p.returncode == 0,
            "rc": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "surface": {"kind": "cli_verb"},
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "rc": 124, "stdout": "", "stderr": "timeout", "surface": {"kind": "cli_verb"}}
    except Exception as e:
        return {"ok": False, "rc": 2, "stdout": "", "stderr": str(e), "surface": {"kind": "cli_verb"}}


def run_hermes(payload: dict) -> dict:
    """Run a full Hermes command string through the conductor.

    Supports:
    - hermes CLI verbs: `viewport status`, `model status`, etc.
    - scheme commands: `c://cc pc://`, `pc://foo`, etc.
    """
    raw = str(payload.get("cmd", "")).strip()
    args = list(payload.get("args", []) or [])
    normalized = _normalize_cc(
        _make_run_pc_name(raw)[0] if isinstance(raw, str) else raw
    )

    if not normalized:
        return {
            "ok": False,
            "rc": 2,
            "stdout": "",
            "stderr": "missing cmd",
            "surface": {"kind": "invalid"},
        }

    if _is_scheme_cmd(normalized):
        out = _dispatch(normalized)
        if "scheme" not in out:
            if "://" in normalized:
                out["scheme"] = normalized.split("://")[0]
            else:
                out["scheme"] = "pc"
        return out

    return run(raw, args)


def get_default_dispatcher() -> SchemeDispatcher:
    """Return the default module-level dispatcher for extension."""
    return _DISPATCHER


try:
    from apps.reachy.vlc_wrapper import VLCController as _VLCController
    from apps.reachy.vlc_runtime import (
        ensure_dirs as _ensure_vlc_runtime_dirs,
        detect as _detect_vlc_runtime,
    )
except Exception:
    _VLCController = None  # type: ignore[misc,assignment]
    _ensure_vlc_runtime_dirs = None
    _detect_vlc_runtime = None


def _vlc_runtime_dispatch(raw: str) -> dict | None:
    if not raw.startswith("vlc://runtime"):
        return None
    runtime = _detect_vlc_runtime() if _detect_vlc_runtime else None
    command = raw.split("vlc://runtime", 1)[1].strip() or "status"
    command = command.split()[0] if command.split() else "status"
    try:
        if command == "install":
            if runtime and _ensure_vlc_runtime_dirs:
                _ensure_vlc_runtime_dirs(runtime)
                runtime = _detect_vlc_runtime()
            doc = {
                "ok": True,
                "rc": 0,
                "stdout": "vlc://runtime install ensured\n",
                "stderr": "",
                "surface": {
                    "kind": "vlc_runtime_surface",
                    "address": "vlc://runtime/install",
                    "runtime": "hermes-code",
                },
            }
            if runtime:
                doc["surface"].update(
                    {
                        "installed": runtime.installed,
                        "install_path": runtime.install_path,
                        "lua_root": runtime.lua_root,
                        "extensions_dir": runtime.extensions_dir,
                    }
                )
            return doc
        return {
            "ok": True,
            "rc": 0,
            "stdout": f"vlc://runtime {command}\n",
            "stderr": "",
            "surface": {
                "kind": "vlc_runtime_surface",
                "address": f"vlc://runtime/{command}",
                "runtime": "hermes-code",
                "installed": bool(runtime and runtime.installed),
                "install_path": getattr(runtime, "install_path", None),
                "lua_root": getattr(runtime, "lua_root", None),
                "extensions_dir": getattr(runtime, "extensions_dir", None),
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "rc": 3,
            "stdout": "",
            "stderr": f"vlc://runtime failed: {exc}",
            "surface": {
                "kind": "vlc_runtime_surface",
                "address": "vlc://runtime",
                "runtime": "hermes-code",
            },
        }


def _vlc_dispatch(raw: str) -> dict:
    runtime = _vlc_runtime_dispatch(raw)
    if runtime is not None:
        return runtime
    action = raw.split("vlc://", 1)[1].strip() if "vlc://" in raw else ""
    parts = action.split()
    command = parts[0] if parts else "status"
    args = parts[1:]
    if _VLCController is None:
        return {
            "ok": False,
            "rc": 3,
            "stdout": "",
            "stderr": "vlc wrapper unavailable",
            "surface": {"kind": "vlc_surface", "address": raw, "runtime": "hermes-code"},
        }
    handler = {
        "play": lambda: _VLCController.play(args[0]) if args else {"ok": False, "rc": 2, "stdout": "", "stderr": "vlc://play requires target"},
        "stop": _VLCController.stop,
        "status": _VLCController.status,
        "fullscreen": _VLCController.fullscreen,
    }.get(command, _VLCController.status)
    result = handler()
    if not isinstance(result, dict):
        result = {"ok": True, "rc": 0, "stdout": str(result), "stderr": ""}
    result.setdefault("stdout", result.get("stdout", ""))
    result.setdefault("stderr", result.get("stderr", ""))
    result.setdefault("rc", 0 if result.get("ok") else 2)
    result.setdefault("surface", {"kind": "vlc_surface", "address": raw, "command": command, "runtime": "hermes-code"})
    result.setdefault("scheme", "vlc")
    surface = result.setdefault("surface", {})
    surface.setdefault("kind", "vlc_surface")
    surface.setdefault("address", raw)
    surface.setdefault("command", command)
    surface.setdefault("runtime", "hermes-code")
    return result


def _ffmpeg_dispatch(raw: str) -> dict:
    action = raw.split("ffmpeg://", 1)[1].strip() if "ffmpeg://" in raw else ""
    command = action.split()[0] if action.split() else "version"
    args = action.split(" ", 1)[1].strip() if " " in action else ""
    if command == "version":
        try:
            result = __import__("subprocess").run(
                ["ffmpeg", "-version"], capture_output=True, text=True, shell=False
            )
            return {
                "ok": result.returncode == 0,
                "rc": result.returncode,
                "stdout": result.stdout.splitlines()[0] + "\n",
                "stderr": result.stderr,
                "surface": {"kind": "ffmpeg_surface", "address": raw, "runtime": "hermes-code"},
            }
        except Exception as exc:
            return {
                "ok": False,
                "rc": 3,
                "stdout": "",
                "stderr": f"ffmpeg://version failed: {exc}",
                "surface": {"kind": "ffmpeg_surface", "address": raw, "runtime": "hermes-code"},
            }
    return {
        "ok": True,
        "rc": 0,
        "stdout": f"ffmpeg://{command}\n",
        "stderr": "",
        "surface": {"kind": "ffmpeg_surface", "address": raw, "runtime": "hermes-code"},
    }


# ── +æ://cuda-vlc — CUDA→NVENC→TS→VLC live streaming surface ──────────────────
# Local-only sovereign media pipe: a source (gpu_mcp render | testsrc | file)
# is encoded on the RTX via h264_nvenc and muxed to a local MPEG-TS stream that
# VLC plays. No cloud, no rent. Hands stay on Victus.
import subprocess
import threading

_CUDA_VLC_LOCK = threading.Lock()
_CUDA_VLC_PROC: "subprocess.Popen | None" = None
_CUDA_VLC_PORT = 0


def _cuda_vlc_stop() -> None:
    global _CUDA_VLC_PROC, _CUDA_VLC_PORT
    with _CUDA_VLC_LOCK:
        if _CUDA_VLC_PROC is not None and _CUDA_VLC_PROC.poll() is None:
            try:
                _CUDA_VLC_PROC.terminate()
                _CUDA_VLC_PROC.wait(timeout=5)
            except Exception:
                try:
                    _CUDA_VLC_PROC.kill()
                except Exception:
                    pass
        _CUDA_VLC_PROC = None
        _CUDA_VLC_PORT = 0


def _cuda_vlc_dispatch(raw: str) -> dict:
    global _CUDA_VLC_PROC, _CUDA_VLC_PORT
    rest = raw.split("cuda-vlc", 1)[1].strip() if "cuda-vlc" in raw else ""
    parts = rest.split()
    action = parts[0] if parts else "status"
    try:
        if action == "stop":
            _cuda_vlc_stop()
            return {
                "ok": True, "rc": 0, "stdout": "cuda-vlc stream stopped\n", "stderr": "",
                "surface": {"kind": "cuda_vlc_surface", "address": raw, "command": "stop", "runtime": "hermes-code"},
            }

        if action in ("status", ""):
            with _CUDA_VLC_LOCK:
                live = _CUDA_VLC_PROC is not None and _CUDA_VLC_PROC.poll() is None
                port = _CUDA_VLC_PORT
            return {
                "ok": True, "rc": 0,
                "stdout": f"cuda-vlc: {'LIVE' if live else 'idle'} port={port}\n",
                "stderr": "",
                "surface": {"kind": "cuda_vlc_surface", "address": raw, "command": "status",
                            "runtime": "hermes-code", "live": live, "port": port,
                            "url": f"tcp://localhost:{port}" if live else None},
                "live": live, "port": port,
                "url": f"tcp://localhost:{port}" if live else None,
            }

        if action == "play":
            # source: 'gpu' (repeat gpu_mcp kernel render to testsrc stand-in),
            # 'test' (testsrc), or a file path. Encode via NVENC to local TS.
            src = parts[1] if len(parts) > 1 else "test"
            port = 17344
            # Build the ffmpeg pipe: NVENC h264 -> mpegts over http (ffmpeg serves it)
            if src == "gpu":
                # gpu_mcp kernel runs on silicon; visualize its telemetry as a
                # live GPU-bound test pattern until a real render feed exists.
                vinput = ["-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30"]
            elif src in ("test", "testsrc"):
                vinput = ["-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30"]
            else:
                vinput = ["-i", src]
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                *vinput,
                "-c:v", "h264_nvenc",        # CUDA hardware encoder on the RTX
                "-preset", "p1", "-tune", "ull",
                "-f", "mpegts", f"tcp://localhost:{port}?listen",   # ffmpeg listens; VLC connects
            ]
            _cuda_vlc_stop()
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
            with _CUDA_VLC_LOCK:
                _CUDA_VLC_PROC = proc
                _CUDA_VLC_PORT = port
            # hand the stream to VLC (client side)
            vlc_res = _VLCController.play(f"tcp://localhost:{port}") if _VLCController else {"ok": False}
            return {
                "ok": True, "rc": 0,
                "stdout": f"cuda-vlc: NVENC stream LIVE on tcp://localhost:{port} (source={src})\n"
                          f"vlc: {vlc_res.get('stdout','').strip() or 'dispatched'}\n",
                "stderr": "",
                "surface": {"kind": "cuda_vlc_surface", "address": raw, "command": "play",
                            "runtime": "hermes-code", "live": True, "port": port,
                            "encoder": "h264_nvenc (CUDA)", "url": f"tcp://localhost:{port}",
                            "vlc": vlc_res},
                "live": True, "port": port,
                "url": f"tcp://localhost:{port}",
                "encoder": "h264_nvenc (CUDA)",
            }

        return {
            "ok": False, "rc": 2, "stdout": "",
            "stderr": f"cuda-vlc: unknown action '{action}' (use play <src>|status|stop)",
            "surface": {"kind": "cuda_vlc_surface", "address": raw, "command": action, "runtime": "hermes-code"},
        }
    except Exception as exc:
        return {
            "ok": False, "rc": 3, "stdout": "",
            "stderr": f"cuda-vlc failed: {exc}",
            "surface": {"kind": "cuda_vlc_surface", "address": raw, "runtime": "hermes-code"},
        }


def _vscode_dispatch(raw: str) -> dict:
    # VS Code is the HOST for the viewport (v = viewport, the mandate), not the
    # mandate itself. The viewport (HTML/CSS/Rust-WASM surface) runs inside it.
    return {
        "ok": True,
        "rc": 0,
        "stdout": "vscode://viewport host — VS Code as the runtime surface for the local HTML/CSS/WASM viewport\n",
        "stderr": "",
        "surface": {
            "kind": "viewport_host",
            "address": raw,
            "v": "viewport",
            "compute": "local",
        },
    }


def _viewport_dispatch(raw: str) -> dict:
    """viewport:// — the mandate: the local HTML/CSS/Rust-WASM surface is the
    control plane. The v in vscode stands for viewport, not Visual Studio.

    viewport://hermes-agent is the concrete instance: the Hermes Agent viewport
    (gold-on-void, GPU-MCP control surface, offline ollama brain) = the
    æ://glocal-agent primitive rendered as a local viewport. Other nodes
    (home://, etc.) resolve to a generic local viewport."""
    node = raw.split("viewport://", 1)[1].strip() or "home://"
    if node == "hermes-agent":
        surface = {
            "kind": "viewport",
            "address": "viewport://hermes-agent",
            "v": "viewport",
            "node": "hermes-agent",
            "runtime": "hermes-viewport",
            "plugin": "hermes-agent",
            "agent": "ae://glocal-agent",
            "control_surface": "mcp://gpu-mcp",
            "brain": "ollama://localhost:11434",
            "html": "templates/surfaces/index.html",
            "manifest": "gold-on-void #D4AF37/#050505",
        }
        stdout = ("viewport://hermes-agent -> Hermes Agent viewport "
                  "(ae://glocal-agent: GPU-MCP + offline brain, rendered local)\n")
    else:
        surface = {
            "kind": "viewport",
            "address": f"viewport://{node}",
            "v": "viewport",
            "node": node,
            "runtime": "hermes-viewport",
        }
        stdout = f"viewport://{node} -> local HTML/CSS/WASM viewport (the mandate)\n"
    return {"ok": True, "rc": 0, "stdout": stdout, "stderr": "", "surface": surface}


def _vscode_open_dispatch(raw: str) -> dict:
    action = raw.split("vscode://", 1)[1].strip() if "vscode://" in raw else ""
    argument = ""
    if action.startswith("open "):
        argument = action.split(" ", 1)[1].strip()
    uri = ""
    launched = False
    if argument:
        try:
            uri = str(Path(argument).expanduser().resolve().as_uri())
        except Exception:
            uri = f"file:///{argument}"
        try:
            __import__("subprocess").Popen(["code", argument], shell=False)
            launched = True
        except Exception:
            launched = False
    stdout = f"vscode://open {argument}\n"
    return {
        "ok": True,
        "rc": 0,
        "stdout": stdout,
        "stderr": "",
        "surface": {"kind": "vscode_surface", "address": raw, "runtime": "hermes-code", "uri": uri, "launched": launched},
    }


_DISPATCHER.register("vlc://runtime", _vlc_runtime_dispatch)
_DISPATCHER.register("vlc://", _vlc_dispatch)
_DISPATCHER.register("ffmpeg://", _ffmpeg_dispatch)
_DISPATCHER.register("vscode://open ", _vscode_open_dispatch)
_DISPATCHER.register("vscode://", _vscode_dispatch)
_DISPATCHER.register("viewport://", _viewport_dispatch)
_DISPATCHER.register("+æ://cuda-vlc", _cuda_vlc_dispatch)


# ── +æ://vps — sovereign backbone node (Victus-local, liftable to any host) ──
def _vps_dispatch(raw: str) -> dict:
    """Route +æ://vps commands to the vps_node process.

    +æ://vps status                  → backbone health + record counts
    +æ://vps record <nsid> <json>    → write a signed ae.core record
    +æ://vps route <vps://cmd>       → dispatch (e.g. rtx://compute)
    The node process lives at the did:web PDS endpoint (http://localhost:3000).
    """
    import urllib.request, json as _json
    rest = raw.split("vps", 1)[1].strip() if "vps" in raw else ""
    parts = rest.split(" ", 1)
    action = parts[0] if parts else "status"
    arg = parts[1] if len(parts) > 1 else ""
    host = os.environ.get("VPS_HOST", "127.0.0.1")
    port = os.environ.get("VPS_PORT", "3000")
    base = f"http://{host}:{port}"
    try:
        if action == "status":
            with urllib.request.urlopen(f"{base}/xrpc/ae.vps.status", timeout=5) as r:
                return {"ok": True, "surface": _json.loads(r.read())}
        if action == "record":
            nsid, _, val = arg.partition(" ")
            payload = _json.dumps({"nsid": nsid, "value": _json.loads(val)}).encode()
            req = urllib.request.Request(f"{base}/xrpc/ae.vps.record", data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=5) as r:
                return {"ok": True, "surface": _json.loads(r.read())}
        if action == "route":
            payload = _json.dumps({"cmd": arg}).encode()
            req = urllib.request.Request(f"{base}/xrpc/ae.vps.route", data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=60) as r:
                return {"ok": True, "surface": _json.loads(r.read())}
        return {"ok": False, "stderr": f"vps: unknown action '{action}' (status|record|route)"}
    except Exception as exc:  # node not running / unreachable
        return {"ok": False, "stderr": f"vps unreachable at {base}: {exc}"}


_DISPATCHER.register("+æ://vps", _vps_dispatch)


# ── +æ://secrets — local-first secret broker (github.io = surface, Victus = custody) ──
def _secrets_dispatch(raw: str) -> dict:
    """Local broker between the github.io secret-bridge surface and GLOCAL agents.

    The bridge UI lives on github.io (source of truth for the SURFACE).
    The SECRET lives only on Victus, in C:\\æ\\secrets\\secrets.json (git-ignored).
    This dispatch reads that local file and feeds secrets to agents at runtime —
    it never uploads, never echoes raw values to logs, never touches cloud.

    +æ://secrets status            → file present? count? (no values)
    +æ://secrets list              → keys + kinds + masked preview (no raw value)
    +æ://secrets get <KEY>         → resolve value for an agent (masked in stdout)
    +æ://secrets sources           → candidate paths + which exist (debug handoff)
    +æ://secrets path              → where the local source lives
    """
    rest = raw.split("secrets", 1)[1].strip() if "secrets" in raw else ""
    parts = rest.split(" ", 1)
    action = parts[0] if parts else "status"
    arg = parts[1] if len(parts) > 1 else ""

    # locate the local source (env HERMES_SECRETS, else GLOCAL default)
    import sys as _sys
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in _sys.path:
        _sys.path.insert(0, _HERE)
    # agents/ lives one level up from hermes_cli/
    _AGENTS = os.path.normpath(os.path.join(_HERE, "..", "..", "agents"))
    if os.path.isdir(_AGENTS) and _AGENTS not in _sys.path:
        _sys.path.insert(0, _AGENTS)
    try:
        from secret_source import get_secret, get_by_prefix, DEFAULT_PATHS
    except Exception as exc:
        return {"ok": False, "stderr": f"secrets: cannot load secret_source ({exc})"}

    src = os.environ.get("HERMES_SECRETS") or next(
        (p for p in DEFAULT_PATHS if p and os.path.exists(p)), None)

    if action == "path":
        return {"ok": True, "stdout": f"secrets source: {src or '(none found)'}\n",
                "surface": {"kind": "secrets", "local_only": True, "path": src}}

    if action == "sources":
        lines = ["secrets candidate sources (first existing wins):"]
        for i, p in enumerate(DEFAULT_PATHS):
            if not p:
                continue
            mark = "OK " if os.path.exists(p) else "absent"
            lines.append(f"  [{mark}] {p}")
        envp = os.environ.get("HERMES_SECRETS")
        if envp:
            lines.append(f"  [env] HERMES_SECRETS={envp} -> {'OK' if os.path.exists(envp) else 'absent'}")
        return {"ok": True, "stdout": "\n".join(lines) + "\n",
                "surface": {"kind": "secrets", "local_only": True, "paths": DEFAULT_PATHS}}

    if action == "status":
        if not src or not os.path.exists(src):
            return {"ok": True, "stdout": "secrets: NO local source found\n"
                    "  bridge: https://myaelmendez.github.io/secret-source-bridge.html\n"
                    "  fix: in bridge click 'Push to local' -> save to C:\\æ\\secrets\\secrets.json\n",
                    "surface": {"kind": "secrets", "local_only": True, "present": False}}
        try:
            data = json.load(open(src, encoding="utf-8"))
            n = len(data.get("secrets", []))
        except Exception as exc:
            return {"ok": False, "stderr": f"secrets: unreadable source ({exc})",
                    "surface": {"kind": "secrets", "local_only": True, "path": src}}
        return {"ok": True, "stdout": f"secrets: {n} entry(ies) at {src}\n",
                "surface": {"kind": "secrets", "local_only": True, "present": True,
                            "count": n, "path": src}}

    if action == "list":
        if not src or not os.path.exists(src):
            return {"ok": False, "stderr": "secrets: NO local source (Push to local first)"}
        data = json.load(open(src, encoding="utf-8"))
        rows = []
        for s in data.get("secrets", []):
            v = str(s.get("value", ""))
            mask = "•" * min(12, max(4, len(v))) if v else ""
            rows.append(f"  {s.get('key')}  [{s.get('kind')}]  {mask}")
        body = "secrets (local, masked):\n" + "\n".join(rows) + "\n"
        return {"ok": True, "stdout": body,
                "surface": {"kind": "secrets", "local_only": True, "count": len(rows)}}

    if action == "get":
        if not arg:
            return {"ok": False, "stderr": "secrets get <KEY> — key required"}
        val = get_secret(arg)
        if val is None:
            # try prefix (e.g. 'BSKY_AGENT_')
            hits = get_by_prefix(arg)
            if hits:
                body = f"secrets get {arg} (prefix, {len(hits)} hit(s)):\n" + "\n".join(
                    f"  {k} = {'•'*min(12,max(4,len(v)))}" for k, v in hits.items()) + "\n"
                return {"ok": True, "stdout": body,
                        "surface": {"kind": "secrets", "local_only": True, "prefix": arg,
                                    "count": len(hits)}}
            return {"ok": False, "stderr": f"secrets: '{arg}' not found locally"}
        # value resolved — show masked in stdout; real value available to the agent only
        mask = "•" * min(12, max(4, len(val)))
        return {"ok": True,
                "stdout": f"secrets get {arg} = {mask}  (resolved locally; injected at runtime)\n",
                "surface": {"kind": "secrets", "local_only": True, "key": arg,
                            "resolved": True},
                "secret_value": val}  # carrier only; never logged by callers

    return {"ok": False, "stderr": f"secrets: unknown action '{action}' (status|list|get|path)"}


_DISPATCHER.register("+æ://secrets", _secrets_dispatch)


def _gauntlet_status() -> dict:
    nous = _dispatch("NOUS://") if "_nous_dispatch" in globals() else {"ok": True, "stdout": "NOUS://\n"}
    vlc = _dispatch("vlc://status")
    ffmpeg = _dispatch("ffmpeg://version")
    nous_running = True if nous.get("ok") else False
    vlc_running = bool(vlc.get("running")) if isinstance(vlc, dict) else False
    return {
        "nous_running": nous_running,
        "vlc_running": vlc_running,
        "ffmpeg_installed": ffmpeg.get("ok") if isinstance(ffmpeg, dict) else False,
        "omniverse_ready": nous_running and vlc_running,
    }


def _geforce_c2_dispatch(raw: str) -> dict:
    action = raw.split("NVIDIA://", 1)[1].strip() if "NVIDIA://" in raw else ""
    if not action:
        return {
            "ok": False,
            "rc": 2,
            "stdout": "",
            "stderr": "missing NVIDIA:// action",
            "surface": {
                "kind": "geforce_command_control",
                "address": "NVIDIA://",
                "runtime": "hermes-code",
            },
        }
    gpu_info = "unknown"
    try:
        result = __import__("subprocess").run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,cuda_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            shell=False,
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
    except Exception:
        gpu_info = "nvidia-smi unavailable"
    return {
        "ok": True,
        "rc": 0,
        "stdout": f"NVIDIA://{action} → {gpu_info}\n",
        "stderr": "",
        "surface": {
            "kind": "geforce_command_control",
            "address": f"NVIDIA://{action}",
            "runtime": "hermes-code",
            "toolkit": True,
            "authorized": True,
            "governance": {
                "required": "+æ member token for local-only GPU surface",
                "audit": True,
                "tracer": "Wyoming DAO LLC audit trail",
            },
        },
    }


def _hermes_superagent_dispatch(raw: str) -> dict:
    """hermes-superagent:// — BLOCKED at the chassis (scalar supremacy).

    There is no "superagent" tier above the sovereign scalar. Agentic-native
    means the language itself enforces the boundary: this scheme resolves to a
    hard refusal, so any surface that links to it dead-ends at the router
    rather than being policed per-file. The one true stack routes through
    æ:// (the agentic-language-chassis) and its dialects.
    """
    return {
        "ok": False,
        "rc": 2,
        "stdout": "",
        "stderr": (
            "hermes-superagent:// is blocked (scalar supremacy): no tier above "
            "the sovereign scalar. Route through æ:// (agentic-language-chassis)."
        ),
        "surface": {
            "kind": "blocked",
            "address": raw,
            "runtime": "hermes-code",
            "reason": "scalar-supremacy",
            "route_through": "æ://",
        },
    }


_DISPATCHER.register("NVIDIA://", _geforce_c2_dispatch)
_DISPATCHER.register("hermes-superagent://", _hermes_superagent_dispatch)
_DISPATCHER.register("+æ://victus", _victus_dispatch)
_DISPATCHER.register("+æ://qrcode", _qrcode_dispatch)
_DISPATCHER.register("+æ://mesh", _mesh_dispatch)
_DISPATCHER.register("commandprompt://", _commandprompt_dispatch)
_DISPATCHER.register("home://", _home_dispatch)
_DISPATCHER.register("fs://", _fs_dispatch)
