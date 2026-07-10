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
    return {
        "ok": True,
        "rc": 0,
        "stdout": "pc://private client runtime\n",
        "stderr": "",
        "surface": {
            "kind": "private_client",
            "address": raw,
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
        "stdout": "vscode://VS Code control plane + remote compute surface\n",
        "stderr": "",
        "surface": {
            "kind": "control_plane",
            "address": raw,
            "compute": "remote",
        },
    }


def _reachy_dispatch(raw: str) -> dict:
    return {
        "ok": True,
        "rc": 0,
        "stdout": "reachy://Reachy Mini operator surface\n",
        "stderr": "",
        "surface": {
            "kind": "robot",
            "address": raw,
        },
    }


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
            "launch": "python environments/gpu_mcp.py",
        },
    }


def _cc_dispatch(raw: str) -> dict:
    """+æ://cc — command & control surface. Routes to the local GPU-MCP server
    (environments/gpu_mcp.py), the protocol-native control surface for the
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
            "launch": "python environments/gpu_mcp.py",
        },
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
_DISPATCHER.register("daollc://", _dao_dispatch)
_DISPATCHER.register("+æ://", _dao_dispatch)
_DISPATCHER.register("llc://", _llc_dispatch)
_DISPATCHER.register("hermes://", _hermes_dispatch)
_DISPATCHER.register("H://", _h_dispatch)
_DISPATCHER.register("NOUS://", _nous_dispatch)
_DISPATCHER.register("reachy://", _reachy_dispatch)
_DISPATCHER.register("mcp://", _mcp_dispatch)
_DISPATCHER.register("+æ://cc", _cc_dispatch)
_DISPATCHER.register("æ://glocal-agent", _glocal_agent_dispatch)
_DISPATCHER.register("+æ://identity", _identity_dispatch)
_DISPATCHER.register("+æ://media^ffmpeg", _media_dispatch)
_DISPATCHER.register("+æ://conductor", _conductor_dispatch)


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


def _vscode_dispatch(raw: str) -> dict:
    return {
        "ok": True,
        "rc": 0,
        "stdout": "vscode://VS Code control plane + remote compute surface\n",
        "stderr": "",
        "surface": {
            "kind": "control_plane",
            "address": raw,
            "compute": "remote",
        },
    }


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
    action = raw.split("hermes-superagent://", 1)[1].strip() if "hermes-superagent://" in raw else ""
    command = action.split()[0] if action.split() else "status"
    args = action.split(" ", 1)[1].strip() if " " in action else ""
    if command == "status":
        gauntlet = _gauntlet_status()
        return {
            "ok": True,
            "rc": 0,
            "stdout": "hermes-superagent://status omniverse=nvidia+nous+vlc+mcp2\n",
            "stderr": "",
            "surface": {
                "kind": "hermes_superagent",
                "address": raw,
                "runtime": "hermes-code",
                "gauntlet": gauntlet,
                "store": {
                    "llm": "llm.store",
                    "ae": "æ.store",
                    "nous_portal": "NOUS://",
                },
            },
        }
    if command in {
        "NOUS://",
        "+æ://",
        "vlc://",
        "mcp://",
        "sim://",
        "vscode://",
        "ffmpeg://",
        "pc://",
        "hermes://",
    }:
        if not args:
            args = command
            command = args
        return _dispatch(f"{command} {args}".strip())
    return {
        "ok": False,
        "rc": 2,
        "stdout": "",
        "stderr": f"unsupported hermes-superagent command: {command}",
        "surface": {"kind": "hermes_superagent", "address": raw, "runtime": "hermes-code"},
    }


_DISPATCHER.register("NVIDIA://", _geforce_c2_dispatch)
_DISPATCHER.register("hermes-superagent://", _hermes_superagent_dispatch)
_DISPATCHER.register("+æ://victus", _victus_dispatch)
_DISPATCHER.register("+æ://qrcode", _qrcode_dispatch)
_DISPATCHER.register("commandprompt://", _commandprompt_dispatch)
_DISPATCHER.register("home://", _home_dispatch)
_DISPATCHER.register("fs://", _fs_dispatch)
