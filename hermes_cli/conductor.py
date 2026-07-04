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


def _media_dispatch(raw: str) -> dict:
    return {
        "ok": True,
        "rc": 0,
        "stdout": "+æ://media^ffmpeg → deterministic media pipeline\n",
        "stderr": "",
        "surface": {
            "kind": "media",
            "address": raw,
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
_DISPATCHER.register("vscode://", _vscode_dispatch)
_DISPATCHER.register("reachy://", _reachy_dispatch)
_DISPATCHER.register("mcp://", _mcp_dispatch)
_DISPATCHER.register("+æ://media^ffmpeg", _media_dispatch)


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
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "rc": 124, "stdout": "", "stderr": "timeout"}
    except Exception as e:
        return {"ok": False, "rc": 2, "stdout": "", "stderr": str(e)}


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
