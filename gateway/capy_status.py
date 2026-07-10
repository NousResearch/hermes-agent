"""Capy capability dashboard/status rendering.

The dashboard is intentionally read-only and secret-safe.  It reports whether
credential-backed capabilities appear configured without printing credential
values.
"""

from __future__ import annotations

import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any


_SECRET_NAMES = (
    "TINKER" + "_API_KEY",
    "WANDB_API_KEY",
    "OPENROUTER_API_KEY",
    "GITHUB_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "NOUS_API_KEY",
)


def _short_command(cmd: list[str], timeout: float = 2.0) -> tuple[int | None, str]:
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        out = proc.stdout or ""
        for name in _SECRET_NAMES:
            val = os.environ.get(name)
            if val:
                out = out.replace(val, "[REDACTED]")
        return proc.returncode, out.strip()
    except Exception as exc:
        return None, str(exc)


def _http_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "capy-dashboard/1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local health checks
            return 200 <= getattr(resp, "status", 0) < 300
    except Exception:
        return False


def _config_model() -> tuple[str, str]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        if isinstance(model_cfg, dict):
            return str(model_cfg.get("provider") or ""), str(model_cfg.get("default") or model_cfg.get("model") or "")
        if isinstance(model_cfg, str):
            return "", model_cfg
    except Exception:
        pass
    return "", ""


def _present(name: str) -> bool:
    return bool((os.environ.get(name) or "").strip())


def collect_capabilities(*, session_id: str = "", workspace: str = "") -> dict[str, Any]:
    code, version = _short_command(["hermes", "--version"])
    provider, model = _config_model()
    home = Path.home()

    return {
        "hermes": {
            "version": version.splitlines()[0] if code == 0 and version else "unknown",
            "provider": provider,
            "model": model,
        },
        "session": {
            "session_id": session_id,
            "workspace": workspace,
        },
        "services": {
            "webui_local": _http_ok("http://127.0.0.1:8787/health"),
            "lmstudio": _http_ok("http://127.0.0.1:1234/v1/models"),
            "comfyui": _http_ok("http://127.0.0.1:8188/system_stats"),
        },
        "rl": {
            "tinker_api_key": _present("TINKER_API_KEY"),
            "wandb_api_key": _present("WANDB_API_KEY"),
            "ready": _present("TINKER_API_KEY") and _present("WANDB_API_KEY"),
            "readiness_script": str(Path(__file__).resolve().parents[1] / "scripts" / "capy_rl_readiness.py"),
        },
        "local_stack": {
            "autonovel": str(home / "workspace" / "autonovel"),
            "comfyui": str(home / "comfy" / "ComfyUI"),
            "piper_voice": str(home / ".local" / "share" / "piper" / "voices" / "en_US-lessac-medium" / "en_US-lessac-medium.onnx"),
            "lmstudio_url": "http://127.0.0.1:1234/v1",
        },
        "runbooks": {
            "visual_qa": str(Path(__file__).resolve().parents[1] / "docs" / "runbooks" / "webui-visual-qa.md"),
            "provider_fallbacks": str(Path(__file__).resolve().parents[1] / "docs" / "runbooks" / "provider-fallbacks.md"),
        },
    }


def _yes(value: bool) -> str:
    return "ok" if value else "needs attention"


def render_capabilities(data: dict[str, Any]) -> str:
    hermes = data.get("hermes", {})
    session = data.get("session", {})
    services = data.get("services", {})
    rl = data.get("rl", {})
    local = data.get("local_stack", {})
    runbooks = data.get("runbooks", {})

    lines = [
        "Capy capability dashboard",
        "=========================",
        f"Hermes: {hermes.get('version') or 'unknown'}",
        f"Model: {hermes.get('provider') or 'default'} / {hermes.get('model') or 'unknown'}",
    ]
    if session.get("session_id"):
        lines.append(f"Session: {session['session_id']}")
    if session.get("workspace"):
        lines.append(f"Workspace: {session['workspace']}")

    lines.extend([
        "",
        "Services:",
        f"  - WebUI local: {_yes(bool(services.get('webui_local')))}",
        f"  - LM Studio: {_yes(bool(services.get('lmstudio')))}",
        f"  - ComfyUI: {_yes(bool(services.get('comfyui')))}",
        "",
        "RL/Tinker:",
        f"  - Ready: {'yes' if rl.get('ready') else 'no'}",
        f"  - TINKER_API_KEY: {'present' if rl.get('tinker_api_key') else 'missing'}",
        f"  - WANDB_API_KEY: {'present' if rl.get('wandb_api_key') else 'missing'}",
        f"  - Readiness: {rl.get('readiness_script')}",
        "",
        "Local stack:",
        f"  - AutoNovel: {local.get('autonovel')}",
        f"  - ComfyUI: {local.get('comfyui')}",
        f"  - Piper voice: {local.get('piper_voice')}",
        f"  - LM Studio URL: {local.get('lmstudio_url')}",
        "",
        "Runbooks:",
        f"  - Visual QA: {runbooks.get('visual_qa')}",
        f"  - Provider fallbacks: {runbooks.get('provider_fallbacks')}",
    ])
    return "\n".join(lines)
