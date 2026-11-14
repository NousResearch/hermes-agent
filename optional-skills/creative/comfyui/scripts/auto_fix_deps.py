#!/usr/bin/env python3
"""
auto_fix_deps.py — Run check_deps.py, then attempt to install whatever is missing.

For local servers:
  - Missing custom nodes → `comfy node install <package>`
  - Missing models → `comfy model download` (only if a URL is supplied via
    --model-source-file or detected via well-known names)

For cloud: prints what would be needed but cannot install (cloud preinstalls
custom nodes and most models server-side; if something genuinely isn't there,
ask Comfy support).

This is conservative: it never installs without an explicit URL for models
(downloading the wrong model is hard to undo). Custom nodes from the registry
are auto-installed by name.

Usage:
    python3 auto_fix_deps.py workflow_api.json
    python3 auto_fix_deps.py workflow_api.json --models-from-file urls.json
    python3 auto_fix_deps.py workflow_api.json --dry-run

Model auth: HF_API_TOKEN / CIVITAI_API_TOKEN in the environment (or
--hf-token / --civitai-token) reach comfy-cli through the child's
environment for the matching download only. They are never put on the
comfy-cli argv: its --set-*-api-token options persist the value to
config.ini and the argv is logged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    DEFAULT_LOCAL_HOST, ENV_API_KEY, emit_json, log, resolve_api_key,
)
from check_deps import check_deps
from _common import unwrap_workflow


def comfy_cli_available() -> str | None:
    """Return command prefix for comfy-cli, or None."""
    if shutil.which("comfy"):
        return "comfy"
    if shutil.which("uvx"):
        return "uvx --from comfy-cli comfy"
    return None


def run_cmd(cmd: list[str], *, dry_run: bool = False,
            env_overlay: dict[str, str] | None = None) -> tuple[int, str]:
    if dry_run:
        return 0, "[dry-run]"
    log(f"$ {' '.join(cmd)}")
    env = {**os.environ, **env_overlay} if env_overlay else None
    # health: allow HX006 -- node installs and multi-GB model downloads have no fair upper bound
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                          check=False, stdin=subprocess.DEVNULL, env=env)
    out = (proc.stdout or "") + (proc.stderr or "")
    for secret in (env_overlay or {}).values():
        out = out.replace(secret, "***")
    return proc.returncode, out


def _host_matches(url: str, domain: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host == domain or host.endswith("." + domain)


def _model_auth_env(url: str, hf_token: str | None,
                    civitai_token: str | None) -> dict[str, str]:
    """Token env for comfy-cli, scoped to the provider that serves ``url``.

    comfy-cli reads HF_API_TOKEN / CIVITAI_API_TOKEN from its environment
    without persisting them, unlike --set-*-api-token which writes config.ini.
    """
    if hf_token and _host_matches(url, "huggingface.co"):
        return {"HF_API_TOKEN": hf_token}
    if civitai_token and _host_matches(url, "civitai.com"):
        return {"CIVITAI_API_TOKEN": civitai_token}
    return {}


def install_node(package: str, *, dry_run: bool = False, comfy_cmd: str = "comfy") -> bool:
    cmd = comfy_cmd.split() + ["--skip-prompt", "node", "install", package]
    code, _ = run_cmd(cmd, dry_run=dry_run)
    return code == 0


def install_model(url: str, folder: str, filename: str | None = None,
                  *, dry_run: bool = False, comfy_cmd: str = "comfy",
                  hf_token: str | None = None, civitai_token: str | None = None) -> bool:
    cmd = comfy_cmd.split() + [
        "--skip-prompt", "model", "download",
        "--url", url,
        "--relative-path", f"models/{folder}",
    ]
    if filename:
        cmd.extend(["--filename", filename])
    code, _ = run_cmd(cmd, dry_run=dry_run,
                      env_overlay=_model_auth_env(url, hf_token, civitai_token))
    return code == 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run check_deps and install whatever is missing")
    p.add_argument("workflow")
    p.add_argument("--host", default=DEFAULT_LOCAL_HOST)
    p.add_argument("--api-key", help=f"or set ${ENV_API_KEY}")
    p.add_argument("--models-from-file",
                   help="JSON file mapping {model_filename: download_url} for models that need install")
    p.add_argument("--hf-token", help="HuggingFace token for downloads (prefer $HF_API_TOKEN)")
    p.add_argument("--civitai-token", help="CivitAI token for downloads (prefer $CIVITAI_API_TOKEN)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be installed without doing it")
    p.add_argument("--no-restart", action="store_true",
                   help="Don't suggest restarting the server after node install")
    args = p.parse_args(argv)

    api_key = resolve_api_key(args.api_key)

    wf_path = Path(args.workflow).expanduser()
    if not wf_path.exists():
        emit_json({"error": f"Workflow not found: {args.workflow}"})
        return 1
    try:
        with wf_path.open(encoding="utf-8-sig") as f:
            workflow = unwrap_workflow(json.load(f))
    except (ValueError, json.JSONDecodeError) as e:
        emit_json({"error": str(e)})
        return 1

    report = check_deps(workflow, host=args.host, api_key=api_key)

    if report["is_ready"]:
        emit_json({"status": "ready", "report": report})
        return 0

    if report["is_cloud"]:
        emit_json({
            "status": "cannot_fix_cloud",
            "reason": "Comfy Cloud preinstalls nodes; if something is genuinely missing, contact support.",
            "report": report,
        })
        return 1

    comfy_cmd = comfy_cli_available()
    if not comfy_cmd:
        emit_json({
            "status": "cannot_fix",
            "reason": "comfy-cli not on PATH; install with `pip install comfy-cli` or `pipx install comfy-cli`",
            "report": report,
        })
        return 1

    actions: list[dict] = []
    failures: list[dict] = []

    # ---- Install missing custom nodes ----
    seen_packages: set[str] = set()
    for entry in report["missing_nodes"]:
        cmd = entry.get("fix_command", "")
        if cmd.startswith("comfy node install "):
            package = cmd.split(" ")[-1]
            if package in seen_packages:
                continue
            seen_packages.add(package)
            ok = install_node(package, dry_run=args.dry_run, comfy_cmd=comfy_cmd)
            (actions if ok else failures).append({
                "kind": "node", "package": package, "node_class": entry["class_type"],
                "ok": ok,
            })
        else:
            failures.append({
                "kind": "node", "node_class": entry["class_type"],
                "ok": False, "reason": "No registry mapping known. " + entry.get("fix_hint", ""),
            })

    # ---- Install missing models (only when URL provided) ----
    sources: dict[str, str] = {}
    if args.models_from_file:
        try:
            sources = json.loads(Path(args.models_from_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log(f"Could not read --models-from-file: {e}")

    for entry in report["missing_models"]:
        filename = entry["value"]
        url = sources.get(filename)
        if not url:
            failures.append({
                "kind": "model", "filename": filename, "folder": entry["folder"],
                "ok": False, "reason": "No URL provided in --models-from-file. "
                                       "Refusing to guess.",
            })
            continue
        ok = install_model(
            url, entry["folder"], filename,
            dry_run=args.dry_run, comfy_cmd=comfy_cmd,
            hf_token=args.hf_token, civitai_token=args.civitai_token,
        )
        (actions if ok else failures).append({
            "kind": "model", "filename": filename, "folder": entry["folder"],
            "url": url, "ok": ok,
        })

    # ---- Embeddings ----
    for entry in report["missing_embeddings"]:
        emb_name = entry["embedding_name"]
        # Try common extensions in user-supplied source map
        url = (sources.get(f"{emb_name}.pt")
               or sources.get(f"{emb_name}.safetensors")
               or sources.get(emb_name))
        if not url:
            failures.append({
                "kind": "embedding", "name": emb_name,
                "ok": False, "reason": "No URL provided in --models-from-file.",
            })
            continue
        target_filename = (
            f"{emb_name}.safetensors" if url.endswith(".safetensors")
            else f"{emb_name}.pt"
        )
        ok = install_model(
            url, "embeddings", target_filename,
            dry_run=args.dry_run, comfy_cmd=comfy_cmd,
            hf_token=args.hf_token, civitai_token=args.civitai_token,
        )
        (actions if ok else failures).append({
            "kind": "embedding", "name": emb_name, "url": url, "ok": ok,
        })

    needs_restart = any(a["kind"] == "node" and a.get("ok") for a in actions)

    emit_json({
        "status": "fixed" if not failures else "partial",
        "actions_taken": actions,
        "failures": failures,
        "needs_server_restart": needs_restart and not args.no_restart,
        "restart_hint": "comfy stop && comfy launch --background",
        "dry_run": args.dry_run,
    })
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
