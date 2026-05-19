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
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    DEFAULT_LOCAL_HOST, ENV_API_KEY, emit_json, log, resolve_api_key,
)
from check_deps import check_deps  # noqa: E402
from _common import unwrap_workflow  # noqa: E402


def comfy_cli_available() -> str | None:
    """Return command prefix for comfy-cli, or None."""
    if shutil.which("comfy"):
        return "comfy"
    if shutil.which("uvx"):
        return "uvx --from comfy-cli comfy"
    return None


def run_cmd(cmd: list[str], *, dry_run: bool = False) -> tuple[int, str]:
    if dry_run:
        return 0, "[dry-run]"
    log(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


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
    if hf_token:
        cmd.extend(["--set-hf-api-token", hf_token])
    if civitai_token:
        cmd.extend(["--set-civitai-api-token", civitai_token])
    code, _ = run_cmd(cmd, dry_run=dry_run)
    return code == 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run check_deps and install whatever is missing")
    p.add_argument("workflow")
    p.add_argument("--host", default=DEFAULT_LOCAL_HOST)
    p.add_argument("--api-key", help=f"or set ${ENV_API_KEY}")
    p.add_argument("--models-from-file",
                   help="JSON file mapping {model_filename: download_url} for models that need install")
    p.add_argument("--hf-token", help="HuggingFace token for downloads")
    p.add_argument("--civitai-token", help="CivitAI token for downloads")
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
        with wf_path.open() as f:
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
            sources = json.loads(Path(args.models_from_file).read_text())
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
