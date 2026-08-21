#!/usr/bin/env python3
"""Verifica integridade dos patches Hermes One no checkout local.

Somente leitura — não altera arquivos. Saída JSON no stdout.
Exit 0 se todos os checks passarem; exit 1 caso contrário.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CHECKS: list[dict] = []


def _record(name: str, status: str, evidence: str) -> None:
    CHECKS.append({"name": name, "status": status, "evidence": evidence})


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        _record(f"read:{path.name}", "fail", str(exc))
        return ""


def check_hermes_one_block(repo: Path) -> None:
    """D-004: logic in hermes_one_model_library.py; thin mount in web_server.py."""
    mod = repo / "hermes_cli" / "hermes_one_model_library.py"
    ws = repo / "hermes_cli" / "web_server.py"
    mod_text = _read(mod)
    ws_text = _read(ws)
    if not mod_text or not ws_text:
        return

    mod_markers = [
        "HERMES_ONE_MODEL_LIBRARY_COMPAT_V1",
        "def _hermes_one_model_library_path",
        "def _hermes_one_read_model_library",
        'def mount_hermes_one_model_library',
        '@app.get("/api/model/library")',
        '@app.post("/api/model/library")',
        '@app.patch("/api/model/library/{model_id:path}")',
        '@app.delete("/api/model/library/{model_id:path}")',
        'get_hermes_home() / "models.json"',
    ]
    ws_markers = [
        "HERMES_ONE_MODEL_LIBRARY_COMPAT_V1",
        "mount_hermes_one_model_library(app)",
        "from hermes_cli.hermes_one_model_library import mount_hermes_one_model_library",
    ]

    missing_mod = [m for m in mod_markers if m not in mod_text]
    missing_ws = [m for m in ws_markers if m not in ws_text]
    if missing_mod or missing_ws:
        bits = []
        if missing_mod:
            bits.append(f"module missing {missing_mod[:3]}")
        if missing_ws:
            bits.append(f"web_server missing {missing_ws}")
        _record("hermes_one_web_server", "fail", "; ".join(bits))
    else:
        _record(
            "hermes_one_web_server",
            "pass",
            f"{mod.name}+{ws.name}: module + mount intact ({len(mod_markers)}+{len(ws_markers)} markers)",
        )


def check_openrouter_prune(repo: Path) -> None:
    path = repo / "agent" / "credential_pool.py"
    text = _read(path)
    if not text:
        return

    markers = [
        'source = "env:OPENROUTER_API_KEY"',
        "INCIDENTE-AUTH-JSON-REWRITE",
        "entries[:] = [e for e in entries if e.source != source]",
    ]
    missing = [m for m in markers if m not in text]
    if missing:
        _record(
            "openrouter_prune",
            "fail",
            f"{path}: missing prune markers: {missing}",
        )
    else:
        _record("openrouter_prune", "pass", f"{path}: OpenRouter stale-entry prune intact")


def check_files_exist(repo: Path) -> None:
    for rel in (
        "hermes_cli/web_server.py",
        "hermes_cli/hermes_one_model_library.py",
        "agent/credential_pool.py",
        "AGENTS.md",
    ):
        path = repo / rel
        if path.is_file():
            _record(f"exists:{rel}", "pass", str(path))
        else:
            _record(f"exists:{rel}", "fail", f"missing {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Hermes One patch integrity guard")
    parser.add_argument(
        "repo_path",
        nargs="?",
        default=str(Path(__file__).resolve().parents[2]),
        help="Root of hermes-agent checkout",
    )
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo = Path(args.repo_path).resolve()

    if not repo.is_dir():
        out = {"ok": False, "checks": [{"name": "repo", "status": "fail", "evidence": f"not a directory: {repo}"}]}
        print(json.dumps(out, indent=2))
        return 1

    check_files_exist(repo)
    check_hermes_one_block(repo)
    check_openrouter_prune(repo)

    ok = all(c["status"] == "pass" for c in CHECKS)
    out = {"ok": ok, "repo_path": str(repo), "checks": CHECKS}
    print(json.dumps(out, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
