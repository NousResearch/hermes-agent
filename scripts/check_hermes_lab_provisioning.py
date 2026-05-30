#!/usr/bin/env python3.11
"""Validate first-run Hermes Lab provisioning."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.dev_control.lab_environment import lab_paths_from_env, validate_lab_environment  # noqa: E402
from gateway.dev_control.lab_process_isolation import audit_current_process_isolation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Hermes Lab provisioning readiness.")
    parser.add_argument("--lab-home", default=None)
    parser.add_argument("--ao-config-path", default=None)
    parser.add_argument("--require-services", action="store_true")
    parser.add_argument("--gateway-port", type=int, default=8662)
    parser.add_argument("--ao-port", type=int, default=3015)
    args = parser.parse_args()

    paths = lab_paths_from_env()
    lab_home = Path(args.lab_home or paths["lab_home"]).expanduser()
    hermes_home = Path(paths["hermes_home"]).expanduser()
    agent = Path(paths["repos_dir"]).expanduser() / "hermes-agent"
    oryn = Path(paths["repos_dir"]).expanduser() / "Oryn"
    ao_config = Path(args.ao_config_path or lab_home / "agent-orchestrator.lab.yaml").expanduser()
    errors: list[str] = []
    warnings: list[str] = []

    safety = validate_lab_environment(
        hermes_home=hermes_home,
        gateway_port=args.gateway_port,
        repo_roots=[agent, oryn],
    )
    errors.extend(safety.get("errors") or [])
    warnings.extend(safety.get("warnings") or [])

    for repo in (agent, oryn):
        if not (repo / ".git").exists():
            errors.append(f"missing lab git clone: {repo}")
    if not ((agent / ".venv/bin/python").exists() or (agent / "venv/bin/python").exists()):
        errors.append(f"missing lab hermes-agent virtualenv under {agent}")
    if not ao_config.exists():
        errors.append(f"missing lab AO config: {ao_config}")
    else:
        text = ao_config.read_text(encoding="utf-8", errors="replace")
        for expected in (str(agent), str(oryn), str(lab_home / "worktrees")):
            if expected not in text:
                errors.append(f"lab AO config does not reference expected lab path: {expected}")
        if "~/.oryn-lab" in text:
            errors.append("lab AO config must be rendered to absolute paths before use with lab HOME")

    codex_config = lab_home / ".codex/config.toml"
    if not codex_config.exists():
        errors.append(f"missing lab Codex trust config: {codex_config}")
    else:
        trust_text = codex_config.read_text(encoding="utf-8", errors="replace")
        for expected in (str(agent), str(oryn), str(lab_home / "worktrees")):
            if f'[projects."{expected}"]' not in trust_text:
                errors.append(f"lab Codex config does not trust expected lab path: {expected}")

    isolation = audit_current_process_isolation()
    if not isolation["ok"]:
        errors.append("current process isolation audit failed")

    service_checks = {}
    if args.require_services:
        service_checks = {
            "gateway": _port_open("127.0.0.1", args.gateway_port),
            "ao": _port_open("127.0.0.1", args.ao_port),
        }
        for name, ok in service_checks.items():
            if not ok:
                errors.append(f"{name} service is not reachable on expected port")

    result = {
        "ok": not errors,
        "object": "hermes.dev_lab_provisioning_check",
        "lab_home": str(lab_home),
        "hermes_home": str(hermes_home),
        "agent_repo": str(agent),
        "oryn_repo": str(oryn),
        "ao_config": str(ao_config),
        "codex_config": str(codex_config),
        "service_checks": service_checks,
        "isolation": isolation,
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["ok"] else 2


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=1.0):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
