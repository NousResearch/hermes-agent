"""Executable runtime preflight checks for Hermes operational scopes."""

from __future__ import annotations

import json as jsonlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hermes_cli.config import get_hermes_home, load_config


@dataclass
class PreflightCheck:
    name: str
    ok: bool
    detail: str = ""


def _configured_model(config: dict[str, Any]) -> str:
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        return str(model_cfg.get("default") or model_cfg.get("name") or "").strip()
    return str(model_cfg or "").strip()


def _configured_provider(config: dict[str, Any]) -> str:
    provider = str(config.get("provider") or "").strip()
    if provider:
        return provider
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        return str(model_cfg.get("provider") or "").strip()
    return ""


def _can_write_probe(directory: Path, label: str) -> PreflightCheck:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".hermes-preflight-write-test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        return PreflightCheck(label, True, str(directory))
    except Exception as exc:
        return PreflightCheck(label, False, f"{directory}: {exc}")


def _guardrails_enabled(config: dict[str, Any]) -> bool:
    kanban_cfg = config.get("kanban") if isinstance(config, dict) else None
    guard_cfg = kanban_cfg.get("runtime_guardrails", {}) if isinstance(kanban_cfg, dict) else {}
    return guard_cfg.get("enabled", True) is not False


def collect_kanban_preflight() -> list[PreflightCheck]:
    """Collect executable checks needed before Kanban worker autonomy."""
    checks: list[PreflightCheck] = []

    try:
        config = load_config() or {}
        checks.append(PreflightCheck("config load", True, "config.yaml loaded"))
    except Exception as exc:
        config = {}
        checks.append(PreflightCheck("config load", False, str(exc)))

    provider = _configured_provider(config)
    model = _configured_model(config)
    checks.append(
        PreflightCheck(
            "provider/model",
            bool(provider and model),
            f"provider={provider or '(missing)'} model={model or '(missing)'}",
        )
    )

    checks.append(
        PreflightCheck(
            "kanban runtime guardrails",
            _guardrails_enabled(config),
            "kanban.runtime_guardrails.enabled must not be false",
        )
    )

    home = Path(os.environ.get("HERMES_HOME") or get_hermes_home()).expanduser()
    checks.append(_can_write_probe(home, "HERMES_HOME writable"))

    try:
        from hermes_cli import kanban_db as kb

        board = kb.get_current_board()
        db_path = kb.kanban_db_path(board)
        checks.append(PreflightCheck("kanban board", True, board))
        checks.append(_can_write_probe(db_path.parent, "kanban db writable"))
        checks.append(_can_write_probe(kb.workspaces_root(board), "kanban workspaces writable"))
    except Exception as exc:
        checks.append(PreflightCheck("kanban paths", False, str(exc)))

    return checks


def _print_human(scope: str, checks: list[PreflightCheck]) -> None:
    print(f"Runtime preflight: {scope}")
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        suffix = f" — {check.detail}" if check.detail else ""
        print(f"[{status}] {check.name}{suffix}")
    print("PASS" if all(check.ok for check in checks) else "FAIL")


def run_runtime_preflight(args) -> int:
    scope = getattr(args, "scope", "kanban") or "kanban"
    if scope != "kanban":
        checks = [PreflightCheck("scope", False, f"unsupported scope: {scope}")]
    else:
        checks = collect_kanban_preflight()

    if getattr(args, "json", False):
        print(jsonlib.dumps({"scope": scope, "ok": all(c.ok for c in checks), "checks": [asdict(c) for c in checks]}, indent=2))
    else:
        _print_human(scope, checks)
    return 0 if all(check.ok for check in checks) else 1


def runtime_command(args) -> None:
    import sys

    action = getattr(args, "runtime_action", None)
    if action == "preflight":
        raise SystemExit(run_runtime_preflight(args))
    print("Unknown runtime subcommand")
    raise SystemExit(1)
