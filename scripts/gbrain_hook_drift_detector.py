#!/usr/bin/env python3
"""Detect Hermes shell-hook config drift before gateway startup."""
from __future__ import annotations

import argparse
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None

from hermes_cli.plugins import VALID_HOOKS

HERMES_HOME = Path(os.path.expanduser(os.environ.get("HERMES_HOME", "~/.hermes")))
DEFAULT_CONFIG = HERMES_HOME / "config.yaml"
DEFAULT_LOG = HERMES_HOME / "logs" / "gbrain-hook-drift-detector.log"
GBRAIN_FLAGS = (
    "GBRAIN_HOOK_BRAIN_FIRST_READ_ENABLED",
    "GBRAIN_HOOK_ENTITY_DETECTOR_ENABLED",
)


def _load_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            out[key.strip()] = value
    except OSError:
        pass
    return out


def _env_value(name: str, dotenv: dict[str, str]) -> str:
    value = os.environ.get(name)
    if value is None:
        value = dotenv.get(name, "")
    return value


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError(f"PyYAML unavailable: {YAML_IMPORT_ERROR}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain a YAML mapping")
    return data


def validate_hooks_config(config: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for agent.shell_hooks config shape."""
    errors: list[str] = []
    warnings: list[str] = []
    hooks = config.get("hooks")
    if hooks in (None, {}):
        warnings.append("hooks block is empty or missing")
        return errors, warnings
    if not isinstance(hooks, dict):
        errors.append(f"hooks must be a mapping of event names to hook-definition lists, got {type(hooks).__name__}")
        return errors, warnings

    for event_name, entries in hooks.items():
        event_label = str(event_name)
        if event_label not in VALID_HOOKS:
            detail = ""
            if isinstance(entries, dict) and "events" in entries:
                inner = entries.get("events")
                detail = f" named-hook object shape detected with inner events={inner!r}; current shell dispatcher expects hooks.<event>: [{{command: ...}}]"
            errors.append(f"dead or invalid hook event {event_label!r}.{detail}")
            continue
        if not isinstance(entries, list):
            errors.append(f"hooks.{event_label} must be a list of hook definitions, got {type(entries).__name__}")
            continue
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errors.append(f"hooks.{event_label}[{idx}] must be a mapping, got {type(entry).__name__}")
                continue
            command = entry.get("command")
            if not isinstance(command, str) or not command.strip():
                errors.append(f"hooks.{event_label}[{idx}] missing non-empty command")
            if "events" in entry:
                errors.append(f"hooks.{event_label}[{idx}] has obsolete events field; event is the parent key")
            if entry.get("enabled") is False:
                warnings.append(f"hooks.{event_label}[{idx}] has enabled=false, shell dispatcher ignores per-entry enabled")
    return errors, warnings


def format_report(config_path: Path, errors: list[str], warnings: list[str], dotenv: dict[str, str]) -> str:
    status = "FAIL" if errors else "PASS"
    lines = [
        f"GBRAIN HOOK DRIFT DETECTOR {status}",
        f"checked_at={datetime.now(timezone.utc).isoformat()}",
        f"config={config_path}",
        f"valid_hooks={','.join(sorted(VALID_HOOKS))}",
    ]
    for flag in GBRAIN_FLAGS:
        raw = _env_value(flag, dotenv)
        state = "ON" if _truthy(raw) else "OFF"
        lines.append(f"feature_flag {flag}={state} raw={raw!r}")
    if errors:
        lines.append("errors:")
        lines.extend(f"- {err}" for err in errors)
    if warnings:
        lines.append("warnings:")
        lines.extend(f"- {warn}" for warn in warnings)
    return "\n".join(lines)


def _write_log(path: Path, report: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(report + "\n\n")
    except OSError:
        pass


def _send_telegram(report: str, dotenv: dict[str, str]) -> None:
    token = _env_value("TELEGRAM_BOT_TOKEN", dotenv)
    chat_id = _env_value("TELEGRAM_HOME_CHANNEL", dotenv) or _env_value("TG_TJ_HOME_CHAT_ID", dotenv)
    if not token or not chat_id:
        return
    text = "🚨 Hermes gbrain hook drift detector alarm\n\n" + report[:3500]
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        urllib.request.urlopen(url, data=body, timeout=5).read()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="config.yaml path")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="alarm log path")
    parser.add_argument("--telegram", action="store_true", help="send Telegram alert when drift is found")
    args = parser.parse_args(argv)

    config_path = Path(os.path.expanduser(args.config))
    dotenv = _load_dotenv(HERMES_HOME / ".env")
    try:
        config = _load_config(config_path)
        errors, warnings = validate_hooks_config(config)
    except Exception as exc:
        errors, warnings = [f"detector failed to read config: {exc}"], []

    report = format_report(config_path, errors, warnings, dotenv)
    stream = sys.stderr if errors else sys.stdout
    print(report, file=stream)
    _write_log(Path(os.path.expanduser(args.log)), report)
    if errors and args.telegram:
        _send_telegram(report, dotenv)
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
