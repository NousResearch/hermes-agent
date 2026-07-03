#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


RECOMMENDED = {
    "agent.tool_use_enforcement": "auto",
    "approvals.mode": "smart",
    "approvals.cron_mode": "deny",
    "terminal.backend": "docker",
}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _rough_yaml_value(text: str, dotted: str) -> str | None:
    parts = dotted.split(".")
    if len(parts) != 2:
        return None
    section, key = parts
    in_section = False
    for raw_line in text.splitlines():
        if raw_line.startswith(f"{section}:"):
            in_section = True
            continue
        if in_section and raw_line and not raw_line.startswith((" ", "\t")):
            in_section = False
        if in_section:
            stripped = raw_line.strip()
            if stripped.startswith(f"{key}:"):
                return stripped.split(":", 1)[1].strip().strip('"').strip("'")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Proactive Hermes MVP setup recommendations.")
    parser.add_argument("--config", default=str(Path.home() / ".hermes" / "config.yaml"))
    parser.add_argument("--obsidian-vault", default="")
    parser.add_argument("--interval-minutes", type=int, default=60)
    args = parser.parse_args()

    config_path = Path(args.config)
    config_text = _read_text(config_path)

    print("Proactive Hermes setup check")
    print(f"config: {config_path}")
    print("This script does not mutate config.yaml.")
    print("")

    if not config_text:
        print("config_status: missing_or_unreadable")
    else:
        print("config_status: readable")
        for dotted, expected in RECOMMENDED.items():
            actual = _rough_yaml_value(config_text, dotted)
            status = "ok" if actual == expected else "review"
            print(f"{status}: {dotted} actual={actual!r} recommended={expected!r}")
        if _rough_yaml_value(config_text, "approvals.mode") == "off":
            print("risk: approvals.mode is off; proactive Hermes should not run without approval gates.")

    if args.obsidian_vault:
        vault = Path(args.obsidian_vault)
        heartbeat = Path(__file__).resolve().parents[1] / "proactive" / "heartbeat.py"
        print("")
        print("manual_cron_command:")
        print(
            f"*/{args.interval_minutes} * * * * "
            f"cd {Path(__file__).resolve().parents[1]} && "
            f"python3 {heartbeat} --obsidian-vault {vault}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
