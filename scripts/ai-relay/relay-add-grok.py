#!/usr/bin/env python3
"""Add Grok to an existing AI Relay config without storing secrets.

This is intentionally small and local-only. It updates:
  .hermes/ai-relay/adapters.yaml
  .hermes/ai-relay/accounts.yaml

It does not log in, read tokens, or write passwords.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - system Python may not have PyYAML
    yaml = None


GROK_ADAPTER = {
    "cmd": [
        "grok",
        "-p",
        "{prompt}",
        "--cwd",
        "{cwd}",
        "--output-format",
        "json",
        "--always-approve",
    ],
    "note": "Grok ใช้บัญชีที่ login ไว้ในเครื่องนี้ ไม่เก็บรหัสลับในไฟล์ AI Relay",
}

DEFAULT_CHAIN = ["grok", "codex", "gemini", "ollama"]


def git_root(start: Path) -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(start),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        return Path(proc.stdout.strip()).resolve()
    return start.resolve()


def read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to update AI Relay YAML files")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return data


def write_yaml(path: Path, data: dict) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to update AI Relay YAML files")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def has_top_level_or_nested_key(text: str, key: str) -> bool:
    return any(line.strip() == f"{key}:" for line in text.splitlines())


def ensure_grok_text_mode(relay_dir: Path) -> dict:
    """Fallback updater for machines whose system Python has no PyYAML."""

    adapters_path = relay_dir / "adapters.yaml"
    accounts_path = relay_dir / "accounts.yaml"
    relay_dir.mkdir(parents=True, exist_ok=True)

    adapter_added = False
    if adapters_path.exists():
        adapters_text = adapters_path.read_text(encoding="utf-8")
    else:
        adapters_text = "tools:\n"
    if "tools:" not in adapters_text:
        adapters_text = "tools:\n" + adapters_text.rstrip() + "\n"
    if not has_top_level_or_nested_key(adapters_text, "grok"):
        adapters_text = adapters_text.rstrip() + """
  grok:
    cmd:
      - grok
      - -p
      - "{prompt}"
      - --cwd
      - "{cwd}"
      - --output-format
      - json
      - --always-approve
    note: "Grok ใช้บัญชีที่ login ไว้ในเครื่องนี้ ไม่เก็บรหัสลับในไฟล์ AI Relay"
"""
        adapter_added = True
    adapters_path.write_text(adapters_text, encoding="utf-8")

    if accounts_path.exists():
        accounts_text = accounts_path.read_text(encoding="utf-8")
    else:
        accounts_text = ""

    if "accounts:" not in accounts_text:
        accounts_text = "accounts:\n" + accounts_text.lstrip()
    if "  grok:" not in accounts_text:
        lines = accounts_text.splitlines()
        insert_at = None
        for idx, line in enumerate(lines):
            if line.strip() == "accounts:":
                insert_at = idx + 1
                break
        block = ["  grok:", "    - id: grok-google-staff"]
        if insert_at is None:
            lines = ["accounts:", *block, *lines]
        else:
            lines[insert_at:insert_at] = block
        accounts_text = "\n".join(lines) + "\n"

    if "fallback:" not in accounts_text:
        accounts_text = accounts_text.rstrip() + "\n\nfallback:\n  code_writing: [grok, codex, gemini, ollama]\n"
    elif "code_writing:" not in accounts_text:
        accounts_text = accounts_text.rstrip() + "\n  code_writing: [grok, codex, gemini, ollama]\n"
    else:
        lines = []
        for line in accounts_text.splitlines():
            if line.lstrip().startswith("code_writing:"):
                indent = line[: len(line) - len(line.lstrip())]
                lines.append(f"{indent}code_writing: [grok, codex, gemini, ollama]")
            else:
                lines.append(line)
        accounts_text = "\n".join(lines) + "\n"

    if "limits:" not in accounts_text:
        accounts_text = accounts_text.rstrip() + """

limits:
  max_rounds_per_issue: 3
  max_calls_per_session: 50
  budget: null
"""
    accounts_path.write_text(accounts_text, encoding="utf-8")

    return {
        "ok": True,
        "adapters_path": str(adapters_path),
        "accounts_path": str(accounts_path),
        "adapter_added": adapter_added,
        "fallback": DEFAULT_CHAIN,
        "secret_written": False,
        "yaml_mode": "text",
    }


def ensure_grok(root: Path) -> dict:
    relay_dir = root / ".hermes" / "ai-relay"
    adapters_path = relay_dir / "adapters.yaml"
    accounts_path = relay_dir / "accounts.yaml"

    if yaml is None:
        result = ensure_grok_text_mode(relay_dir)
        result["root"] = str(root)
        return result

    adapters = read_yaml(adapters_path)
    tools = adapters.setdefault("tools", {})
    adapter_changed = False
    if "grok" not in tools:
        tools["grok"] = GROK_ADAPTER
        adapter_changed = True

    accounts = read_yaml(accounts_path)
    account_map = accounts.setdefault("accounts", {})
    if not isinstance(account_map.get("grok"), list) or not account_map.get("grok"):
        account_map["grok"] = [{"id": "grok-google-staff"}]

    fallback = accounts.setdefault("fallback", {})
    current_chain = fallback.get("code_writing")
    if not isinstance(current_chain, list):
        current_chain = []
    merged_chain = []
    for tool in DEFAULT_CHAIN + current_chain:
        if tool not in merged_chain:
            merged_chain.append(tool)
    fallback["code_writing"] = merged_chain

    accounts.setdefault("limits", {}).setdefault("max_rounds_per_issue", 3)
    accounts.setdefault("limits", {}).setdefault("max_calls_per_session", 50)
    accounts.setdefault("limits", {}).setdefault("budget", None)

    write_yaml(adapters_path, adapters)
    write_yaml(accounts_path, accounts)

    return {
        "ok": True,
        "root": str(root),
        "adapters_path": str(adapters_path),
        "accounts_path": str(accounts_path),
        "adapter_added": adapter_changed,
        "fallback": merged_chain,
        "secret_written": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add Grok to this repo's AI Relay local config.",
    )
    parser.add_argument("--cwd", default=".", help="Repo/worktree path")
    args = parser.parse_args()

    root = git_root(Path(args.cwd).expanduser())
    try:
        result = ensure_grok(root)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
