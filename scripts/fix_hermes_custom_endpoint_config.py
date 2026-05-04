#!/usr/bin/env python3
"""
Fix Hermes custom endpoint config for gateways that serve inference at `/responses`
on the root base URL (no `/v1`), and migrate inline API keys out of config.yaml.

Actions (when --apply):
- Backup ~/.hermes/config.yaml and ~/.hermes/.env (if present)
- Trim model.base_url trailing `/v1` (e.g. http://host/v1 -> http://host)
- Move model.api_key and matching custom_providers[].api_key into ~/.hermes/.env as OPENAI_API_KEY
- Remove api_key fields from config.yaml (avoid plaintext secrets in YAML)
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _mask_secret(secret: str) -> str:
    s = (secret or "").strip()
    if not s:
        return ""
    if len(s) <= 10:
        return s[:2] + "..." + s[-2:]
    return s[:4] + "..." + s[-4:]


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # provided by hermes deps

    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    import yaml

    # Keep it readable; hermes config loader uses safe_load, order not critical.
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
    path.write_text(text, encoding="utf-8")


def _backup_file(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + f".bak.{_timestamp()}")
    bak.write_bytes(path.read_bytes())
    return bak


def _trim_v1(base_url: str) -> Tuple[str, bool]:
    b = (base_url or "").strip().rstrip("/")
    if b.endswith("/v1"):
        return b[:-3], True
    return b, False


def _read_env(path: Path) -> List[str]:
    if not path.exists():
        return []
    # Be forgiving on encoding.
    try:
        return path.read_text(encoding="utf-8").splitlines(True)
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1").splitlines(True)


def _write_env(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


_ENV_KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")


def _set_env_var(lines: List[str], key: str, value: str) -> Tuple[List[str], bool]:
    """
    Set/replace KEY=value. Keeps comments and formatting. Returns (new_lines, changed).
    """
    changed = False
    out: List[str] = []
    found = False
    for line in lines:
        m = _ENV_KEY_RE.match(line)
        if m and m.group(1) == key:
            found = True
            cur = m.group(2).strip()
            if cur != value:
                out.append(f"{key}={value}\n")
                changed = True
            else:
                out.append(line if line.endswith("\n") else (line + "\n"))
            continue
        out.append(line if line.endswith("\n") else (line + "\n"))
    if not found:
        if out and not out[-1].endswith("\n"):
            out[-1] += "\n"
        # Add a separating newline if the file doesn't already end with one.
        if out and out[-1].strip():
            out.append("\n")
        out.append(f"{key}={value}\n")
        changed = True
    return out, changed


def _get_hermes_home() -> Path:
    # Prefer hermes' profile-aware resolver when available.
    try:
        from hermes_constants import get_hermes_home  # type: ignore

        return Path(get_hermes_home())
    except Exception:
        hh = os.getenv("HERMES_HOME", "").strip()
        if hh:
            return Path(hh)
        return Path.home() / ".hermes"


def _collect_keys_to_migrate(cfg: Dict[str, Any], *, match_base_url: str) -> List[str]:
    keys: List[str] = []
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    if isinstance(model_cfg, dict):
        k = str(model_cfg.get("api_key", "") or "").strip()
        if k:
            keys.append(k)
    cps = cfg.get("custom_providers")
    if isinstance(cps, list):
        for entry in cps:
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("base_url", "") or "").strip().rstrip("/")
            if url != match_base_url.rstrip("/"):
                continue
            k = str(entry.get("api_key", "") or "").strip()
            if k:
                keys.append(k)
    # Dedup preserve order
    seen = set()
    out: List[str] = []
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _remove_inline_keys(cfg: Dict[str, Any], *, match_base_url: str) -> int:
    removed = 0
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else None
    if isinstance(model_cfg, dict) and model_cfg.get("api_key"):
        model_cfg.pop("api_key", None)
        removed += 1
    cps = cfg.get("custom_providers")
    if isinstance(cps, list):
        for entry in cps:
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("base_url", "") or "").strip().rstrip("/")
            if url != match_base_url.rstrip("/"):
                continue
            if entry.get("api_key"):
                entry.pop("api_key", None)
                removed += 1
    return removed


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write changes to disk (otherwise dry-run).")
    ap.add_argument("--env-key", default="OPENAI_API_KEY", help="Which env var to write the key to.")
    args = ap.parse_args(argv)

    hermes_home = _get_hermes_home()
    cfg_path = hermes_home / "config.yaml"
    env_path = hermes_home / ".env"

    cfg = _load_yaml(cfg_path)
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    provider = str(model_cfg.get("provider", "") or "").strip().lower() if isinstance(model_cfg, dict) else ""
    base_url = str(model_cfg.get("base_url", "") or "").strip() if isinstance(model_cfg, dict) else ""

    if provider != "custom":
        print(f"Nothing to do: model.provider is not 'custom' (found: {provider or '(empty)'}).")
        return 0
    if not base_url:
        print("Nothing to do: model.base_url is empty.")
        return 0

    trimmed_base, did_trim = _trim_v1(base_url)

    keys_to_migrate = _collect_keys_to_migrate(cfg, match_base_url=base_url)
    key_for_env = keys_to_migrate[0] if keys_to_migrate else ""

    print("Hermes config fix (custom endpoint)")
    print(f"- hermes_home: {hermes_home}")
    print(f"- config:      {cfg_path}")
    print(f"- env:         {env_path}")
    print(f"- base_url:    {base_url.rstrip('/')}")
    if did_trim:
        print(f"- new_base:    {trimmed_base}")
    else:
        print("- new_base:    (unchanged)")
    print(f"- migrate key: {_mask_secret(key_for_env) if key_for_env else '(none found)'} -> {args.env_key}")

    if not args.apply:
        print("\nDry-run only. Re-run with --apply to write changes.")
        return 0

    if not cfg_path.exists():
        print(f"Error: missing config file at {cfg_path}")
        return 2

    bak_cfg = _backup_file(cfg_path)
    bak_env = _backup_file(env_path) if env_path.exists() else None
    print(f"\nBackups:")
    print(f"- {bak_cfg}")
    if bak_env:
        print(f"- {bak_env}")

    # Apply base_url fix
    if isinstance(model_cfg, dict):
        model_cfg["base_url"] = trimmed_base if did_trim else base_url.rstrip("/")
        cfg["model"] = model_cfg

    # Move key into .env, remove from YAML
    if key_for_env:
        env_lines = _read_env(env_path)
        env_lines, changed_env = _set_env_var(env_lines, args.env_key, key_for_env)
        if changed_env:
            _write_env(env_path, env_lines)
            print(f"Wrote {args.env_key} to {env_path}")
        removed = _remove_inline_keys(cfg, match_base_url=base_url)
        if removed:
            print(f"Removed {removed} inline api_key field(s) from config.yaml")

    _dump_yaml(cfg_path, cfg)
    print(f"Updated {cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(list(os.sys.argv[1:])))

