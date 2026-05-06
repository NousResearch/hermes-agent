#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-interactive EvoTraders integration setup for Hermes Agent.

This writes the minimal knobs into the active profile:
- .env:
  - EVOTRADERS_WINAPI_BASE
  - HERMES_PROMPT_PRESET=evotraders_identity,evotraders_policy
- config.yaml:
  - enables the `evotraders` toolset for the CLI platform
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hermes_cli.cli_output import print_info, print_success, print_warning
from hermes_cli.config import ensure_hermes_home, load_config, save_config, save_env_value


def _enable_evotraders_toolset(config: Dict[str, Any]) -> bool:
    """Enable the `evotraders` toolset on CLI platform. Returns True if changed."""
    try:
        # Reuse the canonical save path so we don't fork toolset semantics.
        from hermes_cli.tools_config import _get_platform_tools, _save_platform_tools  # noqa: PLC0415
    except Exception:
        return False

    current = _get_platform_tools(config, "cli", include_default_mcp_servers=False)
    if "evotraders" in current:
        return False
    updated = set(current)
    updated.add("evotraders")
    _save_platform_tools(config, "cli", updated)
    return True


def add_subparser(subparsers) -> None:
    """Register `hermes evotraders setup` subcommand on the main CLI."""
    evo = subparsers.add_parser(
        "evotraders",
        help="EvoTraders integration helpers (WinAPI bridge + prompt preset)",
    )
    evo_sub = evo.add_subparsers(dest="evotraders_command")
    setup = evo_sub.add_parser(
        "setup",
        help="Configure EvoTraders bridge + presets in the current profile",
    )
    setup.add_argument(
        "--winapi-base",
        default="",
        help="WinAPI base URL, e.g. http://192.168.100.168:18880",
    )
    setup.add_argument(
        "--preset",
        default="evotraders_identity,evotraders_policy",
        help="Comma-separated HERMES_PROMPT_PRESET value (default: evotraders_identity,evotraders_policy)",
    )
    setup.set_defaults(func=cmd_evotraders_setup)
    doctor = evo_sub.add_parser(
        "doctor",
        help="Diagnose EvoTraders integration in current profile",
    )
    doctor.add_argument(
        "--timeout-sec",
        type=float,
        default=5.0,
        help="WinAPI health check timeout seconds (default: 5)",
    )
    doctor.set_defaults(func=cmd_evotraders_doctor)
    smoke = evo_sub.add_parser(
        "smoke",
        help="Run EvoTraders minimal smoke checks (health + snapshot + indicator)",
    )
    smoke.add_argument(
        "--stock-code",
        default="000001.SZ",
        help="Stock code for smoke checks (default: 000001.SZ)",
    )
    smoke.add_argument(
        "--timeout-sec",
        type=float,
        default=8.0,
        help="HTTP timeout seconds (default: 8)",
    )
    smoke.add_argument(
        "--out-json",
        default="",
        help="Optional output JSON path (default: <HERMES_HOME>/reports/evotraders_smoke.json)",
    )
    smoke.set_defaults(func=cmd_evotraders_smoke)


def cmd_evotraders_setup(args: argparse.Namespace) -> None:
    ensure_hermes_home()
    winapi = str(getattr(args, "winapi_base", "") or "").strip().rstrip("/")
    preset = str(getattr(args, "preset", "") or "").strip()

    if not winapi:
        print_warning("Missing --winapi-base. Example: hermes evotraders setup --winapi-base http://192.168.100.168:18880")
        return

    save_env_value("EVOTRADERS_WINAPI_BASE", winapi)
    save_env_value("HERMES_PROMPT_PRESET", preset or "evotraders_identity,evotraders_policy")

    cfg = load_config()
    changed = _enable_evotraders_toolset(cfg)
    save_config(cfg)

    print_success("EvoTraders setup saved for this profile.")
    print_info(f"EVOTRADERS_WINAPI_BASE={winapi}")
    print_info(f"HERMES_PROMPT_PRESET={preset or 'evotraders_identity,evotraders_policy'}")
    if changed:
        print_info("Enabled toolset: evotraders (platform=cli)")
    else:
        print_info("Toolset already enabled: evotraders (platform=cli)")
    print_info("Self-check (after restart/new session):")
    print_info("  hermes chat -q \"分析 000001\" --toolsets evotraders")


def apply_evotraders_profile(profile_dir: Path, winapi_base: str, preset: str = "evotraders_identity,evotraders_policy") -> None:
    """Apply EvoTraders env/config to a specific profile directory."""
    old_home = os.environ.get("HERMES_HOME")
    try:
        os.environ["HERMES_HOME"] = str(profile_dir)
        ensure_hermes_home()
        save_env_value("EVOTRADERS_WINAPI_BASE", str(winapi_base).strip().rstrip("/"))
        save_env_value("HERMES_PROMPT_PRESET", str(preset).strip() or "evotraders_identity,evotraders_policy")
        cfg = load_config()
        _enable_evotraders_toolset(cfg)
        save_config(cfg)
    finally:
        if old_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = old_home


def _http_health(url: str, timeout_sec: float = 5.0) -> Dict[str, Any]:
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=max(1.0, float(timeout_sec))) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw.strip() else {}
            return {"ok": True, "status_code": int(getattr(resp, "status", 200)), "data": data}
    except HTTPError as e:
        return {"ok": False, "status_code": int(getattr(e, "code", 500) or 500), "error": "http_error"}
    except URLError as e:
        return {"ok": False, "status_code": 0, "error": f"url_error:{e}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "status_code": 0, "error": f"request_failed:{e}"}


def _http_json(method: str, url: str, body: Dict[str, Any] | None, timeout_sec: float) -> Dict[str, Any]:
    payload = None
    if body is not None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = Request(
        url=url,
        data=payload,
        method=str(method).upper(),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urlopen(req, timeout=max(1.0, float(timeout_sec))) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            data = json.loads(text) if text.strip() else {}
            return {"ok": True, "status_code": int(getattr(resp, "status", 200)), "data": data}
    except HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        return {
            "ok": False,
            "status_code": int(getattr(e, "code", 500) or 500),
            "error": "http_error",
            "raw": raw[:800],
        }
    except URLError as e:
        return {"ok": False, "status_code": 0, "error": f"url_error:{e}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "status_code": 0, "error": f"request_failed:{e}"}


def cmd_evotraders_doctor(args: argparse.Namespace) -> None:
    ensure_hermes_home()
    cfg = load_config()
    current = set()
    try:
        from hermes_cli.tools_config import _get_platform_tools  # noqa: PLC0415

        current = _get_platform_tools(cfg, "cli", include_default_mcp_servers=False)
    except Exception:
        pass

    from hermes_cli.config import get_env_value  # lazy import to avoid cycles

    winapi = str(get_env_value("EVOTRADERS_WINAPI_BASE") or "").strip().rstrip("/")
    preset = str(get_env_value("HERMES_PROMPT_PRESET") or "").strip()

    print_info("EvoTraders doctor report:")
    print_info(f"- EVOTRADERS_WINAPI_BASE: {'set' if winapi else 'missing'}")
    if winapi:
        print_info(f"  value: {winapi}")
    print_info(f"- HERMES_PROMPT_PRESET: {preset or '(missing)'}")
    print_info(f"- cli toolset includes evotraders: {('evotraders' in current)}")

    if winapi:
        h = _http_health(f"{winapi}/v1/health", timeout_sec=float(getattr(args, "timeout_sec", 5.0)))
        print_info(f"- winapi /v1/health: ok={h.get('ok')} status={h.get('status_code')}")
    else:
        print_warning("- winapi /v1/health: skipped (EVOTRADERS_WINAPI_BASE missing)")

    if winapi and preset and ("evotraders" in current):
        print_success("EvoTraders integration looks ready.")
    else:
        print_warning("EvoTraders integration is incomplete; run:")
        print_warning("  hermes evotraders setup --winapi-base http://<WIN-IP>:18880")


def cmd_evotraders_smoke(args: argparse.Namespace) -> None:
    ensure_hermes_home()
    from hermes_cli.config import get_env_value  # lazy import

    base = str(get_env_value("EVOTRADERS_WINAPI_BASE") or "").strip().rstrip("/")
    if not base:
        print_warning("EVOTRADERS_WINAPI_BASE is missing. Run:")
        print_warning("  hermes evotraders setup --winapi-base http://<WIN-IP>:18880")
        return

    timeout_sec = float(getattr(args, "timeout_sec", 8.0) or 8.0)
    stock_code = str(getattr(args, "stock_code", "000001.SZ") or "000001.SZ").strip()

    checks: list[tuple[str, Dict[str, Any]]] = []
    checks.append(("health", _http_json("GET", f"{base}/v1/health", body=None, timeout_sec=timeout_sec)))
    checks.append(
        (
            "market_mainline",
            _http_json(
                "GET",
                f"{base}/v1/proxy/v1/market/sentiment-context?max_age_sec=60",
                body=None,
                timeout_sec=timeout_sec,
            ),
        )
    )
    checks.append(
        (
            "snapshot",
            _http_json(
                "POST",
                f"{base}/v1/tq/get_market_snapshot",
                body={"stock_code": stock_code},
                timeout_sec=timeout_sec,
            ),
        )
    )
    checks.append(
        (
            "indicator",
            _http_json(
                "POST",
                f"{base}/v1/tq/formula_indicator_series",
                body={"stock_code": stock_code, "count": 5},
                timeout_sec=timeout_sec,
            ),
        )
    )

    all_ok = True
    print_info("EvoTraders smoke checks:")
    checks_dict: Dict[str, Any] = {}
    for name, res in checks:
        ok = bool(res.get("ok")) and int(res.get("status_code") or 0) < 400
        all_ok = all_ok and ok
        checks_dict[name] = {
            "ok": ok,
            "status_code": res.get("status_code"),
            "error": res.get("error", ""),
        }
        print_info(f"- {name}: ok={ok} status={res.get('status_code')}")
        if not ok and res.get("error"):
            print_warning(f"  error: {res.get('error')}")

    out_json = str(getattr(args, "out_json", "") or "").strip()
    if not out_json:
        from hermes_cli.config import get_hermes_home  # lazy import

        out_json = str(get_hermes_home() / "reports" / "evotraders_smoke.json")
    out_path = Path(out_json).expanduser()
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "ok": True,
        "schema": "hermes_evotraders_smoke.v1",
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "winapi_base": base,
        "stock_code": stock_code,
        "timeout_sec": timeout_sec,
        "all_ok": all_ok,
        "checks": checks_dict,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_info(f"Smoke report: {out_path}")

    if all_ok:
        print_success("SMOKE PASS: WinAPI relay + core tq calls are healthy.")
    else:
        print_warning("SMOKE FAIL: one or more checks failed.")

