"""CLI commands for Hindsight integration management.

Handles: hermes hindsight setup | status | bank | budget
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

GLOBAL_CONFIG_PATH = Path.home() / ".hindsight" / "config.json"
HOST = "hermes"


def _read_config() -> dict:
    if GLOBAL_CONFIG_PATH.exists():
        try:
            return json.loads(GLOBAL_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _write_config(cfg: dict) -> None:
    GLOBAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_CONFIG_PATH.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _prompt(label: str, default: str | None = None, secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    sys.stdout.write(f"  {label}{suffix}: ")
    sys.stdout.flush()
    if secret:
        if sys.stdin.isatty():
            import getpass
            val = getpass.getpass(prompt="")
        else:
            val = sys.stdin.readline().strip()
    else:
        val = sys.stdin.readline().strip()
    return val or (default or "")


def _ensure_sdk_installed() -> bool:
    """Check hindsight-client is importable; offer to install if not. Returns True if ready."""
    try:
        import hindsight_client  # noqa: F401
        return True
    except ImportError:
        pass

    print("  hindsight-client is not installed.")
    answer = _prompt("Install it now? (hindsight-client>=0.4.0)", default="y")
    if answer.lower() not in ("y", "yes"):
        print("  Skipping install. Run: pip install 'hindsight-client>=0.4.0'\n")
        return False

    import subprocess
    print("  Installing hindsight-client...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "hindsight-client>=0.4.0"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("  Installed.\n")
        return True
    else:
        print(f"  Install failed:\n{result.stderr.strip()}")
        print("  Run manually: pip install 'hindsight-client>=0.4.0'\n")
        return False


def cmd_setup(args) -> None:
    """Interactive Hindsight setup wizard."""
    cfg = _read_config()

    print("\nHindsight memory setup\n" + "─" * 40)
    print("  Hindsight gives Hermes persistent long-term memory across sessions.")
    print("  Config is stored at ~/.hindsight/config.json\n")

    if not _ensure_sdk_installed():
        return

    banks = cfg.setdefault("banks", {})
    hermes_bank = banks.setdefault(HOST, {})

    # API key
    current_key = cfg.get("apiKey", "")
    masked = f"...{current_key[-8:]}" if len(current_key) > 8 else ("set" if current_key else "not set")
    print(f"  Current API key: {masked}")
    new_key = _prompt("Hindsight API key (leave blank to keep current)", secret=True)
    if new_key:
        cfg["apiKey"] = new_key

    effective_key = cfg.get("apiKey", "")
    if not effective_key:
        print("\n  No API key configured. Get your API key at https://app.hindsight.vectorize.io")
        print("  Run 'hermes hindsight setup' again once you have a key.\n")
        return

    # Bank ID
    current_bank = hermes_bank.get("bankId") or HOST
    new_bank = _prompt("Bank ID", default=current_bank)
    if new_bank:
        hermes_bank["bankId"] = new_bank
        hermes_bank["name"] = new_bank  # bank name == bank ID

    # Budget
    current_budget = hermes_bank.get("budget", "mid")
    print(f"\n  Recall budget options:")
    print("    low   — faster, fewer results")
    print("    mid   — balanced (default)")
    print("    high  — thorough, slower")
    new_budget = _prompt("Recall budget", default=current_budget)
    if new_budget in ("low", "mid", "high"):
        hermes_bank["budget"] = new_budget
    else:
        hermes_bank["budget"] = "mid"

    hermes_bank.setdefault("enabled", True)

    _write_config(cfg)
    print(f"\n  Config written to {GLOBAL_CONFIG_PATH}")

    # Test connection — create bank (idempotent)
    print("  Testing connection... ", end="", flush=True)
    try:
        from hindsight_integration.client import HindsightClientConfig, get_hindsight_client, reset_hindsight_client
        reset_hindsight_client()
        hcfg = HindsightClientConfig.from_global_config()
        client = get_hindsight_client(hcfg)
        try:
            client.create_bank(bank_id=hcfg.bank_id, name=hcfg.bank_id)
        except Exception:
            pass  # Already exists — that's fine
        print("OK")
    except Exception as e:
        print(f"FAILED\n  Error: {e}")
        return

    print(f"\n  Hindsight is ready.")
    print(f"  Bank ID:  {hcfg.bank_id}")
    print(f"  Budget:   {hcfg.budget}")
    print(f"\n  Hindsight tools available in chat:")
    print(f"    hindsight_retain  — store information to long-term memory")
    print(f"    hindsight_recall  — search memories with multi-strategy retrieval")
    print(f"    hindsight_reflect — synthesize a reasoned answer from stored memories")
    print(f"\n  Other commands:")
    print(f"    hermes hindsight status       — show config and connection status")
    print(f"    hermes hindsight bank <id>    — show or change bank ID")
    print(f"    hermes hindsight budget <lvl> — show or set recall budget\n")


def cmd_status(args) -> None:
    """Show current Hindsight config and connection status."""
    try:
        import hindsight_client  # noqa: F401
    except ImportError:
        print("  hindsight-client is not installed. Run: hermes hindsight setup\n")
        return

    cfg = _read_config()

    if not cfg:
        print("  No Hindsight config found at ~/.hindsight/config.json")
        print("  Run 'hermes hindsight setup' to configure.\n")
        return

    try:
        from hindsight_integration.client import HindsightClientConfig, get_hindsight_client
        hcfg = HindsightClientConfig.from_global_config()
    except Exception as e:
        print(f"  Config error: {e}\n")
        return

    api_key = hcfg.api_key or ""
    masked = f"...{api_key[-8:]}" if len(api_key) > 8 else ("set" if api_key else "not set")

    print(f"\nHindsight status\n" + "─" * 40)
    print(f"  Enabled:      {hcfg.enabled}")
    print(f"  API key:      {masked}")
    print(f"  Bank ID:      {hcfg.bank_id}")
    print(f"  Budget:       {hcfg.budget}")
    print(f"  Base URL:     {hcfg.base_url}")
    print(f"  Config path:  {GLOBAL_CONFIG_PATH}")

    if hcfg.enabled and hcfg.api_key:
        print("\n  Connection... ", end="", flush=True)
        try:
            get_hindsight_client(hcfg)
            print("OK\n")
        except Exception as e:
            print(f"FAILED ({e})\n")
    else:
        reason = "disabled" if not hcfg.enabled else "no API key"
        print(f"\n  Not connected ({reason})\n")


def cmd_bank(args) -> None:
    """Show or set the bank ID."""
    cfg = _read_config()
    bank_id_arg = getattr(args, "bank_id", None)

    if bank_id_arg is None:
        current = (cfg.get("banks") or {}).get(HOST, {}).get("bankId") or HOST
        print(f"\nHindsight bank\n" + "─" * 40)
        print(f"  Current bank ID: {current}")
        print(f"\n  Set with: hermes hindsight bank <bank-id>\n")
        return

    bank_id = bank_id_arg.strip()
    if not bank_id:
        print("  Bank ID cannot be empty.\n")
        return

    cfg.setdefault("banks", {}).setdefault(HOST, {})["bankId"] = bank_id
    cfg["banks"][HOST]["name"] = bank_id
    _write_config(cfg)
    print(f"  Bank ID → {bank_id}")
    print(f"  Saved to {GLOBAL_CONFIG_PATH}\n")


def cmd_budget(args) -> None:
    """Show or set the recall budget."""
    BUDGETS = {
        "low": "faster, fewer results",
        "mid": "balanced (default)",
        "high": "thorough, slower",
    }
    cfg = _read_config()
    budget_arg = getattr(args, "budget", None)

    if budget_arg is None:
        current = (cfg.get("banks") or {}).get(HOST, {}).get("budget", "mid")
        print(f"\nHindsight recall budget\n" + "─" * 40)
        for b, desc in BUDGETS.items():
            marker = " ←" if b == current else ""
            print(f"  {b:<6}  {desc}{marker}")
        print(f"\n  Set with: hermes hindsight budget [low|mid|high]\n")
        return

    if budget_arg not in BUDGETS:
        print(f"  Invalid budget '{budget_arg}'. Options: {', '.join(BUDGETS)}\n")
        return

    cfg.setdefault("banks", {}).setdefault(HOST, {})["budget"] = budget_arg
    _write_config(cfg)
    print(f"  Budget → {budget_arg}  ({BUDGETS[budget_arg]})\n")


def hindsight_command(args) -> None:
    """Route hindsight subcommands."""
    sub = getattr(args, "hindsight_command", None)
    if sub == "setup" or sub is None:
        cmd_setup(args)
    elif sub == "status":
        cmd_status(args)
    elif sub == "bank":
        cmd_bank(args)
    elif sub == "budget":
        cmd_budget(args)
    else:
        print(f"  Unknown hindsight command: {sub}")
        print("  Available: setup, status, bank, budget\n")
