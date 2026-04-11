"""EverOS CLI commands — hermes everos status/config/reset."""


def _load_config():
    """Load EverOS config from hermes home."""
    import json
    from pathlib import Path
    from hermes_constants import get_hermes_home

    config_path = get_hermes_home() / "everos.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def _get_url():
    """Get EverOS base URL."""
    import os
    cfg = _load_config()
    return os.environ.get("EVEROS_URL", cfg.get("url", "http://localhost:1995"))


def everos_command(args):
    """Handler dispatched by argparse."""
    sub = getattr(args, "everos_command", None)
    if sub == "status":
        _cmd_status(args)
    elif sub == "config":
        _cmd_config(args)
    elif sub == "reset":
        _cmd_reset(args)
    else:
        print("Usage: hermes everos <status|config|reset>")


def _cmd_status(args):
    """Show EverOS connection status and memory stats."""
    import json
    from urllib.request import urlopen, Request
    from urllib.error import URLError

    base_url = _get_url().rstrip("/")
    print(f"EverOS URL: {base_url}")

    # Health check
    try:
        req = Request(f"{base_url}/health")
        with urlopen(req, timeout=5) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        print(f"Status: {health.get('status', 'unknown')}")
    except URLError as e:
        print(f"Status: UNREACHABLE ({e})")
        return
    except Exception as e:
        print(f"Status: ERROR ({e})")
        return

    # Config
    cfg = _load_config()
    user_id = cfg.get("user_id", "hermes-user")
    print(f"User ID: {user_id}")

    # Try to fetch memory count
    try:
        req = Request(f"{base_url}/api/v1/memories")
        req.add_header("Content-Type", "application/json")
        data = json.dumps({"user_id": user_id, "memory_type": "episodic_memory", "limit": 1}).encode()
        req.data = data
        req.method = "GET"
        with urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        mems = result.get("result", {}).get("memories", [])
        print(f"Episodic memories: {len(mems)}+ (limited query)")
    except Exception:
        print("Episodic memories: unable to query")

    print("\nEverOS is active as memory provider.")


def _cmd_config(args):
    """Show current EverOS configuration."""
    import json
    cfg = _load_config()
    if cfg:
        print("Current EverOS config:")
        print(json.dumps(cfg, indent=2))
    else:
        print("No everos.json found. Using defaults:")
        print("  URL: http://localhost:1995")
        print("  User ID: hermes-user")
    print(f"\nConfig file: ~/.hermes/everos.json")


def _cmd_reset(args):
    """Delete all EverOS memories for the configured user."""
    import json
    from urllib.request import urlopen, Request

    base_url = _get_url().rstrip("/")
    cfg = _load_config()
    user_id = cfg.get("user_id", "hermes-user")

    confirm = input(f"Delete ALL memories for user '{user_id}'? This cannot be undone. [y/N] ")
    if confirm.lower() != "y":
        print("Cancelled.")
        return

    try:
        req = Request(
            f"{base_url}/api/v1/memories",
            data=json.dumps({"user_id": user_id}).encode(),
            method="DELETE",
        )
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        count = result.get("result", {}).get("count", "?")
        print(f"Deleted {count} memories.")
    except Exception as e:
        print(f"Delete failed: {e}")


def register_cli(subparser) -> None:
    """Build the hermes everos argparse tree.

    Called by discover_plugin_cli_commands() at argparse setup time.
    """
    subs = subparser.add_subparsers(dest="everos_command")
    subs.add_parser("status", help="Show EverOS connection status and memory stats")
    subs.add_parser("config", help="Show current EverOS configuration")
    subs.add_parser("reset", help="Delete all EverOS memories for current user")
    subparser.set_defaults(func=everos_command)
