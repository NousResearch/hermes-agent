"""One-shot local control-socket request for an MCP server after atomic promotion.

Example: python -m gateway.mcp_reload_cli smart-web --home ~/.hermes
Exit 0: reloaded; 75: still pending (retry later); 1: failed/unavailable.
"""
import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    from gateway.control_socket import reload_gateway_mcp_server
    from hermes_constants import get_hermes_home

    parser = argparse.ArgumentParser(description="Reload one MCP server in the running gateway")
    parser.add_argument("name", help="exact configured MCP server name")
    parser.add_argument("--home", type=Path, default=None, help="gateway control-socket home")
    parser.add_argument("--profile-home", type=Path, default=None, help="served profile home (default: gateway home)")
    parser.add_argument("--drain-timeout", type=float, default=15.0, help="max seconds to drain (0..60)")
    args = parser.parse_args(argv)
    if not 0 <= args.drain_timeout <= 60:
        parser.error("--drain-timeout must be between 0 and 60 seconds")
    home = args.home or Path(get_hermes_home())
    response = reload_gateway_mcp_server(
        home, args.name, profile_home=args.profile_home or home, drain_timeout=args.drain_timeout)
    if response is None:
        response = {"status": "unavailable", "error": "gateway did not acknowledge scoped reload"}
    print(json.dumps(response, ensure_ascii=False, sort_keys=True))
    status = response.get("status")
    return 0 if status == "reloaded" else 75 if status == "pending" and response.get("retry", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
