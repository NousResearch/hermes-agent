"""Start an off-turn OAuth operation on the gateway serving this terminal turn."""
from pathlib import Path


def gateway_oauth_login(args):
    from gateway.control_socket import query_gateway_control
    from gateway.session_context import get_session_env
    from hermes_constants import get_hermes_home

    session = get_session_env("HERMES_SESSION_ID")
    user = get_session_env("HERMES_SESSION_USER_ID")
    if not session or not user:
        raise RuntimeError("Messaging OAuth requires a gateway-originated terminal session")
    home = get_hermes_home()
    params = {"session_id": session, "user_id": user, "home": str(home), "server": args.name,
              "action": "cancel" if getattr(args, "cancel", False) else "start"}
    if getattr(args, "url", None):
        params["url"] = args.url
    # Multiplex gateways own the default profile's socket, while credentials live
    # under the routed profile. The server independently validates that mapping.
    from hermes_cli.profiles import _get_default_hermes_home
    homes = dict.fromkeys((Path(home), _get_default_hermes_home()))
    for socket_home in homes:
        result = query_gateway_control(socket_home, "mcp-oauth", params=params)
        if result is not None:
            print("OAuth operation queued. Authorization and completion will arrive in the originating chat.")
            return
    raise RuntimeError("The originating gateway is unavailable or rejected this OAuth operation")
