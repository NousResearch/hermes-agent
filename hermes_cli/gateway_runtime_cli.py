"""Credential-free ensure discovery and private same-user ticket bootstrap."""
from dataclasses import asdict
import json
import sys
from contextlib import redirect_stdout


def ensure_exit_code(result) -> int:
    if result.reason_code == "deadline":
        return 5
    if result.reason_code in {"authorization", "profile_mismatch"}:
        return 4
    return {"ready": 0, "incompatible": 3, "draining": 6,
            "inaccessible": 7, "conflict": 7, "starting": 5, "absent": 5}[result.state]


def cmd_gateway_ensure(args) -> None:
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import ensure_gateway_runtime

    try:
        # Legacy imported utilities can print; protocol stdout belongs only to us.
        with redirect_stdout(sys.stderr):
            result = ensure_gateway_runtime(get_hermes_home(), timeout=float(args.timeout))
        payload = asdict(result)
        if result.endpoint:
            payload["endpoint"]["capabilities"] = sorted(result.endpoint.capabilities)
        code = ensure_exit_code(result)
    except (ValueError, TypeError):
        payload, code = {"state": "inaccessible", "reason_code": "invalid_invocation", "endpoint": None}, 2
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    raise SystemExit(code)


def cmd_gateway_ticket(args) -> None:
    """Private same-user transport bootstrap; never discover or start another owner.

    SSH carries the request on stdin and the result on its encrypted stdout. The
    exact endpoint must still be served; no credential or scope is accepted in argv.
    """
    from pathlib import Path
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from hermes_cli.gateway_client import _session_ticket

    try:
        request = json.loads(sys.stdin.buffer.read(65537))
        if (not isinstance(request, dict) or set(request) - {"profile_id", "instance_id", "purpose", "profile"} or not {"profile_id", "instance_id", "purpose"} <= set(request)
                or request["purpose"] not in {"interactive", "native-http"}):
            raise ValueError("invalid request")
        home = get_hermes_home().resolve()
        with redirect_stdout(sys.stderr):
            discovery = discover_gateway_endpoint(home)
            endpoint = discovery.endpoint
            if (discovery.state != "ready" or endpoint is None
                    or endpoint.profile_id != request["profile_id"]
                    or endpoint.instance_id != request["instance_id"]):
                raise ValueError("stale endpoint")
            profile = request.get("profile")
            if profile is not None:
                from hermes_cli.profiles import get_profile_dir, profile_exists
                if not isinstance(profile, str) or not profile_exists(profile):
                    raise ValueError("invalid profile")
                target_home = get_profile_dir(profile).resolve()
                target = discover_gateway_endpoint(target_home)
                if (target.state != "ready" or target.endpoint is None
                        or target.endpoint.instance_id != endpoint.instance_id
                        or Path(target.endpoint.control_home or target.endpoint.profile_id).resolve()
                           != Path(endpoint.control_home or endpoint.profile_id).resolve()):
                    raise ValueError("profile is not served by pinned owner")
                home, endpoint = target_home, target.endpoint
            ticket = _session_ticket(home, endpoint, purpose=request["purpose"])
        payload = {"ticket": ticket, "profile_id": endpoint.profile_id,
                   "instance_id": endpoint.instance_id, "runtime_protocol": 1, "profile": profile}
    except Exception:
        # A caller receives bounded diagnostics, never private control/socket data.
        print('{"error":"native_ticket_unavailable"}')
        raise SystemExit(4) from None
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
