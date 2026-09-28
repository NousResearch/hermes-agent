"""Machine-readable ensure boundary. No bootstrap credentials leave this command."""
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
