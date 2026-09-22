"""Read-only guided-routing diagnostics; never open or publish a runtime store."""
import json
import time

from agent.model_selection import _validate_policy, select
from agent.model_selection_types import RoutingBlocked


def run_diagnostic(args) -> int:
    try:
        with open(args.policy_json, encoding="utf-8") as stream:
            policy = json.load(stream)
        _validate_policy(policy)
        if args.routing_action == "validate":
            result = {"valid": True, "policy_id": policy["policy_id"], "revision": policy["revision"],
                      "note": "Schema validation is not capability verification, admission or activation."}
        else:
            with open(args.requirements_json, encoding="utf-8") as stream:
                requirements = json.load(stream)
            result = select(requirements, policy, {}, now=int(time.time()))
        code = 0
    except RoutingBlocked as exc:
        result = {"valid": False, "reason": exc.reason, "detail": exc.detail, "rejections": exc.rejections}
        code = 1
    except (OSError, ValueError) as exc:
        result = {"valid": False, "reason": "schema_invalid", "detail": str(exc)}
        code = 1
    # JSON is also readable at a terminal; one representation avoids divergent explanations.
    print(json.dumps(result, indent=2, sort_keys=True))
    return code
