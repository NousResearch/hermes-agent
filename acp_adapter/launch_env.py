"""Reviewed ACP launch-environment producers and their private wire format.

The client name selects a policy; it does not authenticate the caller. An ACP
client cannot nominate arbitrary environment names or register a policy at runtime.
"""
from dataclasses import dataclass
import json
import re
from typing import TypedDict


class LaunchEnvironment(TypedDict):
    client: str
    env: dict[str, str]


@dataclass(frozen=True)
class _Policy:
    prefix: str
    required: tuple[str, ...]
    excluded: frozenset[str]

    def allows(self, name: str) -> bool:
        return name.startswith(self.prefix) and _NAME.fullmatch(name) is not None and name not in self.excluded


_NAME = re.compile(r"[A-Z][A-Z0-9_]{1,127}")
_POLICIES = {
    "buzz-acp": _Policy(
        prefix="BUZZ_",
        required=("BUZZ_PRIVATE_KEY", "BUZZ_RELAY_URL"),
        # These are Desktop process-ownership stamps, not launch credentials.
        excluded=frozenset({"BUZZ_MANAGED_AGENT", "BUZZ_MANAGED_AGENT_START_NONCE"}),
    ),
}
LAUNCH_ENV_PREFIXES = tuple(policy.prefix for policy in _POLICIES.values())


def collect_launch_env(client: str | None) -> LaunchEnvironment | None:
    """Read only a registered producer's inherited values, before profile dotenv overrides."""
    if (policy := _POLICIES.get(client)) is None:
        return None
    from hermes_cli.env_loader import inherited_process_env

    environ = inherited_process_env()
    if not all(environ.get(name) for name in policy.required):
        return None
    return validate_launch_env({"client": client, "env": {k: v for k, v in environ.items() if policy.allows(k)}})


def validate_launch_env(value) -> LaunchEnvironment:
    """Validate the envelope independently on the receiving side of the private socket."""
    if not isinstance(value, dict) or set(value) != {"client", "env"}:
        raise ValueError("invalid ACP launch environment")
    client, env = value["client"], value["env"]
    policy = _POLICIES.get(client) if isinstance(client, str) else None
    if (policy is None or not isinstance(env, dict) or len(env) > 128
            or any(not isinstance(k, str) or not policy.allows(k)
                   or not isinstance(v, str) or "\0" in v for k, v in env.items())
            or not all(env.get(name) for name in policy.required)
            or len(json.dumps(env)) > 16 * 1024):
        raise ValueError("invalid ACP launch environment")
    return {"client": client, "env": dict(env)}


def launch_env_prefix(launch: LaunchEnvironment) -> str:
    return _POLICIES[launch["client"]].prefix
