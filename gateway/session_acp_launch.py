"""Private ACP launch credentials, scoped to an admitted gateway turn."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import uuid

from acp_adapter.launch_env import LaunchEnvironment, launch_env_prefix
from gateway.session_context import bound_session_key
from persistence.core.errors import AuthorityRefusal

ACP_LAUNCH_BINDING = "acp_launch_binding_v1"
_turn = ContextVar("acp_launch_terminal_turn", default=None)


@dataclass
class _TerminalTurn:
    route: str
    launch: LaunchEnvironment | None = field(repr=False)


def bind_admission(authority, ref, actor, launch: LaunchEnvironment):
    """Only the ACP owner can give its turn the credentials from a private local grant."""
    from gateway.owner_identity import same_owner
    from gateway.session_managed_worker import policy_if_managed
    from gateway.session_policy import policy_for_source

    live = authority.sessions[ref.session_id]
    policy = policy_for_source(authority.runner, live.source)
    if policy is None or policy.source != "acp" or not same_owner(live.source.user_id, actor.subject):
        raise AuthorityRefusal("permission_denied")
    if policy_if_managed(policy) is not None:
        raise AuthorityRefusal("unsupported_managed_policy")
    if live.acp_launch_binding is None:
        live.acp_launch_binding = (uuid.uuid4().hex, launch)
    binding, held = live.acp_launch_binding
    if held != launch:
        raise AuthorityRefusal("admission_conflict", detail="ACP launch credentials changed; start a new session")
    # Only an opaque binding ID goes into the durable FIFO. After a gateway restart it must
    # fail closed rather than execute with ambient credentials or a later client's identity.
    return {ACP_LAUNCH_BINDING: binding}


@contextmanager
def turn_scope(live, payload):
    bound = None
    if (expected := payload.get(ACP_LAUNCH_BINDING)) is not None:
        binding = live.acp_launch_binding
        if binding is None or binding[0] != expected:
            raise AuthorityRefusal("launch_credentials_unavailable")
        bound = _TerminalTurn(live.route, binding[1])
    token = _turn.set(bound)
    try:
        yield
    finally:
        if bound is not None:
            bound.launch = None
        _turn.reset(token)


def active_launch_env() -> LaunchEnvironment | None:
    """The current route's credentials, never the daemon's own `os.environ`."""
    bound = _turn.get()
    return bound.launch if bound is not None and bound.route == bound_session_key() else None


def apply_terminal_env(env: dict) -> None:
    """Replace the launch namespace with the turn's own values after the usual scrub."""
    if (launch := active_launch_env()) is not None:
        prefix = launch_env_prefix(launch)
        for name in [name for name in env if name.upper().startswith(prefix)]:
            del env[name]
        env.update(launch["env"])
