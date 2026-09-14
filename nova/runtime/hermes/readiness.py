"""Whether a materialized agent has the credentials it needs to actually run.

NOVA writes the *name* of every variable an agent needs and never the value, so there is a
gap by design between "configured" and "runnable". This closes it the only way NOVA honestly
can: by reporting, precisely, which variables are missing and which file the operator has to
put them in.

The alternative — writing the secret — is the one thing this whole design exists to avoid.
The other alternative, saying nothing, produces an agent that materializes cleanly and fails
on its first task with a provider error at whatever hour the first task runs.

Resolution mirrors the runtime's own order, verified in
``hermes_cli/env_loader.py::load_hermes_dotenv`` and
``agent/auxiliary_client.py::_scoped_key_env``:

    <profile>/.env   ->   the process environment

``<profile>/.env`` is checked first because it is what the runtime loads with
``override=True``, and it is the file NOVA is forbidden from writing — which makes it
exactly the right place for the operator to own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

# The parser lives in the platform layer: a dotenv file is a format, not a runtime
# concept, and backup needs it too. Re-exported so this module's callers are unaffected.
from nova._env import ENV_FILENAME, read_env_file  # noqa: E402,F401


@dataclass(frozen=True)
class Readiness:
    """What one agent still needs before it can run."""

    agent_id: str
    required: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    env_file: Optional[Path] = None
    #: Where each satisfied variable was found, for an operator debugging the wrong value.
    resolved_from: Mapping[str, str] = field(default_factory=dict)
    #: Variables resolved from the host environment rather than this agent's own file, and
    #: therefore shared with every other agent on the host.
    host_wide: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "ready": self.ready,
            "required": list(self.required),
            "missing": list(self.missing),
            "env_file": str(self.env_file) if self.env_file else "",
            "resolved_from": dict(self.resolved_from),
            "host_wide": list(self.host_wide),
        }

    def explain(self) -> str:
        """What the operator has to do, naming the file and the variables."""
        if self.ready:
            return f"{self.agent_id}: ready"
        names = ", ".join(self.missing)
        where = self.env_file or "<profile>/.env"
        return (
            f"{self.agent_id}: missing {names}. Add them to {where} — NOVA never writes "
            "credentials, so this file is yours and survives every apply"
        )


def check(
    agent_id: str,
    required: Iterable[str],
    *,
    profile_dir: Path,
    environ: Optional[Mapping[str, str]] = None,
) -> Readiness:
    """Which of ``required`` this agent can already resolve, and from where."""
    environ = os.environ if environ is None else environ
    names = tuple(dict.fromkeys(name for name in required if name))
    env_file = profile_dir / ENV_FILENAME
    from_file = read_env_file(env_file)

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        if name in from_file:
            resolved[name] = str(env_file)
        elif environ.get(name):
            # Present, but host-wide: every agent on this box resolves the same value.
            # Worth distinguishing, because it is the configuration that silently stops
            # being per-agent the moment a second tenant lands on the host.
            resolved[name] = "process environment"
        else:
            missing.append(name)

    # A host-wide value is not per-agent. Every agent on this machine resolves the same
    # credential, because a worker inherits the dispatcher's environment and the runtime's
    # multiplex guard is inactive in a spawned worker. Worth saying out loud on a host with
    # more than one agent: it is the difference between the isolation NOVA claims and the
    # isolation this deployment actually has.
    shared = sorted(name for name, where in resolved.items() if where == "process environment")

    return Readiness(
        agent_id=agent_id,
        required=names,
        missing=tuple(missing),
        env_file=env_file,
        resolved_from=resolved,
        host_wide=tuple(shared),
    )


    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export "):].lstrip()
        name, separator, _value = stripped.partition("=")
        name = name.strip()
        if separator and name:
            names[name] = ""  # presence only; the value is not kept
    return names
