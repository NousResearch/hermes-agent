"""Which credentials an agent legitimately needs, and nothing else.

NOVA used to be unable to write a secret at all: ``.env`` is on the materialiser's
``NEVER_WRITE`` list and there was no other path. An administrator asked for credential
entry in the Control Centre, so there is now exactly one — and this module is the reason it
is not also a remote code execution primitive.

**The danger, stated plainly.** ``<profile>/.env`` is loaded into the environment of the
process that runs the agent. A write path that accepted any variable name could set
``LD_PRELOAD``, ``PYTHONPATH``, ``PATH`` or ``BASH_ENV`` and turn "set a Slack token" into
"run my code inside the agent". So a name is writable only if something in the tenant's own
declaration says this agent needs it:

* a channel that grants this agent, contributing its provider manifest's
  ``requires_env`` and ``optional_env``;
* the agent's own model credential variable;
* the deployment's default model credential variable.

Everything else is refused by name. The allowlist is derived per request from the bundle
that is loaded at the time, so revoking a channel grant also revokes the ability to write
that channel's credentials.

**Values go one way.** Nothing here reads a value back. The status of a slot is "set" or
"not set", which is what an operator needs to know and is the most a control plane should
ever be able to say about a secret.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class CredentialSlot:
    """One variable an agent needs, and why it needs it."""

    name: str
    #: What asked for it — a channel id, "model", or "deployment". Shown so an operator can
    #: see why a field is on the screen, and so a stale one is obvious after a revoke.
    source: str
    required: bool = True
    label: str = ""
    description: str = ""
    url: str = ""
    #: Whether the value is a secret. Drives masking; defaults to True because a manifest
    #: that did not say is not a manifest saying "safe to show".
    secret: bool = True

    def to_dict(self, *, present: bool) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "required": self.required,
            "label": self.label,
            "description": self.description,
            "url": self.url,
            "secret": self.secret,
            # Presence only. There is no field here that could ever carry a value.
            "set": present,
        }


def _channels_granting(bundle, agent_id: str) -> Iterable[Any]:
    for channel in getattr(bundle, "channels", ()) or ():
        if not getattr(channel, "enabled", True):
            continue
        if agent_id in (getattr(channel, "allowed_agents", ()) or ()):
            yield channel
            continue
        if any(getattr(r, "agent", "") == agent_id for r in getattr(channel, "routes", ()) or ()):
            yield channel


def slots_for_agent(bundle, agent_id: str) -> tuple[CredentialSlot, ...]:
    """Every variable this agent may have a credential written for.

    Order is stable and grouped by source so the screen does not reshuffle between reads.
    A name reachable from two sources appears once, attributed to the first that asked.
    """
    agent = next((a for a in getattr(bundle, "agents", ()) or () if a.id == agent_id), None)
    if agent is None:
        return ()

    slots: list[CredentialSlot] = []
    seen: set[str] = set()

    def add(slot: CredentialSlot) -> None:
        if slot.name and slot.name not in seen:
            seen.add(slot.name)
            slots.append(slot)

    for channel in _channels_granting(bundle, agent_id):
        provider = channel.catalogue
        declared = {c["name"]: c for c in (getattr(provider, "credentials", ()) or ())}
        for name in getattr(provider, "required_env", ()) or ():
            meta = declared.get(name, {})
            add(CredentialSlot(
                name=name, source=channel.id, required=True,
                label=meta.get("prompt", "") or meta.get("description", ""),
                description=meta.get("description", ""), url=meta.get("url", ""),
                secret=bool(meta.get("secret", True)),
            ))
        for name in getattr(provider, "optional_env", ()) or ():
            meta = declared.get(name, {})
            add(CredentialSlot(
                name=name, source=channel.id, required=False,
                label=meta.get("prompt", "") or meta.get("description", ""),
                description=meta.get("description", ""), url=meta.get("url", ""),
                secret=bool(meta.get("secret", True)),
            ))

    model_env = getattr(getattr(agent, "model", None), "api_key_env", "") or ""
    add(CredentialSlot(
        name=model_env, source="model", required=True,
        label="Model credential",
        description="The key this agent's model provider authenticates with.",
    ))

    deployment = getattr(bundle, "deployment", None)
    provider = getattr(deployment, "provider", None) if deployment else None
    add(CredentialSlot(
        name=getattr(provider, "api_key_env", "") or "", source="deployment",
        required=True, label="Deployment model credential",
        description="The key the deployment's default model provider authenticates with.",
    ))

    return tuple(slots)


def writable_names(bundle, agent_id: str) -> frozenset[str]:
    """The allowlist. A write to any other name is refused."""
    return frozenset(slot.name for slot in slots_for_agent(bundle, agent_id))


def check_writable(bundle, agent_id: str, names: Iterable[str]) -> None:
    """Raise :class:`~nova.errors.SpecError` naming anything outside the allowlist.

    Refused by name rather than silently dropped: an operator who typed a variable that
    this agent has no declared use for should be told, not left believing it was stored.
    """
    from nova.errors import SpecError

    allowed = writable_names(bundle, agent_id)
    rejected = sorted(set(names) - allowed)
    if not rejected:
        return
    raise SpecError(
        f"{', '.join(rejected)}: not a credential {agent_id!r} declares a use for. "
        f"Writable for this agent: {', '.join(sorted(allowed)) or '(none)'}. A variable "
        "reaches the agent's process environment, so only names its channels or its model "
        "actually ask for can be written here"
    )
