"""Creating, editing and retiring an agent — by editing the bundle.

Every function here is a small transformation of one bundle, applied through
:func:`nova.spec.writer.edit`, which validates the whole bundle before anything lands. So
an edit that would produce an agent naming a permission the policy does not define, or a
teammate that does not exist, fails with the same message it would have failed with had
somebody written the YAML by hand. There is no weaker path for edits made from a UI.

**On Soul editing, and why it does not write ``SOUL.md``.** The runtime reads an agent's
identity from ``<profile>/SOUL.md``, and Hermes even exposes an endpoint that writes it.
That file is *derived*: ``nova.runtime.hermes.materialize.build_persona`` composes it from
the bundle's instructions plus tenant branding and the knowledge briefing, and ``apply``
overwrites it every time it runs. Writing there would produce an edit that saves, persists,
survives a reload — and silently disappears the next time anyone applied the bundle. So
:func:`set_instructions` writes the bundle's prompt file, and the runtime picks it up when
the bundle is applied.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Optional

from nova.errors import SpecError
from nova.spec.writer import BundleEdit, dump_yaml, edit, safe_relative, validate_id

#: Where an agent's persona lives inside a bundle.
PROMPTS_DIR = "prompts"

#: Fields the Control Centre may set on an agent, and nothing else.
#:
#: An allowlist rather than "merge whatever arrived": a request body that could set any key
#: could set one the loader ignores today and honours tomorrow, and the bundle would carry
#: a field nobody reviewed. Unknown keys are refused by name so a caller learns what is
#: actually settable instead of watching a value disappear.
AGENT_FIELDS = (
    "name",
    "role",
    "description",
    "enabled",
    "model",
    "tools",
    "knowledge",
    "permissions",
    "approval",
    "limits",
    "delegation",
)


def agent_file(agent_id: str) -> str:
    return f"agents/{agent_id}.yaml"


def prompt_file(agent_id: str) -> str:
    return f"{PROMPTS_DIR}/{agent_id}.md"


def _check_fields(fields: Mapping[str, Any]) -> None:
    unknown = sorted(set(fields) - set(AGENT_FIELDS))
    if unknown:
        raise SpecError(
            f"cannot set {', '.join(unknown)} on an agent. Settable: "
            f"{', '.join(AGENT_FIELDS)}"
        )


def _require_agent(e: BundleEdit, agent_id: str) -> dict[str, Any]:
    document = e.read_yaml(agent_file(agent_id))
    if not document:
        raise SpecError(f"no agent {agent_id!r} in this bundle")
    return document


def create_agent(
    root: Path,
    *,
    agent_id: str,
    fields: Mapping[str, Any],
    instructions: str = "",
):
    """Declare a new agent. Refuses an id that already exists.

    The persona is written as a separate prompt file rather than inline, because that is
    what the example bundles do and what an operator editing by hand will expect to find.
    """
    agent_id = validate_id(agent_id, what="agent id")
    _check_fields(fields)

    def mutate(e: BundleEdit) -> None:
        if e.exists(agent_file(agent_id)):
            raise SpecError(
                f"agent {agent_id!r} already exists. Edit it, or pick another id"
            )
        document: dict[str, Any] = {"id": agent_id}
        document.update({k: copy.deepcopy(v) for k, v in fields.items()})
        document.setdefault("name", agent_id)
        document.setdefault("enabled", True)
        if instructions.strip():
            e.write_text(prompt_file(agent_id), _as_markdown(instructions))
            document["instructions"] = prompt_file(agent_id)
        e.write_yaml(agent_file(agent_id), document)

    return edit(root, mutate)


def update_agent(root: Path, agent_id: str, fields: Mapping[str, Any]):
    """Change declared fields on an existing agent, leaving the rest as they were."""
    agent_id = validate_id(agent_id, what="agent id")
    _check_fields(fields)

    def mutate(e: BundleEdit) -> None:
        document = _require_agent(e, agent_id)
        for key, value in fields.items():
            if value is None:
                document.pop(key, None)
            else:
                document[key] = copy.deepcopy(value)
        document["id"] = agent_id  # never renameable through an edit; see duplicate_agent
        e.write_yaml(agent_file(agent_id), document)

    return edit(root, mutate)


def _as_markdown(text: str) -> str:
    """Persona text, newline-terminated. Nothing else is imposed on it."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text if text.endswith("\n") else text + "\n"


def instructions_of(root: Path, agent_id: str) -> dict[str, Any]:
    """The agent's current persona, and where it comes from.

    ``path`` is what the editor saves back to; ``inline`` says the text lives in the agent
    file rather than a prompt file, which changes nothing for the caller but is worth
    showing rather than hiding.
    """
    from nova.spec import load_bundle

    agent_id = validate_id(agent_id, what="agent id")
    bundle = load_bundle(Path(root))
    for agent in bundle.agents:
        if agent.id == agent_id:
            return {
                "agent_id": agent_id,
                "instructions": agent.instructions,
                "path": agent.instructions_path or prompt_file(agent_id),
                "inline": agent.instructions_path is None and bool(agent.instructions),
            }
    raise SpecError(f"no agent {agent_id!r} in this bundle")


def set_instructions(root: Path, agent_id: str, text: str):
    """Rewrite an agent's persona — the bundle's copy, which is the one that survives.

    Writes to the agent's existing ``instructions`` path when it has one, so an operator
    who organised their prompts a particular way keeps that organisation. An agent that
    carried its persona inline keeps carrying it inline.
    """
    agent_id = validate_id(agent_id, what="agent id")

    def mutate(e: BundleEdit) -> None:
        document = _require_agent(e, agent_id)
        declared_path = str(document.get("instructions") or "").strip()

        if not declared_path and "instructions_text" in document:
            document["instructions_text"] = _as_markdown(text)
            e.write_yaml(agent_file(agent_id), document)
            return

        target = declared_path or prompt_file(agent_id)
        safe_relative(e.root, target)  # refuses anything outside the bundle
        e.write_text(target, _as_markdown(text))
        if not declared_path:
            document["instructions"] = target
            e.write_yaml(agent_file(agent_id), document)

    return edit(root, mutate)


def duplicate_agent(root: Path, agent_id: str, new_id: str, *, name: str = ""):
    """Clone an agent's declaration under a new id, persona included.

    A copy starts **disabled**. A duplicate is a starting point someone is about to edit,
    and an exact copy of a live agent silently joining the workforce — reachable over the
    same channels, holding the same permissions — is not what "duplicate" should mean.
    """
    agent_id = validate_id(agent_id, what="agent id")
    new_id = validate_id(new_id, what="agent id")
    if agent_id == new_id:
        raise SpecError("a duplicate needs a different id")

    def mutate(e: BundleEdit) -> None:
        document = copy.deepcopy(_require_agent(e, agent_id))
        if e.exists(agent_file(new_id)):
            raise SpecError(f"agent {new_id!r} already exists")
        document["id"] = new_id
        document["name"] = name or f"{document.get('name', agent_id)} (copy)"
        document["enabled"] = False

        declared_path = str(document.get("instructions") or "").strip()
        if declared_path:
            persona = e.read_text(declared_path)
            target = prompt_file(new_id)
            e.write_text(target, _as_markdown(persona))
            document["instructions"] = target
        e.write_yaml(agent_file(new_id), document)

    return edit(root, mutate)


def archive_agent(root: Path, agent_id: str, *, enabled: bool = False):
    """Disable an agent without discarding its declaration.

    The reversible half of "delete/archive", and the one the Control Centre offers first: a
    disabled agent stops being scheduled and stops being routed to, while its persona,
    permissions and history remain readable. Set ``enabled=True`` to bring it back.
    """
    return update_agent(root, agent_id, {"enabled": bool(enabled)})


def delete_agent(root: Path, agent_id: str):
    """Remove an agent's declaration and its prompt file.

    Irreversible at the bundle level, and deliberately separate from archiving. What it does
    **not** do is remove the runtime profile: that directory holds the agent's conversation
    history, its memories and its ``.env``, none of which NOVA wrote and none of which it
    will delete on a config edit's say-so.
    """
    agent_id = validate_id(agent_id, what="agent id")

    def mutate(e: BundleEdit) -> None:
        document = _require_agent(e, agent_id)
        declared_path = str(document.get("instructions") or "").strip()
        e.remove(agent_file(agent_id))
        if declared_path:
            e.remove(declared_path)

    return edit(root, mutate)
