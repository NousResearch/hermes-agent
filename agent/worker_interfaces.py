"""Session-stable semantic interfaces for the durable worker service.

The model-facing names in this module are presentation adapters.  Every call
still reaches ``AIAgent._dispatch_delegate_task``; worker identity, authorization,
durability, provider routing, and execution policy remain owned by the existing
delegation service.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from typing import Any, Callable, Iterable, Mapping, Optional


INTERFACE_VERSION = "worker-interface-v1"
CANONICAL_WORKER_TOOL = "delegate_task"
CANONICAL_TEAM_TOOL = "kanban_team"
_INTERFACES = frozenset({"auto", "hermes", "codex", "claude"})
_INTERFACE_NAMES = _INTERFACES - {"auto"}
_SOURCES = frozenset({"explicit", "qualified_exact_match", "canonical_fallback", "legacy_session"})
_QUALIFICATIONS = frozenset({"stable", "experimental_unqualified", "live_evidence_qualified"})
_TOOL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,127}$")
_EVIDENCE_REF = re.compile(r"^[A-Za-z0-9._:/#@+\-]{1,256}$")


@dataclass(frozen=True)
class QualifiedInterfaceProfile:
    """An exact provider/model interface qualification backed by live evidence."""

    provider: str
    model: str
    interface: str
    evidence_ref: str


# Auto-selection is deliberately conservative.  Add an exact entry only after
# recording live model/tool conformance evidence; source-level tests do not
# qualify a provider/model pair.
QUALIFIED_INTERFACE_PROFILES: tuple[QualifiedInterfaceProfile, ...] = ()


@dataclass(frozen=True)
class InterfaceSelection:
    name: str
    source: str
    qualification: str
    provider: str
    model: str
    evidence_ref: Optional[str] = None
    version: str = INTERFACE_VERSION
    # Pairs are (semantic tool name, actual session-advertised name).  This is
    # frozen with the session so later registry refreshes cannot rename calls in
    # replayed history or capture a native/MCP tool that arrived first.
    aliases: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class CanonicalWorkerCall:
    operation: str
    arguments: Mapping[str, Any]


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower()


def _qualified_match(
    provider: str,
    model: str,
    profiles: Iterable[QualifiedInterfaceProfile],
) -> Optional[QualifiedInterfaceProfile]:
    for profile in profiles:
        if (
            _normalized(profile.provider) == provider
            and _normalized(profile.model) == model
            and profile.interface in {"codex", "claude"}
            and isinstance(profile.evidence_ref, str)
            and bool(profile.evidence_ref.strip())
        ):
            return profile
    return None


def resolve_worker_interface(
    config: Mapping[str, Any],
    *,
    provider: Any,
    model: Any,
    qualified_profiles: Iterable[QualifiedInterfaceProfile] = QUALIFIED_INTERFACE_PROFILES,
) -> InterfaceSelection:
    """Resolve explicit override, then an exact qualified match, then Hermes."""

    orchestration = config.get("orchestration", {}) if isinstance(config, Mapping) else {}
    if orchestration is None:
        orchestration = {}
    if not isinstance(orchestration, Mapping):
        raise ValueError("orchestration must be a mapping")
    requested = orchestration.get("interface", "auto")
    if not isinstance(requested, str) or _normalized(requested) not in _INTERFACES:
        raise ValueError("orchestration.interface must be auto, hermes, codex, or claude")
    requested = _normalized(requested)
    normalized_provider, normalized_model = _normalized(provider), _normalized(model)
    if requested != "auto":
        return InterfaceSelection(
            requested,
            "explicit",
            "stable" if requested == "hermes" else "experimental_unqualified",
            normalized_provider,
            normalized_model,
        )
    matched = _qualified_match(normalized_provider, normalized_model, qualified_profiles)
    if matched is not None:
        return InterfaceSelection(
            matched.interface,
            "qualified_exact_match",
            "live_evidence_qualified",
            normalized_provider,
            normalized_model,
            matched.evidence_ref,
        )
    return InterfaceSelection(
        "hermes", "canonical_fallback", "stable", normalized_provider, normalized_model
    )


def _object_schema(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": dict(properties),
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


_TEXT = {"type": "string"}
_TARGET = {"type": "string", "description": "Stable worker id returned by spawn or list."}
_RUN = {"type": "string", "description": "Optional exact run id."}
_REFERENCE = {"type": "string", "description": "Optional typed reference such as worker:<id>, run:<id>, bot:<id>, room:<id>, or task:<id>."}
_PROFILE = {"type": "string", "description": "Optional configured Hermes worker profile."}
_PROVIDER = {"type": "string", "description": "Optional provider override allowed by the selected worker profile."}
_MODEL = {"type": "string", "description": "Optional model override allowed by the selected worker profile."}
_EFFORT = {"type": "string", "description": "Optional reasoning effort override allowed by the selected worker profile."}


def _control_schema(name: str = "worker_control") -> dict[str, Any]:
    return _object_schema(
        name,
        "Read or settle durable worker completions without changing worker execution semantics.",
        {
            "action": {"type": "string", "enum": ["completions", "ack", "reconcile"]},
            "target": _TARGET,
            "run_id": _RUN,
            "disposition": {
                "type": "string",
                "enum": ["confirmed_applied", "confirmed_not_applied", "accepted_unknown_no_replay"],
            },
            "note": _TEXT,
        },
        ("action",),
    )


def _team_schema(name: str) -> dict[str, Any]:
    return _object_schema(
        name,
        "Coordinate dependency-linked Kanban tasks through authorized durable workers and review.",
        {
            "action": {
                "type": "string",
                "enum": [
                    "create", "start", "guide", "submit_review", "accept",
                    "request_changes", "cancel",
                ],
            },
            "task_ref": {"type": "string", "description": "Typed task reference."},
            "title": _TEXT,
            "body": _TEXT,
            "profile": _PROFILE,
            "parent_refs": {"type": "array", "items": {"type": "string"}},
            "targets": {"type": "array", "items": {"type": "string"}},
            "message": _TEXT,
            "reviewer": _PROFILE,
            "summary": _TEXT,
            "idempotency_key": _TEXT,
            "timeout_seconds": {"type": "number", "minimum": 0, "maximum": 60},
        },
        ("action",),
    )


def _codex_schemas() -> tuple[dict[str, Any], ...]:
    return (
        _object_schema("worker_capabilities", "List configured worker profiles and permitted shared references.", {"profile": _PROFILE, "reference": _REFERENCE}),
        _object_schema(
            "spawn_agent",
            "Start one durable worker with fresh conversation context. This does not fork parent history.",
            {
                "message": _TEXT, "context": _TEXT, "profile": _PROFILE,
                "provider": _PROVIDER, "model": _MODEL, "reasoning_effort": _EFFORT,
            },
            ("message",),
        ),
        _object_schema(
            "send_message",
            "Queue a message for a worker. An idle worker is not started.",
            {"target": _TARGET, "message": _TEXT},
            ("target", "message"),
        ),
        _object_schema(
            "followup_task",
            "Queue the worker's next linked turn after its current run, or start it when idle. Cancelled workers stay cancelled.",
            {"target": _TARGET, "message": _TEXT},
            ("target", "message"),
        ),
        _object_schema("inspect_agent", "Inspect one authorized worker and its visible conversation.", {"target": _TARGET, "run_id": _RUN}, ("target",)),
        _object_schema("list_agents", "List workers visible to this agent.", {}),
        _object_schema(
            "wait_agent",
            "Wait for one worker run for at most 60000 milliseconds.",
            {"target": _TARGET, "run_id": _RUN, "timeout_ms": {"type": "integer", "minimum": 0, "maximum": 60000}},
            ("target",),
        ),
        _object_schema("interrupt_agent", "Interrupt one exact current worker run without cancelling its descendants.", {"target": _TARGET, "run_id": _RUN}, ("target",)),
        _object_schema("cancel_agent_tree", "Cancel a worker and its descendant tree.", {"target": _TARGET}, ("target",)),
        _control_schema(),
    )


def _claude_schemas() -> tuple[dict[str, Any], ...]:
    return (
        _object_schema("TaskCapabilities", "List configured worker profiles and permitted shared references.", {"profile": _PROFILE, "reference": _REFERENCE}),
        _object_schema(
            "Agent",
            "Start one durable worker with fresh conversation context. Parent transcript forking is unsupported.",
            {
                "prompt": _TEXT, "context": _TEXT, "subagent_type": _PROFILE,
                "provider": _PROVIDER, "model": _MODEL, "reasoning_effort": _EFFORT,
            },
            ("prompt",),
        ),
        _object_schema(
            "SendMessage",
            "Send to a worker. Set if_idle only when the message should start the next linked turn; cancelled workers stay cancelled.",
            {"recipient": _TARGET, "content": _TEXT, "if_idle": {"type": "boolean", "default": False}},
            ("recipient", "content"),
        ),
        _object_schema(
            "TaskOutput",
            "Inspect a worker, or wait for its selected run when block is true.",
            {
                "task_id": _TARGET,
                "run_id": _RUN,
                "block": {"type": "boolean", "default": False},
                "timeout_ms": {"type": "integer", "minimum": 0, "maximum": 60000},
            },
            ("task_id",),
        ),
        _object_schema("TaskList", "List workers visible to this agent.", {}),
        _object_schema(
            "TaskStop",
            "Stop one run or explicitly cancel its whole worker tree. Graceful process shutdown is unsupported.",
            {"task_id": _TARGET, "run_id": _RUN, "scope": {"type": "string", "enum": ["run", "tree"], "default": "run"}},
            ("task_id",),
        ),
        _control_schema(),
    )


_SCHEMA_FACTORIES = {"codex": _codex_schemas, "claude": _claude_schemas}
_TEAM_SCHEMA_FACTORIES = {
    "codex": lambda: _team_schema("team_task"),
    "claude": lambda: _team_schema("TeamTask"),
}


def _semantic_schemas(selection: InterfaceSelection) -> tuple[dict[str, Any], ...]:
    factory = _SCHEMA_FACTORIES.get(selection.name)
    return factory() if factory is not None else ()


def _team_semantic_schema(selection: InterfaceSelection) -> Optional[dict[str, Any]]:
    factory = _TEAM_SCHEMA_FACTORIES.get(selection.name)
    return factory() if factory is not None else None


def _namespaced_alias(name: str) -> str:
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return f"hermes_worker_{snake}"


def bind_worker_interface(
    selection: InterfaceSelection, definitions: Iterable[Mapping[str, Any]]
) -> InterfaceSelection:
    """Freeze collision-aware advertised aliases against the initial tool catalog."""

    if selection.aliases:
        return selection
    definitions = list(definitions)
    canonical_names = {
        str(item.get("function", {}).get("name")) for item in definitions
    }
    if selection.name == "hermes":
        aliases = [(CANONICAL_WORKER_TOOL, CANONICAL_WORKER_TOOL)]
        if CANONICAL_TEAM_TOOL in canonical_names:
            aliases.append((CANONICAL_TEAM_TOOL, CANONICAL_TEAM_TOOL))
        return replace(selection, aliases=tuple(aliases))
    occupied = {
        str(item.get("function", {}).get("name"))
        for item in definitions
        if item.get("function", {}).get("name") not in {CANONICAL_WORKER_TOOL, CANONICAL_TEAM_TOOL}
    }
    aliases: list[tuple[str, str]] = []
    for schema in _semantic_schemas(selection):
        semantic = str(schema["function"]["name"])
        advertised = semantic
        if advertised in occupied:
            base = _namespaced_alias(semantic)
            advertised = base
            suffix = 2
            while advertised in occupied:
                advertised = f"{base}_{suffix}"
                suffix += 1
        occupied.add(advertised)
        aliases.append((semantic, advertised))
    team_schema = _team_semantic_schema(selection)
    if CANONICAL_TEAM_TOOL in canonical_names and team_schema is not None:
        semantic = str(team_schema["function"]["name"])
        advertised = semantic
        if advertised in occupied:
            base = _namespaced_alias(semantic)
            advertised = base
            suffix = 2
            while advertised in occupied:
                advertised = f"{base}_{suffix}"
                suffix += 1
        aliases.append((semantic, advertised))
    return replace(selection, aliases=tuple(aliases))


def frozen_worker_interface_contract(selection: Optional[InterfaceSelection]) -> dict[str, Any]:
    """Return the sanitized presentation contract stored with a durable worker."""

    if not isinstance(selection, InterfaceSelection):
        selection = InterfaceSelection(
            "hermes", "legacy_session", "stable", "", "",
            aliases=((CANONICAL_WORKER_TOOL, CANONICAL_WORKER_TOOL),),
        )
    contract: dict[str, Any] = {
        "version": selection.version,
        "name": selection.name,
        "source": selection.source,
        "qualification": selection.qualification,
        "aliases": [list(item) for item in selection.aliases],
    }
    if selection.evidence_ref and _EVIDENCE_REF.fullmatch(selection.evidence_ref):
        contract["evidence_ref"] = selection.evidence_ref
    return contract


def restore_worker_interface_contract(
    contract: Optional[Mapping[str, Any]], *, provider: Any, model: Any,
    definitions: Iterable[Mapping[str, Any]],
) -> InterfaceSelection:
    """Validate and restore a retained presentation without expanding authority."""

    definitions = list(definitions)
    if contract is None:
        return bind_worker_interface(
            InterfaceSelection(
                "hermes", "legacy_session", "stable", _normalized(provider), _normalized(model)
            ),
            definitions,
        )
    if not isinstance(contract, Mapping) or contract.get("version") != INTERFACE_VERSION:
        raise ValueError("Retained worker interface contract has an unsupported version.")
    name = contract.get("name")
    source = contract.get("source")
    qualification = contract.get("qualification")
    evidence_ref = contract.get("evidence_ref")
    if name not in _INTERFACE_NAMES or source not in _SOURCES or qualification not in _QUALIFICATIONS:
        raise ValueError("Retained worker interface contract metadata is invalid.")
    if evidence_ref is not None and (
        not isinstance(evidence_ref, str) or not _EVIDENCE_REF.fullmatch(evidence_ref)
    ):
        raise ValueError("Retained worker interface evidence reference is invalid.")
    raw_aliases = contract.get("aliases")
    if not isinstance(raw_aliases, list):
        raise ValueError("Retained worker interface aliases are invalid.")
    aliases: list[tuple[str, str]] = []
    for item in raw_aliases:
        if (
            not isinstance(item, list) or len(item) != 2
            or not all(isinstance(value, str) and _TOOL_NAME.fullmatch(value) for value in item)
        ):
            raise ValueError("Retained worker interface aliases are invalid.")
        aliases.append((item[0], item[1]))
    legacy_expected = (
        {CANONICAL_WORKER_TOOL}
        if name == "hermes"
        else {schema["function"]["name"] for schema in _SCHEMA_FACTORIES[name]()}
    )
    current_expected = set(legacy_expected)
    team_schema = _team_semantic_schema(InterfaceSelection(
        str(name), str(source), str(qualification), "", ""
    ))
    if name == "hermes":
        current_expected.add(CANONICAL_TEAM_TOOL)
    elif team_schema is not None:
        current_expected.add(str(team_schema["function"]["name"]))
    actual_semantics = {semantic for semantic, _ in aliases}
    if frozenset(actual_semantics) not in {
        frozenset(legacy_expected), frozenset(current_expected),
    } or len(aliases) != len(actual_semantics):
        raise ValueError("Retained worker interface aliases do not match the interface version.")
    advertised = [alias for _, alias in aliases]
    if len(advertised) != len(set(advertised)):
        raise ValueError("Retained worker interface aliases contain duplicate tool names.")
    occupied = {
        str(item.get("function", {}).get("name"))
        for item in definitions
        if item.get("function", {}).get("name") not in {CANONICAL_WORKER_TOOL, CANONICAL_TEAM_TOOL}
    }
    team_semantic = (
        CANONICAL_TEAM_TOOL if name == "hermes"
        else str(team_schema["function"]["name"]) if team_schema is not None else ""
    )
    registered_names = {
        str(item.get("function", {}).get("name")) for item in definitions
    }
    if team_semantic in actual_semantics and CANONICAL_TEAM_TOOL not in registered_names:
        raise ValueError("Retained team interface is unavailable in the current registry.")
    collisions = sorted(occupied.intersection(advertised))
    if collisions:
        raise ValueError(
            f"Retained worker interface aliases now collide with registered tools: {collisions}"
        )
    return InterfaceSelection(
        str(name), str(source), str(qualification), _normalized(provider), _normalized(model),
        evidence_ref if isinstance(evidence_ref, str) else None,
        aliases=tuple(aliases),
    )


def project_worker_tool_definitions(
    definitions: Iterable[Mapping[str, Any]], selection: InterfaceSelection
) -> list[dict[str, Any]]:
    """Replace canonical worker presentation at its existing schema position."""

    projected: list[dict[str, Any]] = []
    aliases = dict(selection.aliases)
    reserved_aliases = set(aliases.values())
    replacement = _SCHEMA_FACTORIES.get(selection.name)
    team_replacement = _TEAM_SCHEMA_FACTORIES.get(selection.name)
    for definition in definitions:
        name = definition.get("function", {}).get("name")
        if name == CANONICAL_WORKER_TOOL and replacement is not None:
            for schema in replacement():
                semantic = schema["function"]["name"]
                renamed = {
                    **schema,
                    "function": {**schema["function"], "name": aliases.get(semantic, semantic)},
                }
                projected.append(renamed)
        elif name == CANONICAL_TEAM_TOOL and team_replacement is not None:
            schema = team_replacement()
            semantic = schema["function"]["name"]
            if semantic in aliases:
                projected.append({
                    **schema,
                    "function": {**schema["function"], "name": aliases[semantic]},
                })
            # A retained pre-team contract keeps its original frozen catalog.
        elif name in {CANONICAL_WORKER_TOOL, CANONICAL_TEAM_TOOL} and name not in aliases:
            # Retained Hermes sessions also keep the catalog frozen.
            continue
        elif replacement is not None and name in reserved_aliases:
            raise ValueError(
                f"Frozen worker interface alias '{name}' now collides with a registered tool."
            )
        else:
            projected.append(dict(definition))
    return projected


_CODEX_OPERATIONS = {
    "worker_capabilities": "capabilities",
    "spawn_agent": "spawn",
    "send_message": "message",
    "followup_task": "start_turn",
    "inspect_agent": "inspect",
    "list_agents": "list",
    "wait_agent": "wait",
    "interrupt_agent": "interrupt_run",
    "cancel_agent_tree": "cancel_tree",
    "worker_control": "control",
    "team_task": "team",
}
_CLAUDE_OPERATIONS = {
    "TaskCapabilities": "capabilities",
    "Agent": "spawn",
    "SendMessage": "message",
    "TaskOutput": "inspect",
    "TaskList": "list",
    "TaskStop": "interrupt_run",
    "worker_control": "control",
    "TeamTask": "team",
}
_TEAM_ARGUMENTS = frozenset({
    "action", "task_ref", "title", "body", "profile", "parent_refs", "targets",
    "message", "reviewer", "summary", "idempotency_key", "timeout_seconds",
})
_CODEX_ARGUMENTS = {
    "worker_capabilities": frozenset({"profile", "reference"}),
    "spawn_agent": frozenset({"message", "context", "profile", "provider", "model", "reasoning_effort"}),
    "send_message": frozenset({"target", "message"}),
    "followup_task": frozenset({"target", "message"}),
    "inspect_agent": frozenset({"target", "run_id"}),
    "list_agents": frozenset(),
    "wait_agent": frozenset({"target", "run_id", "timeout_ms"}),
    "interrupt_agent": frozenset({"target", "run_id"}),
    "cancel_agent_tree": frozenset({"target"}),
    "worker_control": frozenset({"action", "target", "run_id", "disposition", "note"}),
    "team_task": _TEAM_ARGUMENTS,
}
_CLAUDE_ARGUMENTS = {
    "TaskCapabilities": frozenset({"profile", "reference"}),
    "Agent": frozenset({"prompt", "context", "subagent_type", "provider", "model", "reasoning_effort"}),
    "SendMessage": frozenset({"recipient", "content", "if_idle"}),
    "TaskOutput": frozenset({"task_id", "run_id", "block", "timeout_ms"}),
    "TaskList": frozenset(),
    "TaskStop": frozenset({"task_id", "run_id", "scope"}),
    "worker_control": frozenset({"action", "target", "run_id", "disposition", "note"}),
    "TeamTask": _TEAM_ARGUMENTS,
}


def advertised_worker_tool_names(selection: InterfaceSelection) -> frozenset[str]:
    if selection.aliases:
        return frozenset(advertised for _, advertised in selection.aliases)
    if selection.name == "codex":
        return frozenset(_CODEX_OPERATIONS)
    if selection.name == "claude":
        return frozenset(_CLAUDE_OPERATIONS)
    return frozenset({CANONICAL_WORKER_TOOL, CANONICAL_TEAM_TOOL})


def semantic_worker_tool(selection: Optional[InterfaceSelection], tool_name: str) -> Optional[str]:
    """Resolve only the aliases frozen for this exact session."""

    if not isinstance(selection, InterfaceSelection):
        return tool_name if tool_name in {CANONICAL_WORKER_TOOL, CANONICAL_TEAM_TOOL} else None
    aliases = selection.aliases
    if not aliases:
        if selection.name == "hermes":
            return tool_name if tool_name in {CANONICAL_WORKER_TOOL, CANONICAL_TEAM_TOOL} else None
        operations = _CODEX_OPERATIONS if selection.name == "codex" else _CLAUDE_OPERATIONS
        return tool_name if tool_name in operations else None
    return next((semantic for semantic, advertised in aliases if advertised == tool_name), None)


def canonical_worker_capability(
    selection: Optional[InterfaceSelection], tool_name: str
) -> str:
    """Return the authority-bearing registry capability for an interface tool."""

    semantic = semantic_worker_tool(selection, tool_name)
    if semantic is not None:
        if semantic in {CANONICAL_TEAM_TOOL, "team_task", "TeamTask"}:
            return CANONICAL_TEAM_TOOL
        return CANONICAL_WORKER_TOOL
    return tool_name


def is_worker_interface_tool(selection: Optional[InterfaceSelection], tool_name: str) -> bool:
    return semantic_worker_tool(selection, tool_name) is not None


def is_worker_spawn_tool(
    selection: Optional[InterfaceSelection], tool_name: str,
    arguments: Optional[Mapping[str, Any]] = None,
) -> bool:
    semantic = semantic_worker_tool(selection, tool_name)
    if semantic in {"spawn_agent", "Agent"}:
        return True
    return semantic == CANONICAL_WORKER_TOOL and _normalized((arguments or {}).get("action")) in {"", "spawn"}


def _hermes_operation(arguments: Mapping[str, Any]) -> str:
    action = _normalized(arguments.get("action")) or "spawn"
    return {
        "discover": "capabilities",
        "status": "list" if not arguments.get("worker_id") else "inspect",
        "resume": "start_turn",
        "cancel": "cancel_tree",
        "interrupt": "interrupt_run",
    }.get(action, action)


def normalize_worker_call(
    selection: InterfaceSelection, tool_name: str, arguments: Mapping[str, Any]
) -> CanonicalWorkerCall:
    """Translate presentation semantics to one canonical delegate call."""

    args = dict(arguments or {})
    advertised_name = tool_name
    tool_name = semantic_worker_tool(selection, advertised_name) or ""
    if selection.name == "hermes":
        if tool_name == CANONICAL_TEAM_TOOL:
            return CanonicalWorkerCall("team", args)
        if tool_name != CANONICAL_WORKER_TOOL:
            raise ValueError(f"Tool '{advertised_name}' is not part of the hermes worker interface.")
        return CanonicalWorkerCall(_hermes_operation(args), args)
    if selection.name == "codex":
        operation = _CODEX_OPERATIONS.get(tool_name)
        if operation is None:
            raise ValueError(f"Tool '{advertised_name}' is not part of the codex worker interface.")
        unsupported = set(args) - _CODEX_ARGUMENTS[tool_name]
        if unsupported:
            raise ValueError(f"Unsupported {tool_name} arguments: {sorted(unsupported)}")
        mapped = {
            "worker_capabilities": lambda: {"action": "discover", "profile": args.get("profile"), "reference": args.get("reference")},
            "spawn_agent": lambda: {
                "goal": args.get("message"), "context": args.get("context"),
                "profile": args.get("profile"), "provider": args.get("provider"),
                "model": args.get("model"), "reasoning_effort": args.get("reasoning_effort"),
            },
            "send_message": lambda: {"action": "message", "worker_id": args.get("target"), "message": args.get("message")},
            "followup_task": lambda: {"action": "resume", "worker_id": args.get("target"), "message": args.get("message")},
            "inspect_agent": lambda: {"action": "inspect", "worker_id": args.get("target"), "run_id": args.get("run_id")},
            "list_agents": lambda: {"action": "status"},
            "wait_agent": lambda: {"action": "wait", "worker_id": args.get("target"), "run_id": args.get("run_id"), "timeout_seconds": float(args.get("timeout_ms", 0)) / 1000},
            "interrupt_agent": lambda: {"action": "interrupt", "worker_id": args.get("target"), "run_id": args.get("run_id")},
            "cancel_agent_tree": lambda: {"action": "cancel", "worker_id": args.get("target")},
            "worker_control": lambda: {
                "action": args.get("action"), "worker_id": args.get("target"),
                "run_id": args.get("run_id"), "reconciliation_disposition": args.get("disposition"),
                "message": args.get("note"),
            },
            "team_task": lambda: args,
        }[tool_name]()
        if tool_name == "worker_control":
            operation = _normalized(args.get("action"))
            if operation not in {"completions", "ack", "reconcile"}:
                raise ValueError("worker_control.action must be completions, ack, or reconcile")
            if operation != "completions" and not args.get("target"):
                raise ValueError(f"worker_control action='{operation}' requires target")
        return CanonicalWorkerCall(operation, {key: value for key, value in mapped.items() if value is not None})
    operation = _CLAUDE_OPERATIONS.get(tool_name)
    if operation is None:
        raise ValueError(f"Tool '{advertised_name}' is not part of the claude worker interface.")
    unsupported = set(args) - _CLAUDE_ARGUMENTS[tool_name]
    if unsupported:
        raise ValueError(f"Unsupported {tool_name} arguments: {sorted(unsupported)}")
    if tool_name == "TaskStop" and args.get("scope", "run") not in {"run", "tree"}:
        raise ValueError("TaskStop.scope must be run or tree")
    if tool_name == "SendMessage" and args.get("if_idle") is True:
        operation = "start_turn"
    if tool_name == "TaskOutput" and args.get("block") is True:
        operation = "wait"
    if tool_name == "TaskStop" and args.get("scope", "run") == "tree":
        operation = "cancel_tree"
    mapped = {
        "TaskCapabilities": lambda: {"action": "discover", "profile": args.get("profile"), "reference": args.get("reference")},
        "Agent": lambda: {
            "goal": args.get("prompt"), "context": args.get("context"),
            "profile": args.get("subagent_type"), "provider": args.get("provider"),
            "model": args.get("model"), "reasoning_effort": args.get("reasoning_effort"),
        },
        "SendMessage": lambda: {"action": "resume" if operation == "start_turn" else "message", "worker_id": args.get("recipient"), "message": args.get("content")},
        "TaskOutput": lambda: {"action": "wait" if operation == "wait" else "inspect", "worker_id": args.get("task_id"), "run_id": args.get("run_id"), "timeout_seconds": float(args.get("timeout_ms", 0)) / 1000},
        "TaskList": lambda: {"action": "status"},
        "TaskStop": lambda: {"action": "cancel" if operation == "cancel_tree" else "interrupt", "worker_id": args.get("task_id"), "run_id": args.get("run_id")},
        "worker_control": lambda: {
            "action": args.get("action"), "worker_id": args.get("target"),
            "run_id": args.get("run_id"), "reconciliation_disposition": args.get("disposition"),
            "message": args.get("note"),
        },
        "TeamTask": lambda: args,
    }[tool_name]()
    if tool_name == "worker_control":
        operation = _normalized(args.get("action"))
        if operation not in {"completions", "ack", "reconcile"}:
            raise ValueError("worker_control.action must be completions, ack, or reconcile")
        if operation != "completions" and not args.get("target"):
            raise ValueError(f"worker_control action='{operation}' requires target")
    return CanonicalWorkerCall(operation, {key: value for key, value in mapped.items() if value is not None})


def _decode_result(result: Any) -> dict[str, Any]:
    try:
        decoded = json.loads(result) if isinstance(result, str) else dict(result)
    except (TypeError, ValueError, json.JSONDecodeError):
        decoded = {"error": "Worker service returned a non-JSON result."}
    return decoded if isinstance(decoded, dict) else {"result": decoded}


def _receipt(
    selection: InterfaceSelection,
    *,
    tool_name: str,
    operation: str,
    effective_action: str,
    canonical_tool: str = CANONICAL_WORKER_TOOL,
) -> dict[str, Any]:
    receipt = {
        "interface": selection.name,
        "version": selection.version,
        "selection_source": selection.source,
        "qualification": selection.qualification,
        "advertised_tool": tool_name,
        "operation": operation,
        "canonical_tool": canonical_tool,
        "effective_action": effective_action,
        "worker_service": (
            "TeamOrchestrationService"
            if canonical_tool == CANONICAL_TEAM_TOOL
            else "AIAgent._dispatch_delegate_task/SubagentLifecycleService"
        ),
        "provider": selection.provider,
        "model": selection.model,
    }
    if selection.evidence_ref:
        receipt["qualification_evidence"] = selection.evidence_ref
    return receipt


def dispatch_worker_interface_call(
    parent_agent: Any,
    tool_name: str,
    arguments: Mapping[str, Any],
    dispatch: Callable[[dict[str, Any]], Any],
) -> str:
    """Dispatch one advertised call through the canonical worker service."""

    selection = getattr(parent_agent, "_worker_interface_selection", None)
    if not isinstance(selection, InterfaceSelection):
        selection = InterfaceSelection(
            "hermes", "legacy_session", "stable", _normalized(getattr(parent_agent, "provider", "")),
            _normalized(getattr(parent_agent, "model", "")),
        )
    try:
        call = normalize_worker_call(selection, tool_name, arguments)
    except (TypeError, ValueError) as exc:
        payload = {"error": str(exc)}
        payload["orchestration_interface"] = _receipt(
            selection, tool_name=tool_name, operation="unsupported", effective_action="none"
        )
        return json.dumps(payload, ensure_ascii=False)

    canonical_args = dict(call.arguments)
    canonical_tool = canonical_worker_capability(selection, tool_name)
    effective_action = str(canonical_args.get("action") or "spawn")
    if canonical_tool == CANONICAL_TEAM_TOOL:
        from agent.team_orchestration import TeamOrchestrationService
        payload = _decode_result(TeamOrchestrationService(parent_agent).dispatch(canonical_args))
    else:
        payload = _decode_result(dispatch(canonical_args))
    payload["orchestration_interface"] = _receipt(
        selection,
        tool_name=tool_name,
        operation=call.operation,
        effective_action=effective_action,
        canonical_tool=canonical_tool,
    )
    return json.dumps(payload, ensure_ascii=False)
