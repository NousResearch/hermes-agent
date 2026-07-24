"""Agents Dashboard core module.

Loads an agent registry from ``$HERMES_HOME/agents.yaml`` (with sensible
defaults derived from existing Hermes profiles when the registry is
missing), renders a manager-friendly team table, emits JSON, creates a
starter registry, and prepares an orchestrated handoff card for the
``orch`` profile via the existing Kanban backend.

This is CLI/edge functionality — it does not touch the core agent loop
or add new model tools.
"""

from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml

from hermes_constants import get_hermes_home


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AgentSpec:
    """Normalized agent record shown in the dashboard."""

    name: str
    profile: str
    role: str
    description: str = ""
    can_edit_files: Any = False  # bool | "limited"
    escalates_to: str = "orch"
    skills: list[str] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    base_url: str = ""
    status: str = "ready"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Default registry / profile discovery
# ---------------------------------------------------------------------------

# Role defaults keyed by profile name. Allows an agent name to differ
# from a profile name while still providing a sensible default role.
_DEFAULT_ROLES: dict[str, str] = {
    "orch": "orchestrator",
    "coding": "implementer",
    "search": "researcher",
    "strong": "architect_reviewer",
    "fast": "quick_worker",
    "default": "default",
}

_DEFAULT_DESCRIPTIONS: dict[str, str] = {
    "orch": "Planeja, delega, acompanha e consolida trabalho do time de IAs.",
    "coding": "Implementa código, roda testes, depura e entrega evidências reais.",
    "search": "Pesquisa documentação, fatos atuais e fontes externas.",
    "strong": "Analisa arquitetura, trade-offs, riscos e decisões difíceis.",
    "fast": "Executa tarefas rápidas, simples e de baixo risco.",
}

# Starter registry content (from the plan). Used by ``agents init``.
STARTER_AGENTS_YAML = """\
version: 1
agents:
  orch:
    profile: orch
    role: orchestrator
    description: Planeja, delega, acompanha e consolida trabalho do time de IAs.
    can_edit_files: false
    escalates_to: user
    skills:
      - kanban-orchestrator

  coding:
    profile: coding
    role: implementer
    description: Implementa código, roda testes, depura e entrega evidências reais.
    can_edit_files: true
    escalates_to: orch
    skills:
      - systematic-debugging
      - test-driven-development
      - requesting-code-review

  fast:
    profile: fast
    role: quick_worker
    description: Executa tarefas rápidas, simples e de baixo risco.
    can_edit_files: limited
    escalates_to: orch
    skills: []

  strong:
    profile: strong
    role: architect_reviewer
    description: Analisa arquitetura, trade-offs, riscos e decisões difíceis.
    can_edit_files: false
    escalates_to: orch
    skills: []

  search:
    profile: search
    role: researcher
    description: Pesquisa documentação, fatos atuais e fontes externas.
    can_edit_files: false
    escalates_to: orch
    skills: []
"""


def _registry_path() -> Path:
    """Return the path to ``$HERMES_HOME/agents.yaml``."""
    return get_hermes_home() / "agents.yaml"


def _discover_profiles() -> list[tuple[str, str, str]]:
    """Discover real profiles on disk.

    Returns a list of ``(name, model, provider)`` tuples. Uses the existing
    ``hermes_cli.profiles`` helpers so we stay consistent with the rest of
    the CLI. Failures are swallowed — a broken profile should never crash
    the agents dashboard.
    """
    try:
        from hermes_cli.profiles import list_profiles
    except Exception:
        return []
    found: list[tuple[str, str, str]] = []
    try:
        for info in list_profiles():
            found.append((
                info.name,
                info.model or "",
                info.provider or "",
            ))
    except Exception:
        pass
    return found


def _build_default_agents() -> list[AgentSpec]:
    """Build a default agent list from discovered profiles.

    When no explicit registry exists, we derive agents from the real
    profiles on disk so the dashboard is useful out of the box.
    """
    agents: list[AgentSpec] = []
    seen: set[str] = set()
    for name, model, provider in _discover_profiles():
        if name in seen:
            continue
        seen.add(name)
        agents.append(AgentSpec(
            name=name,
            profile=name,
            role=_DEFAULT_ROLES.get(name, "agent"),
            description=_DEFAULT_DESCRIPTIONS.get(name, ""),
            model=model,
            provider=provider,
            status="ready",
        ))
    return agents


def _merge_registry_into_agents(
    registry_agents: dict[str, Any],
    discovered: list[AgentSpec],
) -> list[AgentSpec]:
    """Merge explicit registry entries with discovered profile data.

    Registry entries take precedence for role/description/skills/etc.
    Discovered profile model/provider data is merged in when the registry
    entry does not carry its own. Profiles discovered on disk that are
    not in the registry are appended as defaults.
    """
    by_profile: dict[str, AgentSpec] = {}
    for spec in discovered:
        by_profile[spec.profile] = spec

    result: list[AgentSpec] = []
    registry_profiles: set[str] = set()
    for agent_name, raw in (registry_agents or {}).items():
        if not isinstance(raw, dict):
            continue
        profile = raw.get("profile") or agent_name
        registry_profiles.add(profile)
        discovered_spec = by_profile.get(profile)
        spec = AgentSpec(
            name=agent_name,
            profile=profile,
            role=raw.get("role", _DEFAULT_ROLES.get(profile, "agent")),
            description=raw.get("description", ""),
            can_edit_files=raw.get("can_edit_files", False),
            escalates_to=raw.get("escalates_to", "orch"),
            skills=list(raw.get("skills", []) or []),
            model=raw.get("model") or (discovered_spec.model if discovered_spec else ""),
            provider=raw.get("provider") or (discovered_spec.provider if discovered_spec else ""),
            base_url=raw.get("base_url", ""),
            status=raw.get("status", "ready"),
        )
        result.append(spec)

    # Append discovered profiles not present in the registry.
    for spec in discovered:
        if spec.profile not in registry_profiles:
            result.append(spec)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_agents() -> list[AgentSpec]:
    """Load the agent registry and return normalized agent records.

    If ``$HERMES_HOME/agents.yaml`` is missing, derives sensible defaults
    from the real profiles on disk. Missing profiles are marked
    ``missing_profile`` rather than crashing.
    """
    path = _registry_path()
    if not path.is_file():
        return _build_default_agents()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return _build_default_agents()

    registry_agents = data.get("agents", {}) if isinstance(data, dict) else {}
    discovered = _build_default_agents()
    agents = _merge_registry_into_agents(registry_agents, discovered)

    # Mark agents whose profile dir does not exist.
    try:
        from hermes_cli.profiles import list_profiles
        known_profiles = {info.name for info in list_profiles()}
    except Exception:
        known_profiles = set()
    if known_profiles:
        for spec in agents:
            if spec.profile not in known_profiles:
                spec.status = "missing_profile"

    return agents


def render_table(agents: list[AgentSpec]) -> str:
    """Render a manager-friendly team dashboard as plain text."""
    home = str(get_hermes_home())
    header = f"Hermes AI Team\nHERMES_HOME: {home}\n"
    cols = ["Agent", "Profile", "Model", "Provider", "Role", "Status"]
    rows = [[a.name, a.profile, a.model or "-", a.provider or "-", a.role, a.status] for a in agents]
    widths = [len(c) for c in cols]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "  "
    lines = [sep.join(c.ljust(widths[i]) for i, c in enumerate(cols))]
    lines.append(sep.join("-" * widths[i] for i in range(len(cols))))
    for row in rows:
        lines.append(sep.join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    actions = (
        "\nActions:\n"
        '  hermes agents orchestrate "<task>"   Send a task to orch for planning\n'
        "  hermes kanban list                    View team task board\n"
        "  hermes profile list                   Inspect profile configs\n"
    )
    return header + "\n" + "\n".join(lines) + "\n" + actions


def to_json(agents: list[AgentSpec]) -> str:
    """Return a JSON string of normalized agent records."""
    return json.dumps([a.to_dict() for a in agents], indent=2, ensure_ascii=False)


def init_registry(force: bool = False) -> Path:
    """Write the starter registry to ``$HERMES_HOME/agents.yaml``.

    Refuses to overwrite an existing file unless ``force=True``. Returns
    the path written. Raises ``FileExistsError`` when the file exists and
    ``force`` is False.
    """
    path = _registry_path()
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists. Use --force to overwrite."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(STARTER_AGENTS_YAML)
    return path


def orchestrate(task: str, *, create_card: bool = True) -> dict:
    """Prepare a planning handoff for the ``orch`` profile.

    Creates a Kanban card assigned to ``orch`` when the backend is
    available, and always returns a dict describing the handoff (including
    the exact CLI command to start processing). Does NOT execute the task
    itself — that is ``orch``'s job once it picks up the card.

    Returns a dict with keys:
        - ``task``: the user task text
        - ``card_id``: created Kanban card id, or None
        - ``command``: the exact ``hermes kanban create ...`` command
        - ``dispatcher_running``: bool
        - ``dispatcher_message``: human guidance when not running
        - ``body``: the card body handed to orch
    """
    safe_title = task.strip()
    if not safe_title:
        raise ValueError("task text is required")
    title = f"orchestrate: {safe_title}"
    body = _build_orchestrate_body(safe_title)
    assignee = "orch"
    command = _build_kanban_create_command(title, body, assignee)

    result: dict = {
        "task": safe_title,
        "card_id": None,
        "command": command,
        "dispatcher_running": True,
        "dispatcher_message": "",
        "body": body,
    }

    if not create_card:
        return result

    # Attempt to create a Kanban card via the Python API.
    # We use initial_status="running" (the same default as `hermes kanban
    # create`) so the card is immediately visible; the dispatcher/orch
    # picks it up from there. "ready" is not a valid initial_status — it
    # is derived automatically by kanban_db when a task has no pending
    # parents.
    try:
        from hermes_cli import kanban_db as kb
        kb.init_db()
        with kb.connect_closing() as conn:
            card_id = kb.create_task(
                conn,
                title=title,
                body=body,
                assignee=assignee,
                initial_status="running",
            )
        result["card_id"] = card_id
    except Exception as exc:
        # Kanban unavailable — still return the handoff command so the
        # user can run it manually.
        result["dispatcher_message"] = (
            f"Could not create Kanban card automatically: {exc}\n"
            f"Run the command below manually to create the card."
        )

    # Check dispatcher presence so we can warn if the card will sit idle.
    try:
        from hermes_cli.kanban import _check_dispatcher_presence
        running, message = _check_dispatcher_presence()
        result["dispatcher_running"] = running
        if not running and message:
            result["dispatcher_message"] = message
    except Exception:
        pass

    return result


def _build_orchestrate_body(task: str) -> str:
    """Build the card body that instructs orch to plan the work."""
    return (
        f"User task: {task}\n\n"
        "You are the orch profile. Load kanban-orchestrator. "
        "Discover available profiles. Produce a task graph. "
        "Create concrete Kanban cards for coding/search/strong/fast as appropriate. "
        "Do not implement code yourself. "
        "Ask the user only if ambiguity changes scope/architecture/security/cost/UX."
    )


def _build_kanban_create_command(title: str, body: str, assignee: str) -> str:
    """Build a shell-safe ``hermes kanban create`` command string.

    Uses ``shlex.quote`` so the result is safely copiável/colável even
    when the task text contains quotes, ``$`` expansions, backticks, or
    other shell metacharacters.
    """
    return (
        f"hermes kanban create {shlex.quote(title)} "
        f"--assignee {shlex.quote(assignee)} "
        f"--body {shlex.quote(body)}"
    )


def render_orchestrate_summary(result: dict) -> str:
    """Render the orchestrate result as human-friendly text for the CLI.

    When a Kanban card was created (``card_id`` present), we do NOT print
    the ``hermes kanban create ...`` command again — that would invite the
    user to create a duplicate card. Instead we point them to the board /
    gateway as appropriate. When no card was created, we print the exact
    manual command (shell-quoted) so the user can run it themselves.
    """
    lines = [f"Orchestration handoff prepared for: {result['task']}"]
    card_id = result.get("card_id")
    if card_id:
        lines.append(f"Created Kanban card: {card_id}")
    else:
        lines.append("Kanban card was not created automatically.")
    if result.get("dispatcher_message"):
        lines.append("")
        lines.append("⚠  " + result["dispatcher_message"])
    lines.append("")
    if card_id:
        # Card already exists — don't suggest creating another one.
        lines.append("Next step — the card is ready for orch to pick up.")
        running = result.get("dispatcher_running", True)
        if not running:
            lines.append("  Start the gateway so the dispatcher can claim it:")
            lines.append("    hermes gateway start")
        lines.append("  View the board with:")
        lines.append("    hermes kanban list")
    else:
        # No card created — the user needs to run the command manually.
        lines.append("Next step — create the card manually, then orch picks it up:")
        lines.append(f"  {result['command']}")
    return "\n".join(lines)