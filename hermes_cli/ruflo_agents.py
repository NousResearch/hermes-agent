"""Discover and configure ruflo (claude-flow) agent personas.

Ruflo ships ~110 agent .md files under its repo's ``.claude/agents/`` tree.
Each file has YAML frontmatter (``name``, ``description``) and a markdown body
containing the agent's system prompt. This module discovers those agents and
wires them into Hermes's delegation system so:

  1. ``delegate_task(agent_type="researcher", goal=...)`` automatically loads
     the matching ruflo prompt as the child's system prompt prefix.
  2. ``delegate_task`` consults ``delegation.model_by_role`` in config.yaml for
     a per-agent model override (lets users pin "researcher → Haiku,
     security-architect → Opus" once and have every delegated researcher run
     on Haiku without restating the model in every call).
  3. The ``/delegation`` slash command opens an interactive picker so users can
     browse the 110 agents and assign models.

Discovery is pure-filesystem; nothing here calls any of ruflo's runtime tools.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# Default ruflo install location. Configurable via delegation.ruflo_path in
# config.yaml. Resolved lazily so tests can mock without env shims.
DEFAULT_RUFLO_PATH = "~/repos/ruflo"

# Files at the .claude/agents root with these basenames are not real personas
# (they're docs / migration notes). Filter out by name to avoid polluting
# the picker with non-agent entries.
_NON_AGENT_BASENAMES = frozenset({
    "MIGRATION_SUMMARY",
    "README",
    "INDEX",
})

# Directory names under .claude/agents/ that ship pre-canned cloud-only
# integrations we've stripped from the lockdown build. Skip them silently.
_SKIP_CATEGORIES = frozenset({
    "flow-nexus",  # cloud sandbox/auth/payments — not in lockdown build
    "payments",    # agentic-payments — cloud
    "templates",   # base templates, not personas
})


@dataclass(frozen=True)
class RufloAgent:
    """A single ruflo agent persona discovered on disk.

    Attributes:
        name: Stable identifier (basename without .md extension).
            Use this as the ``agent_type`` when calling ``delegate_task``.
        description: One-line description from the file's YAML frontmatter.
            Empty string if the file has no parseable description.
        category: Subdirectory under ``.claude/agents/`` (e.g. ``"swarm"``,
            ``"core"``, ``"github"``). ``"general"`` for files at the root.
        path: Absolute path to the .md file. The full markdown body is the
            agent's system prompt; load with :meth:`load_prompt`.
    """

    name: str
    description: str
    category: str
    path: str

    def load_prompt(self) -> str:
        """Return the markdown body of the agent file (everything after the
        closing ``---`` of the YAML frontmatter). Returns the whole file if
        no frontmatter is present. Returns an empty string on read error.
        """
        try:
            text = Path(self.path).read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            return ""
        return _strip_frontmatter(text)


def _strip_frontmatter(text: str) -> str:
    """Return ``text`` with leading YAML frontmatter (``---\n...\n---\n``)
    stripped. If the text doesn't start with ``---``, return it unchanged.
    """
    if not text.startswith("---"):
        return text
    # Find the closing --- on its own line.
    rest = text[3:]
    closer = rest.find("\n---")
    if closer < 0:
        return text
    after = rest[closer + 4:]
    return after.lstrip("\n")


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Extract ``name`` and ``description`` from YAML frontmatter.

    Doesn't pull in PyYAML — frontmatter here is simple flat key/value pairs.
    Returns an empty dict if no frontmatter is found or it fails to parse.
    Multi-line values are joined into a single description string.
    """
    if not text.startswith("---"):
        return {}
    rest = text[3:]
    closer = rest.find("\n---")
    if closer < 0:
        return {}
    block = rest[:closer].strip()
    out: dict[str, str] = {}
    current_key: Optional[str] = None
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        # Top-level keys (no leading whitespace)
        if not raw_line.startswith((" ", "\t")) and ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            # Strip surrounding quotes if any
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            out[key] = value
            current_key = key
        elif current_key and raw_line.startswith((" ", "\t")):
            # Continuation of the previous value (multi-line description).
            extra = raw_line.strip()
            if extra:
                out[current_key] = (out.get(current_key, "") + " " + extra).strip()
    return out


def _save_to_config_yaml(key_path: str, value: object) -> bool:
    """Persist ``value`` at ``key_path`` (dot-separated) in the active
    config.yaml. Mirrors ``cli.save_config_value`` but lives here to avoid
    importing ``cli`` (which would pull in prompt_toolkit, the agent loop,
    etc.). Idempotent — creates ``~/.hermes/`` and ``config.yaml`` if absent.

    Returns True on success, False on any I/O / YAML failure.
    """
    try:
        import yaml  # type: ignore
    except Exception:
        return False

    home_env = os.environ.get("HERMES_HOME")
    home = home_env or os.path.expanduser("~/.hermes")
    user_path = Path(home) / "config.yaml"
    # Match cli.save_config_value's two-source precedence: user > project,
    # but write to user_path on first run if neither exists.
    # When HERMES_HOME is set explicitly, ALWAYS write to user_path —
    # don't fall back to project_path. This keeps tests / sandboxed
    # invocations from leaking writes into the repo.
    project_path = Path(__file__).resolve().parent.parent / "cli-config.yaml"
    if home_env:
        cfg_path = user_path
    elif user_path.exists():
        cfg_path = user_path
    elif project_path.exists():
        cfg_path = project_path
    else:
        cfg_path = user_path  # Will be created below.
    try:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}
        # Navigate / create dict path.
        keys = key_path.split(".")
        cur = cfg
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = value
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception:
        return False


def get_ruflo_path(config_path: Optional[str] = None) -> Path:
    """Resolve the ruflo install location.

    Precedence: explicit ``config_path`` arg > ``delegation.ruflo_path`` in
    config.yaml > ``RUFLO_PATH`` env > :data:`DEFAULT_RUFLO_PATH`.
    """
    if config_path:
        return Path(os.path.expanduser(config_path)).resolve()
    # Try config file (lazy import — module shouldn't crash if config is broken).
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        delegation = cfg.get("delegation") if isinstance(cfg, dict) else None
        if isinstance(delegation, dict):
            cfg_path = delegation.get("ruflo_path")
            if isinstance(cfg_path, str) and cfg_path.strip():
                return Path(os.path.expanduser(cfg_path.strip())).resolve()
    except Exception:
        pass
    env = os.environ.get("RUFLO_PATH")
    if env:
        return Path(os.path.expanduser(env)).resolve()
    return Path(os.path.expanduser(DEFAULT_RUFLO_PATH)).resolve()


def discover_ruflo_agents(
    ruflo_path: Optional[Path] = None,
) -> list[RufloAgent]:
    """Scan a ruflo install for agent persona .md files.

    Args:
        ruflo_path: Path to the ruflo repo root. Defaults to ``~/repos/ruflo``.

    Returns:
        Sorted list of :class:`RufloAgent`. Deduped by basename — the same
        agent name often appears in multiple ``.claude/agents/`` directories
        across the ruflo monorepo (root, ``v3/@claude-flow/cli/``, etc.); the
        first one encountered (deterministic walk order) wins. Returns an
        empty list if ruflo isn't installed or has no agents directory.

    Filters:
        - Skips legacy v2 tree (``ruflo/v2/...``).
        - Skips ``node_modules`` and ``__tests__``.
        - Skips files whose basename is in :data:`_NON_AGENT_BASENAMES`.
        - Skips entire categories in :data:`_SKIP_CATEGORIES`
          (cloud integrations stripped from the lockdown build).
    """
    base = ruflo_path or get_ruflo_path()
    if not base.is_dir():
        return []

    seen: dict[str, RufloAgent] = {}

    # rglob for .md files under any .claude/agents/ subtree. We filter further
    # by looking for the literal segment in the path.
    for md in base.rglob("*.md"):
        parts = md.parts
        # Need ".claude" then "agents" as adjacent segments.
        try:
            i = parts.index(".claude")
        except ValueError:
            continue
        if i + 1 >= len(parts) or parts[i + 1] != "agents":
            continue
        # Skip legacy / vendor trees.
        if "v2" in parts or "node_modules" in parts or "__tests__" in parts:
            continue
        name = md.stem  # basename without .md
        if name in _NON_AGENT_BASENAMES:
            continue
        # Category = first dir under .claude/agents/, or "general" if file is
        # directly under .claude/agents/.
        rel_after_agents = parts[i + 2 : -1]  # everything between agents/ and the file
        category = rel_after_agents[0] if rel_after_agents else "general"
        if category in _SKIP_CATEGORIES:
            continue
        if name in seen:
            continue  # dedupe — first encounter wins

        # Read just the frontmatter to extract description.
        try:
            with md.open("r", encoding="utf-8", errors="replace") as f:
                head = f.read(2048)  # frontmatter is always tiny
        except OSError:
            continue
        meta = _parse_frontmatter(head)
        description = meta.get("description", "")
        # Some agent files use "name:" in frontmatter — prefer it for display
        # but keep the file basename as the stable identifier.
        seen[name] = RufloAgent(
            name=name,
            description=description,
            category=category,
            path=str(md),
        )

    return sorted(seen.values(), key=lambda a: (a.category, a.name))


def group_by_category(
    agents: Iterable[RufloAgent],
) -> dict[str, list[RufloAgent]]:
    """Group a list of agents by category, preserving sort order within."""
    out: dict[str, list[RufloAgent]] = {}
    for a in agents:
        out.setdefault(a.category, []).append(a)
    return out


# ── Suggested per-role model defaults ─────────────────────────────────────
#
# Curated mapping of ruflo agent → model based on what each persona is
# typically asked to do.  These are *defaults* the user can apply once via
# the `/delegation` slash command (which writes them into
# delegation.model_by_role); individual roles can be re-pinned afterwards.
#
# Mapping rules:
# - Haiku 4.5: cheap retrieval / triage / grep / scanning / lookup work that
#   doesn't require deep reasoning.  Things that mostly read.
# - Sonnet 4.6: balanced default — coders, testers, reviewers, most swarm
#   coordinators, github automation, refactoring, day-to-day analysis.
# - Opus 4.7: deep reasoning, architecture, security audit, novel algorithm
#   design, complex consensus, multi-step planning under uncertainty.

_HAIKU = "claude-haiku-4-5"
_SONNET = "claude-sonnet-4-6"
_OPUS = "claude-opus-4-7"

SUGGESTED_ROLE_MODELS: dict[str, str] = {
    # ── Haiku — retrieval / triage / monitors / scanners ──────────────────
    "researcher": _HAIKU,
    "scout-explorer": _HAIKU,
    "code-analyzer": _HAIKU,
    "analyze-code-quality": _HAIKU,
    "issue-tracker": _HAIKU,
    "pii-detector": _HAIKU,
    "project-board-sync": _HAIKU,
    "sync-coordinator": _HAIKU,
    "performance-monitor": _HAIKU,
    "resource-allocator": _HAIKU,
    "base-template-generator": _HAIKU,
    "release-manager": _HAIKU,
    "workflow-automation": _HAIKU,
    "load-balancer": _HAIKU,
    "test-long-runner": _HAIKU,

    # ── Sonnet — balanced default for code work ───────────────────────────
    "coder": _SONNET,
    "tester": _SONNET,
    "reviewer": _SONNET,
    "planner": _SONNET,
    "code-review-swarm": _SONNET,
    "pr-manager": _SONNET,
    "swarm-pr": _SONNET,
    "swarm-issue": _SONNET,
    "release-swarm": _SONNET,
    "multi-repo-swarm": _SONNET,
    "github-modes": _SONNET,
    "repo-architect": _SONNET,
    "dev-backend-api": _SONNET,
    "data-ml-model": _SONNET,
    "ops-cicd-github": _SONNET,
    "docs-api-openapi": _SONNET,
    "spec-mobile-react-native": _SONNET,
    "production-validator": _SONNET,
    "tdd-london-swarm": _SONNET,
    "test-architect": _SONNET,
    "python-specialist": _SONNET,
    "typescript-specialist": _SONNET,
    "database-specialist": _SONNET,
    "project-coordinator": _SONNET,
    "topology-optimizer": _SONNET,
    "benchmark-suite": _SONNET,
    "performance-benchmarker": _SONNET,
    # SPARC stages — mostly tactical, sonnet-tier
    "specification": _SONNET,
    "pseudocode": _SONNET,
    "refinement": _SONNET,
    # Swarm coordinators (tactical)
    "adaptive-coordinator": _SONNET,
    "hierarchical-coordinator": _SONNET,
    "mesh-coordinator": _SONNET,
    "worker-specialist": _SONNET,
    # Codex-side workers
    "codex-worker": _SONNET,
    "codex-coordinator": _SONNET,
    # Reasoning-bank / memory
    "reasoningbank-learner": _SONNET,
    "memory-specialist": _SONNET,
    "swarm-memory-manager": _SONNET,
    "v3-memory-specialist": _SONNET,
    # Goal planning (tactical)
    "agent": _SONNET,
    "goal-planner": _SONNET,
    "code-goal-planner": _SONNET,
    # Sublinear specialty (matrix/pagerank — math but bounded)
    "matrix-optimizer": _SONNET,
    "pagerank-analyzer": _SONNET,
    "performance-optimizer": _SONNET,
    "consensus-coordinator": _SONNET,
    "trading-predictor": _SONNET,
    # Sona / aidefence runtime guardian (high-volume)
    "sona-learning-optimizer": _SONNET,
    "aidefence-guardian": _SONNET,
    "claims-authorizer": _SONNET,

    # ── Opus — deep reasoning, architecture, security, novel design ───────
    "arch-system-design": _OPUS,
    "architecture": _OPUS,  # SPARC architecture stage
    "adr-architect": _OPUS,
    "security-architect": _OPUS,
    "security-architect-aidefence": _OPUS,
    "security-auditor": _OPUS,
    "v3-security-architect": _OPUS,
    "ddd-domain-expert": _OPUS,
    "performance-engineer": _OPUS,
    "v3-performance-engineer": _OPUS,
    "v3-integration-architect": _OPUS,
    "byzantine-coordinator": _OPUS,
    "raft-manager": _OPUS,
    "quorum-manager": _OPUS,
    "crdt-synchronizer": _OPUS,
    "gossip-coordinator": _OPUS,
    "security-manager": _OPUS,  # consensus-tier security
    "queen-coordinator": _OPUS,
    "v3-queen-coordinator": _OPUS,
    "sparc-orchestrator": _OPUS,
    "injection-analyst": _OPUS,
    "safla-neural": _OPUS,
    "collective-intelligence-coordinator": _OPUS,
    "dual-orchestrator": _OPUS,
}


def apply_suggested_defaults(*, overwrite: bool = False) -> tuple[int, int]:
    """Bulk-apply :data:`SUGGESTED_ROLE_MODELS` to ``delegation.model_by_role``.

    Args:
        overwrite: When True, replace existing assignments. When False
            (default), only fill in roles that have no current assignment —
            user-customised pins are preserved.

    Returns:
        ``(applied, skipped)`` — counts of roles updated and roles whose
        existing assignment was kept (or that weren't in the suggested map).

    Persists the merged dict to ``~/.hermes/config.yaml`` in a single write.
    """
    current = get_role_model_map()
    merged = dict(current)
    applied = 0
    skipped = 0
    for role, model in SUGGESTED_ROLE_MODELS.items():
        if not overwrite and role in current:
            skipped += 1
            continue
        if current.get(role) == model:
            skipped += 1
            continue
        merged[role] = model
        applied += 1
    if applied == 0:
        return (0, skipped)
    if not _save_to_config_yaml("delegation.model_by_role", merged):
        return (0, skipped)
    return (applied, skipped)


# ── Per-role model assignment (config-backed) ─────────────────────────────


def get_role_model_map() -> dict[str, str]:
    """Read ``delegation.model_by_role`` from ~/.hermes/config.yaml.

    Returns an empty dict when the section is missing or unparseable.
    """
    try:
        from hermes_cli.config import load_config
    except Exception:
        return {}
    try:
        cfg = load_config()
    except Exception:
        return {}
    delegation = cfg.get("delegation") if isinstance(cfg, dict) else None
    if not isinstance(delegation, dict):
        return {}
    raw = delegation.get("model_by_role")
    if not isinstance(raw, dict):
        return {}
    # Coerce values to strings; drop any non-string keys/values defensively.
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str) and v.strip():
            out[k] = v.strip()
    return out


def set_role_model(role: str, model: Optional[str]) -> bool:
    """Persist a per-role model assignment to ``~/.hermes/config.yaml``.

    Args:
        role: Agent role/type identifier (e.g. ``"researcher"``).
        model: Model id to pin (e.g. ``"claude-haiku-4-5"``). Pass ``None``
            or empty string to *remove* the assignment (revert to inherit).

    Returns:
        True on success, False on save failure.
    """
    try:
        from hermes_cli.config import load_config
    except Exception:
        return False
    try:
        cfg = load_config() or {}
    except Exception:
        cfg = {}
    delegation = cfg.get("delegation") if isinstance(cfg, dict) else None
    if not isinstance(delegation, dict):
        delegation = {}
    by_role = delegation.get("model_by_role")
    if not isinstance(by_role, dict):
        by_role = {}
    role = role.strip()
    if not role:
        return False
    if model and model.strip():
        by_role[role] = model.strip()
    else:
        by_role.pop(role, None)
    return _save_to_config_yaml("delegation.model_by_role", by_role)


def lookup_model_for_role(role: Optional[str]) -> Optional[str]:
    """Return the configured model for ``role``, or ``None`` if unset.

    Used by ``tools/delegate_tool.py`` to resolve the per-role model when a
    delegate_task() call passes ``agent_type=...`` but doesn't set ``model=``
    explicitly. Falls through to the existing precedence chain
    (top-level ``model`` arg → ``delegation.model`` config → parent's model)
    when None is returned.
    """
    if not role:
        return None
    return get_role_model_map().get(role.strip())


def lookup_agent(name: str) -> Optional[RufloAgent]:
    """Find a discovered ruflo agent by name. Returns None if not found.

    Convenience for ``delegate_task`` to pull the persona prompt for a given
    ``agent_type=...``.
    """
    if not name:
        return None
    needle = name.strip()
    for a in discover_ruflo_agents():
        if a.name == needle:
            return a
    return None
