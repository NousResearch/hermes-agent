"""Discover and configure agent personas for delegated subagents.

Personas are markdown files with YAML frontmatter (``name``, ``description``)
shipped under ``~/.hermes/personas/<category>/<name>.md``. They define system
prompt prefixes that get injected into delegated children when their
``agent_type`` matches a persona name.

Originally these came from ruflo's ``.claude/agents/`` tree. We now store
them locally so:

  * Hermes doesn't depend on a ruflo install at runtime.
  * Users can curate / add their own personas without forking ruflo.
  * The list is portable across machines (just rsync the directory).

Use :func:`sync_from_ruflo` once to populate from a ruflo checkout, then the
ruflo dir can be unwired or deleted.

Public surface (everything :mod:`tools.delegate_tool` and the ``/delegation``
slash command rely on):

  * :class:`Persona` (alias :class:`RufloAgent` for back-compat) — discovered
    persona record.
  * :func:`discover_personas` (alias :func:`discover_ruflo_agents`) — scan
    the personas directory.
  * :func:`lookup_agent` — find one by name.
  * :func:`group_by_category` — bucket by subdir.
  * :data:`SUGGESTED_ROLE_MODELS` and :func:`apply_suggested_defaults` —
    curated per-role model defaults (haiku/sonnet/opus by workload).
  * :func:`get_role_model_map`, :func:`set_role_model`,
    :func:`lookup_model_for_role` — read/write ``delegation.model_by_role``
    in ~/.hermes/config.yaml.
  * :func:`sync_from_ruflo` — one-shot rsync from a ruflo checkout.

All discovery is pure-filesystem; nothing here makes network calls.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# Default personas location.  Configurable via ``delegation.personas_path``
# in ~/.hermes/config.yaml; resolved lazily.
DEFAULT_PERSONAS_PATH = "~/.hermes/personas"

# When syncing from a ruflo checkout, reuse ruflo's own filtering rules so
# we don't pull in non-personas (docs, base templates) or cloud-only
# integrations stripped from the lockdown build.
_NON_AGENT_BASENAMES = frozenset({
    "MIGRATION_SUMMARY",
    "README",
    "INDEX",
})

_SKIP_CATEGORIES_FROM_RUFLO = frozenset({
    "flow-nexus",  # cloud sandbox/auth/payments
    "payments",    # agentic-payments — cloud
    "templates",   # base templates, not personas
})


@dataclass(frozen=True)
class Persona:
    """A discovered persona (system prompt + metadata).

    Attributes:
        name: Stable identifier (basename without .md).  Use this as the
            ``agent_type`` when calling ``delegate_task``.
        description: One-line description from the file's YAML frontmatter.
            Empty string if the file has no parseable description.
        category: Subdirectory under the personas root (e.g. ``"swarm"``,
            ``"core"``, ``"github"``).  ``"general"`` for files at the root.
        path: Absolute path to the .md file.  Use :meth:`load_prompt` to
            read the markdown body (frontmatter stripped).
    """

    name: str
    description: str
    category: str
    path: str

    def load_prompt(self) -> str:
        """Return the markdown body of the persona file (everything after the
        closing ``---`` of the YAML frontmatter).  Returns the whole file if
        there's no frontmatter, or an empty string on read error.
        """
        try:
            text = Path(self.path).read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            return ""
        return _strip_frontmatter(text)


# Back-compat alias — older code (tools/delegate_tool.py before the rename,
# tests imported as RufloAgent) keeps working without churn.
RufloAgent = Persona


# ---------------------------------------------------------------------------
# Frontmatter parsing — kept dependency-free (no PyYAML import).
# ---------------------------------------------------------------------------


def _strip_frontmatter(text: str) -> str:
    """Strip leading YAML frontmatter (``---\\n...\\n---\\n``) if present."""
    if not text.startswith("---"):
        return text
    rest = text[3:]
    closer = rest.find("\n---")
    if closer < 0:
        return text
    after = rest[closer + 4:]
    return after.lstrip("\n")


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Extract ``name`` and ``description`` from YAML frontmatter.

    Frontmatter here is simple flat key/value pairs.  Multi-line values
    (continuation lines indented under the previous key) are joined into
    a single description string.  Returns an empty dict if no frontmatter
    is found.
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
        if not raw_line.startswith((" ", "\t")) and ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            out[key] = value
            current_key = key
        elif current_key and raw_line.startswith((" ", "\t")):
            extra = raw_line.strip()
            if extra:
                out[current_key] = (out.get(current_key, "") + " " + extra).strip()
    return out


# ---------------------------------------------------------------------------
# Config persistence helper — duplicated from cli.save_config_value to avoid
# importing cli (which would pull prompt_toolkit and the agent loop).
# ---------------------------------------------------------------------------


def _save_to_config_yaml(key_path: str, value: object) -> bool:
    """Persist ``value`` at ``key_path`` (dot-separated) in active config.yaml."""
    try:
        import yaml  # type: ignore
    except Exception:
        return False

    home_env = os.environ.get("HERMES_HOME")
    home = home_env or os.path.expanduser("~/.hermes")
    user_path = Path(home) / "config.yaml"
    project_path = Path(__file__).resolve().parent.parent / "cli-config.yaml"
    if home_env:
        cfg_path = user_path
    elif user_path.exists():
        cfg_path = user_path
    elif project_path.exists():
        cfg_path = project_path
    else:
        cfg_path = user_path
    try:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}
        if not isinstance(cfg, dict):
            cfg = {}
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


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def get_personas_path(config_path: Optional[str] = None) -> Path:
    """Resolve the personas directory.

    Precedence: explicit ``config_path`` arg > ``delegation.personas_path``
    in config.yaml > ``HERMES_PERSONAS_PATH`` env > :data:`DEFAULT_PERSONAS_PATH`.
    """
    if config_path:
        return Path(os.path.expanduser(config_path)).resolve()
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        delegation = cfg.get("delegation") if isinstance(cfg, dict) else None
        if isinstance(delegation, dict):
            cfg_path = delegation.get("personas_path")
            if isinstance(cfg_path, str) and cfg_path.strip():
                return Path(os.path.expanduser(cfg_path.strip())).resolve()
    except Exception:
        pass
    env = os.environ.get("HERMES_PERSONAS_PATH")
    if env:
        return Path(os.path.expanduser(env)).resolve()
    return Path(os.path.expanduser(DEFAULT_PERSONAS_PATH)).resolve()


# Back-compat alias — older code called this ``get_ruflo_path``.  Keep the
# old name working so callers in tools/, tests/, and skills don't break.
def get_ruflo_path(config_path: Optional[str] = None) -> Path:
    """Deprecated alias for :func:`get_personas_path`."""
    return get_personas_path(config_path)


def discover_personas(
    personas_path: Optional[Path] = None,
) -> list[Persona]:
    """Scan the personas directory for .md files.

    Args:
        personas_path: Personas root.  Defaults to :data:`DEFAULT_PERSONAS_PATH`.

    Returns:
        Sorted list of :class:`Persona` objects.  Ordered by (category, name).
        Returns an empty list if the directory is missing or empty.

    Layout convention:
        ``<root>/<category>/<name>.md`` — top-level files use ``"general"``
        as their category.

    Filters out ``_NON_AGENT_BASENAMES`` (README/INDEX/etc.) so users can
    safely drop documentation alongside personas without it appearing in the
    picker.
    """
    base = personas_path or get_personas_path()
    if not base.is_dir():
        return []

    seen: dict[str, Persona] = {}
    for md in base.rglob("*.md"):
        if not md.is_file():
            continue
        name = md.stem
        if name in _NON_AGENT_BASENAMES:
            continue
        # Category = directory name relative to the personas root.
        # Files at the root use "general".
        try:
            rel = md.relative_to(base)
        except ValueError:
            continue
        if len(rel.parts) > 1:
            category = rel.parts[0]
        else:
            category = "general"
        if name in seen:
            continue  # dedupe — first encounter wins
        try:
            with md.open("r", encoding="utf-8", errors="replace") as f:
                head = f.read(2048)
        except OSError:
            continue
        meta = _parse_frontmatter(head)
        description = meta.get("description", "")
        seen[name] = Persona(
            name=name,
            description=description,
            category=category,
            path=str(md),
        )
    return sorted(seen.values(), key=lambda a: (a.category, a.name))


# Back-compat alias — older imports used ``discover_ruflo_agents``.
def discover_ruflo_agents(
    ruflo_path: Optional[Path] = None,
) -> list[Persona]:
    """Deprecated alias for :func:`discover_personas`."""
    return discover_personas(ruflo_path)


def group_by_category(
    personas: Iterable[Persona],
) -> dict[str, list[Persona]]:
    """Group personas by category, preserving sort order within each bucket."""
    out: dict[str, list[Persona]] = {}
    for p in personas:
        out.setdefault(p.category, []).append(p)
    return out


def lookup_agent(name: str) -> Optional[Persona]:
    """Find a discovered persona by name.  Returns None if not found.

    Used by ``tools/delegate_tool.py`` to load the persona prompt for a given
    ``agent_type=...`` argument on ``delegate_task``.
    """
    if not name:
        return None
    needle = name.strip()
    for p in discover_personas():
        if p.name == needle:
            return p
    return None


# ---------------------------------------------------------------------------
# One-shot sync helper — pulls a ruflo checkout's .claude/agents tree into
# the personas directory.  Idempotent.  Use to refresh after upstream ruflo
# updates, or as a one-time bootstrap.
# ---------------------------------------------------------------------------


def sync_from_ruflo(
    ruflo_root: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    dest: Optional[Path] = None,
) -> tuple[int, int]:
    """Copy persona .md files from a ruflo checkout to the personas dir.

    Args:
        ruflo_root: Path to a ruflo repo checkout (e.g. ``~/repos/ruflo``).
        overwrite: When True, replace files that already exist.  Default
            False — first sync wins, subsequent syncs only add new files.
        dest: Override the destination personas directory.  Defaults to
            :func:`get_personas_path`.

    Returns:
        ``(copied, skipped)`` — counts of files copied vs. skipped (because
        they already existed and ``overwrite=False``).

    Filtering matches the rules used by the original ruflo discovery code:
    skip ``v2/``, ``node_modules/``, ``__tests__/``, the
    :data:`_NON_AGENT_BASENAMES` set, and the
    :data:`_SKIP_CATEGORIES_FROM_RUFLO` cloud-integration categories.
    First-encounter-wins dedup across the ruflo monorepo.
    """
    src_root = Path(os.path.expanduser(str(ruflo_root))).resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"ruflo checkout not found: {src_root}")
    dst_root = (dest or get_personas_path()).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    seen: dict[str, tuple[Path, str]] = {}
    for md in src_root.rglob("*.md"):
        parts = md.parts
        try:
            i = parts.index(".claude")
        except ValueError:
            continue
        if i + 1 >= len(parts) or parts[i + 1] != "agents":
            continue
        if "v2" in parts or "node_modules" in parts or "__tests__" in parts:
            continue
        name = md.stem
        if name in _NON_AGENT_BASENAMES:
            continue
        rel_after = parts[i + 2 : -1]
        category = rel_after[0] if rel_after else "general"
        if category in _SKIP_CATEGORIES_FROM_RUFLO:
            continue
        if name in seen:
            continue
        seen[name] = (md, category)

    copied = 0
    skipped = 0
    for name, (src, category) in seen.items():
        dst_dir = dst_root / category
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{name}.md"
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    return (copied, skipped)


# ---------------------------------------------------------------------------
# Suggested per-role model defaults — curated mapping of persona → model
# based on the workload each persona typically performs.  Apply once via
# the ``/delegation defaults`` command; individual roles can be re-pinned
# afterwards.  Kept in lockstep with :data:`SUGGESTED_ROLE_MODELS` in the
# v1 ``ruflo_agents.py`` module that this replaces.
#
# Mapping rules:
#   Haiku 4.5 — cheap retrieval / triage / monitors / scanners / glue.
#               Anything that mostly reads state, routes work, emits status.
#               Coordinators are here when their reasoning happens in their
#               workers, not their own prompts.
#   Sonnet 4.6 — balanced default for code work: coders, testers, reviewers,
#                most swarm coordinators, github automation, refactoring.
#   Opus 4.7 — deep reasoning: architecture, security, novel algorithm
#              design, complex consensus, multi-step planning under
#              uncertainty.
# ---------------------------------------------------------------------------

_HAIKU = "claude-haiku-4-5"
_SONNET = "claude-sonnet-4-6"
_OPUS = "claude-opus-4-7"

SUGGESTED_ROLE_MODELS: dict[str, str] = {
    # ── Haiku — retrieval / triage / monitors / scanners / glue ───────────
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
    "swarm-issue": _HAIKU,
    "swarm-pr": _HAIKU,
    "release-swarm": _HAIKU,
    "pr-manager": _HAIKU,
    "aidefence-guardian": _HAIKU,
    "claims-authorizer": _HAIKU,

    # ── Sonnet — balanced default for code work ───────────────────────────
    "coder": _SONNET,
    "tester": _SONNET,
    "reviewer": _SONNET,
    "planner": _SONNET,
    "code-review-swarm": _SONNET,
    "multi-repo-swarm": _SONNET,
    "github-modes": _SONNET,
    "dev-backend-api": _SONNET,
    "data-ml-model": _SONNET,
    "ops-cicd-github": _SONNET,
    "docs-api-openapi": _SONNET,
    "spec-mobile-react-native": _SONNET,
    "production-validator": _SONNET,
    "test-architect": _SONNET,
    "python-specialist": _SONNET,
    "typescript-specialist": _SONNET,
    "database-specialist": _SONNET,
    "project-coordinator": _SONNET,
    "topology-optimizer": _SONNET,
    "benchmark-suite": _SONNET,
    "performance-benchmarker": _SONNET,
    # SPARC stages — mostly tactical (architecture stage is in Opus below).
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
    # Memory subsystem (storage/index work; not novel design)
    "memory-specialist": _SONNET,
    "swarm-memory-manager": _SONNET,
    "v3-memory-specialist": _SONNET,
    # Goal planning (tactical)
    "agent": _SONNET,
    "goal-planner": _SONNET,
    "code-goal-planner": _SONNET,
    # Sublinear specialty (matrix / pagerank — bounded math)
    "matrix-optimizer": _SONNET,
    "pagerank-analyzer": _SONNET,
    "performance-optimizer": _SONNET,
    "consensus-coordinator": _SONNET,
    "trading-predictor": _SONNET,
    # Sona learning loops (orchestration of LoRA/SAFLA pipelines)
    "sona-learning-optimizer": _SONNET,
    "safla-neural": _SONNET,
    # Well-defined consensus algorithms — implementation, not novel design.
    "crdt-synchronizer": _SONNET,
    "gossip-coordinator": _SONNET,

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
    "byzantine-coordinator": _OPUS,  # adversarial — needs the depth
    "raft-manager": _OPUS,           # subtle ordering / leader election
    "quorum-manager": _OPUS,         # dynamic membership reasoning
    "security-manager": _OPUS,       # consensus-tier security
    "queen-coordinator": _OPUS,
    "v3-queen-coordinator": _OPUS,
    "sparc-orchestrator": _OPUS,
    "injection-analyst": _OPUS,
    "collective-intelligence-coordinator": _OPUS,
    "dual-orchestrator": _OPUS,
    "repo-architect": _OPUS,
    "reasoningbank-learner": _OPUS,
    "tdd-london-swarm": _OPUS,
}


def apply_suggested_defaults(*, overwrite: bool = False) -> tuple[int, int]:
    """Bulk-apply :data:`SUGGESTED_ROLE_MODELS` to ``delegation.model_by_role``.

    Args:
        overwrite: When True, replace existing assignments.  When False
            (default), only fill in roles that have no current assignment —
            user-customised pins are preserved.

    Returns:
        ``(applied, skipped)`` — counts of roles updated and roles whose
        existing assignment was kept (or that weren't in the suggested map).
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


# ---------------------------------------------------------------------------
# Per-role model assignment (config-backed)
# ---------------------------------------------------------------------------


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
    out: dict[str, str] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str) and v.strip():
            out[k] = v.strip()
    return out


def set_role_model(role: str, model: Optional[str]) -> bool:
    """Persist a per-role model assignment to ~/.hermes/config.yaml.

    Pass ``model=None`` or empty string to remove the assignment.
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
    explicitly.  Falls through to the existing precedence chain (top-level
    ``model`` arg → ``delegation.model`` config → parent's model) when
    None is returned.
    """
    if not role:
        return None
    return get_role_model_map().get(role.strip())
