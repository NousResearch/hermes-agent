"""Kanban agent-profile registry (seven profiles: orchestrator + six lanes).

Pure, fail-closed, side-effect-free definitions. No configuration,
filesystem, database, process, or network access. Dispatchers,
decomposers, and worker sessions consume this module; they never
mutate it. Profile enforcement (actually applying these boundaries to
a live session) is a separate wiring layer and is explicitly out of
scope here.

Structure per profile:
- name, role (what the profile exists to do)
- description (human-readable, stable across builds)
- toolsets: exact allowlist of Hermes toolset names the profile may load
- terminal_scope / file_scope: boundary semantics only (strings), since
  real path scoping is enforced at wiring time, not in this registry
- credential_scopes: names of credential scopes (no secrets here)
- capabilities: behavioral flags the dispatcher may rely on
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

_REGISTRY_VERSION: Final = 1
_PROFILE_NAME_RE: Final = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

#: Toolsets a purely-coordinating profile may load: no execution surface.
_ORCHESTRATOR_TOOLSETS: Final = ("delegation",)
#: Toolsets profiles need to read context and produce artifacts.
_READ_TOOLSETS: Final = ("web", "file")
#: Baseline every lane profile gets: read/search plus delegation up only.
_LANE_BASE_TOOLSETS: Final = ("web", "file", "delegation")

_CREDENTIAL_SCOPE_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_/]{0,95}$")


@dataclass(frozen=True, slots=True)
class AgentProfile:
    """One immutable profile definition."""

    name: str
    role: str
    description: str
    toolsets: Mapping[str, None]
    terminal_scope: str
    file_scope: str
    credential_scopes: Mapping[str, None]
    capabilities: Mapping[str, bool]
    registry_version: int = _REGISTRY_VERSION

    def __post_init__(self) -> None:
        for field in ("toolsets", "credential_scopes", "capabilities"):
            object.__setattr__(
                self, field, MappingProxyType(dict(getattr(self, field)))
            )

    @property
    def allowed_toolsets(self) -> frozenset[str]:
        return frozenset(self.toolsets)

    @property
    def allowed_credential_scopes(self) -> frozenset[str]:
        return frozenset(self.credential_scopes)


def _profile(
    name: str,
    role: str,
    description: str,
    toolsets: tuple[str, ...],
    terminal_scope: str,
    file_scope: str,
    credential_scopes: tuple[str, ...],
    capabilities: Mapping[str, bool],
) -> AgentProfile:
    return AgentProfile(
        name=name,
        role=role,
        description=description,
        toolsets={t: None for t in toolsets},
        terminal_scope=terminal_scope,
        file_scope=file_scope,
        credential_scopes={s: None for s in credential_scopes},
        capabilities=dict(capabilities),
    )


#: The canonical registry. Lane names match kanban_route_contract lanes
#: one-to-one; the orchestrator profile backs the routing contract's
#: ``orchestrator_profile`` field.
PROFILES: Final[Mapping[str, AgentProfile]] = MappingProxyType(
    {
        p.name: p
        for p in (
            _profile(
                "kanban-orchestrator",
                role="orchestrator",
                description=(
                    "Coordinates the board only: routes cards, tracks "
                    "generations, records launch outcomes. Never edits "
                    "repository code and never executes project commands."
                ),
                toolsets=_ORCHESTRATOR_TOOLSETS,
                terminal_scope="none",
                file_scope="board-state-read-only",
                credential_scopes=("kanban/board",),
                capabilities={
                    "may_delegate": True,
                    "may_execute_terminal": False,
                    "may_edit_code": False,
                    "may_approve_external_effects": False,
                },
            ),
            _profile(
                "kanban-planning",
                role="planning",
                description=(
                    "Decomposes approved cards into isolated implementation "
                    "slices and writes plans. Read-only research over repo "
                    "and docs; delegates coding."
                ),
                toolsets=_LANE_BASE_TOOLSETS,
                terminal_scope="none",
                file_scope="repo-read-only+plans",
                credential_scopes=("kanban/board",),
                capabilities={
                    "may_delegate": True,
                    "may_execute_terminal": False,
                    "may_edit_code": False,
                    "may_approve_external_effects": False,
                },
            ),
            _profile(
                "kanban-research",
                role="research",
                description=(
                    "Read-only evidence gathering: web search, source "
                    "extraction, benchmark notes. Produces cited findings; "
                    "never mutates repository or infrastructure."
                ),
                toolsets=_READ_TOOLSETS,
                terminal_scope="none",
                file_scope="scratch-notes-only",
                credential_scopes=("kanban/board",),
                capabilities={
                    "may_delegate": False,
                    "may_execute_terminal": False,
                    "may_edit_code": False,
                    "may_approve_external_effects": False,
                },
            ),
            _profile(
                "kanban-coding",
                role="coding",
                description=(
                    "Implements one isolated slice in a dedicated worktree "
                    "under TDD gates. Terminal restricted to the slice "
                    "worktree; no board writes, no deployment."
                ),
                toolsets=_LANE_BASE_TOOLSETS + ("terminal",),
                terminal_scope="slice-worktree-only",
                file_scope="slice-worktree-rw",
                credential_scopes=("kanban/board", "git/commit_local"),
                capabilities={
                    "may_delegate": False,
                    "may_execute_terminal": True,
                    "may_edit_code": True,
                    "may_approve_external_effects": False,
                },
            ),
            _profile(
                "kanban-review",
                role="review",
                description=(
                    "Independent exact-tree review of a candidate commit. "
                    "Read-only over the reviewed tree; findings only, no "
                    "fixes, no merges."
                ),
                toolsets=_READ_TOOLSETS,
                terminal_scope="none",
                file_scope="reviewed-tree-read-only",
                credential_scopes=("kanban/board",),
                capabilities={
                    "may_delegate": False,
                    "may_execute_terminal": False,
                    "may_edit_code": False,
                    "may_approve_external_effects": False,
                },
            ),
            _profile(
                "kanban-security",
                role="security",
                description=(
                    "Supply-chain and secret-exposure review of a candidate "
                    "tree. Read-only; findings only. Never holds broad "
                    "credentials."
                ),
                toolsets=_READ_TOOLSETS,
                terminal_scope="none",
                file_scope="reviewed-tree-read-only",
                credential_scopes=("kanban/board",),
                capabilities={
                    "may_delegate": False,
                    "may_execute_terminal": False,
                    "may_edit_code": False,
                    "may_approve_external_effects": False,
                },
            ),
            _profile(
                "kanban-ops",
                role="ops",
                description=(
                    "Approval-gated deployment and infrastructure "
                    "execution. Terminal limited to the approval bundle; "
                    "every external effect requires a recorded approval."
                ),
                toolsets=_LANE_BASE_TOOLSETS + ("terminal",),
                terminal_scope="approval-bundle-only",
                file_scope="approval-bundle-rw",
                credential_scopes=("kanban/board", "ops/deploy"),
                capabilities={
                    "may_delegate": False,
                    "may_execute_terminal": True,
                    "may_edit_code": False,
                    "may_approve_external_effects": True,
                },
            ),
        )
    }
)

#: Lane → profile-name mapping that the routing contract's lanes must
#: agree with (single source of truth for lane separation).
LANE_PROFILE_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "planning": PROFILES["kanban-planning"].name,
        "research": PROFILES["kanban-research"].name,
        "coding": PROFILES["kanban-coding"].name,
        "review": PROFILES["kanban-review"].name,
        "security": PROFILES["kanban-security"].name,
        "ops": PROFILES["kanban-ops"].name,
    }
)


def registry_hash() -> str:
    """Canonical SHA-256 over the sorted, canonical JSON registry."""
    payload = {
        "registry_version": _REGISTRY_VERSION,
        "profiles": {
            name: {
                "name": p.name,
                "role": p.role,
                "description": p.description,
                "toolsets": sorted(p.toolsets),
                "terminal_scope": p.terminal_scope,
                "file_scope": p.file_scope,
                "credential_scopes": sorted(p.credential_scopes),
                "capabilities": dict(sorted(p.capabilities.items())),
                "registry_version": p.registry_version,
            }
            for name, p in sorted(PROFILES.items())
        },
        "lane_profile_names": dict(sorted(LANE_PROFILE_NAMES.items())),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _expected_toolsets(role: str) -> tuple[str, ...]:
    base = _LANE_BASE_TOOLSETS
    if role == "orchestrator":
        return _ORCHESTRATOR_TOOLSETS
    if role == "research":
        return _READ_TOOLSETS
    if role in ("coding", "ops"):
        return base + ("terminal",)
    if role == "planning":
        return base
    return _READ_TOOLSETS   # review, security


#: Exact expected registry content. validate_registry() compares every
#: profile field against this spec, so any post-construction tampering
#: (including via object.__setattr__) is detected as a mismatch.
_EXPECTED_CAPABILITIES: Final[Mapping[str, Mapping[str, bool]]] = MappingProxyType(
    {
        # Nested mappings are individually frozen below via _deep_freeze.
        "orchestrator": {
            "may_delegate": True,
            "may_execute_terminal": False,
            "may_edit_code": False,
            "may_approve_external_effects": False,
        },
        "planning": {
            "may_delegate": True,
            "may_execute_terminal": False,
            "may_edit_code": False,
            "may_approve_external_effects": False,
        },
        "research": {
            "may_delegate": False,
            "may_execute_terminal": False,
            "may_edit_code": False,
            "may_approve_external_effects": False,
        },
        "coding": {
            "may_delegate": False,
            "may_execute_terminal": True,
            "may_edit_code": True,
            "may_approve_external_effects": False,
        },
        "review": {
            "may_delegate": False,
            "may_execute_terminal": False,
            "may_edit_code": False,
            "may_approve_external_effects": False,
        },
        "security": {
            "may_delegate": False,
            "may_execute_terminal": False,
            "may_edit_code": False,
            "may_approve_external_effects": False,
        },
        "ops": {
            "may_delegate": False,
            "may_execute_terminal": True,
            "may_edit_code": False,
            "may_approve_external_effects": True,
        },
    }
)
_EXPECTED_CAPABILITIES = MappingProxyType(
    {r: MappingProxyType(m) for r, m in _EXPECTED_CAPABILITIES.items()}
)

_EXPECTED_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "orchestrator": "kanban-orchestrator",
        "planning": "kanban-planning",
        "research": "kanban-research",
        "coding": "kanban-coding",
        "review": "kanban-review",
        "security": "kanban-security",
        "ops": "kanban-ops",
    }
)

_EXPECTED_TERMINAL_SCOPE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "orchestrator": "none",
        "planning": "none",
        "research": "none",
        "coding": "slice-worktree-only",
        "review": "none",
        "security": "none",
        "ops": "approval-bundle-only",
    }
)

_EXPECTED_FILE_SCOPE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "orchestrator": "board-state-read-only",
        "planning": "repo-read-only+plans",
        "research": "scratch-notes-only",
        "coding": "slice-worktree-rw",
        "review": "reviewed-tree-read-only",
        "security": "reviewed-tree-read-only",
        "ops": "approval-bundle-rw",
    }
)


_EXPECTED_DESCRIPTIONS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "orchestrator": (
            "Coordinates the board only: routes cards, tracks generations, "
            "records launch outcomes. Never edits repository code and never "
            "executes project commands."
        ),
        "planning": (
            "Decomposes approved cards into isolated implementation slices "
            "and writes plans. Read-only research over repo and docs; "
            "delegates coding."
        ),
        "research": (
            "Read-only evidence gathering: web search, source extraction, "
            "benchmark notes. Produces cited findings; never mutates "
            "repository or infrastructure."
        ),
        "coding": (
            "Implements one isolated slice in a dedicated worktree under "
            "TDD gates. Terminal restricted to the slice worktree; no board "
            "writes, no deployment."
        ),
        "review": (
            "Independent exact-tree review of a candidate commit. Read-only "
            "over the reviewed tree; findings only, no fixes, no merges."
        ),
        "security": (
            "Supply-chain and secret-exposure review of a candidate tree. "
            "Read-only; findings only. Never holds broad credentials."
        ),
        "ops": (
            "Approval-gated deployment and infrastructure execution. "
            "Terminal limited to the approval bundle; every external effect "
            "requires a recorded approval."
        ),
    }
)

_EXPECTED_CREDENTIAL_SCOPES: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "orchestrator": frozenset({"kanban/board"}),
        "planning": frozenset({"kanban/board"}),
        "research": frozenset({"kanban/board"}),
        "coding": frozenset({"kanban/board", "git/commit_local"}),
        "review": frozenset({"kanban/board"}),
        "security": frozenset({"kanban/board"}),
        "ops": frozenset({"kanban/board", "ops/deploy"}),
    }
)


def _is_frozen_mapping(value: object) -> bool:
    return isinstance(value, MappingProxyType)


def validate_registry() -> None:
    """Fail closed on any registry invariant violation or content drift."""
    if set(PROFILES) != set(_EXPECTED_NAMES.values()):
        raise ValueError(
            f"profile name set mismatch: {sorted(PROFILES)} vs "
            f"{sorted(_EXPECTED_NAMES.values())}"
        )
    seen_roles: set[str] = set()
    for name, profile in PROFILES.items():
        if not _PROFILE_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid profile name: {name!r}")
        if name != profile.name:
            raise ValueError(f"registry key {name!r} != profile.name {profile.name!r}")
        role = profile.role
        if role not in _EXPECTED_NAMES:
            raise ValueError(f"invalid role for {name!r}: {role!r}")
        if role in seen_roles:
            raise ValueError(f"duplicate role: {role!r}")
        seen_roles.add(role)
        if name != _EXPECTED_NAMES[role]:
            raise ValueError(f"role {role!r} must use profile name {_EXPECTED_NAMES[role]!r}")
        if type(profile.registry_version) is not int or profile.registry_version != _REGISTRY_VERSION:
            raise ValueError(f"registry_version drift for {name!r}")
        if not profile.description or len(profile.description) > 400:
            raise ValueError(f"description out of bounds for {name!r}")
        # Frozen-container checks: a tampered (mutable) replacement is a
        # validation failure even if its current content matches.
        for field in ("toolsets", "credential_scopes", "capabilities"):
            if not _is_frozen_mapping(getattr(profile, field)):
                raise ValueError(f"{field!r} is not a frozen mapping for {name!r}")
            if field != "capabilities" and any(
                v is not None for v in getattr(profile, field).values()
            ):
                raise ValueError(f"{field!r} must carry None values only for {name!r}")
        # Exact content checks.
        expected_toolsets = frozenset(_expected_toolsets(role))
        if profile.allowed_toolsets != expected_toolsets:
            raise ValueError(
                f"toolset mismatch for {name!r}: {sorted(profile.allowed_toolsets)} "
                f"!= {sorted(expected_toolsets)}"
            )
        if profile.terminal_scope != _EXPECTED_TERMINAL_SCOPE[role]:
            raise ValueError(f"terminal_scope mismatch for {name!r}")
        if profile.file_scope != _EXPECTED_FILE_SCOPE[role]:
            raise ValueError(f"file_scope mismatch for {name!r}")
        if profile.description != _EXPECTED_DESCRIPTIONS[role]:
            raise ValueError(f"description mismatch for {name!r}")
        if profile.allowed_credential_scopes != _EXPECTED_CREDENTIAL_SCOPES[role]:
            raise ValueError(f"credential scope mismatch for {name!r}")
        for scope in profile.credential_scopes:
            if not _CREDENTIAL_SCOPE_RE.fullmatch(scope):
                raise ValueError(f"invalid credential scope {scope!r} in {name!r}")
        expected_caps = _EXPECTED_CAPABILITIES[role]
        caps = dict(profile.capabilities)
        if set(caps) != set(expected_caps):
            raise ValueError(f"capability key mismatch for {name!r}")
        for key, value in caps.items():
            if type(value) is not bool:
                raise ValueError(f"capability {key!r} is not a bool for {name!r}")
            if value != expected_caps[key]:
                raise ValueError(
                    f"capability {key!r}={value!r} does not match expected "
                    f"{expected_caps[key]!r} for {name!r}"
                )
    # Lane mapping consistency.
    expected_lanes = {"planning", "research", "coding", "review", "security", "ops"}
    if set(LANE_PROFILE_NAMES) != expected_lanes:
        raise ValueError("lane profile mapping must cover exactly the six lanes")
    for lane, pname in LANE_PROFILE_NAMES.items():
        if pname not in PROFILES:
            raise ValueError(f"lane {lane!r} maps to unknown profile {pname!r}")
        if PROFILES[pname].role != lane:
            raise ValueError(f"lane {lane!r} maps to role {PROFILES[pname].role!r}")
    if not _is_frozen_mapping(LANE_PROFILE_NAMES):
        raise ValueError("LANE_PROFILE_NAMES is not a frozen mapping")


def get_profile(name: str) -> AgentProfile:
    """Return the profile or raise KeyError (fail closed)."""
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown agent profile: {name!r}") from None
