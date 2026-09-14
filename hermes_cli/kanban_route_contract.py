"""Pure, fail-closed versioned routing contract for Kanban.

This module deliberately has no configuration, filesystem, database, process,
or network access.  Dispatcher, decomposer, dashboard, and CLI wiring are
separate layers; callers supply the parsed contract mapping and the exact
card routing inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

_SUPPORTED_VERSION: Final = 1
_LANES: Final = frozenset(
    {"planning", "research", "coding", "review", "security", "ops"}
)
_EXECUTIONS: Final = frozenset(
    {"read_only", "remote_isolated", "remote_approval_gated"}
)
_SEPARATION_GROUPS: Final = frozenset(
    {"coding", "review", "security", "ops"}
)
_CONTRACT_KEYS: Final = frozenset(
    {"routing_contract_version", "default_assignee", "orchestrator_profile", "lanes"}
)
_LANE_KEYS: Final = frozenset({"profile", "execution"})
_PROFILE_RE: Final = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


@dataclass(frozen=True, slots=True)
class RouteContract:
    routing_contract_version: int
    default_assignee: str
    orchestrator_profile: str
    lanes: Mapping[str, Mapping[str, str]]
    hash: str
    valid: bool = True

    def __post_init__(self) -> None:
        # Deep immutability: freeze nested lane maps so a parsed contract
        # cannot be mutated after hashing while retaining its identity.
        object.__setattr__(
            self,
            "lanes",
            MappingProxyType(
                {lane: MappingProxyType(spec) for lane, spec in self.lanes.items()}
            ),
        )

    @classmethod
    def invalid(cls) -> RouteContract:
        return cls(
            routing_contract_version=_SUPPORTED_VERSION,
            default_assignee="",
            orchestrator_profile="",
            lanes={},
            hash="",
            valid=False,
        )

    @classmethod
    def from_mapping(cls, raw: object) -> RouteContract:
        if not isinstance(raw, Mapping) or set(raw) != _CONTRACT_KEYS:
            return cls.invalid()

        version = raw.get("routing_contract_version")
        default_assignee = raw.get("default_assignee")
        orchestrator: object = raw.get("orchestrator_profile")
        lanes_raw: object = raw.get("lanes")

        if type(version) is not int or version != _SUPPORTED_VERSION:
            return cls.invalid()
        if default_assignee != "":
            return cls.invalid()
        if not _valid_profile(orchestrator):
            return cls.invalid()
        if not isinstance(lanes_raw, Mapping) or set(lanes_raw) != _LANES:
            return cls.invalid()

        lanes: dict[str, dict[str, str]] = {}
        for lane in sorted(_LANES):
            spec = lanes_raw[lane]
            if not isinstance(spec, Mapping) or set(spec) != _LANE_KEYS:
                return cls.invalid()
            profile = spec.get("profile")
            execution = spec.get("execution")
            if not _valid_profile(profile):
                return cls.invalid()
            if not isinstance(execution, str) or execution not in _EXECUTIONS:
                return cls.invalid()
            lanes[lane] = {"profile": profile, "execution": execution}

        separated = [lanes[lane]["profile"] for lane in sorted(_SEPARATION_GROUPS)]
        if len(set(separated)) != len(separated):
            return cls.invalid()

        digest = hashlib.sha256(
            json.dumps(
                {
                    "routing_contract_version": version,
                    "default_assignee": default_assignee,
                    "orchestrator_profile": orchestrator,
                    "lanes": lanes,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return cls(
            routing_contract_version=version,
            default_assignee=default_assignee,
            orchestrator_profile=orchestrator,
            lanes=lanes,
            hash=digest,
        )


@dataclass(frozen=True, slots=True)
class RouteDecision:
    status: str
    profile: str | None
    reason: str
    routing_contract_version: int
    routing_contract_hash: str


def resolve_route(
    contract: RouteContract,
    task_id: str,
    explicit_assignee: str | None = None,
    lane: str | None = None,
) -> RouteDecision:
    if not contract.valid:
        return RouteDecision(
            status="blocked",
            profile=None,
            reason="routing contract is missing, malformed, or unsupported",
            routing_contract_version=0,
            routing_contract_hash="",
        )
    if explicit_assignee is not None:
        if _valid_profile(explicit_assignee):
            return RouteDecision(
                status="routed",
                profile=explicit_assignee,
                reason="explicit assignee is authoritative",
                routing_contract_version=contract.routing_contract_version,
                routing_contract_hash=contract.hash,
            )
        return RouteDecision(
            status="triage",
            profile=None,
            reason="diagnostic: explicit assignee is invalid; no fallback route",
            routing_contract_version=contract.routing_contract_version,
            routing_contract_hash=contract.hash,
        )
    if lane is not None and lane in contract.lanes:
        return RouteDecision(
            status="routed",
            profile=contract.lanes[lane]["profile"],
            reason="lane resolved at admission time",
            routing_contract_version=contract.routing_contract_version,
            routing_contract_hash=contract.hash,
        )
    return RouteDecision(
        status="triage",
        profile=None,
        reason="diagnostic: no valid explicit assignee or lane; no default route",
        routing_contract_version=contract.routing_contract_version,
        routing_contract_hash=contract.hash,
    )


def _valid_profile(value: object) -> bool:
    return (
        type(value) is str
        and value != ""
        and value.strip() == value
        and _PROFILE_RE.match(value) is not None
    )
