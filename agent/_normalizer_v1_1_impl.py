"""Private v1.1 implementation module for the Producer Normalizer.

Encapsulates:
- Registry[T] generic abstraction.
- Concrete ExtractorRegistry, PolicyRegistry.
- ExtractorFactory, PolicyDefinition dataclasses.
- Concrete Extractor implementations (regex_count, etc.).
- PolicyValidator (structural validation).
- PolicyEvaluator (uniform condition evaluation).

This module is NOT part of the contract. The contract uses
`extractor_kind` and `policy_kind` strings. This module is loaded
in bootstrap by agent.producer_normalizer and is the only place
where Python class names for extractors live.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Engine STOP exception (local; re-exported by agent.producer_normalizer)
# ---------------------------------------------------------------------------


class EngineStop(Exception):
    def __init__(self, detail: str, **context: Any) -> None:
        self.detail = detail
        self.context = context
        super().__init__(f"engine_stop: {detail}")


# ---------------------------------------------------------------------------
# Registry[T] common abstraction
# ---------------------------------------------------------------------------


class Registry(Generic[T]):
    """Common registry abstraction. Lookup by stable string key."""

    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, key: str, value: T) -> None:
        if key in self._items:
            raise ValueError(f"duplicate key: {key}")
        self._items[key] = value

    def resolve(self, key: str) -> T:
        if key not in self._items:
            raise EngineStop("registry_key_unknown", key=key)
        return self._items[key]

    def contains(self, key: str) -> bool:
        return key in self._items

    def keys(self) -> set[str]:
        return set(self._items.keys())

    def is_empty(self) -> bool:
        return len(self._items) == 0


# ---------------------------------------------------------------------------
# Extractor interface + concrete extractors
# ---------------------------------------------------------------------------


class Extractor:
    """Base interface for extractors. Concrete impls live below."""

    def run(self, body: str) -> Any:
        raise NotImplementedError


class _RegexCountExtractor(Extractor):
    def __init__(self, pattern: str) -> None:
        self._pattern = re.compile(pattern)

    def run(self, body: str) -> int:
        return len(self._pattern.findall(body))


class _RegexInverseCountExtractor(Extractor):
    """Returns total length of body minus regex match count.

    Used for `non_whitespace_chars`: pattern matches whitespace;
    inverse = total chars - whitespace count.
    """

    def __init__(self, pattern: str) -> None:
        self._pattern = re.compile(pattern)

    def run(self, body: str) -> int:
        ws_matches = self._pattern.findall(body)
        return len(body) - len(ws_matches)


class _JsonParseBooleanExtractor(Extractor):
    def run(self, body: str) -> bool:
        try:
            parsed = json.loads(body)
        except Exception:
            return False
        return isinstance(parsed, (dict, list))


class _JsonTopLevelCountExtractor(Extractor):
    def run(self, body: str) -> int:
        try:
            parsed = json.loads(body)
        except Exception:
            return 0
        if isinstance(parsed, dict):
            return len(parsed.keys())
        if isinstance(parsed, list):
            return len(parsed)
        return 0


class _JsonArrayCountExtractor(Extractor):
    def __init__(self, paths: list, aggregation: str = "max") -> None:
        self._paths = paths
        self._aggregation = aggregation

    def run(self, body: str) -> int:
        try:
            parsed = json.loads(body)
        except Exception:
            return 0
        lengths: list[int] = []
        for path in self._paths:
            val = _resolve_jsonpath(parsed, path)
            if isinstance(val, list):
                lengths.append(len(val))
        if not lengths:
            return 0
        if self._aggregation == "max":
            return max(lengths)
        if self._aggregation == "sum":
            return sum(lengths)
        return max(lengths)


def _resolve_jsonpath(obj, path: str):
    """Minimal JSONPath resolver supporting only $.key.subkey and $.key[N]."""
    if not path.startswith("$"):
        return None
    parts = path[1:].split(".")
    cur = obj
    for p in parts:
        if not p:
            continue
        if "[" in p and p.endswith("]"):
            key, idx = p[:-1].split("[")
            idx = int(idx)
            if key:
                cur = cur.get(key) if isinstance(cur, dict) else None
            if isinstance(cur, list) and idx < len(cur):
                cur = cur[idx]
            else:
                return None
        else:
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                return None
        if cur is None:
            return None
    return cur


# ---------------------------------------------------------------------------
# ExtractorFactory + ExtractorRegistry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractorFactory:
    extractor_kind: str
    deterministic: bool
    allowed_parameters: frozenset
    _create_fn: Callable[[dict], Extractor] = field(repr=False)

    def create(self, parameters: dict) -> Extractor:
        try:
            return self._create_fn(parameters)
        except Exception as e:
            raise EngineStop("extractor_create_failed", kind=self.extractor_kind) from e


def build_extractor_registry() -> "Registry[ExtractorFactory]":
    reg = Registry()
    reg.register("regex_count", ExtractorFactory(
        extractor_kind="regex_count",
        deterministic=True,
        allowed_parameters=frozenset({"pattern"}),
        _create_fn=lambda params: _RegexCountExtractor(pattern=params["pattern"]),
    ))
    reg.register("regex_count_inverse", ExtractorFactory(
        extractor_kind="regex_count_inverse",
        deterministic=True,
        allowed_parameters=frozenset({"pattern"}),
        _create_fn=lambda params: _RegexInverseCountExtractor(pattern=params["pattern"]),
    ))
    reg.register("json_parse_boolean", ExtractorFactory(
        extractor_kind="json_parse_boolean",
        deterministic=True,
        allowed_parameters=frozenset(),
        _create_fn=lambda params: _JsonParseBooleanExtractor(),
    ))
    reg.register("json_top_level_count", ExtractorFactory(
        extractor_kind="json_top_level_count",
        deterministic=True,
        allowed_parameters=frozenset(),
        _create_fn=lambda params: _JsonTopLevelCountExtractor(),
    ))
    reg.register("json_array_count", ExtractorFactory(
        extractor_kind="json_array_count",
        deterministic=True,
        allowed_parameters=frozenset({"paths", "aggregation"}),
        _create_fn=lambda params: _JsonArrayCountExtractor(
            paths=params.get("paths", []),
            aggregation=params.get("aggregation", "max"),
        ),
    ))
    return reg


# ---------------------------------------------------------------------------
# PolicyDefinition + PolicyRegistry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyDefinition:
    policy_kind: str
    deterministic: bool
    allowed_conditions: frozenset
    allowed_actions: frozenset


def build_policy_registry() -> "Registry[PolicyDefinition]":
    reg = Registry()
    base_conditions = frozenset({"json_valid", "substantive_content", "artifact_id"})
    base_actions = frozenset({
        "issue_kind", "severity", "normalizer_verdict_hint", "continue_to_reviewer",
    })
    for kind in ("structural_validation", "consistency_validation",
                 "integrity_validation", "ambiguity_validation"):
        reg.register(kind, PolicyDefinition(
            policy_kind=kind,
            deterministic=True,
            allowed_conditions=base_conditions,
            allowed_actions=base_actions,
        ))
    return reg


# ---------------------------------------------------------------------------
# PolicyValidator (structural validation only)
# ---------------------------------------------------------------------------


VALID_WHEN_KEYS = frozenset({"json_valid", "substantive_content", "artifact_id"})
VALID_THEN_KEYS = frozenset({
    "issue_kind", "severity", "normalizer_verdict_hint", "continue_to_reviewer",
})
VALID_SEVERITIES = frozenset({"informational", "warning", "blocker"})


def _is_semver(s: str) -> bool:
    if not isinstance(s, str):
        return False
    parts = s.split(".")
    if len(parts) != 3:
        return False
    return all(p.isdigit() for p in parts)


class PolicyValidator:
    """Structural validation of policies."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def validate(self, policy: dict) -> PolicyDefinition:
        if not isinstance(policy.get("policy_id"), str):
            raise EngineStop("policy_invalid_id")
        pkind = policy.get("policy_kind")
        if pkind is None:
            raise EngineStop("policy_kind_missing")
        if not self._registry.contains(pkind):
            raise EngineStop("policy_kind_unknown", kind=pkind)
        definition = self._registry.resolve(pkind)
        if not _is_semver(policy.get("policy_version", "")):
            raise EngineStop("policy_invalid_version")
        applies = policy.get("applies_to", [])
        if not applies or not all(isinstance(a, str) for a in applies):
            raise EngineStop("policy_invalid_applies_to")
        when = policy.get("when", {})
        if not isinstance(when, dict):
            raise EngineStop("policy_invalid_when")
        unknown_when = set(when.keys()) - VALID_WHEN_KEYS
        if unknown_when:
            raise EngineStop("policy_invalid_when_keys", keys=sorted(unknown_when))
        then = policy.get("then", {})
        if not isinstance(then, dict):
            raise EngineStop("policy_invalid_then")
        unknown_then = set(then.keys()) - VALID_THEN_KEYS
        if unknown_then:
            raise EngineStop("policy_invalid_then_keys", keys=sorted(unknown_then))
        sev = then.get("severity")
        if sev not in VALID_SEVERITIES:
            raise EngineStop("policy_invalid_severity", severity=sev)
        if not definition.deterministic:
            raise EngineStop("policy_nondeterministic", kind=pkind)
        return definition


# ---------------------------------------------------------------------------
# PolicyEvaluator (uniform condition evaluation)
# ---------------------------------------------------------------------------


class PolicyEvaluator:
    """Evaluate a (validated) policy against derived_state."""

    def evaluate(self, policy: dict, derived_state: dict) -> Optional[dict]:
        when = policy.get("when", {})
        for cond_key, expected in when.items():
            actual = derived_state.get(cond_key)
            if actual != expected:
                return None
        return policy.get("then", {})
