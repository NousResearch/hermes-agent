"""Deterministic authority attenuation and effect settlement (DR Task 3.2).

Pure contract module: no I/O, no environment reads, no wall-clock time.
Monotone authority is enforced structurally — a child scope can never exceed
its parent, every consumption is a typed receipt, and all state transitions
are deterministic and replayable.
"""

from __future__ import annotations

from typing import Any, Mapping

from .collaboration import ContractError, digest

SCHEMA_VERSION = "authority-scope-v1"

_SCOPE_FIELDS = ("tool", "target", "time", "use_count")
_SETTLEMENT_STATES = frozenset({"reserved", "committed", "released", "indeterminate"})
_OPEN_STATES = frozenset({"reserved"})
_DEFAULT_HOLDER = "holder:unassigned"


def _require_str(value: Any, code: str, field: str = "") -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(code, field)
    return value


def normalize_scope(raw: Any) -> dict[str, Any]:
    """Canonicalize a scope record; reject unknown fields and bad types."""
    if not isinstance(raw, Mapping):
        raise ContractError("INVALID_SCOPE")
    unknown = set(raw) - set(_SCOPE_FIELDS)
    if unknown:
        raise ContractError("UNKNOWN_FIELD", sorted(unknown)[0])
    normalized: dict[str, Any] = {}
    for field in _SCOPE_FIELDS:
        if field not in raw:
            continue
        value = raw[field]
        if field == "use_count":
            if type(value) is not int or value < 1:
                raise ContractError("INVALID_USE_COUNT")
            normalized[field] = value
        elif field == "time":
            if not isinstance(value, Mapping):
                raise ContractError("INVALID_TIME_SCOPE")
            unknown_time = set(value) - {"not_before", "not_after"}
            if unknown_time:
                raise ContractError("UNKNOWN_FIELD", sorted(unknown_time)[0])
            if not value:
                raise ContractError("INVALID_TIME_SCOPE")
            for bound in ("not_before", "not_after"):
                if bound in value:
                    _require_str(value[bound], "INVALID_TIME_BOUND", "time." + bound)
            if "not_before" in value and "not_after" in value and value["not_before"] > value["not_after"]:
                raise ContractError("INVALID_TIME_SCOPE")
            normalized[field] = dict(value)
        else:
            normalized[field] = _require_str(value, "INVALID_" + field.upper(), field)
    if not normalized:
        raise ContractError("EMPTY_SCOPE")
    return normalized


def scope_fingerprint(scope: Mapping[str, Any]) -> str:
    return digest({"schema": SCHEMA_VERSION, "scope": normalize_scope(scope)})


def is_subset_scope(subset: Mapping[str, Any], superset: Mapping[str, Any]) -> bool:
    """True iff `subset`'s authority is entirely within `superset`'s.

    Absence in the superset means unrestricted for that field.  Absence in
    the subset ALSO means unrestricted for that field, so a subset must
    restate every restriction its parent imposes (strict monotone semantics;
    silent field loss can never widen an emitted grant).  use_count must be
    <=; tool/target must match exactly; time bounds must be contained.  An
    empty subset imposes no authority of its own and is trivially contained;
    grant creation (normalize_scope) still rejects an empty scope fail-closed.
    """
    if not isinstance(subset, Mapping) or not isinstance(superset, Mapping):
        raise ContractError("INVALID_SCOPE")
    if not subset:
        return True
    if not superset:
        return False
    a = normalize_scope(subset)
    b = normalize_scope(superset)
    for field in _SCOPE_FIELDS:
        in_a, in_b = field in a, field in b
        if in_b and not in_a:
            return False  # child drops a parent restriction -> widening
        if not in_a or not in_b:
            continue  # child-only restriction is narrower
        if field == "use_count":
            if a[field] > b[field]:
                return False
        elif field == "time":
            sa, sb = a[field], b[field]
            if "not_before" in sb and ("not_before" not in sa or sa["not_before"] < sb["not_before"]):
                return False
            if "not_after" in sb and ("not_after" not in sa or sa["not_after"] > sb["not_after"]):
                return False
        else:
            if a[field] != b[field]:
                return False
    return True


class AuthorityScopeV1:
    """Immutable attenuable authority scope with generation-bound allocation."""

    def __init__(self, *, scope: Mapping[str, Any], generation: int, holder: str | None = None):
        self._scope = normalize_scope(scope)
        if type(generation) is not int or generation < 0:
            raise ContractError("INVALID_GENERATION")
        self._generation = generation
        self._holder = _DEFAULT_HOLDER if holder is None else _require_str(holder, "INVALID_HOLDER")
        self._settlements: dict[str, dict[str, Any]] = {}
        self._open_count = 0
        self._consumed_count = 0

    @property
    def scope(self) -> dict[str, Any]:
        return dict(self._scope)

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def holder(self) -> str:
        return self._holder

    def fingerprint(self) -> str:
        return scope_fingerprint(self._scope)

    def is_subset(self, other: "AuthorityScopeV1") -> bool:
        if not isinstance(other, AuthorityScopeV1):
            raise ContractError("INVALID_SCOPE_OBJECT")
        return is_subset_scope(self._scope, other._scope)

    def attenuate(self, child_scope: Mapping[str, Any], *, child_generation: int, child_holder: str | None = None) -> "AuthorityScopeV1":
        """Derive a child scope; monotone authority is mandatory, not optional.

        The child restates every parent restriction (strict subset
        semantics); unspecified child fields would otherwise silently widen.
        """
        if type(child_generation) is not int or child_generation <= self._generation:
            raise ContractError("NON_MONOTONE_GENERATION")
        normalized_child = normalize_scope(child_scope)
        inherited = dict(self._scope)
        inherited.update(normalized_child)
        if not is_subset_scope(inherited, self._scope):
            raise ContractError("ATTENUATION_ESCALATION")
        return AuthorityScopeV1(scope=inherited, generation=child_generation, holder=child_holder)

    def reserve(self, *, consumption_ref: str, args_digest: str, target_ref: str | None = None) -> dict[str, Any]:
        """Open a reservation against finite remaining use."""
        consumption_ref = _require_str(consumption_ref, "INVALID_CONSUMPTION_REF")
        args_digest = _require_str(args_digest, "INVALID_ARGS_DIGEST")
        if consumption_ref in self._settlements:
            raise ContractError("DUPLICATE_CONSUMPTION_REF")
        if "use_count" in self._scope and self._open_count >= self._scope["use_count"]:
            raise ContractError("USE_COUNT_EXHAUSTED")
        if target_ref is not None:
            target_ref = _require_str(target_ref, "INVALID_TARGET_REF")
            if "target" in self._scope and target_ref != self._scope["target"]:
                raise ContractError("TARGET_OUTSIDE_SCOPE")
        self._settlements[consumption_ref] = {
            "consumption_ref": consumption_ref,
            "state": "reserved",
            "args_digest": args_digest,
            "target_ref": target_ref,
        }
        self._open_count += 1
        return self._receipt(consumption_ref)

    def commit(self, consumption_ref: str, *, effect_receipt_digest: str) -> dict[str, Any]:
        return self._settle(consumption_ref, "committed", effect_receipt_digest=effect_receipt_digest)

    def release(self, consumption_ref: str, *, reason: str = "operator_release") -> dict[str, Any]:
        return self._settle(consumption_ref, "released", reason=reason)

    def mark_indeterminate(self, consumption_ref: str, *, reason: str) -> dict[str, Any]:
        if not _require_str(reason, "INVALID_REASON"):
            raise ContractError("INVALID_REASON")
        return self._settle(consumption_ref, "indeterminate", reason=reason)

    def _settle(self, consumption_ref: str, state: str, **extra: str) -> dict[str, Any]:
        _require_str(consumption_ref, "INVALID_CONSUMPTION_REF")
        record = self._settlements.get(consumption_ref)
        if record is None:
            raise ContractError("UNKNOWN_CONSUMPTION_REF")
        if record["state"] != "reserved":
            raise ContractError("ALREADY_SETTLED")
        record["state"] = state
        record.update(extra)
        self._open_count -= 1
        if state == "committed":
            self._consumed_count += 1
        return self._receipt(consumption_ref)

    def _receipt(self, consumption_ref: str) -> dict[str, Any]:
        record = self._settlements[consumption_ref]
        receipt = {
            "schema": SCHEMA_VERSION,
            "scope_fingerprint": self.fingerprint(),
            "generation": self._generation,
            "holder": self._holder,
            "record": dict(record),
            "open_reservations": self._open_count,
            "consumed_total": self._consumed_count,
        }
        receipt["receipt_digest"] = digest(receipt)
        return receipt

    def settlement(self, consumption_ref: str) -> Mapping[str, Any] | None:
        record = self._settlements.get(consumption_ref)
        return dict(record) if record else None

    def subset_witness(self, parent: "AuthorityScopeV1") -> dict[str, Any]:
        """Deterministic witness that self is contained in parent."""
        if not isinstance(parent, AuthorityScopeV1):
            raise ContractError("INVALID_SCOPE_OBJECT")
        contained = self.is_subset(parent)
        witness = {
            "schema": SCHEMA_VERSION,
            "witness_kind": "subset",
            "child_fingerprint": self.fingerprint(),
            "parent_fingerprint": parent.fingerprint(),
            "child_generation": self._generation,
            "parent_generation": parent._generation,
            "contained": contained,
        }
        witness["witness_digest"] = digest(witness)
        if not contained:
            raise ContractError("ATTENUATION_ESCALATION")
        return witness
