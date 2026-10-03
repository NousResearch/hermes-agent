"""Shared per-call model-override object for ``delegate_task``, Kanban and ``cronjob``.

One override shape, one validator, three surfaces.  Before this module each
tool grew its own ad-hoc parsing, so a route could be expressed three
different ways and the flagship/firepower guard could be bypassed by picking
the surface that had not implemented it yet.

The object is::

    {"model": "<id>", "provider": "<id>", "reasoning_effort": "high",
     "firepower": "<reason>", "ttl": "2h"}

Every key is optional and any subset is valid (Ace 2026-09-21), so
``{"provider": "claude-bpx-19"}`` legitimately means "same model, different
provider".  ``ttl`` is accepted only where a time-boxed override makes sense
(the Kanban lane-model row); the per-call surfaces reject it rather than
silently storing a value nothing will ever expire.

Why a hard rejection instead of a coercion
------------------------------------------
``cronjob`` historically accepted a bare string for ``model`` and folded it
into the object shape.  That tolerance is what let an explicit input get
dropped on the OTHER surfaces: ``delegate_task`` registers with
``strict_args=True`` precisely because an undeclared flat ``model=`` "would be
silently dropped and the children would run on the config default"
(``tools/delegate_tool.py``, 2026-09-09).  A flat string is therefore an
ERROR here, and the message names the object shape so the caller can fix the
call in one edit instead of re-reading a schema.  Silence is the bug class
this module exists to close.

The flagship/firepower predicate is NOT defined here — it is imported from
``hermes_cli.model_policy`` so Kanban, delegation and cron share one
classifier.  Forking it would let a model banned on one surface stay
reachable on another.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from hermes_cli.model_policy import firepower_guard_error, is_firepower_model

__all__ = [
    "ModelOverrideError",
    "OVERRIDE_OBJECT_HINT",
    "ModelOverride",
    "parse_model_override",
    "format_route",
]


#: Canonical shape, quoted verbatim in every refusal so the caller never has to
#: guess which key it missed.
OVERRIDE_OBJECT_HINT = (
    '{"model": "<id>", "provider": "<id>", "reasoning_effort": "<level>"}'
)

#: Keys the object accepts. ``ttl`` is gated per-surface (see ``allow_ttl``).
_BASE_KEYS = {"model", "provider", "reasoning_effort", "firepower"}

#: Accepted reasoning levels, sourced from the same constant the agent uses so
#: this validator cannot drift from what the runtime will actually honour.
def _valid_efforts() -> Tuple[str, ...]:
    try:
        from hermes_constants import VALID_REASONING_EFFORTS

        return tuple(VALID_REASONING_EFFORTS)
    except Exception:  # pragma: no cover - constants module always present
        return ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")


class ModelOverrideError(ValueError):
    """Raised when an override object is malformed or violates policy.

    Carries a caller-facing message only — never a stack trace — because every
    consumer surfaces it directly to the model or the operator.
    """


class ModelOverride:
    """A validated override.  Immutable by convention; construct via the parser."""

    __slots__ = ("model", "provider", "reasoning_effort", "firepower", "ttl")

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        firepower: Optional[str] = None,
        ttl: Optional[str] = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        self.firepower = firepower
        self.ttl = ttl

    def __bool__(self) -> bool:
        """Truthy when the override actually pins something."""
        return bool(
            self.model or self.provider or self.reasoning_effort is not None
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ModelOverride):
            return NotImplemented
        return self.as_dict() == other.as_dict()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ModelOverride({self.as_dict()!r})"

    def as_dict(self) -> Dict[str, Any]:
        """Round-trip form: only keys that were actually supplied."""
        out: Dict[str, Any] = {}
        for key in ("model", "provider", "reasoning_effort", "firepower", "ttl"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    def merged_over(self, base: Optional["ModelOverride"]) -> "ModelOverride":
        """Return self layered over *base*, field by field.

        Used for per-task-beats-top-level: a task that pins only ``provider``
        keeps the batch-level ``model`` instead of blanking it.  ``firepower``
        is inherited too, so a batch-level justification covers the tasks it
        authorised rather than forcing the reason to be repeated N times.
        """
        if base is None:
            return self
        return ModelOverride(
            model=self.model if self.model is not None else base.model,
            provider=(
                self.provider if self.provider is not None else base.provider
            ),
            reasoning_effort=(
                self.reasoning_effort
                if self.reasoning_effort is not None
                else base.reasoning_effort
            ),
            firepower=(
                self.firepower if self.firepower is not None else base.firepower
            ),
            ttl=self.ttl if self.ttl is not None else base.ttl,
        )

    def as_delegation_cfg(self) -> Dict[str, Any]:
        """Project onto the ``delegation:`` config shape.

        ``_resolve_delegation_credentials`` consumes a config-shaped mapping;
        emitting that shape lets the per-call route reuse the exact credential
        resolution the config pin already goes through, instead of a parallel
        code path that could resolve providers differently.
        """
        cfg: Dict[str, Any] = {}
        if self.model:
            cfg["model"] = self.model
        if self.provider:
            cfg["provider"] = self.provider
        return cfg


def _coerce_optional_text(value: Any, field: str) -> Optional[str]:
    """Accept a string (or None); reject anything else by name."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ModelOverrideError(
            f"model.{field} must be a string, got {type(value).__name__}. "
            f"Expected {OVERRIDE_OBJECT_HINT}."
        )
    text = str(value).strip()
    return text or None


def parse_model_override(
    raw: Any,
    *,
    field: str = "model",
    allow_ttl: bool = False,
    enforce_firepower: bool = True,
) -> Optional[ModelOverride]:
    """Validate a per-call override object.

    Args:
        raw: the caller-supplied value. ``None``/absent means "no override".
        field: the argument name, used in error messages so the caller knows
            whether the batch-level or a per-task object was at fault.
        allow_ttl: accept a ``ttl`` key (Kanban lane-model only).
        enforce_firepower: apply the flagship guard. Callers pass ``False``
            when the value did NOT come from the call site (e.g. it was read
            back out of config), because the guard exists to make an explicit
            per-call escalation deliberate — not to retroactively refuse a
            standing default the config gate already owns.

    Returns:
        A ``ModelOverride``, or ``None`` when nothing was supplied.

    Raises:
        ModelOverrideError: malformed shape, unknown key, bad reasoning level,
            or an unjustified flagship route.
    """
    if raw is None:
        return None

    # The load-bearing rejection. A bare string is the shape callers reach for
    # by habit; accepting it here would re-open the silent-fallthrough this
    # module exists to close, so name the object shape and refuse.
    if isinstance(raw, str):
        raise ModelOverrideError(
            f"{field} must be an object, not a bare string {raw.strip()!r}. "
            f"Use {OVERRIDE_OBJECT_HINT} — e.g. "
            f'{{"model": "{raw.strip()}"}}' + (
                " (add \"provider\" when the model is not on the default provider)."
            )
        )
    if not isinstance(raw, dict):
        raise ModelOverrideError(
            f"{field} must be an object, got {type(raw).__name__}. "
            f"Expected {OVERRIDE_OBJECT_HINT}."
        )

    allowed = set(_BASE_KEYS)
    if allow_ttl:
        allowed.add("ttl")
    unknown = sorted(k for k in raw if k not in allowed)
    if unknown:
        # An unknown key is almost always a misremembered name
        # ("effort" for "reasoning_effort"). Dropping it silently would pin a
        # route the caller did not ask for, so refuse and list what is valid.
        raise ModelOverrideError(
            f"{field} has unknown key(s) {', '.join(repr(k) for k in unknown)}. "
            f"Valid keys: {', '.join(sorted(allowed))}. "
            f"Expected {OVERRIDE_OBJECT_HINT}."
        )

    model = _coerce_optional_text(raw.get("model"), "model")
    provider = _coerce_optional_text(raw.get("provider"), "provider")
    firepower = _coerce_optional_text(raw.get("firepower"), "firepower")
    ttl = _coerce_optional_text(raw.get("ttl"), "ttl") if allow_ttl else None

    reasoning_effort = None
    if raw.get("reasoning_effort") is not None:
        effort_text = _coerce_optional_text(
            raw.get("reasoning_effort"), "reasoning_effort"
        )
        if effort_text is not None:
            normalized = effort_text.lower()
            # Mirror parse_reasoning_effort's disable aliases so
            # reasoning_effort:"none" means "no thinking", not "invalid".
            if normalized in {"none", "false", "disabled"}:
                reasoning_effort = "none"
            elif normalized in _valid_efforts():
                reasoning_effort = normalized
            else:
                raise ModelOverrideError(
                    f"{field}.reasoning_effort {effort_text!r} is not a valid "
                    f"level. Valid: none, {', '.join(_valid_efforts())}."
                )

    override = ModelOverride(
        model=model,
        provider=provider,
        reasoning_effort=reasoning_effort,
        firepower=firepower,
        ttl=ttl,
    )

    if not override and not firepower and not ttl:
        # An empty object is a no-op the caller probably did not intend; say so
        # rather than letting it look like a successful pin.
        raise ModelOverrideError(
            f"{field} is empty — supply at least one of model, provider or "
            f"reasoning_effort. Expected {OVERRIDE_OBJECT_HINT}."
        )

    if enforce_firepower:
        guard = firepower_guard_error(
            override.model, override.firepower, reason_field=f'{field}.firepower'
        )
        if guard:
            raise ModelOverrideError(
                f"{guard}. Flagship models are reserved: route a batch on the "
                f"standard workhorse unless this specific job needs the "
                f"flagship, and say why."
            )

    return override


def format_route(
    provider: Optional[str],
    model: Optional[str],
    source: str,
    *,
    ttl_remaining: Optional[str] = None,
) -> str:
    """Render the ANNOUNCE line shared by all three surfaces.

    ``route=<provider>/<model> source=<source>`` — one grep-able format so an
    operator can answer "what did this actually run on, and who chose it?"
    from delegation, cron and Kanban logs with the same query.
    """
    route = f"{provider}/{model}" if provider and model else (model or provider or "inherit")
    line = f"route={route} source={source}"
    if ttl_remaining:
        line = f"{line}(ttl {ttl_remaining})"
    if is_firepower_model(model):
        line = f"{line} firepower=yes"
    return line
