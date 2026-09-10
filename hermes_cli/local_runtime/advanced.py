"""Typed advanced launch planning for the managed llama.cpp runtime.

The automatic Local Models path remains the default.  This module only turns an
explicit user request into a validated, reproducible plan; it never starts a
server, downloads a model, or writes configuration.

Recipe terminology and the candidate/evidence split are informed by TurboFit
(https://github.com/SouthpawIN/turbofit, MIT, source snapshot
98b45598785c4ca8efe5a5d5ea0835782f4ee007).  The implementation deliberately
uses Hermes' own runtime lifecycle and memory estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from hermes_cli.local_runtime.context_policy import RUNTIME_OVERHEAD_BYTES, ub_logits_bytes
from hermes_cli.local_runtime.estimator import HardwareBudget, ModelProfile, ctx_bytes

_KV_TYPES = frozenset({"q8_0", "f16"})
_SPECULATION = frozenset({"auto", "off", "mtp"})
_MAX_SLOTS = 16


@dataclass(frozen=True)
class LaunchRequest:
    """One persisted, per-model advanced request.

    ``context_tokens`` is the usable window of *each* simultaneous request.
    llama.cpp's ``-c`` is the aggregate allocation, so launch code derives it
    as context_tokens * slots rather than accidentally cutting every user to a
    fraction of the requested window.
    """

    context_tokens: int | None = None
    slots: int = 1
    kv_cache: str = "q8_0"
    speculation: str = "auto"
    mtp_draft_depth: int | None = None

    @classmethod
    def from_mapping(cls, value: object) -> "LaunchRequest":
        if not isinstance(value, dict):
            return cls()
        context = value.get("context_tokens")
        slots = value.get("slots", 1)
        depth = value.get("mtp_draft_depth")
        if isinstance(context, bool) or (context is not None and not isinstance(context, int)):
            raise ValueError("context_tokens must be an integer or null")
        if isinstance(slots, bool) or not isinstance(slots, int):
            raise ValueError("slots must be an integer")
        if isinstance(depth, bool) or (depth is not None and not isinstance(depth, int)):
            raise ValueError("mtp_draft_depth must be an integer or null")
        return cls(
            context_tokens=context, slots=slots,
            kv_cache=str(value.get("kv_cache", "q8_0")),
            speculation=str(value.get("speculation", "auto")), mtp_draft_depth=depth,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "context_tokens": self.context_tokens, "slots": self.slots,
            "kv_cache": self.kv_cache, "speculation": self.speculation,
            "mtp_draft_depth": self.mtp_draft_depth,
        }


@dataclass(frozen=True)
class LaunchPlan:
    request: LaunchRequest
    effective_context_tokens: int
    aggregate_context_tokens: int
    mtp_enabled: bool
    mtp_draft_depth: int | None
    estimated_bytes: int
    available_bytes: int
    fits: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "request": self.request.to_mapping(),
            "effective_context_tokens": self.effective_context_tokens,
            "aggregate_context_tokens": self.aggregate_context_tokens,
            "mtp_enabled": self.mtp_enabled,
            "mtp_draft_depth": self.mtp_draft_depth,
            "estimated_bytes": self.estimated_bytes,
            "available_bytes": self.available_bytes,
            "fits": self.fits,
            "reasons": list(self.reasons),
        }


def validate_request(request: LaunchRequest, *, native_context: int, mtp_supported: bool) -> tuple[bool, int | None, tuple[str, ...]]:
    """Validate user intent without silently degrading it."""
    problems: list[str] = []
    if not 1 <= request.slots <= _MAX_SLOTS:
        problems.append(f"slots must be between 1 and {_MAX_SLOTS}")
    if request.kv_cache not in _KV_TYPES:
        problems.append("kv_cache must be q8_0 or f16")
    if request.speculation not in _SPECULATION:
        problems.append("speculation must be auto, off, or mtp")
    mtp_enabled = request.speculation == "mtp" or (request.speculation == "auto" and mtp_supported)
    if request.speculation == "mtp" and not mtp_supported:
        problems.append("this model does not have a validated MTP recipe")
    depth = request.mtp_draft_depth
    if depth is not None and not 1 <= depth <= 8:
        problems.append("mtp_draft_depth must be between 1 and 8")
    if depth is not None and not mtp_enabled:
        problems.append("mtp_draft_depth requires MTP")
    if request.context_tokens is not None and not 1 <= request.context_tokens <= native_context:
        problems.append(f"context_tokens must be between 1 and {native_context}")
    return mtp_enabled, depth, tuple(problems)


def plan_launch(profile: ModelProfile, budget: HardwareBudget, request: LaunchRequest, *,
                default_context_tokens: int, mtp_supported: bool, default_mtp_depth: int = 3,
                fixed_overhead_bytes: int = RUNTIME_OVERHEAD_BYTES) -> LaunchPlan:
    """Price one model with shared weights and per-slot request state.

    We intentionally account weights once.  KV, logits/work buffers and MTP's
    draft context are per slot.  The value is a conservative admission plan,
    not a replacement for the engine's measured allocator.
    """
    native = profile.n_ctx_train or default_context_tokens
    mtp_enabled, depth, problems = validate_request(request, native_context=native, mtp_supported=mtp_supported)
    context = request.context_tokens if request.context_tokens is not None else min(default_context_tokens, native)
    if context < 1:
        context = 1
    slots = max(1, request.slots)
    planned_profile = replace(profile, kv_scale=1.2 if mtp_enabled else 1.0)
    kv = ctx_bytes(planned_profile, context, flash_attention=request.kv_cache == "q8_0")
    logits = ub_logits_bytes(profile.n_vocab, mtp_capable=mtp_enabled)
    estimated = profile.weights_bytes + fixed_overhead_bytes + slots * (kv + logits)
    available = budget.usable_vram_bytes + budget.ram_available_bytes
    reasons = list(problems)
    if not problems and estimated > available:
        reasons.append("requested context and slots exceed available VRAM plus RAM")
    if not problems and estimated <= available:
        reasons.append("shared weights plus per-slot KV and work buffers fit the planning budget")
    return LaunchPlan(
        request=request, effective_context_tokens=context,
        aggregate_context_tokens=context * slots, mtp_enabled=mtp_enabled,
        mtp_draft_depth=(depth if depth is not None else (default_mtp_depth if mtp_enabled else None)),
        estimated_bytes=estimated, available_bytes=available,
        fits=not problems and estimated <= available, reasons=tuple(reasons),
    )
