"""Detect xAI models retired on May 15, 2026.

Source: https://docs.x.ai/developers/migration/may-15-retirement

Pure logic: walks a Hermes config dict, returns issues for any reference
to a retired xAI model. No I/O, no CLI dependencies — testable in isolation
and reusable from both `hermes doctor` and a future `hermes migrate xai`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


MIGRATION_GUIDE_URL = "https://docs.x.ai/developers/migration/may-15-retirement"
RETIREMENT_DATE = "May 15, 2026"


# Official mapping per xAI migration guide.
# Some entries set ``reasoning_effort`` because non-reasoning variants don't
# have a one-to-one replacement: ``grok-4.3`` reasons by default, so emulating
# ``*-non-reasoning`` behavior on it requires ``reasoning_effort="none"``.
_RETIRED_MODELS: Dict[str, Dict[str, Optional[str]]] = {
    "grok-4":                       {"replacement": "grok-4.3", "reasoning_effort": None,  "note": "ambiguous (reasoning vs non-reasoning) — defaulting to grok-4.3"},
    "grok-4-0709":                  {"replacement": "grok-4.3", "reasoning_effort": None,  "note": None},
    "grok-4-fast":                  {"replacement": "grok-4.3", "reasoning_effort": None,  "note": "ambiguous variant — verify reasoning vs non-reasoning intent"},
    "grok-4-fast-reasoning":        {"replacement": "grok-4.3", "reasoning_effort": None,  "note": None},
    "grok-4-fast-non-reasoning":    {"replacement": "grok-4.3", "reasoning_effort": "none", "note": None},
    "grok-4-1-fast":                {"replacement": "grok-4.3", "reasoning_effort": None,  "note": "ambiguous variant — verify reasoning vs non-reasoning intent"},
    "grok-4-1-fast-reasoning":      {"replacement": "grok-4.3", "reasoning_effort": None,  "note": None},
    "grok-4-1-fast-non-reasoning":  {"replacement": "grok-4.3", "reasoning_effort": "none", "note": None},
    "grok-code-fast-1":             {"replacement": "grok-4.3", "reasoning_effort": None,  "note": None},
    "grok-imagine-image-pro":       {"replacement": "grok-imagine-image-quality", "reasoning_effort": None, "note": None},
}


@dataclass(frozen=True)
class RetirementIssue:
    """A reference to a retired xAI model found in a Hermes config."""

    config_path: str            # e.g. "principal.model" or "auxiliary.vision.model"
    current_model: str          # exact value found in config (preserves casing/prefix)
    replacement: str            # recommended xAI replacement
    reasoning_effort: Optional[str] = None  # set if non-reasoning variant migration
    note: Optional[str] = None  # disambiguation note when applicable


def _normalize(model_id: str) -> str:
    """Strip provider prefix (``x-ai/grok-4`` → ``grok-4``) and lowercase."""
    m = model_id.strip().lower()
    for prefix in ("x-ai/", "xai/"):
        if m.startswith(prefix):
            m = m[len(prefix):]
            break
    return m


def _looks_like_xai(model_id: Optional[str]) -> bool:
    if not isinstance(model_id, str) or not model_id.strip():
        return False
    return _normalize(model_id).startswith("grok-")


def find_retired_xai_refs(config: Dict[str, Any]) -> List[RetirementIssue]:
    """Walk all model slots in a Hermes config and return retirement issues.

    Slots scanned:
      - ``principal.model``
      - ``auxiliary.<any>.model`` (introspective — covers future aux slots)
      - ``delegation.model``
      - ``tts.xai.model``
      - ``plugins.image_gen.xai.model``
    """
    issues: List[RetirementIssue] = []

    def _check(path: str, model: Any) -> None:
        if not _looks_like_xai(model):
            return
        norm = _normalize(model)
        entry = _RETIRED_MODELS.get(norm)
        if entry is None:
            return
        issues.append(RetirementIssue(
            config_path=path,
            current_model=model,
            replacement=entry["replacement"],
            reasoning_effort=entry.get("reasoning_effort"),
            note=entry.get("note"),
        ))

    if not isinstance(config, dict):
        return issues

    principal = config.get("principal")
    if isinstance(principal, dict):
        _check("principal.model", principal.get("model"))

    aux = config.get("auxiliary")
    if isinstance(aux, dict):
        for slot_name, slot_cfg in aux.items():
            if isinstance(slot_cfg, dict):
                _check(f"auxiliary.{slot_name}.model", slot_cfg.get("model"))

    delegation = config.get("delegation")
    if isinstance(delegation, dict):
        _check("delegation.model", delegation.get("model"))

    tts = config.get("tts")
    if isinstance(tts, dict):
        tts_xai = tts.get("xai")
        if isinstance(tts_xai, dict):
            _check("tts.xai.model", tts_xai.get("model"))

    plugins = config.get("plugins")
    if isinstance(plugins, dict):
        image_gen = plugins.get("image_gen")
        if isinstance(image_gen, dict):
            ig_xai = image_gen.get("xai")
            if isinstance(ig_xai, dict):
                _check("plugins.image_gen.xai.model", ig_xai.get("model"))

    return issues


def format_issue(issue: RetirementIssue) -> str:
    """One-line human-readable rendering of a retirement issue."""
    parts = [
        f"{issue.config_path}: {issue.current_model!r} → use {issue.replacement!r}"
    ]
    if issue.reasoning_effort:
        parts.append(f'(set reasoning_effort: "{issue.reasoning_effort}")')
    if issue.note:
        parts.append(f"[note: {issue.note}]")
    return " ".join(parts)
