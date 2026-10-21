"""Model-id matching helpers shared by local metadata probes."""

from __future__ import annotations

from typing import Any


def _model_id_matches(candidate_id: Any, lookup_model: Any) -> bool:
    """Exact match, or ``publisher/slug`` whose slug equals the configured name."""
    if not isinstance(candidate_id, str) or not isinstance(lookup_model, str):
        return False
    return candidate_id == lookup_model or ("/" in candidate_id and candidate_id.rsplit("/", 1)[1] == lookup_model)


def _lmstudio_model_match_rank(model: dict[str, Any], lookup_model: str) -> int:
    """Rank an LM Studio entry for a requested model or loaded-instance id."""
    instances = model.get("loaded_instances", [])
    if not isinstance(instances, list):
        instances = []
    if any(isinstance(instance, dict) and instance.get("id") == lookup_model for instance in instances):
        return 5
    if model.get("key") == lookup_model or model.get("id") == lookup_model:
        return 4
    parent_ids = (model.get("key", ""), model.get("id", ""))
    if any(_model_id_matches(candidate_id, lookup_model) for candidate_id in parent_ids):
        return 3 if instances else 1
    return 0


def _lmstudio_model_matches(model: dict[str, Any], lookup_model: str) -> bool:
    """Match a parent model or one of its loaded-instance aliases."""
    return _lmstudio_model_match_rank(model, lookup_model) > 0
