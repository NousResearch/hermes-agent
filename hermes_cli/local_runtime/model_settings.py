"""Persistent launch settings owned by the shared model directory's config."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from hermes_cli.local_runtime.estimator import ModelProfile
from hermes_cli.local_runtime.gguf import model_id_from_stem


class InvalidModelSettings(ValueError):
    """An explicit launch policy cannot be honored; stale or stock fitting is unsafe."""


@dataclass(frozen=True)
class ModelSettings:
    context_cap: int | None = None
    backend_defaults: bool = False

    def apply(self, profile: ModelProfile) -> ModelProfile:
        if self.context_cap is None:
            return profile
        return replace(profile, n_ctx_train=min(profile.n_ctx_train or self.context_cap,
                                                self.context_cap))


def settings_for_model(gguf: Path) -> ModelSettings:
    """Bind settings to the machine asset, never the current conversation profile.

    Models live at <shared root>/models; boot, idle refits and growth must all read the same
    config even when a named profile or an unbound background thread invokes them.
    """
    from hermes_cli.config_effective import load_user_config_effective

    config_path = gguf.parent.parent / "config.yaml"
    try:
        config = load_user_config_effective(config_path, fail_closed=True)
    except Exception as exc:
        raise InvalidModelSettings(f"cannot read local model policy from {config_path}") from exc
    section = config.get("local_runtime") or {}
    if not isinstance(section, dict):
        raise InvalidModelSettings("local_runtime must be a mapping")
    settings = section.get("model_settings", {})
    if not isinstance(settings, dict):
        raise InvalidModelSettings("local_runtime.model_settings must be a mapping")
    model_id = model_id_from_stem(gguf.stem)
    raw = settings.get(model_id, {})
    if not isinstance(raw, dict):
        raise InvalidModelSettings(f"local_runtime.model_settings[{model_id!r}] must be a mapping")
    unknown = raw.keys() - {"context_cap", "batch_posture"}
    if unknown:
        raise InvalidModelSettings(f"{model_id}: unknown model settings {sorted(map(str, unknown))}")
    cap = raw.get("context_cap")
    if "context_cap" in raw and (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0):
        raise InvalidModelSettings(f"{model_id}: context_cap must be a positive integer")
    batch = raw.get("batch_posture", "auto")
    if batch not in ("auto", "backend-default"):
        raise InvalidModelSettings(f"{model_id}: batch_posture must be auto or backend-default")
    return ModelSettings(cap, batch == "backend-default")
