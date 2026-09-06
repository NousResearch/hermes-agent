"""Provider-owned auxiliary targets for virtual and external-process runtimes."""
from typing import Any


def profile_auxiliary_target(
    provider: str, runtime: dict[str, Any], task: str | None,
) -> tuple[str, str, str, Any, str] | None:
    from providers import get_provider_profile

    profile = get_provider_profile(provider)
    resolve = getattr(profile, 'resolve_auxiliary_runtime', None)
    if resolve is None:
        return None
    target = resolve(main_runtime=dict(runtime), task=task)
    if target is None:
        return None
    if not isinstance(target, dict):
        raise ValueError('Provider auxiliary runtime must be a mapping or None')
    destination = target.get('provider')
    model = target.get('model')
    if not isinstance(destination, str) or not destination.strip() or not isinstance(model, str) or not model.strip():
        raise ValueError('Provider auxiliary runtime requires a provider and model')
    return (
        destination.strip(), model.strip(), str(target.get('base_url') or ''),
        target.get('api_key') or '', str(target.get('api_mode') or ''),
    )
