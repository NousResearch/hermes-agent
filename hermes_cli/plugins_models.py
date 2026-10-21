"""``ctx.models``: the read-only model catalog a plugin needs for a model picker.

Thin wrappers over the functions behind ``hermes model``, so a plugin stops importing private
``hermes_cli.models`` helpers. Every call runs in the plugin's own profile (its manager's home and
that profile's secrets), never the ambient context: one process serves several profiles, and an
unbound credential read would answer a secondary profile with the launch profile's keys.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _secret_mapping(home: Path) -> dict[str, str]:
    """The profile's credentials as a turn sees them: the launch profile keeps its process env under
    its ``.env``; a served profile gets its own ``.env`` and secret sources only."""
    from agent.secret_scope import _is_process_home, build_profile_secret_scope
    from hermes_cli.env_loader import hydrate_profile_secret_sources
    hydrate_profile_secret_sources(home)  # once per home; a profile may not have been served yet
    if _is_process_home(home):
        from tui_gateway.launch_profile_policy import launch_secret_scope
        return launch_secret_scope(home)
    return build_profile_secret_scope(home)


@contextmanager
def _profile_scope(home: Path) -> Iterator[None]:
    from agent.secret_scope import _is_process_home, reset_secret_scope, set_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    home_token = set_hermes_home_override(str(home))
    try:
        secret_token = set_secret_scope(
            _secret_mapping(home), profile_home=None if _is_process_home(home) else str(home))
        try:
            yield
        finally:
            reset_secret_scope(secret_token)
    finally:
        reset_hermes_home_override(home_token)


class PluginModels:
    """Provider ids, credential readiness, model listings and ``provider:model`` parsing."""

    def __init__(self, home: Path):
        self._home = Path(home)

    def providers(self) -> list[str]:
        """Provider ids the ``hermes model`` picker offers (``list_available_providers``' rows)."""
        from hermes_cli.models_catalog_static import listed_canonical_providers
        with _profile_scope(self._home):
            return [entry.slug for entry in listed_canonical_providers()] + ["custom"]

    def ready(self, provider: str) -> bool:
        """Whether this profile holds usable credentials for *provider* (the picker's check)."""
        from hermes_cli.models import _provider_has_credentials
        with _profile_scope(self._home):
            return _provider_has_credentials(provider)

    def list(self, provider: str) -> list[str]:
        """Model ids *provider* lists, from the picker's disk-cached catalog. May fetch live on a
        cold cache; ``[]`` when the provider cannot list models (custom endpoint, offline)."""
        from hermes_cli.models import cached_provider_model_ids
        with _profile_scope(self._home):
            return list(cached_provider_model_ids(provider))

    def parse(self, spec: str) -> tuple[str, str]:
        """``"provider:model"`` -> ``(canonical provider, model)``. Raises ``ValueError`` for a bare
        model, an empty side or a provider Hermes does not know."""
        from hermes_cli.models import parse_model_input
        spec = (spec or "").strip()
        with _profile_scope(self._home):
            provider, model = parse_model_input(spec, "")
        if not provider or not model:
            raise ValueError(f"{spec!r} is not provider:model with a known provider (run `hermes model`)")
        return provider, model
