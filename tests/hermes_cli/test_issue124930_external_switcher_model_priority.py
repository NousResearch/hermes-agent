"""An external switcher's selected model must beat a stale per-provider catalog model (#124930).

CC Switch (and friends) manage API plans by rewriting the **top-level**
``model`` block of ``config.yaml``::

    model:
      default: doubao-seed-evolving
      provider: huoshan-copy

The same plan also lives under ``custom_providers``, where each entry carries
its own trailing ``model:`` field the switcher does **not** sync. When the
active plan is resolved with no explicit target model,
``_apply_custom_provider_extras`` let that per-provider catalog value win, so
the runtime resolved — and ``/status`` displayed — the first model of the
plan's catalog instead of the model the operator actually selected.

Contract: for the **active** provider (the one named by the top-level
``model.provider``), with no explicit target model, the top-level
``model.default`` wins over the entry's trailing ``model:``. Everything else
keeps its priority, verified here:

- an explicit target model (``/model`` pin, auxiliary slots) still wins
- a provider that is NOT the active one keeps its own configured default
- a custom_provider entry with no trailing ``model:`` is unaffected
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

# The suite's home_io_guard refuses stdlib reads that happen after it is installed
# (the runtime interpreter's zoneinfo → sysconfig walk is the usual victim), so
# pull the heavy imports in at module scope, before any fixture runs.
import agent.moa_loop  # noqa: F401
import tests.conftest  # noqa: F401
import pytest

from hermes_cli import runtime_provider_custom as rpc


def _config(model_default: str = "", model_provider: str = "", custom_providers: Optional[Dict[str, Any]] = None):
    """A config.yaml shape as an external switcher leaves it."""
    cfg: Dict[str, Any] = {}
    if model_default or model_provider:
        cfg["model"] = {}
        if model_default:
            cfg["model"]["default"] = model_default
        if model_provider:
            cfg["model"]["provider"] = model_provider
    if custom_providers:
        cfg["custom_providers"] = custom_providers
    return cfg


class _FakeRuntimeProviders:
    """Minimal stand-in for ``hermes_cli.runtime_provider``."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    def load_config(self) -> Dict[str, Any]:
        return self._config

    def _get_model_config(self) -> Dict[str, Any]:
        model_cfg = self._config.get("model")
        if not isinstance(model_cfg, dict):
            return {}
        return dict(model_cfg)

    def has_usable_secret(self, value) -> bool:
        return bool(str(value or "").strip())

    def _get_named_custom_provider(self, name: str) -> Optional[Dict[str, Any]]:
        entries = self._config.get("custom_providers") or {}
        entry = entries.get(name)
        return dict(entry) if isinstance(entry, dict) else None

    def _resolves_to_custom(self, _name: str) -> bool:
        return False

    def _try_resolve_from_custom_pool(self, *a, **k):
        return None

    def _detect_api_mode_for_url(self, _base_url):
        return None

    def _host_gated_env_key_candidates(self, *_a, **_k):
        return []

    def _runtime(self, provider, api_mode, base_url, api_key, **k):
        return {
            "provider": provider,
            "api_mode": api_mode,
            "base_url": base_url,
            "api_key": api_key,
            **{key: value for key, value in k.items()},
        }

    def _key_env_secret(self, _provider, _label):
        return ""


@pytest.fixture
def resolver(monkeypatch):
    """Load ``runtime_provider_custom`` with a controllable config + fake collaborator."""
    holder = SimpleNamespace(config={})

    def _patch(config: Dict[str, Any]):
        holder.config = config
        monkeypatch.setattr(rpc, "_rp", lambda: _FakeRuntimeProviders(config), raising=False)
        return holder

    monkeypatch.setattr(rpc, "_clean", lambda v: (str(v or "").strip()), raising=False)
    return _patch


def _resolve(resolver, config, provider="huoshan-copy", target_model=None):
    resolver(config)
    return rpc._resolve_named_custom_runtime(
        requested_provider=provider,
        explicit_api_key="k",
        target_model=target_model,
    )


def test_active_provider_uses_the_external_selection_not_the_stale_catalog_model(resolver):
    """The top-level ``model.default`` wins for the active provider.

    The switcher wrote ``doubao-seed-evolving`` at the top level but never
    synced the entry's trailing ``model: glm-5.3`` — the first model of that
    plan's catalog.
    """
    runtime = _resolve(
        resolver,
        _config(
            model_default="doubao-seed-evolving",
            model_provider="huoshan-copy",
            custom_providers={
                "huoshan-copy": {
                    "name": "huoshan-copy",
                    "base_url": "https://example.invalid/v1",
                    "api_key": "k",
                    "model": "glm-5.3",
                }
            },
        ),
    )

    assert runtime is not None
    assert runtime["model"] == "doubao-seed-evolving"


def test_an_inactive_provider_keeps_its_own_configured_model(resolver):
    """Another plan's default is not ours to override."""
    runtime = _resolve(
        resolver,
        _config(
            model_default="doubao-seed-evolving",
            model_provider="huoshan-copy",
            custom_providers={
                "other-plan": {
                    "name": "other-plan",
                    "base_url": "https://example.invalid/v1",
                    "api_key": "k",
                    "model": "glm-5.3",
                }
            },
        ),
        provider="other-plan",
    )

    assert runtime is not None
    assert runtime["model"] == "glm-5.3"


def test_an_explicit_target_model_still_wins(resolver):
    """``/model`` selections and auxiliary slots resolve a concrete model; they
    must not fall back to the top-level default (the existing contract)."""
    runtime = _resolve(
        resolver,
        _config(
            model_default="doubao-seed-evolving",
            model_provider="huoshan-copy",
            custom_providers={
                "huoshan-copy": {
                    "name": "huoshan-copy",
                    "base_url": "https://example.invalid/v1",
                    "api_key": "k",
                    "model": "glm-5.3",
                }
            },
        ),
        target_model="claude-opus-5-5",
    )

    assert runtime is not None
    assert runtime["model"] == "claude-opus-5-5"


def test_a_provider_entry_without_a_trailing_model_is_unaffected(resolver):
    """No stale catalog value to override: the top-level selection still applies."""
    runtime = _resolve(
        resolver,
        _config(
            model_default="doubao-seed-evolving",
            model_provider="huoshan-copy",
            custom_providers={
                "huoshan-copy": {
                    "name": "huoshan-copy",
                    "base_url": "https://example.invalid/v1",
                    "api_key": "k",
                }
            },
        ),
    )

    assert runtime is not None
    assert runtime["model"] == "doubao-seed-evolving"


def test_no_top_level_default_keeps_the_entries_configured_model(resolver):
    """A plan whose operator never set a top-level default: nothing to prefer,
    so the entry's own model stands (previous behaviour)."""
    runtime = _resolve(
        resolver,
        _config(
            custom_providers={
                "huoshan-copy": {
                    "name": "huoshan-copy",
                    "base_url": "https://example.invalid/v1",
                    "api_key": "k",
                    "model": "glm-5.3",
                }
            },
        ),
    )

    assert runtime is not None
    assert runtime["model"] == "glm-5.3"
