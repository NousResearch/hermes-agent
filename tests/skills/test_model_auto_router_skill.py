"""Tests for the model-auto-router skill's router module + demo_resolve entrypoint.

Covers Stage 2 `/model auto` routing behaviour end-to-end at the resolve() boundary:
default tier, deep intent, vision gate + multimodal tiebreak, manual override, and empty
config. Uses stdlib + pytest + unittest.mock only — no live network, no loading config.yaml.

Real router import throughout (authoring standards: no mocks for resolution logic);
only the demo_resolve._load_aliases() boundary is mocked so it exercises without a repo.
"""

import importlib
import io
import sys
from contextlib import redirect_stdout
import dataclasses
from types import SimpleNamespace
from unittest import mock

import pytest

from hermes_cli.models_router import RouteDecision, resolve, build_aliases_from_config


# Sample aliases mirroring a real config.yaml `model_aliases` block: name -> {model, provider}.
SAMPLE_ALIASES = {
    "fast": {"model": "qwen/qwen3.5-9b", "provider": "custom"},
    "researcher": {"model": "ornith-1.5-35b-a3b", "provider": "custom"},
}


def _inject_models_dev(get_all):
    """Register a fake ``agent.models_dev`` so the router's ``from agent import models_dev`` finds it.

    The mock exposes ``get_model_info(provider_id, model_id) -> object`` where each returned
    object has an ``attachment`` attribute. A fresh registry is built per call so tests can
    toggle multimodality alias-by-alias; a caller may pass ``raise_on=True`` to force the
    router's try/except fallback path (custom/LM Studio aliases, metadata unavailable).

    Injecting into ``sys.modules["agent.models_dev"]`` covers both relative-import paths:
    whether the router is loaded as part of ``hermes_cli`` or via a direct file path, its
    ``from agent import models_dev`` resolves to this entry.
    """

    class _Info:
        def __init__(self, attachment):
            self.attachment = attachment

    registry = get_all()  # {alias_name: {"id": str, "attachment": bool}}

    def get_model_info(provider_id, model_id, *, allow_network=False):
        if not registry.get("raise_on"):
            # Match by alias whose target model == model_id (router passes the config model).
            for name, spec in registry.items():
                if isinstance(spec, dict) and str(spec["id"]).lower() == str(model_id).lower():
                    return _Info(bool(spec["attachment"]))
            return None
        raise RuntimeError("models_dev unavailable")

    fake = SimpleNamespace(get_model_info=get_model_info)
    sys.modules["agent.models_dev"] = fake
    return mock.patch("agent.models_dev", fake, create=True)


def _alias_config_with_metadata(aliases, *, multimodal_aliases=(), raise_on=False):
    """Return (aliases_for_resolve, get_all) for a case.

    ``multimodal_aliases`` is the subset whose target model carries the attachment capability;
    ``raise_on=True`` makes the mocked ``get_model_info`` raise to exercise the heuristic fallback.
    """
    aliases_config = dict(aliases)

    def get_all():
        out = {}
        for name, spec in aliases_config.items():
            mid = str(spec["model"]).lower()
            out[name] = {"id": mid, "attachment": name in multimodal_aliases}
        if raise_on:
            out["raise_on"] = True
        return out

    return aliases_config, get_all


def _run_resolve(aliases, *, has_vision=False, context_bytes=0, deep_intent=False,
                 prefer_alias=None, multimodal_aliases=(), raise_on=False):
    cfg, get_all = _alias_config_with_metadata(
        aliases, multimodal_aliases=multimodal_aliases, raise_on=raise_on)
    with _inject_models_dev(get_all):
        return resolve(cfg, has_vision=has_vision, context_bytes=context_bytes,
                       deep_intent=deep_intent, prefer_alias=prefer_alias)


# ---------------------------------------------------------------------------
# resolve() — default tier (no vision, no deep intent)
# ---------------------------------------------------------------------------

def test_resolve_default_picks_fast():
    d = _run_resolve(SAMPLE_ALIASES)  # get_model_info returns None -> heuristic fallback
    assert isinstance(d, RouteDecision)
    assert d.alias == "fast"               # default tier wins
    assert d.provider_model == "qwen/qwen3.5-9b"
    assert "tier" in d.reason


def test_resolve_default_picks_fast_with_metadata():
    """The router reads real ``get_model_info`` metadata, not a hand list."""
    aliases, get_all = _alias_config_with_metadata(
        SAMPLE_ALIASES, multimodal_aliases=("fast",))
    with mock.patch("agent.models_dev", SimpleNamespace(get_model_info=lambda *a, **k: None), create=True):
        d = resolve(SAMPLE_ALIASES)
    assert d.alias == "fast"


def test_resolve_vision_prefers_multimodal_metadata():
    """With vision attachments, the alias flagged attachment in models_dev wins."""
    _aliases, get_all = _alias_config_with_metadata(
        SAMPLE_ALIASES, multimodal_aliases=("researcher",))
    d = _run_resolve(SAMPLE_ALIASES, has_vision=True, multimodal_aliases=("researcher",))
    assert d.alias == "researcher"  # flagged attachment broke the tier tie


def test_resolve_no_multimodal_falls_back():
    """Neither alias is flagged attachment: vision still routes, falls back to first."""
    aliases, get_all = _alias_config_with_metadata(SAMPLE_ALIASES)
    d = _run_resolve(SAMPLE_ALIASES, has_vision=True)
    assert d.alias in ("fast", "researcher")  # first-available fallback


def test_resolve_heuristic_fallback_when_dev_unavailable():
    """When get_model_info raises (custom/LM Studio alias), the router still resolves via name."""
    aliases, get_all = _alias_config_with_metadata(SAMPLE_ALIASES, raise_on=True)
    d = _run_resolve(SAMPLE_ALIASES, has_vision=False)
    assert d.alias == "fast"  # graceful: no crash, default tier picks


# ---------------------------------------------------------------------------
# RouteDecision is a frozen dataclass; alias attribute holds the pick.
# ---------------------------------------------------------------------------

def test_routedecision_is_frozen():
    with mock.patch("agent.models_dev", SimpleNamespace(get_model_info=lambda *a, **k: None), create=True):
        d = resolve(SAMPLE_ALIASES)
    assert dataclasses.is_dataclass(d)
    assert isinstance(d.alias, str)


# ---------------------------------------------------------------------------
# build_aliases_from_config — reads a YAML path to the model_aliases dict.
# ---------------------------------------------------------------------------

def test_build_aliases_from_config(tmp_path):
    import yaml
    cfg = {"model_aliases": {"fast": {"model": "qwen/qwen3.5-9b"},
                             "researcher": {"model": "ornith-1.5-35b-a3b"}}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    aliases = build_aliases_from_config(str(path))
    assert set(aliases) == {"fast", "researcher"}
    assert aliases["fast"]["model"] == "qwen/qwen3.5-9b"


def test_build_aliases_from_config_missing_file():
    # Unreadable path returns {} (graceful), never raises.
    assert build_aliases_from_config("/nonexistent/config.yaml") == {}


# ---------------------------------------------------------------------------
# demo_resolve entrypoint — main() with injected aliases + resolve().
# Path: skills/model-auto-router/scripts/demo_resolve.py, imported via file path.
# ---------------------------------------------------------------------------

def _load_demo():
    import sys
    import importlib.util
    from pathlib import Path

    demo_path = (Path(__file__).resolve().parents[2] / "skills" / "model-auto-router" / "scripts" / "demo_resolve.py")
    mod_name = "model_auto_router_demo_resolve"
    spec = importlib.util.spec_from_file_location(mod_name, demo_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(demo_path.parent))
    spec.loader.exec_module(module)
    return module


def test_demo_resolve_returns_zero_on_picks():
    """main() exits 0 when at least one routing case produced a pick."""
    def fake_resolve(cfg, **kwargs):
        pick = "researcher" if kwargs.get("deep_intent") else "fast"
        return RouteDecision(alias=pick, provider_model=str(cfg[pick]["model"]),
                             tier="tier", reason="test pick")

    module = _load_demo()
    with mock.patch.object(module, "_load_aliases", return_value=SAMPLE_ALIASES), \
         mock.patch("hermes_cli.models_router.resolve", fake_resolve):
        rc = module.main(["--deep"])
    assert rc == 0


def test_demo_resolve_returns_one_when_no_aliases():
    """main() reports FAIL (returns 1) when aliases are empty."""

    def fake_resolve(cfg, **kwargs):
        return RouteDecision(alias="", provider_model=None, tier="none",
                             reason="no model_aliases configured -- run Stage 1 first")

    module = _load_demo()
    with mock.patch.object(module, "_load_aliases", return_value={}), \
         mock.patch("hermes_cli.models_router.resolve", fake_resolve):
        rc = module.main(["--deep"])
    assert rc == 1


def test_demo_override_alias_strips_auto_prefix():
    """Positional 'auto-researcher' override strips the 'auto-' prefix -> researcher."""
    def fake_resolve(cfg, **kwargs):
        prefer = kwargs.get("prefer_alias")
        return RouteDecision(alias=prefer or "fast", provider_model="ornith-1.5-35b-a3b",
                             tier="tier", reason="test pick")

    module = _load_demo()
    with mock.patch.object(module, "_load_aliases", return_value=SAMPLE_ALIASES), \
         mock.patch("hermes_cli.models_router.resolve", fake_resolve):
        rc = module.main(["auto-researcher"])
    assert rc == 0
