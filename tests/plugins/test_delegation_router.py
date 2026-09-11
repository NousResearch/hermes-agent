"""Profile-pool delegation — ``plugins/delegation_router``.

Three contracts are pinned here:

* core honours a **plugin-owned** ``delegate_task`` handler and never a built-in one, so the
  inline dispatch site (``AIAgent._dispatch_delegate_task``) can reach a plugin that took the
  tool over with the ``tools.override`` capability;
* the router's resolution order (explicit profile → bucket → project+role → keyword → default)
  picks the profile that owns the work;
* anything unresolvable falls through to stock delegation instead of being swallowed.

The plugin is imported against a temp ``HERMES_HOME`` with fake profiles on disk, so no test
touches the real home.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_INIT = REPO_ROOT / "plugins" / "delegation_router" / "__init__.py"

ROUTING = {
    "version": 1,
    "enabled": True,
    "default_profile": "",
    "max_depth": 1,
    "lease_policy": "ignore",
    "projects": {"acme": {"dev": "acme-dev", "growth": "acme-growth"}},
    "buckets": {
        "coding": {"role": "dev", "candidates": ["acme-dev", "coder"],
                   "keywords": ["refactor", "bug", "api"]},
        "qa": {"candidates": ["qa-absent", "acme-dev"], "keywords": ["regression"]},
        "ops": {"candidates": [], "keywords": ["fleet"]},
    },
}


def _write_home(tmp_path: Path, profiles: list[str], routing: dict | None) -> Path:
    home = tmp_path / ".hermes"
    (home / "profiles").mkdir(parents=True, exist_ok=True)
    for name in profiles:
        prof = home / "profiles" / name
        prof.mkdir(parents=True, exist_ok=True)
        (prof / "config.yaml").write_text("model:\n  default: test-model\n", encoding="utf-8")
    if routing is not None:
        plug = home / "plugins" / "delegation-router"
        plug.mkdir(parents=True, exist_ok=True)
        (plug / "routing.yaml").write_text(yaml.safe_dump(routing), encoding="utf-8")
    return home


@pytest.fixture
def router(tmp_path, monkeypatch):
    """The bundled plugin loaded against a temp home holding acme-dev / coder profiles."""
    home = _write_home(tmp_path, ["acme-dev", "coder"], ROUTING)
    # Profiles are HOME-anchored by design (hermes profile operations see the whole fleet).
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_DELEGATION_ROUTER_DEPTH", raising=False)
    spec = importlib.util.spec_from_file_location("delegation_router_under_test", PLUGIN_INIT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── core hook: who owns delegate_task ───────────────────────────────────────

def _plugin_owned_handler(tmp_path: Path):
    """A handler defined in a plugin module namespace, as a real plugin's handler is.

    ``registry`` decides ownership by the handler's defining module, so the handler has to be
    imported from a ``hermes_plugins.*`` namespace to exercise the real rule."""
    pkg = tmp_path / "hermes_plugins"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "acme_delegation.py").write_text(
        "def handler(args, **kwargs):\n    return 'plugin-result'\n", encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module("hermes_plugins.acme_delegation").handler
    finally:
        sys.path.remove(str(tmp_path))


def test_only_a_plugin_owned_handler_is_offered(tmp_path):
    """A private registry, so the process-wide one is never mutated by a test."""
    from tools.registry import ToolRegistry

    reg = ToolRegistry()
    builtin = lambda args, **kwargs: "builtin"  # noqa: E731 - defined in this test module
    reg.register(name="delegate_task", toolset="delegation", schema={"name": "delegate_task"}, handler=builtin)
    assert reg.plugin_handler("delegate_task") is None

    plugin_handler = _plugin_owned_handler(tmp_path)
    reg.register(name="delegate_task", toolset="delegation", schema={"name": "delegate_task"},
                 handler=plugin_handler, override=True)
    assert reg.plugin_handler("delegate_task") is plugin_handler


def test_inline_dispatch_hands_the_call_to_a_plugin_handler(tmp_path, monkeypatch):
    """The inline dispatch site runs the plugin's handler — no monkey-patching involved."""
    import run_agent
    import tools.registry as registry_module
    from tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register(name="delegate_task", toolset="delegation", schema={"name": "delegate_task"},
                 handler=_plugin_owned_handler(tmp_path), override=True)
    monkeypatch.setattr(registry_module, "registry", reg)

    agent = types.SimpleNamespace(session_id="session-under-test", _delegate_depth=0)
    assert run_agent.AIAgent._dispatch_delegate_task(agent, {"goal": "anything"}) == "plugin-result"


def test_inline_dispatch_stays_builtin_without_a_plugin(monkeypatch):
    """With no plugin owning the tool the lookup is inert, so stock delegation is untouched."""
    import run_agent
    import tools.registry as registry_module
    from tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register(name="delegate_task", toolset="delegation", schema={"name": "delegate_task"},
                 handler=lambda args, **kwargs: "builtin")
    monkeypatch.setattr(registry_module, "registry", reg)
    assert run_agent._plugin_delegate_dispatch_override() is None


# ── resolution order ────────────────────────────────────────────────────────

def test_bucket_picks_the_first_candidate_that_exists(router):
    decision = router.resolve_target({"routing": "coding", "goal": "fix the api"})
    assert decision["action"] == "route"
    assert decision["profile"] == "acme-dev"


def test_bucket_skips_missing_candidates(router):
    decision = router.resolve_target({"routing": "qa", "goal": "regression run"})
    assert (decision["action"], decision["profile"]) == ("route", "acme-dev")


def test_project_plus_role_selects_the_project_profile(router):
    decision = router.resolve_target({"routing": "coding", "project": "acme", "goal": "ship it"})
    assert (decision["action"], decision["profile"], decision["project"]) == ("route", "acme-dev", "acme")


def test_explicit_profile_wins_and_unknown_profile_is_an_error(router):
    routed = router.resolve_target({"profile": "coder", "goal": "anything"})
    assert (routed["action"], routed["profile"]) == ("route", "coder")

    unknown = router.resolve_target({"profile": "nope", "goal": "anything"})
    assert unknown["action"] == "error"
    assert "acme-dev" in unknown["reason"]  # names what IS available


def test_keyword_match_routes_without_an_explicit_bucket(router):
    decision = router.resolve_target({"goal": "please refactor the parser"})
    assert (decision["action"], decision["bucket"]) == ("route", "coding")
    assert "keyword" in decision["reason"]


def test_no_bucket_and_empty_default_passes_through(router):
    decision = router.resolve_target({"goal": "write a haiku about spreadsheets"})
    assert decision["action"] == "passthrough"


def test_unknown_bucket_is_an_error_naming_valid_buckets(router):
    decision = router.resolve_target({"routing": "nonsense", "goal": "x"})
    assert decision["action"] == "error"
    assert "coding" in decision["reason"]


def test_bucket_without_a_live_profile_passes_through(router):
    decision = router.resolve_target({"routing": "ops", "goal": "x"})
    assert decision["action"] == "passthrough"


def test_depth_cap_stops_further_routing(router, monkeypatch):
    monkeypatch.setenv("HERMES_DELEGATION_ROUTER_DEPTH", str(router.load_routing().get("max_depth", 1)))
    decision = router.resolve_target({"routing": "coding", "goal": "fix the api"})
    assert decision["action"] == "passthrough"
    assert "max_depth" in decision["reason"]


def test_deeper_allowance_routes_one_generation_further(router, tmp_path, monkeypatch):
    """max_depth: 2 = a routed profile may route its own work once; its children may not."""
    plug = tmp_path / ".hermes" / "plugins" / "delegation-router"
    (plug / "routing.yaml").write_text(yaml.safe_dump(dict(ROUTING, max_depth=2)), encoding="utf-8")
    monkeypatch.setenv("HERMES_DELEGATION_ROUTER_DEPTH", "1")
    assert router.resolve_target({"routing": "coding", "goal": "fix the api"})["action"] == "route"
    monkeypatch.setenv("HERMES_DELEGATION_ROUTER_DEPTH", "2")
    assert router.resolve_target({"routing": "coding", "goal": "fix the api"})["action"] == "passthrough"


def test_disabled_or_missing_table_passes_through(tmp_path, monkeypatch):
    home = _write_home(tmp_path, ["acme-dev"], None)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    spec = importlib.util.spec_from_file_location("delegation_router_no_table", PLUGIN_INIT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.load_routing() == {}
    assert module.resolve_target({"routing": "coding", "goal": "fix"})["action"] == "passthrough"


def test_disabled_table_keeps_stock_behaviour(router, tmp_path, monkeypatch):
    routing = dict(ROUTING, enabled=False)
    plug = tmp_path / ".hermes" / "plugins" / "delegation-router"
    (plug / "routing.yaml").write_text(yaml.safe_dump(routing), encoding="utf-8")
    assert router.resolve_target({"routing": "coding", "goal": "fix"})["action"] == "passthrough"


# ── model-facing schema ─────────────────────────────────────────────────────

def test_schema_exposes_routing_project_and_profile(router):
    schema = router._build_schema()
    params = schema["parameters"]["properties"]
    assert {"routing", "project", "profile"} <= set(params)
    assert "profile" in params  # an explicit escape hatch for the caller
