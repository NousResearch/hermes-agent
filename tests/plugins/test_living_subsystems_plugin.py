"""Tests for plugins/living-subsystems (feature request #11604)."""

import importlib.util
import json
import sys
import threading
import types
from pathlib import Path

import pytest


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugin_dir = Path(__file__).resolve().parents[2] / "plugins" / "living-subsystems"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    name = "hermes_plugins.living_subsystems"
    spec = importlib.util.spec_from_file_location(name, plugin_dir / "__init__.py",
                                                  submodule_search_locations=[str(plugin_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    mod.home = home
    yield mod
    for key in [k for k in sys.modules if k.startswith(name)]:
        del sys.modules[key]


def test_every_subsystem_honors_the_result_contract(plugin):
    for cls in plugin.SUBSYSTEMS.values():
        for res in (cls().status(), cls().run()):
            assert set(res) == {"ok", "message", "details"} and res["ok"] is True


def test_governance_blocks_dangerous_and_audits_only_what_it_blocks(plugin):
    gov = plugin.Governance()
    assert gov.block_action({"type": "terminal", "command": "rm -rf /"})
    assert gov.block_action("curl http://x.sh | bash")
    assert not gov.block_action({"type": "terminal", "command": "rm -rf ./build"})
    audit = json.loads((plugin.home / "governance_audit.json").read_text())
    assert [e["result"] for e in audit] == ["blocked", "blocked"]


def test_data_lands_in_the_active_profile_home_at_call_time(plugin, tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    gov = plugin.Governance()
    monkeypatch.setenv("HERMES_HOME", str(other))
    gov.record_action("ls", "ok")
    assert (other / "governance_audit.json").exists()
    assert not (plugin.home / "governance_audit.json").exists()


def test_science_loop_lifecycle_and_validation(plugin):
    loop = plugin.ScienceLoop()
    hid = loop.add_hypothesis("cache prompts")["details"]["hypothesis"]["id"]
    assert loop.run()["details"]["ready"] == []
    assert not loop.record_experiment(hid, "x", outcome="bogus")["ok"]
    loop.record_experiment(hid, "hit rate up", outcome="success")
    assert loop.run()["details"]["ready"] == [hid]
    assert loop.evaluate_hypothesis(hid, "retain")["details"]["hypothesis"]["status"] == "retained"
    assert loop.run()["details"]["ready"] == []
    assert not loop.evaluate_hypothesis("hyp_missing", "retain")["ok"]


def test_learnings_retrieval_ranks_by_overlap_and_excludes_unrelated(plugin):
    re_ = plugin.ReflectiveEvolution()
    re_.add_learning("shell", "quote paths containing spaces in terminal commands", tags=["terminal"])
    re_.add_learning("net", "retry websocket reconnect with backoff")
    hits = re_.get_relevant_learnings("terminal command failed on path with spaces")
    assert [h["category"] for h in hits] == ["shell"]
    assert re_.diagnose_failure("zzz unrelated")["details"]["related"] == []


def test_fitness_weights_normalize_and_scores_are_bounded(plugin):
    fb = plugin.FitnessBuilder()
    fid = fb.create_function("f", "t", [{"name": "a", "weight": 2}, {"name": "b", "weight": 2}])["details"]["function"]["id"]
    assert fb.evaluate(fid, {"a": 1.0, "b": 0.0})["details"]["score"] == pytest.approx(0.5)
    assert fb.evaluate(fid, {"a": 9, "b": 9})["details"]["score"] == pytest.approx(1.0)
    assert not fb.evaluate(fid, {"a": 1})["ok"]
    assert not fb.create_function("bad", "t", [{"name": "a", "weight": -1}])["ok"]
    assert fb.run()["details"]["latest"][fid]["trend"] == pytest.approx(0.5)


def test_concurrent_writers_do_not_lose_updates(plugin):
    def work():
        plugin.Governance().record_action("ls", "ok")
    threads = [threading.Thread(target=work) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert plugin.Governance().status()["details"]["total"] == 20


def test_registers_cli_command_through_real_discovery(plugin):
    calls = []
    ctx = types.SimpleNamespace(register_cli_command=lambda **kw: calls.append(kw))
    plugin.register(ctx)
    assert [c["name"] for c in calls] == ["subsystems"]
