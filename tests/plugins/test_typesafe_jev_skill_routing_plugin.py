"""Tests for the typesafe-jev-skill-routing bundled plugin."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "typesafe-jev-skill-routing"


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "skills").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    yield hermes_home


def _write_skill(root: Path, name: str, description: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_dir.joinpath("SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n",
        encoding="utf-8",
    )


def _load_router():
    spec = importlib.util.spec_from_file_location(
        "jev_skill_router_under_test",
        PLUGIN_DIR / "jev_skill_router.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_plugin():
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.typesafe_jev_skill_routing",
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.typesafe_jev_skill_routing"
    mod.__path__ = [str(PLUGIN_DIR)]
    sys.modules["hermes_plugins.typesafe_jev_skill_routing"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestRouterCore:
    def test_gate_mean_inverts_prose_suffices(self):
        router = _load_router()
        answers = {
            "gate::acts_on_user_system": {"noul": 0.8},
            "gate::would_follow_documented_procedure": {"noul": 0.7},
            "gate::prose_suffices": {"noul": 0.9},
        }
        assert router.gate_mean(answers) == pytest.approx((0.8 + 0.7 + 0.1) / 3)

    def test_should_skip_slash_and_existing_tag(self):
        router = _load_router()
        assert router.should_skip_request("/jev status") is True
        assert router.should_skip_request("hello <skill_relevance> x") is True
        assert router.should_skip_request("queue the album on the speaker") is False

    def test_routing_enabled_modes(self):
        router = _load_router()
        assert router.routing_enabled({"mode": "off"}, api_key_present=True) is False
        assert router.routing_enabled({"mode": "on"}, api_key_present=False) is True
        assert router.routing_enabled({"mode": "auto"}, api_key_present=False) is False
        assert router.routing_enabled({"mode": "auto"}, api_key_present=True) is True

    def test_load_roster_from_dirs_dedupes_and_caps(self, _isolate_env):
        router = _load_router()
        skills_root = _isolate_env / "skills"
        _write_skill(skills_root, "alpha", "First skill")
        _write_skill(skills_root, "beta", "Second skill")
        roster = router.load_roster_from_dirs([skills_root])
        names = [s.name for s in roster]
        assert names == ["alpha", "beta"]

    def test_suggest_skill_injects_block_when_gate_passes(self):
        router = _load_router()
        skills = [
            router.Skill("alpha", "Does alpha things"),
            router.Skill("beta", "Does beta things"),
        ]

        captured = {}

        def fake_post(**kwargs):
            captured.update(kwargs)
            return {
                "model": "jev-latest",
                "answers": {
                    "which": {
                        "choice": "alpha",
                        "probabilities": {"alpha": 0.72},
                    },
                    "gate::acts_on_user_system": {"noul": 0.8},
                    "gate::would_follow_documented_procedure": {"noul": 0.7},
                    "gate::prose_suffices": {"noul": 0.9},
                },
            }

        result = router.suggest_skill(
            "run the alpha workflow on my inbox",
            skills,
            api_key="ts_test",
            post_fn=fake_post,
        )
        assert result is not None
        assert result.skill == "alpha"
        assert "alpha" in result.block()
        assert captured["state"]["request"].startswith("run the alpha")
        assert "which" in captured["questions"]

    def test_suggest_skill_silent_when_gate_fails(self):
        router = _load_router()
        skills = [
            router.Skill("alpha", "Does alpha things"),
            router.Skill("beta", "Does beta things"),
        ]

        def fake_post(**kwargs):
            return {
                "answers": {
                    "which": {"choice": "alpha", "probabilities": {"alpha": 0.9}},
                    "gate::acts_on_user_system": {"noul": 0.05},
                    "gate::would_follow_documented_procedure": {"noul": 0.05},
                    "gate::prose_suffices": {"noul": 0.95},
                },
            }

        assert router.suggest_skill("hello", skills, api_key="ts_test", post_fn=fake_post) is None

    def test_systemone_url_normalizes_base(self):
        router = _load_router()
        assert router.systemone_url("https://api.typesafe.ai/v1") == "https://api.typesafe.ai/v1/systemone"
        assert router.systemone_url("https://api.typesafe.ai") == "https://api.typesafe.ai/v1/systemone"


class TestPluginRegistration:
    def test_register_wires_hook_and_commands(self):
        plugin = _load_plugin()
        ctx = MagicMock()
        plugin.register(ctx)
        ctx.register_hook.assert_called_once()
        assert ctx.register_hook.call_args.args[0] == "pre_llm_call"
        ctx.register_command.assert_called_once()
        assert ctx.register_command.call_args.kwargs["handler"] is not None
        ctx.register_cli_command.assert_called_once()
        assert ctx.register_cli_command.call_args.args[0] == "jev"

    def test_pre_llm_call_returns_context_dict(self, _isolate_env, monkeypatch):
        plugin = _load_plugin()
        router = plugin.router
        skills_root = _isolate_env / "skills"
        _write_skill(skills_root, "alpha", "Alpha skill")
        _write_skill(skills_root, "beta", "Beta skill")

        monkeypatch.setenv("TYPESAFE_API_KEY", "ts_test")
        monkeypatch.setattr(
            plugin.router,
            "suggest_skill",
            lambda *a, **k: router.SuggestResult("alpha", 0.5, 0.8, 0.1, "jev-latest"),
        )

        ctx = MagicMock()
        ctx.get_config.side_effect = lambda key, default=None: {
            "mode": "on",
            "gate": 0.3,
            "timeout_sec": 2.5,
            "model": "jev-latest",
            "base_url": "",
            "suggest_chars": 4000,
        }.get(key, default)

        plugin.register(ctx)
        hook = ctx.register_hook.call_args.args[1]
        payload = hook(user_message="please run alpha on my files", platform="cli")
        assert isinstance(payload, dict)
        assert "<skill_relevance>" in payload["context"]
        assert "alpha" in payload["context"]

    def test_slash_status_without_key(self, _isolate_env):
        plugin = _load_plugin()
        text = plugin._handle_slash("")
        assert "TypeSafe Jev skill routing" in text
        assert "MISSING" in text

