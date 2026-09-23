"""StepFun provider wiring: four ids, one per (region x endpoint family).

StepFun accounts are regional — a key issued on api.stepfun.ai returns 401 on api.stepfun.com —
so region is a provider id with its own key env var, never a base-url toggle over a shared key.
"""

from __future__ import annotations

import sys
import types

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv


# (id, base url, key env var, base-url override var, models.dev catalog)
STEPFUN_IDS = (
    ("stepfun", "https://api.stepfun.ai/v1", "STEPFUN_API_KEY", "STEPFUN_BASE_URL", "stepfun-ai"),
    ("stepfun-cn", "https://api.stepfun.com/v1", "STEPFUN_CN_API_KEY", "STEPFUN_CN_BASE_URL", "stepfun"),
    ("stepfun-plan", "https://api.stepfun.ai/step_plan/v1", "STEPFUN_API_KEY",
     "STEPFUN_STEP_PLAN_BASE_URL", "stepfun-ai-step-plan"),
    ("stepfun-plan-cn", "https://api.stepfun.com/step_plan/v1", "STEPFUN_CN_API_KEY",
     "STEPFUN_CN_STEP_PLAN_BASE_URL", "stepfun-step-plan"),
)


class TestStepfunRegistry:
    def test_every_id_registered_with_its_own_endpoint_and_key(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        for pid, base_url, key_var, base_var, _mdev in STEPFUN_IDS:
            entry = PROVIDER_REGISTRY[pid]
            assert entry.inference_base_url == base_url
            assert entry.api_key_env_vars == (key_var,)
            assert entry.base_url_env_var == base_var

    def test_china_ids_never_read_the_international_key(self):
        """The whole point of the split: a China id must not fall back to STEPFUN_API_KEY."""
        from hermes_cli.auth import PROVIDER_REGISTRY

        for pid in ("stepfun-cn", "stepfun-plan-cn"):
            assert "STEPFUN_API_KEY" not in PROVIDER_REGISTRY[pid].api_key_env_vars

    def test_overlays_match_the_registry(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        for pid, base_url, key_var, base_var, _mdev in STEPFUN_IDS:
            overlay = HERMES_OVERLAYS[pid]
            assert overlay.transport == "openai_chat"
            assert overlay.base_url_override == base_url
            assert overlay.base_url_env_var == base_var
            assert overlay.extra_env_vars == (key_var,)

    def test_profiles_registered(self):
        from providers import get_provider_profile

        for pid, base_url, key_var, _base_var, _mdev in STEPFUN_IDS:
            profile = get_provider_profile(pid)
            assert profile is not None
            assert profile.base_url == base_url
            assert profile.env_vars == (key_var,)
            # Vision is a per-model fact (step-3.5-flash is text-only, step-3.7-flash is not).
            assert profile.supports_vision is False

    def test_aliases(self):
        from hermes_cli.auth import resolve_provider
        from hermes_cli.providers import normalize_provider

        expected = {
            "step": "stepfun-plan", "stepfun-coding-plan": "stepfun-plan",
            "stepfun-step-plan": "stepfun-plan", "stepfun-ai": "stepfun",
            "stepfun-china": "stepfun-cn", "stepfun_cn": "stepfun-cn",
            "stepfun-step-plan-cn": "stepfun-plan-cn", "stepfun-coding-plan-cn": "stepfun-plan-cn",
        }
        for alias, canonical in expected.items():
            assert normalize_provider(alias) == canonical, alias
            assert resolve_provider(alias) == canonical, alias


class TestStepfunCatalogs:
    def test_each_id_maps_to_its_own_models_dev_catalog(self):
        """Collapsing the ids would lose step-router-v1, which only the China plan catalog has."""
        from agent.models_dev import PROVIDER_TO_MODELS_DEV

        mapped = {pid: PROVIDER_TO_MODELS_DEV[pid] for pid, _u, _k, _b, _m in STEPFUN_IDS}
        assert mapped == {pid: mdev for pid, _u, _k, _b, mdev in STEPFUN_IDS}
        assert len(set(mapped.values())) == 4

    def test_default_model_is_step_5_preview_everywhere(self):
        """First entry of _PROVIDER_MODELS is the non-interactive default, so order is contract."""
        from hermes_cli.models import _PROVIDER_MODELS, get_default_model_for_provider

        for pid, _u, _k, _b, _m in STEPFUN_IDS:
            assert _PROVIDER_MODELS[pid][0] == "step-5-preview"
            assert get_default_model_for_provider(pid) == "step-5-preview"

    def test_router_is_china_plan_only(self):
        """Live /models: step-router-v1 is absent from both api.stepfun.ai catalogs."""
        from hermes_cli.models import _PROVIDER_MODELS

        assert "step-router-v1" in _PROVIDER_MODELS["stepfun-plan-cn"]
        for pid in ("stepfun", "stepfun-plan"):
            assert "step-router-v1" not in _PROVIDER_MODELS[pid]

    def test_aux_fallback_is_a_flash_model(self):
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS_FALLBACK as AUX

        for pid, _u, _k, _b, _m in STEPFUN_IDS:
            assert AUX[pid] == "step-3.7-flash"


class TestStepfunBuiltinMetadata:
    def test_step_5_preview_described_on_every_id(self):
        """models.dev stops at the step-3.x family; without the builtin every id would fall back
        to _UNKNOWN_MODEL_BASE (200K context, 8192 output, no vision)."""
        from unittest.mock import patch

        from agent.models_dev import get_model_capabilities

        with patch("agent.models_dev.fetch_models_dev", return_value={}):
            caps = {pid: get_model_capabilities(pid, "step-5-preview")
                    for pid, _u, _k, _b, _m in STEPFUN_IDS}

        for pid, value in caps.items():
            assert value is not None, pid
            assert value.context_window == 1_024_000, pid
            # An absent limit.output would clamp generations to the 8192 default.
            assert value.max_output_tokens == 1_024_000, pid
            assert value.supports_vision is True, pid
            assert value.supports_tools is True, pid
            assert value.supports_reasoning is True, pid

    def test_router_builtin_is_china_only(self):
        from unittest.mock import patch

        from agent.models_dev import get_model_capabilities

        with patch("agent.models_dev.fetch_models_dev", return_value={}):
            cn = get_model_capabilities("stepfun-plan-cn", "step-router-v1")
            cn_std = get_model_capabilities("stepfun-cn", "step-router-v1")
            intl = get_model_capabilities("stepfun-plan", "step-router-v1")

        assert cn is not None and cn.context_window == 262_144
        assert cn.supports_vision is False
        assert cn_std is not None
        assert intl is None

    def test_family_context_fallback(self):
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS

        assert DEFAULT_CONTEXT_LENGTHS["step-5-preview"] == 1_024_000


class TestStepfunMigration:
    """v46 → v47: the single `stepfun` id becomes four, and a China key moves var."""

    def _run(self, tmp_path, monkeypatch, model_cfg, env_seed=None):
        """Run migrate_config against an isolated temp hermes home.

        ``env_seed`` maps env-var names to values written into the temp ``.env`` BEFORE the
        migration runs, so the migration's get_env_value/save_env_value helpers read and write
        the temp home and never the developer's real ~/.hermes.
        """
        import yaml

        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            yaml.safe_dump({"_config_version": 46, "model": model_cfg}), encoding="utf-8")
        (home / ".env").write_text(
            "\n".join(f"{k}={v}" for k, v in (env_seed or {}).items()) + "\n", encoding="utf-8")

        from hermes_cli import config as config_mod

        # migrate_config resolves paths via hermes_cli.config.get_hermes_home (re-exported at
        # import), NOT the hermes_constants original — patching this symbol isolates reads AND
        # writes, env file included.
        monkeypatch.setattr(config_mod, "get_hermes_home", lambda: home)
        monkeypatch.setattr(config_mod, "_RAW_CONFIG_CACHE", {})
        config_mod.invalidate_env_cache()
        for var in ("STEPFUN_BASE_URL", "STEPFUN_STEP_PLAN_BASE_URL", "STEPFUN_CN_BASE_URL",
                    "STEPFUN_CN_STEP_PLAN_BASE_URL", "STEPFUN_API_KEY", "STEPFUN_CN_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        config_mod.migrate_config(interactive=False, quiet=True)
        out = yaml.safe_load((home / "config.yaml").read_text()) or {}
        out["_env"] = dict(
            line.split("=", 1)
            for line in (home / ".env").read_text(encoding="utf-8").splitlines() if "=" in line)
        return out

    def test_unset_endpoint_keeps_the_pre_split_default(self, tmp_path, monkeypatch):
        """Pre-split bare `stepfun` WAS international Step Plan, so that is where it must land."""
        out = self._run(tmp_path, monkeypatch, {"provider": "stepfun", "default": "step-3.5-flash"})
        assert out["model"]["provider"] == "stepfun-plan"

    def test_config_base_url_picks_the_id(self, tmp_path, monkeypatch):
        cases = {
            "https://api.stepfun.ai/step_plan/v1": "stepfun-plan",
            "https://api.stepfun.com/step_plan/v1": "stepfun-plan-cn",
            "https://api.stepfun.ai/v1": "stepfun",
            "https://api.stepfun.com/v1": "stepfun-cn",
        }
        for base_url, expected in cases.items():
            out = self._run(
                tmp_path / expected, monkeypatch,
                {"provider": "stepfun", "base_url": base_url, "default": "step-3.5-flash"})
            assert out["model"]["provider"] == expected, base_url

    def test_env_override_moves_to_the_new_ids_variable(self, tmp_path, monkeypatch):
        plan_url = "https://api.stepfun.ai/step_plan/v1"
        out = self._run(
            tmp_path, monkeypatch, {"provider": "stepfun", "default": "step-3.5-flash"},
            env_seed={"STEPFUN_BASE_URL": plan_url, "STEPFUN_API_KEY": "intl-key"})
        assert out["model"]["provider"] == "stepfun-plan"
        assert out["_env"].get("STEPFUN_STEP_PLAN_BASE_URL") == plan_url
        assert not out["_env"].get("STEPFUN_BASE_URL")
        # An international config keeps its key where it is.
        assert out["_env"].get("STEPFUN_API_KEY") == "intl-key"

    def test_china_key_moves_to_the_china_variable(self, tmp_path, monkeypatch):
        """Without this the upgrade silently breaks auth: the new id reads STEPFUN_CN_API_KEY,
        but the China user's key is sitting in STEPFUN_API_KEY."""
        cn_url = "https://api.stepfun.com/step_plan/v1"
        out = self._run(
            tmp_path, monkeypatch,
            {"provider": "stepfun", "base_url": cn_url, "default": "step-3.5-flash"},
            env_seed={"STEPFUN_API_KEY": "china-key"})
        assert out["model"]["provider"] == "stepfun-plan-cn"
        assert out["_env"].get("STEPFUN_CN_API_KEY") == "china-key"
        assert not out["_env"].get("STEPFUN_API_KEY")

    def test_non_stepfun_config_untouched(self, tmp_path, monkeypatch):
        out = self._run(
            tmp_path, monkeypatch, {"provider": "openrouter", "default": "z-ai/glm-5.2"})
        assert out["model"]["provider"] == "openrouter"
