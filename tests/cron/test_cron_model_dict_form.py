"""A cron job whose ``model`` is the structured dict form must resolve to its model.

Jobs written by older schemas or edited by hand can store
``model={"model": "<name>", "provider": "<p>"}`` instead of the plain string. The dict is
truthy, so model resolution accepted it, skipped the config fallback, then failed the
fail-fast ``isinstance(model, str)`` guard and raised "has no model configured" while
quoting the model it was given -- every tick.
"""
import pytest

import cron.scheduler as sched
from cron.scheduler import _coerce_job_model

DICT_MODEL = {"model": "gpt-5.6-sol", "provider": "openai-codex"}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    return tmp_path


def _job(model):
    return {"id": "j", "name": "j", "prompt": "x", "model": model}


class TestCoerceJobModel:
    def test_dict_form_yields_the_inner_model_string(self):
        assert _coerce_job_model(DICT_MODEL) == "gpt-5.6-sol"

    def test_plain_string_is_preserved_and_stripped(self):
        assert _coerce_job_model("  claude-opus-5  ") == "claude-opus-5"

    def test_dict_alias_keys(self):
        assert _coerce_job_model({"name": "a"}) == "a"
        assert _coerce_job_model({"default": "b"}) == "b"

    @pytest.mark.parametrize(
        "value", [None, "", "   ", {}, {"provider": "openai-codex"}, {"model": ""}, 123, [], object()])
    def test_unusable_values_yield_none(self, value):
        """None means "no pin here", so the env/config fallback still runs."""
        assert _coerce_job_model(value) is None


class TestLoadCronJobConfig:
    def test_dict_model_resolves_instead_of_raising(self, home):
        cfg = sched._load_cron_job_config(_job(DICT_MODEL), "j", "j")
        assert cfg.model == "gpt-5.6-sol"

    def test_dict_pin_wins_over_config_defaults(self, home):
        (home / "config.yaml").write_text("model:\n  default: main-model\ncron:\n  model: cron-default\n")
        cfg = sched._load_cron_job_config(_job(DICT_MODEL), "j", "j")
        assert cfg.model == "gpt-5.6-sol"

    def test_unusable_dict_falls_back_to_config(self, home):
        (home / "config.yaml").write_text("model:\n  default: main-model\n")
        cfg = sched._load_cron_job_config(_job({"provider": "openai-codex"}), "j", "j")
        assert cfg.model == "main-model"

    def test_no_model_anywhere_still_fails_fast(self, home):
        with pytest.raises(RuntimeError, match="no model configured"):
            sched._load_cron_job_config(_job(None), "j", "j")


def test_preflight_passes_the_flattened_model(home, monkeypatch):
    from cron.scheduler_preflight import _preflight_check_provider_key
    import hermes_cli.runtime_provider as rp

    captured = {}
    monkeypatch.setattr(rp, "resolve_runtime_provider", lambda **kw: captured.update(kw) or {})
    _preflight_check_provider_key(_job(DICT_MODEL), {"cron": {}})
    assert captured["target_model"] == "gpt-5.6-sol"
