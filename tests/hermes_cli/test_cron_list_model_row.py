"""`hermes cron list` shows which model each agent-backed job runs on, and its fallback.

An unpinned job follows ``cron.model`` or the main model at fire time, so switching the main
model silently re-routes it, and it inherits the global fallback chain. A pinned job keeps its own
route and never borrows that chain. The listing must say which case applies. Model and provider
names below are placeholders: the behaviour does not depend on any vendor.
"""

import json
from unittest.mock import patch

import pytest

from cron.jobs import create_job
from cron.scheduler import _CronJobConfig, _resolve_job_runtime
from hermes_cli.auth import AuthError
from hermes_cli.cron import cron_list
from hermes_constants import get_hermes_home


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("hermes_cli.cron._warn_if_gateway_not_running", lambda: None)
    return tmp_path


def _write_config(cfg: dict) -> None:
    (get_hermes_home() / "config.yaml").write_text(json.dumps(cfg), encoding="utf-8")  # JSON is valid YAML


def _row(capsys, label: str) -> str:
    out = capsys.readouterr().out
    lines = [ln.strip() for ln in out.splitlines() if ln.strip().startswith(f"{label}:")]
    assert len(lines) == 1, out
    return lines[0]


def test_pinned_job_shows_its_own_model_and_provider(tmp_cron_dir, capsys):
    _write_config({"model": {"default": "main-model", "provider": "main-prov"}})
    create_job(prompt="p", schedule="0 9 * * *", model="pin-model", provider="pin-prov")
    cron_list()
    line = _row(capsys, "Model")
    assert "pin-model (pin-prov)" in line
    assert "[pinned]" in line
    assert "main-model" not in line


def test_unpinned_job_names_the_main_model_it_follows(tmp_cron_dir, capsys):
    _write_config({"model": {"default": "main-model", "provider": "main-prov"}})
    create_job(prompt="p", schedule="0 9 * * *")
    cron_list()
    line = _row(capsys, "Model")
    assert "main-model (main-prov)" in line
    assert "follows main model" in line


def test_main_model_switch_changes_the_row(tmp_cron_dir, capsys):
    create_job(prompt="p", schedule="0 9 * * *")
    _write_config({"model": {"default": "model-a", "provider": "prov-a"}})
    cron_list()
    assert "model-a (prov-a)" in _row(capsys, "Model")
    _write_config({"model": {"default": "model-b", "provider": "prov-b"}})
    cron_list()
    assert "model-b (prov-b)" in _row(capsys, "Model")


def test_cron_model_default_beats_main_model(tmp_cron_dir, capsys):
    _write_config({
        "model": {"default": "main-model", "provider": "main-prov"},
        "cron": {"model": "cron-model", "model_provider": "cron-prov"},
    })
    create_job(prompt="p", schedule="0 9 * * *")
    cron_list()
    line = _row(capsys, "Model")
    assert "cron-model (cron-prov)" in line
    assert "follows cron.model" in line


def test_cron_model_without_provider_uses_the_main_provider(tmp_cron_dir, capsys):
    _write_config({
        "model": {"default": "main-model", "provider": "main-prov"},
        "cron": {"model": "cron-model"},
    })
    create_job(prompt="p", schedule="0 9 * * *")
    cron_list()
    assert "cron-model (main-prov)" in _row(capsys, "Model")


def test_main_model_shorthand_string_is_understood(tmp_cron_dir, capsys):
    _write_config({"model": "main-model"})
    create_job(prompt="p", schedule="0 9 * * *")
    cron_list()
    assert "main-model" in _row(capsys, "Model")


def test_unpinned_job_lists_the_fallback_chain_in_order(tmp_cron_dir, capsys):
    _write_config({
        "model": {"default": "main-model", "provider": "main-prov"},
        "fallback_providers": [
            {"provider": "fb-prov-1", "model": "fb-model-1"},
            {"provider": "fb-prov-2", "model": "fb-model-2"},
        ],
    })
    create_job(prompt="p", schedule="0 9 * * *")
    cron_list()
    assert _row(capsys, "Fallback") == "Fallback:  fb-model-1 (fb-prov-1) → fb-model-2 (fb-prov-2)"


def test_legacy_fallback_model_key_is_listed(tmp_cron_dir, capsys):
    _write_config({
        "model": {"default": "main-model", "provider": "main-prov"},
        "fallback_model": {"provider": "old-prov", "model": "old-model"},
    })
    create_job(prompt="p", schedule="0 9 * * *")
    cron_list()
    assert "old-model (old-prov)" in _row(capsys, "Fallback")


def test_unpinned_job_without_a_chain_says_none_configured(tmp_cron_dir, capsys):
    _write_config({"model": {"default": "main-model", "provider": "main-prov"}})
    create_job(prompt="p", schedule="0 9 * * *")
    cron_list()
    assert "none configured" in _row(capsys, "Fallback")


def test_pinned_job_does_not_borrow_the_global_chain(tmp_cron_dir, capsys):
    _write_config({
        "model": {"default": "main-model", "provider": "main-prov"},
        "fallback_providers": [{"provider": "fb-prov", "model": "fb-model"}],
    })
    create_job(prompt="p", schedule="0 9 * * *", model="pin-model", provider="pin-prov")
    cron_list()
    line = _row(capsys, "Fallback")
    assert "fb-model" not in line
    assert "pinned job does not use the fallback chain" in line


def test_no_agent_job_has_no_model_or_fallback_row(tmp_cron_dir, capsys):
    _write_config({
        "model": {"default": "main-model", "provider": "main-prov"},
        "fallback_providers": [{"provider": "fb-prov", "model": "fb-model"}],
    })
    script = get_hermes_home() / "scripts" / "tick.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('x')\n", encoding="utf-8")
    create_job(prompt="", schedule="0 9 * * *", script="tick.py", no_agent=True)
    cron_list()
    out = capsys.readouterr().out
    assert "Model:" not in out
    assert "Fallback:" not in out


def test_provider_only_pin_runs_the_cron_default_model(tmp_cron_dir, capsys):
    _write_config({
        "model": {"default": "main-model", "provider": "main-prov"},
        "cron": {"model": "cron-model"},
    })
    create_job(prompt="p", schedule="0 9 * * *", provider="pin-prov")
    cron_list()
    line = _row(capsys, "Model")
    assert "cron-model (pin-prov)" in line
    assert "[pinned]" in line


def test_provider_only_pin_without_cron_model_runs_the_main_model(tmp_cron_dir, capsys):
    _write_config({"model": {"default": "main-model", "provider": "main-prov"}})
    create_job(prompt="p", schedule="0 9 * * *", provider="pin-prov")
    cron_list()
    assert "main-model (pin-prov)" in _row(capsys, "Model")


def test_model_pin_without_provider_uses_the_cron_default_provider(tmp_cron_dir, capsys):
    _write_config({
        "model": {"default": "main-model", "provider": "main-prov"},
        "cron": {"model": "cron-model", "model_provider": "cron-prov"},
    })
    create_job(prompt="p", schedule="0 9 * * *", model="pin-model")
    cron_list()
    assert "pin-model (cron-prov)" in _row(capsys, "Model")


def test_cron_provider_alone_applies_to_the_main_model(tmp_cron_dir, capsys):
    _write_config({
        "model": {"default": "main-model", "provider": "main-prov"},
        "cron": {"model_provider": "cron-prov"},
    })
    create_job(prompt="p", schedule="0 9 * * *")
    cron_list()
    line = _row(capsys, "Model")
    assert "main-model (cron-prov)" in line
    assert "follows main model" in line


def test_env_model_is_shown_when_the_config_names_no_model(tmp_cron_dir, capsys, monkeypatch):
    _write_config({})
    monkeypatch.setenv("HERMES_MODEL", "env-model")
    create_job(prompt="p", schedule="0 9 * * *")
    cron_list()
    line = _row(capsys, "Model")
    assert "env-model" in line
    assert "follows HERMES_MODEL" in line


def test_config_model_beats_the_env_model(tmp_cron_dir, capsys, monkeypatch):
    _write_config({"model": {"default": "main-model", "provider": "main-prov"}})
    monkeypatch.setenv("HERMES_MODEL", "env-model")
    create_job(prompt="p", schedule="0 9 * * *")
    cron_list()
    assert "main-model (main-prov)" in _row(capsys, "Model")


# --- provider outage: what the listing promises is what the scheduler does -------------------

_MAIN = {"default": "main-model", "provider": "main-prov"}
_CHAIN = [
    {"provider": "fb-prov-1", "model": "fb-model-1"},
    {"provider": "fb-prov-2", "model": "fb-model-2"},
]


def _runtime(provider):
    return {"api_key": "k", "base_url": "https://example.invalid/v1", "provider": provider,
            "api_mode": "chat_completions"}


def _outage(*, down):
    """``resolve_runtime_provider`` stand-in: providers in *down* raise AuthError, the rest resolve."""
    def resolve(**kwargs):
        requested = kwargs.get("requested") or _MAIN["provider"]
        if requested in down:
            raise AuthError(f"{requested}: usage limit reached")
        return _runtime(requested)
    return resolve


def _listed_fallback(capsys) -> str:
    return _row(capsys, "Fallback").removeprefix("Fallback:").strip()


def _resolved(job, cfg, *, down):
    jc = _CronJobConfig(cfg=cfg, model=_MAIN["default"], model_cfg=cfg.get("model", {}),
                        cron_default_provider="")
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=_outage(down=down)):
        runtime, model = _resolve_job_runtime(job, job["id"], jc)
    return runtime["provider"], model


def test_outage_lands_on_the_first_listed_fallback(tmp_cron_dir, capsys):
    cfg = {"model": _MAIN, "fallback_providers": _CHAIN}
    _write_config(cfg)
    job = create_job(prompt="p", schedule="0 9 * * *")
    cron_list()
    listed = _listed_fallback(capsys)
    landed = _resolved(job, cfg, down={"main-prov"})
    assert landed == ("fb-prov-1", "fb-model-1")
    assert listed.startswith("fb-model-1 (fb-prov-1)")


def test_outage_skips_a_dead_fallback_in_listed_order(tmp_cron_dir, capsys):
    cfg = {"model": _MAIN, "fallback_providers": _CHAIN}
    _write_config(cfg)
    job = create_job(prompt="p", schedule="0 9 * * *")
    cron_list()
    assert _listed_fallback(capsys) == "fb-model-1 (fb-prov-1) → fb-model-2 (fb-prov-2)"
    assert _resolved(job, cfg, down={"main-prov", "fb-prov-1"}) == ("fb-prov-2", "fb-model-2")


def test_outage_on_a_pinned_job_fails_and_the_listing_says_why_and_how_to_fix(tmp_cron_dir, capsys):
    cfg = {"model": _MAIN, "fallback_providers": _CHAIN}
    _write_config(cfg)
    job = create_job(prompt="p", schedule="0 9 * * *", model="pin-model", provider="pin-prov")
    cron_list()
    listed = _listed_fallback(capsys)
    with pytest.raises(RuntimeError, match="usage limit reached"):
        _resolved(job, cfg, down={"pin-prov"})
    assert "does not use the fallback chain" in listed
    assert f"hermes cron edit {job['id']} --unpin" in listed


def test_the_unpin_remedy_makes_the_same_outage_recoverable(tmp_cron_dir, capsys):
    cfg = {"model": _MAIN, "fallback_providers": _CHAIN}
    _write_config(cfg)
    job = create_job(prompt="p", schedule="0 9 * * *", model="main-model", provider="main-prov")
    with pytest.raises(RuntimeError):
        _resolved(job, cfg, down={"main-prov"})
    unpinned = {**job, "model": None, "provider": None, "base_url": None}
    assert _resolved(unpinned, cfg, down={"main-prov"}) == ("fb-prov-1", "fb-model-1")


def test_outage_without_any_chain_fails_and_the_listing_names_the_command(tmp_cron_dir, capsys):
    cfg = {"model": _MAIN}
    _write_config(cfg)
    job = create_job(prompt="p", schedule="0 9 * * *")
    cron_list()
    listed = _listed_fallback(capsys)
    with pytest.raises(RuntimeError, match="usage limit reached"):
        _resolved(job, cfg, down={"main-prov"})
    assert "none configured" in listed
    assert "hermes fallback add" in listed


def test_pinned_job_with_no_global_chain_offers_no_misleading_unpin_hint(tmp_cron_dir, capsys):
    _write_config({"model": _MAIN})
    create_job(prompt="p", schedule="0 9 * * *", model="pin-model", provider="pin-prov")
    cron_list()
    assert "--unpin" not in _listed_fallback(capsys)
