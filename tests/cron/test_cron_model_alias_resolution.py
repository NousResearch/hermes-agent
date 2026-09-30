"""Cron model pins resolve through the seat alias table sessions use (#126655).

A job pinned ``--model claude-opus`` used to reach the provider as the literal string
while ordinary sessions routed the same seat alias fine; the pin must go through the
same ``model.aliases`` / ``model_aliases:`` table, and an explicit provider pin still
wins over the alias's own provider label.
"""

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def alias_home(tmp_path, monkeypatch):
    """Tmp hermes home whose config.yaml declares string-form ``model.aliases``."""
    home = tmp_path / "hermes"
    (home / "cron").mkdir(parents=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  aliases:\n"
        "    claude-opus: anthropic/claude-opus-5-5\n"
        "    sol: openai-codex/gpt-6-sol\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    token = set_hermes_home_override(str(home))
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def _load(job_extra=None):
    import cron.scheduler as sched
    job = {"id": "j", "name": "j", "prompt": "x"}
    job.update(job_extra or {})
    return sched._load_cron_job_config(job, "j", "j")


def test_alias_pin_resolves_to_target_model_and_provider(alias_home):
    """``--model claude-opus`` routes like a session default: target model + alias provider."""
    jc = _load({"model": "claude-opus"})
    assert jc.model == "claude-opus-5-5"
    assert jc.alias_provider == "anthropic"


def test_explicit_job_provider_pin_wins_over_alias_provider(alias_home):
    """The model still resolves, but an explicit per-job provider keeps its route."""
    jc = _load({"model": "claude-opus", "provider": "openrouter"})
    assert jc.model == "claude-opus-5-5"
    assert jc.alias_provider == ""


def test_unaliased_pin_passes_through_untouched(alias_home):
    """A concrete (non-alias) model name is handed on verbatim with no provider injected."""
    jc = _load({"model": "openai/gpt-5.2"})
    assert jc.model == "openai/gpt-5.2"
    assert jc.alias_provider == ""


def test_config_default_model_resolves_aliases_too(alias_home):
    """The unpinned path (config ``model.default``) goes through the same resolution."""
    (alias_home / "config.yaml").write_text(
        "model:\n"
        "  default: claude-opus\n"
        "  aliases:\n"
        "    claude-opus: anthropic/claude-opus-5-5\n"
    )
    jc = _load()
    assert jc.model == "claude-opus-5-5"
    assert jc.alias_provider == "anthropic"


def test_resolve_job_runtime_uses_alias_provider_when_nothing_explicit(alias_home):
    """Provider precedence per-job > cron.model_provider > alias label > persisted config."""
    import cron.scheduler as sched
    jc = sched._CronJobConfig(
        cfg={}, model="claude-opus-5-5", model_cfg={},
        cron_default_provider="", alias_provider="anthropic")
    captured = {}

    def fake_resolve(**kw):
        captured.update(kw)
        return {"provider": "anthropic"}

    import hermes_cli.runtime_provider as rp
    original = rp.resolve_runtime_provider
    rp.resolve_runtime_provider = fake_resolve
    try:
        runtime, model = sched._resolve_job_runtime({"id": "j", "prompt": "x"}, "j", jc)
        assert model == "claude-opus-5-5"
        assert captured["requested"] == "anthropic"
        assert captured["target_model"] == "claude-opus-5-5"
    finally:
        rp.resolve_runtime_provider = original


def test_resolve_job_runtime_fleet_provider_beats_alias_label(alias_home):
    """``cron.model_provider`` is an explicit fleet pin and outranks the alias's own label."""
    import cron.scheduler as sched
    jc = sched._CronJobConfig(
        cfg={}, model="claude-opus-5-5", model_cfg={},
        cron_default_provider="openrouter", alias_provider="anthropic")
    captured = {}

    def fake_resolve(**kw):
        captured.update(kw)
        return {"provider": "openrouter"}

    import hermes_cli.runtime_provider as rp
    original = rp.resolve_runtime_provider
    rp.resolve_runtime_provider = fake_resolve
    try:
        sched._resolve_job_runtime({"id": "j", "prompt": "x"}, "j", jc)
        assert captured["requested"] == "openrouter"
    finally:
        rp.resolve_runtime_provider = original


def _notice(job, error, monkeypatch):
    """``_summarize_cron_failure_for_delivery`` with no fallback chain configured."""
    import cron.scheduler as sched
    monkeypatch.setattr(sched, "load_config", lambda: {})
    monkeypatch.setattr(sched, "get_fallback_chain", lambda cfg: [])
    return sched._summarize_cron_failure_for_delivery(job, error)


def test_model_not_found_notice_says_alias_not_resolved_on_prefix_gateway(alias_home, monkeypatch):
    """PR #126745 field feedback: a pin that IS an alias key still 404ing means the running
    gateway predates #126655 — the notice must say alias-not-resolved and name the target."""
    msg = _notice(
        {"name": "J", "id": "ab12cd34", "model": "claude-opus"},
        "Error code: 404 - model: claude-opus is not a valid model", monkeypatch)
    assert "seat alias" in msg, msg
    assert "claude-opus-5-5" in msg, msg
    assert "#126655" in msg and "gateway" in msg, msg


def test_model_not_found_notice_lists_seat_aliases_for_a_non_alias_pin(alias_home, monkeypatch):
    """A pin matching no alias still names the seat's valid aliases (bounded, actionable)."""
    msg = _notice(
        {"name": "J", "id": "ab12cd34", "model": "claude-opuss"},
        "Error code: 404 - model: claude-opuss is not a valid model", monkeypatch)
    assert "claude-opus -> claude-opus-5-5" in msg, msg
    assert "`hermes cron edit ab12cd34 --model" in msg, msg


def test_model_not_found_without_a_pin_keeps_the_plain_notice(alias_home, monkeypatch):
    """No model pin (fleet default) — no alias appendix; the base notice is untouched."""
    msg = _notice(
        {"name": "J", "id": "ab12cd34"},
        "Error code: 404 - model: claude-opus is not a valid model", monkeypatch)
    assert "seat alias" not in msg, msg
    assert "`hermes cron edit ab12cd34 --model" in msg, msg  # base model_not_found action
