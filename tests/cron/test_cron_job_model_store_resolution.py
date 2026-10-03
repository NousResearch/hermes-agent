"""A cron job's user-set model resolves exactly like ``/model`` at the store seam every surface
shares (``hermes cron create/edit --model``, dashboard, tool): a ``model_aliases:`` tier lands as its
concrete route, a catalog short name expands on the job's provider, a full id passes through, and an
ambiguous short name is refused with ``/model``'s candidate list instead of being stored."""

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def alias_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    (home / "cron").mkdir(parents=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  aliases:\n"
        "    fast:\n"
        "      model: qwen3-coder\n"
        "      provider: custom\n"
        "      base_url: http://127.0.0.1:11434/v1\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    token = set_hermes_home_override(str(home))
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def test_store_resolves_aliases_like_slash_model(alias_home, monkeypatch):
    import hermes_cli.model_switch as ms
    from cron import jobs

    monkeypatch.setattr(ms, "list_provider_models", lambda provider, **kw: ["kimi-k2.5"] if provider == "testprov" else [])
    with jobs.use_cron_store(alias_home / "cron"):
        tier = jobs.create_job(prompt="p", schedule="every 1h", model="fast")
        assert (tier["model"], tier["provider"], tier["base_url"]) == (
            "qwen3-coder", "custom", "http://127.0.0.1:11434/v1")

        catalog = jobs.create_job(prompt="p", schedule="every 1h", model="kimi", provider="testprov")
        assert (catalog["model"], catalog["provider"]) == ("kimi-k2.5", "testprov")

        full = jobs.create_job(prompt="p", schedule="every 1h", model="openai/gpt-5.2", provider="openrouter")
        assert (full["model"], full["provider"], full["base_url"]) == ("openai/gpt-5.2", "openrouter", None)

        # `hermes cron edit --model fast` on a job that already pins a provider keeps that pin.
        edited = jobs.update_job(full["id"], {"model": "fast"})
        assert (edited["model"], edited["provider"], edited["base_url"]) == ("qwen3-coder", "openrouter", None)


def test_ambiguous_alias_is_refused_not_stored(alias_home, monkeypatch):
    import hermes_cli.model_switch as ms
    from cron import jobs

    monkeypatch.setattr(ms, "list_provider_models", lambda provider, **kw: ["kimi-k2.5", "kimi-k2.6"])
    with jobs.use_cron_store(alias_home / "cron"):
        with pytest.raises(ValueError) as err:
            jobs.create_job(prompt="p", schedule="every 1h", model="kimi", provider="testprov")
        assert "kimi-k2.5" in str(err.value) and "kimi-k2.6" in str(err.value)
        assert jobs.list_jobs() == []
