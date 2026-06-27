from types import SimpleNamespace

from agent.capacity_governor import (
    check_capacity,
    release,
    reserve,
    task_class_for_auxiliary_task,
)


def _enabled_config(blocked=None):
    return {
        "capacity_governor": {
            "enabled": True,
            "protect_interactive_main": True,
            "block_task_classes": blocked
            or [
                "background_review",
                "title",
                "session_search",
                "cron_judgment",
            ],
        }
    }


def test_disabled_governor_allows_codex_background(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: {"rate_limited": True, "reset_at": 123.0},
    )

    decision = check_capacity(
        provider="openai-codex",
        model="gpt-5.5",
        task_class="background_review",
        config={"capacity_governor": {"enabled": False}},
    )

    assert decision.allowed is True


def test_codex_exhausted_blocks_background_when_enabled(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: {"rate_limited": True, "reset_at": 123.0},
    )

    decision = check_capacity(
        provider="codex",
        model="gpt-5.5",
        task_class="background_review",
        config=_enabled_config(),
    )

    assert decision.allowed is False
    assert decision.action == "defer"
    assert decision.reset_at == 123.0


def test_governor_preserves_interactive_and_compression(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: {"rate_limited": True, "reset_at": 123.0},
    )
    cfg = _enabled_config()

    assert check_capacity(
        provider="openai-codex",
        model="gpt-5.5",
        task_class="interactive_main",
        config=cfg,
    ).allowed is True
    assert check_capacity(
        provider="openai-codex",
        model="gpt-5.5",
        task_class="compression_critical",
        config=cfg,
    ).allowed is True


def test_governor_allows_non_codex_provider(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: (_ for _ in ()).throw(AssertionError("should not check codex")),
    )

    decision = check_capacity(
        provider="anthropic",
        model="claude-sonnet",
        task_class="background_review",
        config=_enabled_config(),
    )

    assert decision.allowed is True


def test_task_class_for_auxiliary_task_maps_protected_and_low_risk_tasks():
    assert task_class_for_auxiliary_task("compression") == "compression_critical"
    assert task_class_for_auxiliary_task("title_generation") == "title"
    assert task_class_for_auxiliary_task("curator_merge") == "curator"
    assert task_class_for_auxiliary_task(None, raw_codex=True) == "interactive_main"


def test_reserve_release_api_is_available(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: {"rate_limited": False},
    )

    assert reserve(
        provider="openai-codex",
        model="gpt-5.5",
        task_class="background_review",
        config=_enabled_config(),
    ).allowed is True
    assert release(provider="openai-codex", task_class="background_review") is None


def test_aux_auto_skips_codex_main_for_blocked_low_risk_task(monkeypatch):
    import agent.auxiliary_client as aux

    fallback_client = SimpleNamespace(name="fallback")

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _enabled_config())
    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: {"rate_limited": True, "reset_at": 123.0},
    )
    monkeypatch.setattr(aux, "_read_main_provider", lambda: "openai-codex")
    monkeypatch.setattr(aux, "_read_main_model", lambda: "gpt-5.5")
    monkeypatch.setattr(aux, "_build_codex_client", lambda model: (SimpleNamespace(name="codex"), model))
    monkeypatch.setattr(aux, "_try_configured_fallback_chain", lambda *a, **k: (None, None, ""))
    monkeypatch.setattr(aux, "_try_main_fallback_chain", lambda *a, **k: (None, None, ""))
    monkeypatch.setattr(aux, "_get_provider_chain", lambda: [("openrouter", lambda: (fallback_client, "fallback-model"))])

    client, model = aux._resolve_auto(task="title_generation")

    assert client is fallback_client
    assert model == "fallback-model"


def test_aux_auto_keeps_codex_main_for_compression(monkeypatch):
    import agent.auxiliary_client as aux

    codex_client = SimpleNamespace(name="codex")

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _enabled_config())
    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: {"rate_limited": True, "reset_at": 123.0},
    )
    monkeypatch.setattr(aux, "_read_main_provider", lambda: "openai-codex")
    monkeypatch.setattr(aux, "_read_main_model", lambda: "gpt-5.5")
    monkeypatch.setattr(aux, "_build_codex_client", lambda model: (codex_client, model))

    client, model = aux._resolve_auto(task="compression")

    assert client is codex_client
    assert model == "gpt-5.5"
