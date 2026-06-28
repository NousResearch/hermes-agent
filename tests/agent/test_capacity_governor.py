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


# --- Adversarial-review hardening (2026-06-27): foot-gun + fail-open ---


def test_protected_class_never_blocked_even_if_misconfigured(monkeypatch):
    """A misconfigured block list must NOT be able to defer the interactive
    main loop or compression — that would crash the user's turn via
    conversation_loop's RuntimeError. PROTECTED is unconditionally excluded."""
    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: {"rate_limited": True, "reset_at": 123.0},
    )
    hostile_cfg = {
        "capacity_governor": {
            "enabled": True,
            # Operator turns protection off AND adds critical classes to the
            # block list — the worst-case misconfiguration.
            "protect_interactive_main": False,
            "block_task_classes": ["interactive_main", "compression_critical"],
        }
    }
    for task in ("interactive_main", "compression_critical"):
        decision = check_capacity(
            provider="openai-codex",
            model="gpt-5.5",
            task_class=task,
            config=hostile_cfg,
        )
        assert decision.allowed is True, f"{task} must never be deferred"


def test_failopen_when_codex_status_raises(monkeypatch):
    """The #1 safety property: if the codex status lookup throws, the gate must
    allow (fail-open), never block a real call."""

    def _boom():
        raise RuntimeError("auth lookup exploded")

    monkeypatch.setattr("hermes_cli.auth.get_codex_auth_status", _boom)

    decision = check_capacity(
        provider="openai-codex",
        model="gpt-5.5",
        task_class="background_review",  # would be blocked if status said exhausted
        config=_enabled_config(),
    )
    assert decision.allowed is True


def test_failopen_when_config_missing(monkeypatch):
    """No config (None) → governor disabled by default → allow."""
    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: {"rate_limited": True, "reset_at": 123.0},
    )
    decision = check_capacity(
        provider="openai-codex",
        model="gpt-5.5",
        task_class="background_review",
        config=None,
    )
    assert decision.allowed is True


def test_allows_exhausted_codex_for_task_outside_block_list(monkeypatch):
    """Even when codex is exhausted, a task that is not in the block list must
    pass — only explicitly blocked classes are deferred."""
    monkeypatch.setattr(
        "hermes_cli.auth.get_codex_auth_status",
        lambda: {"rate_limited": True, "reset_at": 123.0},
    )
    decision = check_capacity(
        provider="openai-codex",
        model="gpt-5.5",
        task_class="some_unlisted_task",
        config=_enabled_config(),  # block list does not contain this task
    )
    assert decision.allowed is True
