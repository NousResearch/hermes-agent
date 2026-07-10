from types import SimpleNamespace


ROUTER_CONFIG = {
    "model_router": {
        "enabled": True,
        "default": "standard",
        "routes": {
            "quick": {"model": "gpt-5.6-luna"},
            "standard": {"model": "gpt-5.6-terra"},
            "complex": {"model": "gpt-5.6-sol"},
        },
    }
}


def test_select_model_routes_quick_standard_and_complex():
    from hermes_cli.model_router import select_model_for_turn

    assert select_model_for_turn("what time is it?", "gpt-5.5", ROUTER_CONFIG) == (
        "gpt-5.6-luna",
        "quick",
    )
    assert select_model_for_turn("implement the auth route", "gpt-5.5", ROUTER_CONFIG) == (
        "gpt-5.6-terra",
        "standard",
    )
    assert select_model_for_turn("draft an architectural migration plan", "gpt-5.5", ROUTER_CONFIG) == (
        "gpt-5.6-sol",
        "complex",
    )


def test_select_model_supports_explicit_route_overrides():
    from hermes_cli.model_router import select_model_for_turn

    assert select_model_for_turn("/sol advise on the data model", "gpt-5.5", ROUTER_CONFIG) == (
        "gpt-5.6-sol",
        "complex",
    )
    assert select_model_for_turn("route:luna summarize this", "gpt-5.5", ROUTER_CONFIG) == (
        "gpt-5.6-luna",
        "quick",
    )


def test_select_model_disabled_keeps_base_model():
    from hermes_cli.model_router import select_model_for_turn

    assert select_model_for_turn(
        "architect the platform",
        "gpt-5.5",
        {"model_router": {"enabled": False}},
    ) == ("gpt-5.5", None)


def test_session_router_preserves_existing_pin_when_disabled():
    from hermes_cli.model_router import select_model_for_session_turn

    assert select_model_for_session_turn(
        "follow up on the implementation",
        "gpt-5.6-terra",
        pinned_model="gpt-5.6-luna",
        config={"model_router": {"enabled": False}},
    ) == ("gpt-5.6-luna", None)


def test_cli_turn_route_uses_model_router(monkeypatch):
    import cli as cli_mod
    import hermes_cli.model_router as model_router

    monkeypatch.setattr(
        model_router,
        "select_model_for_session_turn",
        lambda message, base_model, pinned_model=None: ("gpt-5.6-sol", "complex"),
    )
    stub = SimpleNamespace(
        model="gpt-5.6-terra",
        api_key="primary-key",
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openai-codex",
        api_mode="codex_responses",
        acp_command=None,
        acp_args=[],
        _credential_pool=None,
        service_tier="",
    )

    route = cli_mod.HermesCLI._resolve_turn_agent_config(stub, "architect this")

    assert route["model"] == "gpt-5.6-sol"
    assert route["router_tier"] == "complex"
    assert route["signature"][0] == "gpt-5.6-sol"
    assert route["runtime"]["provider"] == "openai-codex"


def test_cli_marks_explicit_route_as_ephemeral(monkeypatch):
    import cli as cli_mod
    import hermes_cli.model_router as model_router

    monkeypatch.setattr(
        model_router,
        "select_model_for_session_turn",
        lambda message, base_model, pinned_model=None: ("gpt-5.6-sol", "complex"),
    )
    monkeypatch.setattr(
        model_router,
        "explicit_model_router_tier",
        lambda message, config=None: "complex",
    )
    stub = SimpleNamespace(
        model="gpt-5.6-terra",
        api_key="primary-key",
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openai-codex",
        api_mode="codex_responses",
        acp_command=None,
        acp_args=[],
        _credential_pool=None,
        service_tier="",
    )

    route = cli_mod.HermesCLI._resolve_turn_agent_config(stub, "route:sol audit this")

    assert route["model"] == "gpt-5.6-sol"
    assert route["router_ephemeral"] is True
    assert route["router_restore_model"] == "gpt-5.6-terra"


def test_cli_ephemeral_route_clears_active_signature():
    import cli as cli_mod

    stub = SimpleNamespace()
    cli_mod.HermesCLI._set_active_agent_route_signature(
        stub,
        "gpt-5.6-sol",
        {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
            "command": None,
            "args": [],
        },
        router_ephemeral=True,
    )

    assert stub._active_agent_route_signature is None


def test_cli_restores_session_pin_after_ephemeral_route():
    import cli as cli_mod

    calls = []
    stub = SimpleNamespace(
        session_id="session-1",
        agent=SimpleNamespace(session_id="session-1"),
        _session_db=SimpleNamespace(
            update_session_model=lambda session_id, model: calls.append((session_id, model))
        ),
    )
    cli_mod.HermesCLI._restore_ephemeral_router_session_model(
        stub,
        router_ephemeral=True,
        restore_model="gpt-5.6-terra",
    )

    assert calls == [("session-1", "gpt-5.6-terra")]


def test_cli_ephemeral_route_restores_durable_agent_and_system_prompt():
    import cli as cli_mod

    calls = []

    class SessionDB:
        def get_session(self, session_id):
            assert session_id == "session-1"
            return {"system_prompt": "Model: gpt-5.6-terra"}

        def update_session_model(self, session_id, model):
            calls.append(("model", session_id, model))

        def update_system_prompt(self, session_id, prompt):
            calls.append(("prompt", session_id, prompt))

    durable = SimpleNamespace(
        session_id="session-1", _cached_system_prompt="Model: gpt-5.6-terra"
    )
    released = []
    temporary = SimpleNamespace(
        session_id="session-1", release_clients=lambda: released.append(True)
    )
    signature = ("gpt-5.6-terra", "openai-codex", "", "", None, ())
    stub = SimpleNamespace(
        session_id="session-1",
        agent=durable,
        _active_agent_route_signature=signature,
        _session_db=SessionDB(),
    )
    stub._release_agent_clients = lambda candidate: cli_mod.HermesCLI._release_agent_clients(
        stub, candidate
    )

    cli_mod.HermesCLI._capture_ephemeral_router_state(stub, router_ephemeral=True)
    stub.agent = temporary
    stub._active_agent_route_signature = None
    cli_mod.HermesCLI._restore_ephemeral_router_session_model(
        stub,
        router_ephemeral=True,
        restore_model="gpt-5.6-terra",
    )

    assert stub.agent is durable
    assert stub._active_agent_route_signature == signature
    assert released == [True]
    assert calls == [
        ("model", "session-1", "gpt-5.6-terra"),
        ("prompt", "session-1", "Model: gpt-5.6-terra"),
    ]


def test_cli_ephemeral_route_preserves_compressed_child_session_id():
    import cli as cli_mod

    calls = []

    class SessionDB:
        def update_session_model(self, session_id, model):
            calls.append(("model", session_id, model))

        def update_system_prompt(self, session_id, prompt):
            calls.append(("prompt", session_id, prompt))

    durable = SimpleNamespace(
        session_id="parent-session", _cached_system_prompt="Model: gpt-5.6-terra"
    )
    temporary = SimpleNamespace(session_id="child-session")
    stub = SimpleNamespace(
        session_id="parent-session",
        agent=durable,
        _active_agent_route_signature=("gpt-5.6-terra",),
        _session_db=SessionDB(),
    )

    cli_mod.HermesCLI._capture_ephemeral_router_state(stub, router_ephemeral=True)
    stub.agent = temporary
    cli_mod.HermesCLI._restore_ephemeral_router_session_model(
        stub,
        router_ephemeral=True,
        restore_model="gpt-5.6-terra",
    )

    assert stub.agent is None
    assert stub._last_ephemeral_router_session_id == "child-session"
    assert calls == [
        ("model", "child-session", "gpt-5.6-terra"),
        ("prompt", "child-session", "Model: gpt-5.6-terra"),
    ]


def test_cli_explicit_model_switch_replaces_router_pin():
    import cli as cli_mod

    calls = []
    stub = SimpleNamespace(
        session_id="session-1",
        _session_db=SimpleNamespace(
            update_session_model=lambda session_id, model: calls.append((session_id, model))
        ),
    )
    stub._set_active_agent_route_signature = (
        lambda model, runtime, router_ephemeral=False: cli_mod.HermesCLI._set_active_agent_route_signature(
            stub, model, runtime, router_ephemeral=router_ephemeral
        )
    )
    runtime = {
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
        "command": None,
        "args": [],
    }

    cli_mod.HermesCLI._set_explicit_session_model_pin(
        stub, "custom-explicit-model", runtime
    )

    assert stub._active_agent_route_signature[0] == "custom-explicit-model"
    assert stub._explicit_session_model_pin == "custom-explicit-model"
    assert calls == [("session-1", "custom-explicit-model")]


def test_cli_explicit_model_pin_beats_stale_session_route_after_restore_failure(monkeypatch):
    import cli as cli_mod
    import hermes_cli.model_router as model_router

    captured = {}

    def select_model(message, base_model, pinned_model=None):
        captured["pinned_model"] = pinned_model
        return pinned_model or base_model, "standard"

    monkeypatch.setattr(model_router, "explicit_model_router_tier", lambda message, config=None: None)
    monkeypatch.setattr(model_router, "select_model_for_session_turn", select_model)
    stub = SimpleNamespace(
        model="custom-explicit-model",
        session_id="session-1",
        agent=SimpleNamespace(session_id="session-1"),
        _explicit_session_model_pin="custom-explicit-model",
        _active_agent_route_signature=None,
        _session_db=SimpleNamespace(
            get_session=lambda session_id: {
                "model": "temporary-route-model", "message_count": 2
            },
            update_session_model=lambda session_id, model: (_ for _ in ()).throw(
                RuntimeError("database unavailable")
            ),
        ),
        api_key="primary-key",
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openai-codex",
        api_mode="codex_responses",
        acp_command=None,
        acp_args=[],
        _credential_pool=None,
        service_tier="",
    )

    cli_mod.HermesCLI._restore_ephemeral_router_session_model(
        stub,
        router_ephemeral=True,
        restore_model="custom-explicit-model",
    )
    route = cli_mod.HermesCLI._resolve_turn_agent_config(stub, "continue normally")

    assert captured["pinned_model"] == "custom-explicit-model"
    assert route["model"] == "custom-explicit-model"


def test_cli_model_switch_pins_explicit_model_and_discards_failed_agent(monkeypatch):
    import cli as cli_mod
    import hermes_cli.model_switch as model_switch

    monkeypatch.setattr(cli_mod, "_cprint", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        model_switch, "resolve_display_context_length", lambda *args, **kwargs: None
    )
    switch_calls = []
    pin_calls = []

    class FailingAgent:
        session_id = "agent-session"
        _config_context_length = None

        def switch_model(self, **kwargs):
            switch_calls.append(kwargs)
            raise RuntimeError("in-place swap failed")

    stub = SimpleNamespace(
        session_id="cli-session",
        model="gpt-5.6-luna",
        provider="openai-codex",
        requested_provider="openai-codex",
        api_key="existing-key",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        acp_command=None,
        acp_args=[],
        agent=FailingAgent(),
        _session_db=SimpleNamespace(
            update_session_model=lambda session_id, model: pin_calls.append(
                (session_id, model)
            )
        ),
    )
    stub._set_active_agent_route_signature = (
        lambda model, runtime, router_ephemeral=False: cli_mod.HermesCLI._set_active_agent_route_signature(
            stub, model, runtime, router_ephemeral=router_ephemeral
        )
    )
    stub._set_explicit_session_model_pin = lambda model, runtime: cli_mod.HermesCLI._set_explicit_session_model_pin(
        stub, model, runtime
    )
    result = SimpleNamespace(
        success=True,
        new_model="custom-explicit-model",
        target_provider="openai-codex",
        api_key=None,
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        provider_label="OpenAI Codex",
        model_info=None,
        warning_message=None,
        provider_changed=False,
    )

    cli_mod.HermesCLI._apply_model_switch_result(stub, result, persist_global=False)

    assert switch_calls == [{
        "new_model": "custom-explicit-model",
        "new_provider": "openai-codex",
        "api_key": None,
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
    }]
    assert stub.agent is None
    assert stub._active_agent_route_signature[0] == "custom-explicit-model"
    assert pin_calls == [("agent-session", "custom-explicit-model")]

    import hermes_cli.model_router as model_router

    captured = {}
    monkeypatch.setattr(
        model_router,
        "explicit_model_router_tier",
        lambda message, config=None: "complex",
    )
    monkeypatch.setattr(
        model_router,
        "select_model_for_session_turn",
        lambda message, base_model, pinned_model=None: (
            captured.setdefault("pinned_model", pinned_model) and "gpt-5.6-sol",
            "complex",
        ),
    )
    next_route = cli_mod.HermesCLI._resolve_turn_agent_config(
        stub, "route:sol review this"
    )

    assert captured["pinned_model"] == "custom-explicit-model"
    assert next_route["router_restore_model"] == "custom-explicit-model"


def test_gateway_turn_route_uses_model_router(monkeypatch):
    import gateway.run as gateway_run
    import hermes_cli.model_router as model_router

    monkeypatch.setattr(
        model_router,
        "select_model_for_session_turn",
        lambda message, base_model, pinned_model=None: ("gpt-5.6-luna", "quick"),
    )
    runner = SimpleNamespace(_service_tier="")
    runtime_kwargs = {
        "api_key": "***",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "command": None,
        "args": [],
        "credential_pool": None,
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner,
        "what is the status?",
        "gpt-5.6-terra",
        runtime_kwargs,
    )

    assert route["model"] == "gpt-5.6-luna"
    assert route["router_tier"] == "quick"
    assert route["signature"][0] == "gpt-5.6-luna"
    assert route["runtime"]["provider"] == "openai-codex"


def test_gateway_marks_explicit_route_as_ephemeral(monkeypatch):
    import gateway.run as gateway_run
    import hermes_cli.model_router as model_router

    monkeypatch.setattr(
        model_router,
        "select_model_for_session_turn",
        lambda message, base_model, pinned_model=None: ("gpt-5.6-sol", "complex"),
    )
    monkeypatch.setattr(
        model_router,
        "explicit_model_router_tier",
        lambda message, config=None: "complex",
    )
    runner = SimpleNamespace(_service_tier="")
    runtime_kwargs = {
        "api_key": "***", "base_url": "https://chatgpt.com/backend-api/codex",
        "provider": "openai-codex", "api_mode": "codex_responses",
        "command": None, "args": [], "credential_pool": None,
    }

    route = gateway_run.GatewayRunner._resolve_turn_agent_config(
        runner, "route:sol audit", "gpt-5.6-terra", runtime_kwargs
    )

    assert route["model"] == "gpt-5.6-sol"
    assert route["router_ephemeral"] is True


def test_gateway_restores_session_pin_after_ephemeral_route():
    import gateway.run as gateway_run

    calls = []
    runner = SimpleNamespace(
        _session_db=SimpleNamespace(
            update_session_model=lambda session_id, model: calls.append((session_id, model))
        )
    )

    gateway_run.GatewayRunner._restore_ephemeral_router_session_model(
        runner,
        "session-1",
        {
            "router_ephemeral": True,
            "router_restore_model": "gpt-5.6-luna",
        },
    )

    assert calls == [("session-1", "gpt-5.6-luna")]


def test_gateway_ephemeral_route_restores_durable_system_prompt():
    import gateway.run as gateway_run

    calls = []

    class SessionDB:
        def get_session(self, session_id):
            assert session_id == "session-1"
            return {"system_prompt": "Model: gpt-5.6-terra"}

        def update_session_model(self, session_id, model):
            calls.append(("model", session_id, model))

        def update_system_prompt(self, session_id, prompt):
            calls.append(("prompt", session_id, prompt))

    turn_route = {
        "router_ephemeral": True,
        "router_restore_model": "gpt-5.6-terra",
    }
    runner = SimpleNamespace(_session_db=SessionDB())

    gateway_run.GatewayRunner._capture_ephemeral_router_system_prompt(
        runner, "session-1", turn_route
    )
    gateway_run.GatewayRunner._restore_ephemeral_router_session_model(
        runner, "session-1", turn_route
    )

    assert calls == [
        ("model", "session-1", "gpt-5.6-terra"),
        ("prompt", "session-1", "Model: gpt-5.6-terra"),
    ]


def test_gateway_does_not_overwrite_durable_cache_for_ephemeral_route():
    import gateway.run as gateway_run

    durable = object()
    cache = {"session-key": (durable, ("gpt-5.6-terra",), 2)}
    runner = SimpleNamespace(_agent_cache=cache, _agent_cache_lock=None)

    gateway_run.GatewayRunner._cache_agent_for_turn(
        runner,
        "session-key",
        object(),
        ("gpt-5.6-sol",),
        2,
        router_ephemeral=True,
    )

    assert cache["session-key"][0] is durable


def test_session_turn_pins_the_first_effective_model():
    from hermes_cli.model_router import select_model_for_session_turn

    assert select_model_for_session_turn(
        "draft an architectural migration plan",
        "gpt-5.6-terra",
        pinned_model="gpt-5.6-luna",
        config=ROUTER_CONFIG,
    ) == ("gpt-5.6-luna", "quick")


def test_session_turn_explicit_directive_overrides_pinned_model():
    from hermes_cli.model_router import select_model_for_session_turn

    assert select_model_for_session_turn(
        "route:sol draft an architecture plan",
        "gpt-5.6-terra",
        pinned_model="gpt-5.6-luna",
        config=ROUTER_CONFIG,
    ) == ("gpt-5.6-sol", "complex")


def test_session_turn_multimodal_explicit_directive_overrides_pinned_model():
    from hermes_cli.model_router import select_model_for_session_turn

    message = [{"type": "text", "text": "route:sol draft an architecture plan"}]
    assert select_model_for_session_turn(
        message,
        "gpt-5.6-terra",
        pinned_model="gpt-5.6-luna",
        config=ROUTER_CONFIG,
    ) == ("gpt-5.6-sol", "complex")


def test_gateway_router_pin_uses_persisted_session_model():
    import gateway.run as gateway_run

    runner = SimpleNamespace(
        _session_db=SimpleNamespace(
            get_session=lambda session_id: {
                "id": session_id,
                "model": "gpt-5.6-luna",
                "message_count": 2,
            }
        )
    )
    assert gateway_run.GatewayRunner._get_pinned_session_router_model(
        runner, "session-1"
    ) == ("gpt-5.6-luna", 2)


def test_gateway_router_does_not_pin_empty_session():
    import gateway.run as gateway_run

    runner = SimpleNamespace(
        _session_db=SimpleNamespace(
            get_session=lambda session_id: {
                "id": session_id,
                "model": "gpt-5.6-luna",
                "message_count": 0,
            }
        )
    )
    assert gateway_run.GatewayRunner._get_pinned_session_router_model(
        runner, "session-1"
    ) == (None, 0)


def test_gateway_router_preserves_explicit_model_override():
    import gateway.run as gateway_run

    runner = SimpleNamespace(
        _session_model_overrides={"session-1": {"model": "custom-explicit-model"}}
    )

    assert gateway_run.GatewayRunner._get_explicit_session_model_override(
        runner, "session-1"
    ) == "custom-explicit-model"


def test_gateway_router_explicit_override_beats_persisted_route():
    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_db = SimpleNamespace(
        get_session=lambda session_id: {
            "id": session_id,
            "model": "gpt-5.6-luna",
            "message_count": 2,
        }
    )
    runner._session_model_overrides = {
        "session-1": {"model": "custom-explicit-model"}
    }

    assert runner._resolve_session_router_pin("db-session", "session-1") == (
        "custom-explicit-model",
        2,
    )
