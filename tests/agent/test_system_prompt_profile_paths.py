from types import SimpleNamespace


def _minimal_agent() -> SimpleNamespace:
    return SimpleNamespace(
        context_compressor=None,
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=[],
        _kanban_worker_guidance=None,
        provider="",
        model="",
        _tool_use_enforcement="auto",
        platform="cli",
        _platform_hint_overrides={},
        _memory_store=None,
        _memory_enabled=False,
        _user_profile_enabled=False,
        _memory_manager=None,
        pass_session_id=False,
        session_id=None,
    )


def _stub_prompt_runtime(monkeypatch):
    import agent.system_prompt as system_prompt

    runtime = SimpleNamespace(
        load_soul_md=lambda *_args, **_kwargs: None,
        build_nous_subscription_prompt=lambda *_args, **_kwargs: "",
        build_environment_hints=lambda *_args, **_kwargs: "",
        build_context_files_prompt=lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr(system_prompt, "_ra", lambda: runtime)
    return system_prompt


def _active_profile_line(stable_prompt: str) -> str:
    return next(
        line
        for line in stable_prompt.splitlines()
        if line.startswith("Active Hermes profile:")
    )


def test_named_profile_prompt_uses_custom_hermes_root(monkeypatch):
    monkeypatch.setenv("HERMES_HOME", "/opt/hermes/profiles/coder")
    system_prompt = _stub_prompt_runtime(monkeypatch)

    stable = system_prompt.build_system_prompt_parts(_minimal_agent())["stable"]

    line = _active_profile_line(stable)
    assert "Active Hermes profile: coder." in line
    assert "reads and writes /opt/hermes/profiles/coder/." in line
    assert "default profile's data lives at /opt/hermes/skills/" in line
    assert "/opt/hermes/plugins/" in line
    assert "/opt/hermes/cron/" in line
    assert "/opt/hermes/memories/" in line
    assert "~/.hermes" not in line


def test_default_profile_prompt_uses_custom_hermes_root(monkeypatch):
    monkeypatch.setenv("HERMES_HOME", "/opt/hermes")
    system_prompt = _stub_prompt_runtime(monkeypatch)

    stable = system_prompt.build_system_prompt_parts(_minimal_agent())["stable"]

    line = _active_profile_line(stable)
    assert "Active Hermes profile: default." in line
    assert "under /opt/hermes/profiles/<name>/." in line
    assert "~/.hermes" not in line
