from types import SimpleNamespace


def test_runtime_preflight_kanban_passes_with_writable_home(monkeypatch, capsys, tmp_path):
    from hermes_cli import runtime_preflight as mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.setattr(
        mod,
        "load_config",
        lambda: {
            "model": "gpt-5.5",
            "provider": "openai-codex",
            "kanban": {"runtime_guardrails": {"enabled": True}},
        },
    )

    rc = mod.run_runtime_preflight(SimpleNamespace(scope="kanban", json=False))

    out = capsys.readouterr().out
    assert rc == 0
    assert "Runtime preflight: kanban" in out
    assert "provider/model" in out
    assert "kanban db writable" in out
    assert "PASS" in out


def test_runtime_preflight_kanban_fails_when_guardrails_disabled(monkeypatch, capsys, tmp_path):
    from hermes_cli import runtime_preflight as mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.setattr(
        mod,
        "load_config",
        lambda: {
            "model": "gpt-5.5",
            "provider": "openai-codex",
            "kanban": {"runtime_guardrails": {"enabled": False}},
        },
    )

    rc = mod.run_runtime_preflight(SimpleNamespace(scope="kanban", json=False))

    out = capsys.readouterr().out
    assert rc == 1
    assert "kanban runtime guardrails" in out
    assert "FAIL" in out
