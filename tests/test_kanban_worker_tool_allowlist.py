from __future__ import annotations

import yaml


TERMINAL = {"kanban_complete", "kanban_block"}


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _invalidate_config_cache() -> None:
    import hermes_cli.config as cfg_mod

    invalidate = getattr(cfg_mod, "_invalidate_load_config_cache", None)
    if callable(invalidate):
        invalidate()


def test_dispatcher_worker_allowlist_filters_only_kanban_tools(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    home.joinpath("config.yaml").write_text(
        yaml.safe_dump(
            {
                "kanban": {
                    "worker_tools": [
                        "kanban_show",
                        "kanban_comment",
                        "kanban_complete",
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_local")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    _invalidate_config_cache()

    import model_tools

    tools = [
        _schema("kanban_show"),
        _schema("kanban_comment"),
        _schema("kanban_complete"),
        _schema("kanban_block"),
        _schema("kanban_heartbeat"),
        _schema("todo"),
    ]
    filtered = model_tools._apply_worker_kanban_tool_allowlist(tools)
    names = {tool["function"]["name"] for tool in filtered}

    assert names == {
        "kanban_show",
        "kanban_comment",
        "kanban_complete",
        "kanban_block",
        "todo",
    }


def test_worker_allowlist_always_preserves_terminal_tools(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    home.joinpath("config.yaml").write_text(
        yaml.safe_dump({"kanban": {"worker_tools": ["kanban_show", "not_a_tool"]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_local")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    _invalidate_config_cache()

    import model_tools

    tools = [
        _schema("kanban_show"),
        _schema("kanban_complete"),
        _schema("kanban_block"),
        _schema("kanban_heartbeat"),
    ]
    names = {
        tool["function"]["name"]
        for tool in model_tools._apply_worker_kanban_tool_allowlist(tools)
    }

    assert TERMINAL <= names
    assert "kanban_show" in names
    assert "kanban_heartbeat" not in names


def test_allowlist_does_not_change_normal_or_delegated_sessions(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    home.joinpath("config.yaml").write_text(
        yaml.safe_dump({"kanban": {"worker_tools": ["kanban_complete"]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    _invalidate_config_cache()

    import model_tools

    tools = [
        _schema("kanban_show"),
        _schema("kanban_complete"),
        _schema("kanban_block"),
        _schema("kanban_heartbeat"),
    ]

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert model_tools._apply_worker_kanban_tool_allowlist(tools) == tools

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_inherited")
    monkeypatch.setattr(model_tools, "_is_delegated_child_context", lambda: True)
    assert model_tools._apply_worker_kanban_tool_allowlist(tools) == tools


def test_get_tool_definitions_applies_worker_allowlist_end_to_end(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    home.joinpath("config.yaml").write_text(
        yaml.safe_dump(
            {
                "kanban": {
                    "worker_tools": [
                        "kanban_show",
                        "kanban_comment",
                        "kanban_complete",
                        "kanban_block",
                        "kanban_heartbeat",
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_e2e")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "1")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "test-claim")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    _invalidate_config_cache()

    import model_tools

    model_tools._tool_defs_cache.clear()
    definitions = model_tools.get_tool_definitions(
        enabled_toolsets=["kanban"], quiet_mode=True
    )
    kanban_names = {
        tool["function"]["name"]
        for tool in definitions
        if tool["function"]["name"].startswith("kanban_")
    }

    assert kanban_names == {
        "kanban_show",
        "kanban_comment",
        "kanban_complete",
        "kanban_block",
        "kanban_heartbeat",
    }


def test_absent_allowlist_preserves_existing_worker_surface(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    home.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_default")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    _invalidate_config_cache()

    import model_tools

    tools = [
        _schema("kanban_show"),
        _schema("kanban_complete"),
        _schema("kanban_block"),
        _schema("kanban_heartbeat"),
    ]
    assert model_tools._apply_worker_kanban_tool_allowlist(tools) == tools
