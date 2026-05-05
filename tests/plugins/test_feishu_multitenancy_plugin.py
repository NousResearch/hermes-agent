"""Tests for the bundled Feishu multitenancy plugin."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_KEY = "platforms/feishu-multitenancy"
PLUGIN_DIR = REPO_ROOT / "plugins" / "platforms" / "feishu-multitenancy"


def _forget_multitenancy_modules() -> None:
    for name in list(sys.modules):
        if (
            name == "hermes_multitenancy"
            or name.startswith("hermes_multitenancy.")
            or name.startswith("hermes_plugins.platforms__feishu_multitenancy")
        ):
            sys.modules.pop(name, None)


@contextmanager
def _bundled_plugin_path():
    _forget_multitenancy_modules()
    sys.path.insert(0, str(PLUGIN_DIR))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(PLUGIN_DIR))
        except ValueError:
            pass
        _forget_multitenancy_modules()


def test_bundled_plugin_loads_when_enabled(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled:\n"
        f"    - {PLUGIN_KEY}\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from hermes_cli.plugins import PluginManager

    _forget_multitenancy_modules()
    manager = PluginManager()
    manager.discover_and_load(force=True)

    loaded = manager._plugins[PLUGIN_KEY]
    assert loaded.enabled
    assert loaded.error is None
    assert "pre_gateway_dispatch" in loaded.hooks_registered
    assert any(
        getattr(callback, "__name__", "") == "on_pre_gateway_dispatch"
        for callback in manager._hooks["pre_gateway_dispatch"]
    )


def test_route_sync_wrapper_applies_users(tmp_path):
    users_json = tmp_path / "users.json"
    db_path = tmp_path / "multitenancy.db"
    users_json.write_text(
        json.dumps(
            [
                {
                    "user_id": "tenant-a",
                    "profile_name": "alice",
                    "open_id": "ou_test_alice",
                    "union_id": "on_test_alice",
                },
                {
                    "user_id": "tenant-b",
                    "profile_name": "bob",
                    "open_id": "ou_test_bob",
                },
            ]
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PLUGIN_DIR / "sync.py"),
            "apply",
            str(users_json),
            "--db",
            str(db_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert json.loads(result.stdout) == {
        "upserted": 2,
        "soft_deleted": 0,
        "kept": 0,
    }

    with _bundled_plugin_path():
        from hermes_multitenancy.routing import RoutingTable

        table = RoutingTable(db_path)
        try:
            row = table.lookup_by_open_id("ou_test_alice")
            assert row is not None
            assert row.profile_name == "alice"
            assert row.union_id == "on_test_alice"
        finally:
            table.close()


def test_router_prefers_feishu_open_id_from_raw_event():
    with _bundled_plugin_path():
        from hermes_multitenancy.router import _resolve_sender_for_routing

        event = SimpleNamespace(
            source=SimpleNamespace(user_id="tenant-local-id", user_id_alt="on_union"),
            raw={
                "event": {
                    "sender": {
                        "sender_id": {
                            "open_id": "ou_real_sender",
                            "union_id": "on_union",
                        }
                    }
                }
            },
        )

        assert _resolve_sender_for_routing(event) == "ou_real_sender"


def test_auto_profile_config_does_not_invent_a_default_model():
    with _bundled_plugin_path():
        from hermes_multitenancy.router import _normalize_profile_config

        assert _normalize_profile_config({}) == {}
        assert _normalize_profile_config({"tools": ["web"]}) == {"tools": ["web"]}


def test_aiagent_toolsets_merge_explicit_feishu_entries_with_defaults():
    with _bundled_plugin_path():
        from hermes_multitenancy.agent_real import _resolve_enabled_toolsets

        seen: dict[str, object] = {}

        def fake_get_platform_tools(config, platform, *, include_default_mcp_servers=True):
            seen["platform_toolsets"] = config.get("platform_toolsets")
            seen["platform"] = platform
            seen["include_default_mcp_servers"] = include_default_mcp_servers
            return {"web", "browser", "feishu_doc"}

        config = {"platform_toolsets": {"feishu": ["feishu_drive"]}}

        assert _resolve_enabled_toolsets(
            config,
            "feishu",
            platform_tools_resolver=fake_get_platform_tools,
        ) == ["browser", "feishu_doc", "feishu_drive", "web"]
        assert seen == {
            "platform_toolsets": {},
            "platform": "feishu",
            "include_default_mcp_servers": True,
        }


def test_aiagent_toolsets_allow_explicit_mode(monkeypatch):
    with _bundled_plugin_path():
        from hermes_multitenancy.agent_real import _resolve_enabled_toolsets

        def fake_get_platform_tools(*_args, **_kwargs):
            raise AssertionError("explicit mode must not resolve platform defaults")

        monkeypatch.setenv("HERMES_MULTITENANCY_TOOLSETS_MODE", "explicit")

        assert _resolve_enabled_toolsets(
            {"platform_toolsets": {"feishu": ["feishu_drive", "web"]}},
            "feishu",
            platform_tools_resolver=fake_get_platform_tools,
        ) == ["feishu_drive", "web"]


def test_aiagent_subprocess_payload_carries_router_messages(tmp_path):
    with _bundled_plugin_path():
        from hermes_multitenancy.agent_real import _event_to_subprocess_payload

        event = SimpleNamespace(
            text="send me the link",
            message_id="om_current",
            source=SimpleNamespace(
                platform=SimpleNamespace(value="feishu"),
                chat_id="oc_test",
                user_id="ou_test",
            ),
        )
        messages = [
            {"role": "user", "content": "create a doc"},
            {"role": "assistant", "content": "created doc doxcn123"},
            {"role": "user", "content": "send me the link"},
        ]

        payload = _event_to_subprocess_payload(event, tmp_path, messages=messages)

        assert payload["messages"] == messages


def test_aiagent_session_id_is_stable_and_user_isolated(tmp_path):
    with _bundled_plugin_path():
        from hermes_multitenancy.agent_real import _resolve_aiagent_session_id

        profile_home = tmp_path / "profiles" / "coder"
        event = SimpleNamespace(
            text="hello",
            message_id="om_first",
            source=SimpleNamespace(
                platform=SimpleNamespace(value="feishu"),
                chat_id="oc_test",
                chat_type="dm",
                user_id="ou_fallback",
                message_id="om_source_first",
            ),
        )
        next_event = SimpleNamespace(
            text="next",
            message_id="om_second",
            source=SimpleNamespace(
                platform=SimpleNamespace(value="feishu"),
                chat_id="oc_test",
                chat_type="dm",
                user_id="ou_fallback",
                message_id="om_source_second",
            ),
        )

        first_session = _resolve_aiagent_session_id(event, profile_home, "ou_sender")
        next_session = _resolve_aiagent_session_id(next_event, profile_home, "ou_sender")

        assert first_session == next_session
        assert "profile:coder" in first_session
        assert "chat:oc_test" in first_session
        assert "user:ou_sender" in first_session
        assert first_session != _resolve_aiagent_session_id(
            event,
            profile_home,
            "ou_other",
        )


def test_processing_outcome_fallback_without_gateway_package(monkeypatch):
    with _bundled_plugin_path():
        from hermes_multitenancy.router import _processing_outcome

        monkeypatch.delitem(sys.modules, "gateway.platforms.base", raising=False)

        assert str(_processing_outcome(failed=False)) == "ProcessingOutcome.SUCCESS"
        assert str(_processing_outcome(failed=True)) == "ProcessingOutcome.FAILURE"
