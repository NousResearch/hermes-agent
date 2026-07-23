from unittest.mock import Mock

from agent.transports.codex_app_server import (
    CodexAppServerClient,
    CodexAppServerError,
    CodexPluginMention,
    CodexPluginSummary,
)
from agent.transports.codex_app_server_session import CodexAppServerSession


def test_inventory_prefers_installed_then_falls_back_flattens_and_filters():
    client = object.__new__(CodexAppServerClient)
    client.request = Mock(
        side_effect=[
            CodexAppServerError(-32601, "Method not found"),
            {
                "marketplaces": [
                    {
                        "plugins": [
                            {
                                "id": "acme/tools",
                                "name": "acme",
                                "installed": True,
                                "enabled": True,
                                "availability": "AVAILABLE",
                                "keywords": ["one", "two"],
                                "interface": {
                                    "displayName": "Acme Tools",
                                    "shortDescription": "Useful",
                                },
                            },
                            {
                                "id": "disabled",
                                "installed": True,
                                "enabled": False,
                                "interface": {},
                            },
                            {
                                "id": "unavailable",
                                "installed": True,
                                "enabled": True,
                                "availability": "UNAVAILABLE",
                                "interface": {},
                            },
                        ]
                    }
                ]
            },
        ]
    )
    plugins = client.list_plugins()
    assert [call.args[0] for call in client.request.call_args_list] == [
        "plugin/installed",
        "plugin/list",
    ]
    assert plugins == [
        CodexPluginSummary("acme/tools", "acme", "Acme Tools", "Useful", ("one", "two"))
    ]
    assert plugins[0].path == "plugin://acme/tools"


def test_inventory_fallback_accepts_structured_error_message():
    client = object.__new__(CodexAppServerClient)
    client.request = Mock(
        side_effect=[
            CodexAppServerError(-32000, {"error": "Unknown method"}),
            [],
        ]
    )

    assert client.list_plugins() == []
    assert [call.args[0] for call in client.request.call_args_list] == [
        "plugin/installed",
        "plugin/list",
    ]


def test_mentions_validate_canonical_name_path_dedupe_and_wire_shape(monkeypatch):
    session = CodexAppServerSession()
    monkeypatch.setattr(
        session,
        "list_plugins",
        Mock(return_value=[CodexPluginSummary("acme/tools", "acme", "Acme Tools")]),
    )
    assert session._validated_plugin_mentions([
        {"name": "acme", "path": "plugin://acme/tools"},
        {"name": "acme", "path": "plugin://acme/tools"},
        {"name": "wrong", "path": "plugin://acme/tools"},
        {"name": "acme", "path": "plugin://unknown"},
        None,
    ]) == [CodexPluginMention("Acme Tools", "plugin://acme/tools")]


def test_mentions_are_idempotent_after_canonicalization(monkeypatch):
    session = CodexAppServerSession()
    monkeypatch.setattr(
        session,
        "list_plugins",
        Mock(return_value=[CodexPluginSummary("acme/tools", "acme", "Acme Tools")]),
    )

    emitted = session._validated_plugin_mentions([
        {"name": "acme", "path": "plugin://acme/tools"}
    ])
    replayed = session._validated_plugin_mentions([
        {"name": mention.name, "path": mention.path} for mention in emitted
    ])

    assert (
        replayed == emitted == [CodexPluginMention("Acme Tools", "plugin://acme/tools")]
    )


def test_turn_start_structured_input_and_plain_compatibility(monkeypatch):
    client = Mock()
    client.request.side_effect = [
        {"turn": {"id": "one"}},
        {"turn": {"id": "two"}},
    ]
    client.take_server_request.return_value = None
    client.take_notification.side_effect = [
        {"method": "turn/completed", "params": {"turn": {"id": "one"}}},
        {"method": "turn/completed", "params": {"turn": {"id": "two"}}},
    ]
    session = CodexAppServerSession()
    session._client = client
    session._thread_id = "thread"
    monkeypatch.setattr(
        session,
        "list_plugins",
        lambda: [CodexPluginSummary("acme/tools", "acme", "Acme Tools")],
    )
    session.run_turn(
        "hello", plugin_mentions=[{"name": "acme", "path": "plugin://acme/tools"}]
    )
    session.run_turn("plain")
    starts = [call.args[1]["input"] for call in client.request.call_args_list]
    assert starts[0] == [
        {"type": "text", "text": "hello"},
        {"type": "mention", "name": "Acme Tools", "path": "plugin://acme/tools"},
    ]
    assert starts[1] == [{"type": "text", "text": "plain"}]
