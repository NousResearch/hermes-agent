"""``pre_tool_call`` dispatch accepts and forwards the principal identity kwargs.

The directive resolver has an explicit keyword signature, so a new payload key that is not
declared there is a ``TypeError`` for every caller — not a silently dropped field.
"""

from __future__ import annotations

from hermes_cli.plugins import PluginManager, _dispatch_pre_tool_call_hooks

_IDENTITY = dict(
    user_id="x", user_name="Alice", platform="discord", chat_id="123", chat_type="channel",
    thread_id="456", gateway_session_key="agent:main:discord:channel:123",
)


def test_dispatch_forwards_identity_and_keeps_narrow_legacy_callbacks_working(monkeypatch):
    wide_seen: dict = {}
    narrow_seen: list = []

    def wide_hook(tool_name, args, **kwargs):
        wide_seen.update(kwargs)

    def narrow_hook(tool_name, args, task_id):  # pre-identity signature: must not break
        narrow_seen.append((tool_name, task_id))

    manager = PluginManager()
    manager._discovered = True
    manager._hooks = {"pre_tool_call": [wide_hook, narrow_hook]}
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    block_message, modified_args = _dispatch_pre_tool_call_hooks("terminal", {"command": "ls"}, task_id="t", **_IDENTITY)

    assert (block_message, modified_args) == (None, None)
    assert narrow_seen == [("terminal", "t")]
    for key, value in _IDENTITY.items():
        assert wide_seen[key] == value, key


def test_identity_defaults_to_empty_strings_when_caller_omits_it(monkeypatch):
    seen: dict = {}
    manager = PluginManager()
    manager._discovered = True
    manager._hooks = {"pre_tool_call": [lambda tool_name, args, **kwargs: seen.update(kwargs)]}
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)

    _dispatch_pre_tool_call_hooks("terminal", {"command": "ls"}, task_id="t")

    assert {k: seen[k] for k in _IDENTITY} == {k: "" for k in _IDENTITY}
