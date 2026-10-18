"""Plugin slash-command handlers receive only the calling-session context they declare."""

from hermes_cli.plugin_command_dispatch import invoke_plugin_command

CONTEXT = {"source": "src", "session_id": "sid", "session_key": "key", "platform": "telegram"}


def test_handlers_get_only_the_context_they_declare():
    def legacy(raw_args):
        return (raw_args,)

    def named(raw_args, *, session_id, platform):
        return raw_args, session_id, platform

    def catch_all(args, **kwargs):
        return args, kwargs

    def positional_only(raw_args, session_id=None, /):  # cannot be passed by keyword: no opt-in
        return raw_args, session_id

    assert invoke_plugin_command(legacy, "a", **CONTEXT) == ("a",)
    assert invoke_plugin_command(named, "a", **CONTEXT) == ("a", "sid", "telegram")
    assert invoke_plugin_command(catch_all, "a", **CONTEXT) == ("a", CONTEXT)
    assert invoke_plugin_command(positional_only, "a", **CONTEXT) == ("a", None)
