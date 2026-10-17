"""Compatibility behavior for gateway source delivery to plugin slash handlers."""

import pytest

from hermes_cli.plugin_command_dispatch import invoke_plugin_command


def test_invoke_plugin_command_delivers_source_only_to_keyword_capable_parameters():
    source = object()
    seen = []

    def keyword_only(raw_args, *, source):
        seen.append((raw_args, source))
        return "ok"

    def keyword_capture(raw_args, **kwargs):
        seen.append((raw_args, kwargs["source"]))
        return "ok"

    def positional_only(raw_args, source=None, /):
        return raw_args, source

    def positional_only_with_kwargs(raw_args, source=None, /, **kwargs):
        seen.append((raw_args, kwargs["source"]))
        return "ok"

    assert invoke_plugin_command(keyword_only, "args", source=source) == "ok"
    assert invoke_plugin_command(keyword_capture, "args", source=source) == "ok"
    assert invoke_plugin_command(positional_only, "args", source=source) == ("args", None)
    assert invoke_plugin_command(positional_only_with_kwargs, "args", source=source) == "ok"
    assert seen == [("args", source)] * 3


def test_invoke_plugin_command_keeps_legacy_contract_when_signature_is_opaque(monkeypatch):
    calls = []

    def legacy(raw_args):
        calls.append(raw_args)
        return "legacy"

    monkeypatch.setattr("hermes_cli.plugin_command_dispatch.inspect.signature", lambda _handler: (_ for _ in ()).throw(ValueError()))

    assert invoke_plugin_command(legacy, "args", source=object()) == "legacy"
    assert calls == ["args"]


def test_invoke_plugin_command_does_not_retry_handler_type_error():
    calls = []

    def failing(raw_args, *, source):
        calls.append((raw_args, source))
        raise TypeError("handler body failed")

    source = object()
    with pytest.raises(TypeError, match="handler body failed"):
        invoke_plugin_command(failing, "args", source=source)
    assert calls == [("args", source)]
