"""Errors crossing the plugin-host wire keep the builtin class callers catch."""

import json

import pytest

from hermes_cli.plugin_host_wire import PluginHostError, error_payload, raise_remote


def _roundtrip(exc: BaseException):
    raise_remote(json.loads(json.dumps(error_payload(exc))))


def test_a_value_error_subclass_is_still_a_value_error_on_the_other_side():
    try:
        json.loads("{not json")
    except json.JSONDecodeError as exc:
        original = exc
    with pytest.raises(ValueError, match="^JSONDecodeError: Expecting property name"):
        _roundtrip(original)


def test_an_exact_builtin_keeps_its_class():
    with pytest.raises(KeyError):
        _roundtrip(KeyError("k"))


def test_an_unmapped_error_is_a_plugin_host_error():
    class Custom(Exception):
        pass

    with pytest.raises(PluginHostError) as info:
        _roundtrip(Custom("x"))
    assert info.value.type_name == "Custom"


def test_a_runtime_error_subclass_stays_a_plugin_host_error_with_its_name():
    class TransportError(RuntimeError):
        pass

    with pytest.raises(PluginHostError) as info:
        _roundtrip(TransportError("down"))
    assert info.value.type_name == "TransportError"


def test_system_exit_is_not_mapped_to_a_builtin():
    with pytest.raises(PluginHostError):
        _roundtrip(SystemExit(3))
