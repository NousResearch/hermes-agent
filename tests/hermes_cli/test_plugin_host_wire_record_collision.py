"""Generic records must not decode as memory wire records by name alone."""

import dataclasses
import json

from agent.memory_provider import MemoryObservation, MemoryPrefetchResult
from hermes_cli.plugin_host_wire import Record, decode, encode


def test_generic_records_named_like_memory_types_round_trip():
    for name in ("MemoryObservation", "MemoryPrefetchResult", "MemoryObservationTuple", "MemoryObservationPayload"):
        cls = dataclasses.make_dataclass(name, [("value", str)])
        result = decode(json.loads(json.dumps(encode(cls("kept")))))
        assert isinstance(result, Record)
        assert result.value == "kept"
        model = type(name, (), {"model_dump": lambda self: {"value": "kept"}})()
        result = decode(json.loads(json.dumps(encode(model))))
        assert isinstance(result, Record)
        assert result.value == "kept"


def test_actual_memory_records_still_round_trip():
    observation = MemoryObservation(source_kind="test", schema="test", version=1, payload={"x": 1})
    result = decode(json.loads(json.dumps(encode(MemoryPrefetchResult(context="hi", observations=(observation,))))))
    assert isinstance(result, MemoryPrefetchResult)
    assert isinstance(result.observations[0], MemoryObservation)
    assert result.observations[0].payload["x"] == 1
