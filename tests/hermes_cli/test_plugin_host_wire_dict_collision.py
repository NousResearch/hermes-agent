import json

from hermes_cli.plugin_host_wire import decode, encode


def test_dictionary_record_markers_round_trip():
    for value in (
        {"__record__": "MemoryObservationPayload", "fields": {"x": 1}, "other": "lost"},
        {"__record__": "MemoryObservation", "fields": {"__record__": "nested"}},
        {"__record__": "generic:dict", "fields": []},
    ):
        assert type(decode(json.loads(json.dumps(encode(value))))) is dict
        assert decode(json.loads(json.dumps(encode(value)))) == value
