"""Behaviour contract for the custom-endpoint ``context_length`` write path.

The desktop panel's "Context" field stores an optional ``context_length``
override on a custom provider (blank / 0 = auto-detect). Saving it must be able
to CLEAR a stored pin: an explicit 0 used to be silently ignored, so the old
value stayed on disk and the panel reloaded it — the field appeared to revert
the instant it was saved.

An *omitted* field is a partial update and must leave the override alone. That
is the same contract the Settings write-back documents in web_server_config.py.
"""

from hermes_cli.web_models import CustomEndpointUpdate
from hermes_cli.web_routers.config_env import _write_custom_endpoint

_BASE_URL = "http://192.168.1.69:11434/v1"
_MODEL = "qwen3:8b"


def _cfg_with_pin() -> dict:
    return {
        "providers": {
            "nasty": {
                "name": "NASty",
                "base_url": _BASE_URL,
                "model": _MODEL,
                "context_length": 131072,
                "models": {_MODEL: {"context_length": 131072}},
            }
        }
    }


def _body(**overrides) -> CustomEndpointUpdate:
    payload = {"id": "nasty", "name": "NASty", "base_url": _BASE_URL, "model": _MODEL}
    payload.update(overrides)
    return CustomEndpointUpdate(**payload)


def test_explicit_zero_clears_the_stored_pin():
    cfg = _cfg_with_pin()

    _write_custom_endpoint(cfg, _body(context_length=0))

    entry = cfg["providers"]["nasty"]
    assert "context_length" not in entry
    assert "context_length" not in entry["models"][_MODEL]


def test_positive_context_length_is_written_to_both_levels():
    cfg = _cfg_with_pin()

    _write_custom_endpoint(cfg, _body(context_length=65536))

    entry = cfg["providers"]["nasty"]
    assert entry["context_length"] == 65536
    assert entry["models"][_MODEL]["context_length"] == 65536


def test_omitted_context_length_leaves_the_pin_untouched():
    cfg = _cfg_with_pin()

    _write_custom_endpoint(cfg, _body())  # context_length defaults to None

    entry = cfg["providers"]["nasty"]
    assert entry["context_length"] == 131072
    assert entry["models"][_MODEL]["context_length"] == 131072
