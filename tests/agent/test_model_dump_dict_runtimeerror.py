"""Cron/agent serialization: model_dump raising RuntimeError/AttributeError (#107967).

After Hermes v0.21, agent-mode cron jobs fail with
``RuntimeError: 'dict' object has no attribute 'model_dump'``. Nested SDK
serializers (or duck objects) raise that from ``.model_dump()``; helpers that
only catch ``TypeError`` let it escape and cron wraps it as the job error.
"""

import pytest

from agent.anthropic_message_convert import _to_plain_data
from agent.chat_completion_helpers import _dump_if_model, _model_dump_safe


class Boom:
    """Duck object whose model_dump reproduces the nested-dict serializer error."""

    def model_dump(self, **kwargs):
        raise RuntimeError("'dict' object has no attribute 'model_dump'")


class AttrBoom:
    """Same seam via AttributeError (pydantic/SDK nested-dict path)."""

    def model_dump(self, **kwargs):
        raise AttributeError("'dict' object has no attribute 'model_dump'")


class Duck:
    def model_dump(self, **kwargs):
        return {"quack": True, "kwargs": bool(kwargs)}


def test_to_plain_data_boom_does_not_raise():
    result = _to_plain_data(Boom())
    assert isinstance(result, dict) or result is None or result == {}


def test_model_dump_safe_boom_does_not_raise():
    _model_dump_safe(Boom())


def test_dump_if_model_boom_does_not_raise():
    _dump_if_model(Boom())


def test_to_plain_data_attr_boom_does_not_raise():
    result = _to_plain_data(AttrBoom())
    assert isinstance(result, dict) or result is None or result == {}


def test_model_dump_safe_attr_boom_does_not_raise():
    _model_dump_safe(AttrBoom())


def test_dump_if_model_already_dict_pass_through():
    payload = {"already": True}
    assert _dump_if_model(payload) == {"already": True}
    assert _dump_if_model(payload) is payload


def test_to_plain_data_already_dict_pass_through():
    payload = {"already": True}
    assert _to_plain_data(payload) == {"already": True}


def test_working_duck_model_dump_still_serializes():
    assert _to_plain_data(Duck())["quack"] is True
    assert _model_dump_safe(Duck())["quack"] is True
    assert _dump_if_model(Duck())["quack"] is True


def test_real_pydantic_model_still_dumps():
    pydantic = pytest.importorskip("pydantic")

    class Item(pydantic.BaseModel):
        name: str
        count: int = 1

    item = Item(name="cron")
    assert _to_plain_data(item)["name"] == "cron"
    assert _model_dump_safe(item)["name"] == "cron"
    assert _dump_if_model(item)["name"] == "cron"
