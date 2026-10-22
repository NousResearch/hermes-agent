"""Native task inputs (``images``, ``output_schema``) on the public subagent lifecycle API (#138828).

Only the model conversation is stubbed. Launch, child input attachment, the native worker
(``_run_child_lifecycle`` -> ``_run_single_child``), first-turn image plumbing and the one-shot schema repair all run
for real.
"""

import base64
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.subagent_lifecycle import (
    _REGISTRY,
    SubagentLaunchRequest,
    SubagentLifecycleError,
    SubagentLifecycleService,
    SubagentState,
)
from hermes_cli.plugin_host_wire import decode, encode

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="
)
ADDRESS_SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}


class ScriptedChild:
    """A child whose only fake part is the model: ``run_conversation`` replays scripted answers."""

    tool_progress_callback = None
    _delegate_saved_tool_names: list = []
    _credential_pool = None
    _delegate_role = "leaf"
    _delegate_depth = 1
    _parent_subagent_id = None
    provider = "test"
    model = "test-model"
    session_prompt_tokens = session_completion_tokens = session_reasoning_tokens = 0
    session_estimated_cost_usd = 0.0

    def __init__(self, ident, responses):
        self._subagent_id = ident
        self.responses = list(responses)
        self.turns = []

    def get_activity_summary(self):
        return {"api_call_count": 1, "max_iterations": 5, "current_tool": None}

    def run_conversation(self, user_message, task_id=None, **_kwargs):
        self.turns.append(user_message)
        return {"final_response": self.responses.pop(0), "completed": True, "api_calls": 1, "messages": []}

    def close(self):
        return None


class Harness:
    """Launches through the real service; ``wire=True`` first roundtrips the request over the plugin-host wire."""

    def __init__(self, service):
        self.service, self.builds, self.children, self.answers, self.wire = service, [], [], [], False

    def build(self, **kwargs):
        child = ScriptedChild(f"sa-inputs-{len(self.builds)}", self.answers)
        self.builds.append(kwargs)
        self.children.append(child)
        return child

    def launch(self, request, *answers):
        self.answers = list(answers)
        handle = self.service.launch(decode(encode(request)) if self.wire else request)
        return self.service, (self.children[-1] if self.children else None), (self.builds[-1] if self.builds else None), handle


@pytest.fixture
def harness(monkeypatch):
    parent = SimpleNamespace(
        session_id="parent-inputs", enabled_toolsets=["file"], _current_task_id=None, _delegate_depth=0,
        _touch_activity=lambda _desc: None,
    )
    h = Harness(SubagentLifecycleService(lambda: parent))
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", h.build)
    return h


def _finish(service, handle):
    assert service.wait(handle, timeout_seconds=10).state is SubagentState.SUCCEEDED
    return service.result(handle)


@pytest.mark.parametrize("wire", [False, True])
def test_images_reach_the_childs_first_turn_natively(harness, tmp_path, wire):
    harness.wire = wire
    shot = tmp_path / "shot.png"
    shot.write_bytes(_PNG)
    url = "https://example.invalid/mock.png"
    with patch("agent.image_routing.decide_image_input_mode", return_value="native"):
        service, child, _, handle = harness.launch(
            SubagentLaunchRequest(goal="Review the supplied mock", images=(str(shot), url)), "looks fine")
        assert _finish(service, handle).summary == "looks fine"
    (first,) = child.turns
    assert first[0]["type"] == "text" and "Review the supplied mock" in first[0]["text"]
    sent = [part["image_url"]["url"] for part in first if part["type"] == "image_url"]
    assert url in sent and any(u.startswith("data:image/png;base64,") for u in sent)
    assert sum(part["type"] == "image_url" for part in first) == 2  # native image parts, one per source
    assert service.result(handle).structured_payload is None  # no schema requested -> no receipt


def test_legacy_request_keeps_text_only_first_turn_and_no_payload(harness):
    service, child, build, handle = harness.launch(SubagentLaunchRequest(goal="plain goal", context="ctx"), "done")
    result = _finish(service, handle)
    assert child.turns == ["plain goal"] and build["context"] == "ctx"
    assert result.structured_payload is None
    assert not getattr(child, "_delegate_output_schema", None) and not getattr(child, "_delegate_images", None)


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("answers, valid, turns, summary", [
    (['{"city": "Berlin"}'], True, 1, '{"city": "Berlin"}'),
    (["not json", '{"city": "Oslo"}'], True, 2, '{"city": "Oslo"}'),
    (["nope", "still nope"], False, 2, "still nope"),
])
def test_output_schema_validates_with_one_repair_and_returns_receipt(harness, wire, answers, valid, turns, summary):
    harness.wire = wire
    service, child, build, handle = harness.launch(
        SubagentLaunchRequest(goal="produce the address", context="base", output_schema=ADDRESS_SCHEMA), *answers)
    result = _finish(service, handle)
    assert "OUTPUT CONTRACT" in build["context"] and build["context"].startswith("base")
    assert len(child.turns) == turns  # native single repair turn, never a second child
    assert len(harness.builds) == 1
    assert result.summary == summary  # raw final answer retained, even when the contract was missed
    payload = result.structured_payload
    assert payload["schema_valid"] is valid
    assert set(payload) <= {"schema_valid", "schema_errors", "schema_note"}
    assert bool(payload.get("schema_errors")) is (not valid) and ("schema_note" in payload) is (not valid)
    if turns == 2:
        assert "rejected" in child.turns[1]
    wired = decode(encode(result))
    assert wired.structured_payload == payload and wired.summary == summary and wired.terminal_state == "SUCCEEDED"


@pytest.mark.parametrize("bad", [
    {"images": 5},
    {"images": ("a.png", "")},
    {"images": tuple(f"{i}.png" for i in range(9))},
    {"output_schema": 5},
    {"output_schema": "not json"},
    {"output_schema": {"type": "no-such-type"}},
    {"output_schema": {"enum": [{1, 2}]}},  # not JSON-serializable
    {"output_schema": ADDRESS_SCHEMA, "context": "c" * 32_001},  # existing caller-context limit remains
])
def test_malformed_inputs_are_rejected_before_any_child_is_built(harness, bad):
    before = dict(_REGISTRY.records)
    with pytest.raises(SubagentLifecycleError):
        harness.launch(SubagentLaunchRequest(goal="g", **bad), "unused")
    assert harness.builds == [] and _REGISTRY.records == before


@pytest.mark.parametrize("wire", [False, True])
@pytest.mark.parametrize("schema,context", [
    ({"description": "x" * 20_000}, "caller facts"),
    ({"type": "object", "description": "x" * 9_000}, "c" * 25_000),
    ({"type": "object", "$defs": {"unused": {"const": [{}] * 2600}}}, "facts " * 1000),
], ids=["large-schema", "large-contract", "expanded-schema"])
def test_native_accepted_schemas_do_not_gain_lifecycle_size_limits(harness, wire, schema, context):
    from tools.delegate_tool_tasks import _coerce_task_schemas
    from tools.delegation_output_schema import append_output_contract

    native, error = _coerce_task_schemas([{"goal": "g", "context": context, "output_schema": schema}], None)
    assert error is None and native == [schema]
    harness.wire = wire
    service, child, build, handle = harness.launch(
        SubagentLaunchRequest(goal="g", context=context, output_schema=schema), "{}")
    assert child._delegate_output_schema == schema
    assert build["context"] == append_output_contract(context, child._delegate_output_schema)
    assert _finish(service, handle).structured_payload == {"schema_valid": True}


def test_schema_is_snapshotted_at_launch(harness):
    schema = json.loads(json.dumps(ADDRESS_SCHEMA))
    service, child, _, handle = harness.launch(SubagentLaunchRequest(goal="g", output_schema=schema), '{"city": "x"}')
    schema["required"].append("zip")
    assert child._delegate_output_schema == ADDRESS_SCHEMA
    assert _finish(service, handle).structured_payload == {"schema_valid": True}
