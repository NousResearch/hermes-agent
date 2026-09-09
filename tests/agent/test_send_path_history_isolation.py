"""Send-path transforms must never write through into persisted history.

The send path builds ``api_messages`` from the persisted conversation
history and then rewrites the copies IN PLACE (tool-call argument
canonicalization/repair, surrogate sanitization, non-ASCII sanitization,
content strips). The build previously used a shallow ``msg.copy()``, which
decouples only top-level fields: nested containers (tool_calls entries and
their function dicts, multimodal content-part lists, reasoning_details)
stayed aliased to the history's objects, so those in-place transforms
silently rewrote the stored transcript. Incident #80498: an unrepairable
``write_file`` argument string was replaced with ``{}`` in the persisted
turn, destroying the streamed file content.

``_clone_message_for_send`` closes the whole class at the chokepoint: it
clones every container while sharing immutable leaves. These tests pin the
invariant CLASS-WIDE — every send-path in-place transform runs over an
adversarial fixture and the history must remain byte-identical — so any
future transform added to the pipeline inherits the guarantee (or fails
here loudly).
"""

import copy
import json

import agent.conversation_loop as cl
from agent.message_sanitization import (
    _sanitize_messages_non_ascii,
    _sanitize_messages_surrogates,
)

TRUNCATED_ARGS = '{"content": "# chapter draft\\nline one'  # unrepairable
VALID_ARGS = json.dumps({"path": "a.txt", "text": "héllo"})
LONE_SURROGATE = "hello \ud83d world"


def _adversarial_history():
    """Every nested container + every dirty-leaf shape the transforms touch."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this " + LONE_SURROGATE},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        },
        {
            "role": "assistant",
            "content": "ok " + LONE_SURROGATE,
            "reasoning_content": "thinking … " + LONE_SURROGATE,
            "reasoning_details": [
                {"type": "reasoning.text", "text": "chaîne " + LONE_SURROGATE}
            ],
            "tool_calls": [
                {
                    "id": "c1" + LONE_SURROGATE,
                    "type": "function",
                    "function": {"name": "write_file", "arguments": TRUNCATED_ARGS},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": VALID_ARGS},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "done"},
    ]


def _api_copy(history):
    """Exactly the send path's build shape."""
    return [cl._clone_message_for_send(m) for m in history]


# Every send-path transform that rewrites api_messages in place. Add new
# transforms here when the pipeline grows — the invariant is class-wide.
def _run_full_pipeline(api_messages):
    for am in api_messages:
        if isinstance(am.get("content"), str):
            am["content"] = am["content"].strip()
    cl._canonicalize_api_tool_calls(api_messages)
    _sanitize_messages_surrogates(api_messages)
    _sanitize_messages_non_ascii(api_messages)


class TestSendPathNeverMutatesHistory:
    def test_full_pipeline_leaves_history_byte_identical(self):
        history = _adversarial_history()
        before = copy.deepcopy(history)

        api_messages = _api_copy(history)
        _run_full_pipeline(api_messages)

        assert history == before, (
            "a send-path transform wrote through the api copy into the "
            "persisted history — the clone is no longer structural"
        )

    def test_each_transform_in_isolation(self):
        transforms = {
            "canonicalize/repair": cl._canonicalize_api_tool_calls,
            "surrogate sanitizer": _sanitize_messages_surrogates,
            "non-ascii sanitizer": _sanitize_messages_non_ascii,
        }
        for name, fn in transforms.items():
            history = _adversarial_history()
            before = copy.deepcopy(history)
            fn(_api_copy(history))
            assert history == before, f"{name} mutated persisted history"

    def test_clone_decouples_every_nested_container(self):
        history = _adversarial_history()
        api = _api_copy(history)

        # Write into every nested container of the copy.
        api[0]["content"][0]["text"] = "MUTATED"
        api[0]["content"][1]["image_url"]["url"] = "MUTATED"
        api[1]["tool_calls"][0]["function"]["arguments"] = "{}"
        api[1]["tool_calls"][1]["id"] = "MUTATED"
        api[1]["reasoning_details"][0]["text"] = "MUTATED"

        assert history[0]["content"][0]["text"].startswith("look at this")
        assert history[0]["content"][1]["image_url"]["url"].startswith("data:")
        assert (
            history[1]["tool_calls"][0]["function"]["arguments"]
            == TRUNCATED_ARGS
        )
        assert history[1]["tool_calls"][1]["id"] == "c2"
        assert history[1]["reasoning_details"][0]["text"].startswith("chaîne")

    def test_clone_shares_immutable_leaves(self):
        """Cost model: containers copied, strings shared (not duplicated)."""
        history = _adversarial_history()
        api = _api_copy(history)
        # Same string object — no byte copy of large payloads.
        assert (
            api[1]["tool_calls"][1]["function"]["arguments"]
            is history[1]["tool_calls"][1]["function"]["arguments"]
        )
        # Different container objects at every level.
        assert api[1] is not history[1]
        assert api[1]["tool_calls"] is not history[1]["tool_calls"]
        assert api[1]["tool_calls"][0] is not history[1]["tool_calls"][0]
        assert (
            api[1]["tool_calls"][0]["function"]
            is not history[1]["tool_calls"][0]["function"]
        )

    def test_non_dict_messages_pass_through(self):
        sentinel = object()
        assert cl._clone_message_for_send(sentinel) is sentinel
        assert cl._clone_message_for_send("plain") == "plain"
        assert cl._clone_message_for_send(None) is None


class TestSendPathBuildIsWiredToTheClone:
    """The api_messages build must actually USE the structural clone.

    The isolation tests above exercise ``_clone_message_for_send`` directly,
    so they cannot notice the build site quietly reverting to the shallow
    ``msg.copy()`` (the mutation that recreates #80498). This AST contract
    pins the wiring: inside ``run_conversation_loop``'s api_messages build,
    the per-message copy expression must be a ``_clone_message_for_send``
    call, and no ``.copy()``-shaped fallback may reappear on the history
    iteration variable.
    """

    def test_history_build_calls_the_clone(self):
        import ast
        import inspect

        import agent.turn_context as tc
        import agent.turn_request_assembly as ra

        # The history build lives in turn_context.build_api_messages; the
        # prefill insert in turn_request_assembly.assemble_api_request. Scan all homes.
        nodes = [
            node
            for mod in (cl, tc, ra)
            for node in ast.walk(ast.parse(inspect.getsource(mod)))
        ]

        clone_calls = []
        shallow_copies = []
        for node in nodes:
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id == "_clone_message_for_send":
                    if (
                        node.args
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id in ("msg", "pfm")
                    ):
                        clone_calls.append(node.args[0].id)
                if (
                    isinstance(fn, ast.Attribute)
                    and fn.attr == "copy"
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id in ("msg", "pfm")
                ):
                    shallow_copies.append(fn.value.id)

        assert "msg" in clone_calls, (
            "the api_messages history build no longer clones via "
            "_clone_message_for_send(msg) — shallow aliasing (#80498) is back"
        )
        assert "pfm" in clone_calls, (
            "the prefill insert no longer clones via "
            "_clone_message_for_send(pfm)"
        )
        assert not shallow_copies, (
            f"shallow .copy() reappeared on send-path message variables: "
            f"{shallow_copies} — nested containers alias the persisted history"
        )
