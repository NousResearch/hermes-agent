"""Regression tests for #112135 — the opencode-go wire must not carry the OpenAI
``name`` field on tool-result rows.

``https://opencode.ai/zen/go/v1`` forwards Chat Completions payloads to an
upstream that answers HTTP 400 ``messages[N]: "name" is not supported by this
endpoint`` as soon as a tool-result row carries ``name`` — which is what
``agent.tool_dispatch_helpers.make_tool_result_message`` stamps on every tool
row (``name`` for the wire, ``tool_name`` for the session DB). The rejected row
stays in history, so every later call in the session dies too.

The strip lives ONCE, role-qualified, in the shared transport
(``chat_completions._sanitize_message``): ``name`` is schema-foreign on
``role: tool`` only, so it must survive on user/assistant rows. These tests pin
the reporter's configuration — the opencode-go *profile* build path, plus the
legacy path — and the invariants a provider-scoped patch would also have owed:
the stored trajectory row is never mutated in place (copy-on-write) and the
content bytes that make up the cached prefix are untouched.
"""

from __future__ import annotations

import copy

import pytest

from agent.transports import get_transport
from providers import get_provider_profile

MODEL = "glm-5.3-flash"


@pytest.fixture
def transport():
    import agent.transports.chat_completions  # noqa: F401
    return get_transport("chat_completions")


@pytest.fixture
def go_profile():
    profile = get_provider_profile("opencode-go")
    assert profile is not None and profile.base_url.endswith("/zen/go/v1")
    return profile


def _tool_result(name: str, content: str, call_id: str) -> dict:
    """Tool row exactly as ``make_tool_result_message`` builds it."""
    return {
        "role": "tool",
        "name": name,
        "tool_name": name,
        "content": content,
        "tool_call_id": call_id,
    }


def _assistant_call(name: str, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}],
    }


def _reported_conversation() -> list:
    """The issue's minimal reproduction transcript."""
    return [
        {"role": "user", "content": "Use the test_echo tool with text=hello.", "name": "sylvain"},
        _assistant_call("test_echo", "call_test1"),
        _tool_result("test_echo", "echoed: hello", "call_test1"),
        {"role": "user", "content": "Now tell me what the tool returned."},
    ]


def _wire(transport, messages, profile=None) -> list:
    params = {"model_lower": MODEL.lower()}
    if profile is not None:
        params["provider_profile"] = profile
    return transport.build_kwargs(MODEL, messages, **params)["messages"]


class TestOpenCodeGoToolResultName:
    """#112135: the opencode-go profile build path sends no ``name`` on tool rows."""

    def test_profile_path_drops_name_on_tool_results(self, transport, go_profile):
        wire = _wire(transport, _reported_conversation(), go_profile)

        assert wire[2] == {"role": "tool", "tool_call_id": "call_test1", "content": "echoed: hello"}
        # ``name`` is the reported 400; ``tool_name`` is the Hermes-internal twin.
        assert "name" not in wire[2] and "tool_name" not in wire[2]
        # Schema-valid elsewhere: the user row keeps it, the tool_call keeps its shape.
        assert wire[0]["name"] == "sylvain"
        assert wire[1]["tool_calls"][0]["function"]["name"] == "test_echo"

    def test_legacy_path_matches_profile_path(self, transport, go_profile):
        """No profile in params (custom/legacy route) must produce the same wire rows."""
        assert _wire(transport, _reported_conversation()) == _wire(
            transport, _reported_conversation(), go_profile
        )

    def test_name_field_is_never_role_widened(self, transport, go_profile):
        """A tool row that ALSO has a user/assistant row with ``name``: only tool loses it."""
        msgs = [
            {"role": "system", "content": "sys", "name": "hermes"},
            {"role": "assistant", "content": "hi", "name": "hermes"},
            _assistant_call("terminal", "call_1"),
            _tool_result("terminal", "ok", "call_1"),
        ]
        wire = _wire(transport, msgs, go_profile)

        assert "name" not in wire[3]
        assert [m.get("name") for m in wire[:2]] == ["hermes", "hermes"]


class TestProtectionFace:
    """No other provider's payload may change: the strip is role-qualified and
    shared, not opencode-go-specific, so no provider can depend on tool-row
    ``name`` (the Chat Completions schema does not define it on ``role: tool``)."""

    @pytest.mark.parametrize(
        "provider, profile",
        [
            ("nvidia", True),
            ("openrouter", False),
            ("", False),
        ],
    )
    def test_other_providers_get_identical_message_rows(self, transport, go_profile, provider, profile):
        expected = _wire(transport, _reported_conversation(), go_profile)
        other = get_provider_profile(provider) if profile else None
        assert _wire(transport, _reported_conversation(), other) == expected

    def test_clean_transcript_is_identity(self, transport, go_profile):
        """Nothing to strip -> the very same message objects go out (no copy churn)."""
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
        out = _wire(transport, msgs, go_profile)
        assert out[0] is msgs[0] and out[1] is msgs[1]


class TestCopyOnWrite:
    """Storage side must be untouched — only the outbound copy loses ``name``."""

    def test_stored_rows_not_mutated(self, transport, go_profile):
        msgs = _reported_conversation()
        before = copy.deepcopy(msgs)

        _wire(transport, msgs, go_profile)

        assert msgs == before
        assert msgs[2]["name"] == "test_echo" and msgs[2]["tool_name"] == "test_echo"

    def test_cached_prefix_content_unchanged(self, transport, go_profile):
        """Cache-relevant bytes: the strip drops a key, never rewrites content."""
        msgs = _reported_conversation()
        wire = _wire(transport, msgs, go_profile)
        assert [m.get("content") for m in wire] == [m.get("content") for m in msgs]
        assert wire[2]["tool_call_id"] == msgs[2]["tool_call_id"]


class TestMultiTurn:
    """Four rounds of tool use: every tool row stays clean and shape-stable."""

    def test_multi_turn_tool_loop(self, transport, go_profile):
        msgs = [{"role": "user", "content": "do the thing"}]
        for turn in range(4):
            call_id = f"call_{turn}"
            msgs.append(_assistant_call("terminal", call_id))
            msgs.append(_tool_result("terminal", f"out {turn}", call_id))
            msgs.append({"role": "user", "content": f"and now step {turn}"})

        wire = _wire(transport, msgs, go_profile)

        tool_rows = [m for m in wire if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_rows] == [f"call_{i}" for i in range(4)]
        assert all(set(m) == {"role", "tool_call_id", "content"} for m in tool_rows)
        # Identical key sets per turn keeps the serialized prefix stable across turns.
        assert len({tuple(sorted(m)) for m in tool_rows}) == 1

    def test_multiline_tool_content_survives(self, transport, go_profile):
        payload = "line1\nline2\n{\"json\": true}\n"
        msgs = [{"role": "user", "content": "run"}, _assistant_call("terminal", "call_x"),
                _tool_result("terminal", payload, "call_x")]
        wire = _wire(transport, msgs, go_profile)
        assert wire[2]["content"] == payload
        assert "name" not in wire[2]
