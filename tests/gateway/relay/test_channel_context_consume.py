"""Unit tests for relay channel-context consumption (design relay-channel-context).

Covers:
  - CapabilityDescriptor.supports_context: default False + JSON round-trip +
    forward-compat (older gateway ignores it / newer connector sends it).
  - _event_from_wire mapping the connector's read-only `context` array into the
    existing MessageEvent.channel_context injection field, and leaving it unset
    (byte-identical to today) when absent/empty/malformed.
  - The trigger text is never affected by context (read-only invariant).

Pure unit tests: no socket, no websockets dependency.
"""

from __future__ import annotations

from gateway.relay.descriptor import CapabilityDescriptor
from gateway.relay.ws_transport import _event_from_wire, _render_relay_context


def _descriptor_kwargs(**overrides):
    base = dict(
        contract_version=1,
        platform="discord",
        label="Discord",
        max_message_length=2000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="discord",
        len_unit="chars",
    )
    base.update(overrides)
    return base


class TestDescriptorSupportsContext:
    def test_defaults_false(self):
        d = CapabilityDescriptor(**_descriptor_kwargs())
        assert d.supports_context is False

    def test_round_trip_true(self):
        d = CapabilityDescriptor(**_descriptor_kwargs(supports_context=True))
        restored = CapabilityDescriptor.from_json(d.to_json())
        assert restored.supports_context is True

    def test_from_json_absent_defaults_false(self):
        # An older connector that never sends the key -> default False.
        payload = (
            '{"contract_version":1,"platform":"discord","label":"Discord",'
            '"max_message_length":2000,"supports_draft_streaming":false,'
            '"supports_edit":true,"supports_threads":true,'
            '"markdown_dialect":"discord","len_unit":"chars"}'
        )
        d = CapabilityDescriptor.from_json(payload)
        assert d.supports_context is False

    def test_from_json_ignores_unknown_keys(self):
        # Forward-compat: a newer connector sending extra keys must not break.
        payload = (
            '{"contract_version":1,"platform":"discord","label":"Discord",'
            '"max_message_length":2000,"supports_draft_streaming":false,'
            '"supports_edit":true,"supports_threads":true,'
            '"markdown_dialect":"discord","len_unit":"chars",'
            '"supports_context":true,"some_future_field":123}'
        )
        d = CapabilityDescriptor.from_json(payload)
        assert d.supports_context is True


class TestRenderRelayContext:
    def test_none_and_empty_return_none(self):
        assert _render_relay_context(None) is None
        assert _render_relay_context([]) is None
        assert _render_relay_context("not a list") is None

    def test_renders_author_and_text_oldest_first(self):
        ctx = [
            {"text": "first", "source": {"user_name": "alice"}},
            {"text": "second", "source": {"user_name": "bob"}},
        ]
        out = _render_relay_context(ctx)
        assert out is not None
        assert "alice: first" in out
        assert "bob: second" in out
        # order preserved (oldest -> newest)
        assert out.index("first") < out.index("second")

    def test_falls_back_to_user_id_then_bare_text(self):
        ctx = [
            {"text": "has id", "source": {"user_id": "u123"}},
            {"text": "no author", "source": {}},
        ]
        out = _render_relay_context(ctx)
        assert "u123: has id" in out
        assert "no author" in out

    def test_skips_malformed_items_without_raising(self):
        ctx = ["not a dict", {"no_text": True}, {"text": "", "source": {}}, 42]
        # Nothing usable -> None, and definitely no exception.
        assert _render_relay_context(ctx) is None


class TestEventFromWireContext:
    def _wire(self, **overrides):
        base = {
            "text": "@bot repeat what they said above",
            "message_type": "text",
            "source": {
                "platform": "discord",
                "chat_id": "chan-1",
                "chat_type": "channel",
                "user_id": "author-1",
            },
            "message_id": "m-100",
        }
        base.update(overrides)
        return base

    def test_context_maps_into_channel_context(self):
        ev = _event_from_wire(
            self._wire(
                context=[
                    {"text": "earlier", "source": {"user_name": "alice"}},
                ]
            )
        )
        assert ev.channel_context is not None
        assert "alice: earlier" in ev.channel_context

    def test_no_context_leaves_channel_context_unset(self):
        ev = _event_from_wire(self._wire())
        assert ev.channel_context is None

    def test_empty_context_leaves_channel_context_unset(self):
        ev = _event_from_wire(self._wire(context=[]))
        assert ev.channel_context is None

    def test_context_does_not_alter_trigger_text(self):
        # Read-only invariant: the addressed text is untouched by context.
        ev = _event_from_wire(
            self._wire(context=[{"text": "noise", "source": {"user_name": "x"}}])
        )
        assert ev.text == "@bot repeat what they said above"
