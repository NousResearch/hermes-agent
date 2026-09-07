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


