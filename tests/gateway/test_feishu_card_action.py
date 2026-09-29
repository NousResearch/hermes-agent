"""Card-action dispatch must not use the callback token as a reply target.

Feishu card-action callbacks carry an opaque callback id (c-...) in event.token,
not an open_message_id (om-...). Routing the synthetic event with that token as
its message_id makes the eventual reply target an invalid open_message_id and
every send fails with Feishu error 99992354 ("not a valid open_message_id").
"""

import asyncio
from types import SimpleNamespace


def _make_card_action_data(token: str) -> SimpleNamespace:
    return SimpleNamespace(
        event=SimpleNamespace(
            token=token,
            context=SimpleNamespace(open_chat_id="oc_1"),
            operator=SimpleNamespace(open_id="ou_1"),
            action=SimpleNamespace(tag="button", value={"probe": 1}),
        )
    )


class TestCardActionMessageId:
    def _dispatch(self, token: str) -> dict:
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter

        adapter = FeishuAdapter(PlatformConfig())
        captured: dict = {}

        async def _capture(**kwargs) -> None:
            captured.update(kwargs)

        adapter._dispatch_synthetic_event = _capture
        asyncio.run(adapter._handle_card_action_event(_make_card_action_data(token)))
        return captured

    def test_callback_token_is_not_used_as_message_id(self):
        """A c-... callback id must never become the reply target (Feishu 99992354)."""
        captured = self._dispatch("c-1730000000000")
        assert captured["message_id"] is None

    def test_open_message_id_passes_through(self):
        """A real om-... id is kept so replies can quote the source message."""
        captured = self._dispatch("om_a1b2c3")
        assert captured["message_id"] == "om_a1b2c3"

    def test_missing_token_yields_no_message_id(self):
        """No token at all also means no reply target (no synthetic uuid fallback)."""
        captured = self._dispatch("")
        assert captured["message_id"] is None

    def test_synthetic_text_carries_action(self):
        """Sanity: the button click still routes as a /card command."""
        captured = self._dispatch("om_a1b2c3")
        assert captured["text"].startswith("/card button")
