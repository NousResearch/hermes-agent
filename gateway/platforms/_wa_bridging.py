"""WhatsApp self-send bridge mixin — fork-only, not in upstream.

When a WhatsApp user messages their own number (self-chat), WhatsApp
suppresses phone notifications for replies. This mixin intercepts the
outbound-send path and redirects to the Telegram home channel instead.

**How it plugs in**

GatewayRunner inherits from WABridgeMixin (one extra base class). The
mixin overrides a set of virtual hook/resolver methods that GatewayRunner
calls at key points:

  _pre_agent(event, source)              before _handle_message_with_agent
  _on_agent_error(event, source)         when _handle_message_with_agent raises
  _post_agent(event, source, result)     after goal-continuation — returns None
                                         to signal "WA bridge handled delivery"
  _bridged(source, adapter, chat_id)     redirect every outbound send to TG
  _bridge_session_key_for_source(src)    approval session key for bridge channel
  _register_bridge_approval(bk, ak)     link TG session → WA approval queue
  _unregister_bridge_approval(bk)        clean up at end of turn
  _apply_bridge_progress_context(...)    strip WA threading before TG redirect
  _apply_bridge_status_target(...)       redirect status/interim msgs to TG

**Config** (in profile's config.yaml):

  platforms:
    whatsapp:
      whatsapp_bridging: true

Requires a live Telegram adapter with a home channel configured.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from hermes_cli.event import MessageEvent
    from hermes_cli.session_source import SessionSource

logger = logging.getLogger(__name__)


class WABridgeMixin:
    """Redirects WA self-chat replies to the Telegram home channel.

    All WA bridge state and logic live here, keeping gateway/run.py free of
    platform-specific bridge code. Remove this from GatewayRunner's base list
    to retire the feature.
    """

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def _init_wa_bridge_state(self) -> None:
        """Initialise per-instance bridge state. Called from GatewayRunner.__init__."""
        # Key: bridge (Telegram) session_key → Value: WA session_key.
        # Maps the TG session to the WA session's pending-approval queue so
        # /approve and /deny typed in Telegram resolve the right WA session.
        self._wa_bridge_approval_redirects: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Bridge detection
    # ------------------------------------------------------------------

    def _whatsapp_bridge_target(self, source: "SessionSource"):
        """Return (wa_adapter, tg_adapter, tg_chat_id) if bridging applies, else None."""
        try:
            from hermes_cli.platforms import Platform
        except ImportError:
            return None
        if source.platform != Platform.WHATSAPP or not source.user_id:
            return None
        if source.user_id != source.chat_id:
            return None
        wa_cfg = self.config.platforms.get(Platform.WHATSAPP)  # type: ignore[attr-defined]
        if not wa_cfg or not wa_cfg.extra.get("whatsapp_bridging"):
            return None
        wa_adapter = self.adapters.get(Platform.WHATSAPP)  # type: ignore[attr-defined]
        tg_adapter = self.adapters.get(Platform.TELEGRAM)  # type: ignore[attr-defined]
        tg_home = self.config.get_home_channel(Platform.TELEGRAM)  # type: ignore[attr-defined]
        if not wa_adapter or not tg_adapter or not tg_home:
            return None
        return wa_adapter, tg_adapter, tg_home.chat_id

    # ------------------------------------------------------------------
    # _handle_message lifecycle hooks
    # ------------------------------------------------------------------

    async def _pre_agent(self, event: "MessageEvent", source: "SessionSource") -> None:
        """Send 🕶️ ack reaction on WA self-chat before agent processing begins."""
        _wa_bridge = self._whatsapp_bridge_target(source)
        if not _wa_bridge:
            return
        wa_adapter, _, _ = _wa_bridge
        try:
            await wa_adapter.send_reaction(source.chat_id, event.message_id, "\U0001F576")  # 🕶️
        except Exception:
            logger.debug("WhatsApp bridging: failed to send ack reaction", exc_info=True)

    async def _on_agent_error(self, event: "MessageEvent", source: "SessionSource") -> None:
        """Send ❌ reaction on WA self-chat when agent processing raises."""
        _wa_bridge = self._whatsapp_bridge_target(source)
        if not _wa_bridge:
            return
        wa_adapter, _, _ = _wa_bridge
        try:
            await wa_adapter.send_reaction(source.chat_id, event.message_id, "❌")
        except Exception:
            logger.debug("WhatsApp bridging: failed to send error reaction", exc_info=True)

    async def _post_agent(self, event: "MessageEvent", source: "SessionSource", agent_result: Any) -> Optional[Any]:
        """Relay final response to Telegram home channel + update WA reaction.

        Returns None to signal that the WA bridge handled delivery — the caller
        (GatewayRunner._handle_message) returns None to its own caller, which is
        correct: the response was sent to Telegram, not back to WhatsApp.
        """
        _wa_bridge = self._whatsapp_bridge_target(source)
        if not _wa_bridge:
            return agent_result

        wa_adapter, tg_adapter, tg_chat_id = _wa_bridge
        _ack_emoji = "❌"
        _already_sent_to_tg = bool(
            isinstance(agent_result, dict) and agent_result.get("already_sent")
        )
        # Recompute final_text from agent_result (avoids moving the _final_text
        # block in _handle_message, keeping that function identical to upstream).
        if isinstance(agent_result, dict):
            final_text = str(agent_result.get("final_response") or "")
        elif isinstance(agent_result, str):
            final_text = agent_result
        else:
            final_text = ""

        if final_text.strip():
            if _already_sent_to_tg:
                # The stream consumer / interim-assistant callback was already
                # redirected to Telegram during the run and delivered this text
                # — avoid sending it twice.
                _ack_emoji = "\U0001F44D"  # 👍
            else:
                _quote = (event.text or "").strip()
                if len(_quote) > 500:
                    _quote = _quote[:500] + "…"
                _bridge_text = f"[WhatsApp] {_quote}\n\n{final_text}"
                try:
                    _tg_result = await tg_adapter.send(tg_chat_id, _bridge_text)
                    if getattr(_tg_result, "success", False):
                        _ack_emoji = "\U0001F44D"  # 👍
                except Exception:
                    logger.exception("WhatsApp bridging: Telegram delivery failed")
        else:
            # Nothing to relay — clear the "still working" reaction.
            _ack_emoji = "\U0001F44D"  # 👍

        try:
            await wa_adapter.send_reaction(source.chat_id, event.message_id, _ack_emoji)
        except Exception:
            logger.debug("WhatsApp bridging: failed to send result reaction", exc_info=True)

        return None  # signal: bridge handled delivery; caller returns None

    # ------------------------------------------------------------------
    # _handle_message_with_agent run-level hooks
    # ------------------------------------------------------------------

    def _bridged(self, source: "SessionSource", default_adapter: Any, default_chat_id: str) -> tuple:
        """Resolve outbound (adapter, chat_id) — redirects to Telegram on WA self-chat.

        Called at every outbound send site in _handle_message_with_agent so that
        tool progress, status updates, heartbeats, approval prompts, and interim
        messages all go to Telegram rather than the silent WA self-chat.
        """
        _wa_bridge = self._whatsapp_bridge_target(source)
        if _wa_bridge:
            _, tg_adapter, tg_chat_id = _wa_bridge
            return tg_adapter, tg_chat_id
        return default_adapter, default_chat_id

    def _bridge_session_key_for_source(self, source: "SessionSource") -> Optional[str]:
        """Return the Telegram home-channel session key when WA bridging is active, else None.

        Used to register the cross-channel /approve and /deny redirect so commands
        typed in the Telegram home channel resolve the WA session's approval queue.
        """
        _wa_bridge = self._whatsapp_bridge_target(source)
        if not _wa_bridge:
            return None
        _, _, tg_chat_id = _wa_bridge
        try:
            from hermes_cli.platforms import Platform
            from hermes_cli.session_source import SessionSource as _SS
        except ImportError:
            return None
        return self._session_key_for_source(  # type: ignore[attr-defined]
            _SS(platform=Platform.TELEGRAM, chat_id=tg_chat_id, chat_type="dm")
        )

    def _register_bridge_approval(self, bridge_session_key: Optional[str], approval_session_key: str) -> None:
        """Link bridge session key → WA approval queue so /approve in Telegram works."""
        if bridge_session_key:
            self._wa_bridge_approval_redirects[bridge_session_key] = approval_session_key

    def _unregister_bridge_approval(self, bridge_session_key: Optional[str]) -> None:
        """Remove the cross-channel approval redirect at end of turn."""
        if bridge_session_key:
            self._wa_bridge_approval_redirects.pop(bridge_session_key, None)

    def _resolve_approval_session_key(self, session_key: str) -> str:
        """Remap session key through the WA bridge approval-redirect table.

        Called by /approve and /deny handlers when the primary session key has
        no blocking approval — checks whether a WA self-chat session was
        bridged and its pending approval is actually under a different key.
        """
        redirect_key = self._wa_bridge_approval_redirects.get(session_key)
        if redirect_key:
            try:
                from tools.approval import has_blocking_approval
                if has_blocking_approval(redirect_key):
                    return redirect_key
            except ImportError:
                pass
        return session_key

    def _apply_bridge_progress_context(
        self,
        source: "SessionSource",
        thread_id: Any,
        metadata: Any,
        reply_to: Any,
    ) -> tuple:
        """Strip WA-specific threading targets when progress is redirected to Telegram.

        Telegram home channel is a flat DM — WhatsApp thread/reply context makes
        no sense there and can cause errors.
        """
        if self._whatsapp_bridge_target(source):
            return None, None, None
        return thread_id, metadata, reply_to

    def _apply_bridge_status_target(
        self,
        source: "SessionSource",
        status_adapter: Any,
        status_chat_id: str,
        status_metadata: Any,
    ) -> tuple:
        """Redirect status/interim messages to Telegram home channel on WA self-chat."""
        _wa_bridge = self._whatsapp_bridge_target(source)
        if _wa_bridge:
            _, tg_adapter, tg_chat_id = _wa_bridge
            return tg_adapter, tg_chat_id, None
        return status_adapter, status_chat_id, status_metadata
