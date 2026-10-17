"""Session-title callback wiring for gateway turn runners."""

import logging

from gateway.run_turn_progress import GatewayTurnProgressMixin, local_viewer_turn

logger = logging.getLogger("gateway.run")


class SessionTitleCallbacks(GatewayTurnProgressMixin):
    def _attach_session_title_callback(self, agent, ctx) -> None:
        """Wire the platform thread-rename lane onto the agent as `_on_session_title`.

        The titler runs in the turn prologue, so attach before the run, not after it.
        """
        try:
            if local_viewer_turn(self):
                return super()._attach_session_title_callback(agent, ctx)
            # Gateway auto-title failures are not user-actionable, so never surface them as messages;
            # overriding the failure sink keeps CLI on _emit_auxiliary_failure while gateway logs debug.
            agent._title_failure_callback = lambda task, exc: logger.debug(
                "Gateway auto-title failure suppressed (not user-visible): %s: %s", task, exc,
            )
            session_id = getattr(agent, "session_id", None)
            runner = self._runner
            source = runner._recover_discord_auto_thread_source(ctx.source, ctx.session_key)
            # Both lanes spend a rate-limited platform call per title, so they use the model's title
            # only (TitleCallback); renaming twice burns Discord's 2-per-10-min budget on a throwaway.
            # Relay Discord predicate is shape-only: whether the connector auto-threaded our reply is
            # only knowable AFTER delivery, so register eagerly and let the rename lane look up the
            # cache at fire time — gating registration on the cache read meant it never registered.
            if runner._is_telegram_topic_lane(source):
                lane = "_schedule_telegram_topic_title_rename"
            elif runner._is_discord_auto_thread_lane(source) or runner._is_relay_discord_channel_lane(source):
                lane = "_schedule_discord_semantic_thread_rename"
            elif getattr(source.platform, "value", source.platform) == "slack" and source.chat_type == "dm" and source.thread_id:
                lane = "_schedule_slack_assistant_thread_title"
            else:
                return
            agent._on_session_title = lambda title, title_source: (
                title_source == "llm" and getattr(runner, lane)(source, session_id, title)
            )
        except Exception:
            logger.debug("Failed to attach session title callback", exc_info=True)
