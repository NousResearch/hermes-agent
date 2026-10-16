"""Slack model picker action dispatch, split from the platform adapter."""

import logging

from agent.i18n import t

try:
    from .block_kit import sanitize_blocks
except ImportError:  # pragma: no cover - plugin loaded outside package context
    from block_kit import sanitize_blocks  # type: ignore

logger = logging.getLogger("plugins.platforms.slack.adapter")

# Model picker Block Kit action IDs. The picker is a two-step drill-down:
# provider static_select → model static_select, plus Back/Cancel buttons.
_MODEL_PICKER_PROVIDER_ACTION = "hermes_model_provider"
_MODEL_PICKER_MODEL_ACTION = "hermes_model_model"
_MODEL_PICKER_BACK_ACTION = "hermes_model_back"
_MODEL_PICKER_CANCEL_ACTION = "hermes_model_cancel"
# Rendered when a live-looking picker message can no longer resolve (gateway
# restart, aged-out state entry, or a value the stored state no longer
# covers): the message is rewritten to this so the control visibly dies.


def _model_picker_expired_notice() -> str:
    return t("platform.shared.model_picker_expired")


_MODEL_PICKER_ACTION_IDS = (
    _MODEL_PICKER_PROVIDER_ACTION,
    _MODEL_PICKER_MODEL_ACTION,
    _MODEL_PICKER_BACK_ACTION,
    _MODEL_PICKER_CANCEL_ACTION,
)


async def handle_model_picker_action(self, ack, body, action) -> None:
    """Handle a model picker Block Kit interaction.

    Dispatches on the action_id: provider static_select advances to the
    model stage, model static_select runs ``on_model_selected``, Back
    returns to the provider stage, Cancel dismisses the picker.
    """
    await ack()

    team_id = self._event_team_id({}, body)
    action_id = action.get("action_id", "")
    message = body.get("message", {})
    msg_ts = message.get("ts", "")
    channel_id = body.get("channel", {}).get("id", "")
    user_name = body.get("user", {}).get("name", "unknown")
    user_id = body.get("user", {}).get("id", "")

    if not self._enterprise_interaction_allowed(str(team_id), str(user_id), str(channel_id)):
        return
    if not self._enterprise_write_allowed(str(channel_id), {"slack_team_id": str(team_id)}):
        return

    if not self._is_interactive_user_authorized(
        user_id,
        channel_id=channel_id,
        user_name=user_name,
        team_id=team_id,
    ):
        logger.warning(
            "[Slack] Unauthorized model picker click by %s (%s) - ignoring",
            user_name, user_id,
        )
        return

    # Look up the picker state. The send path may have stored it under a
    # bare ts (metadata-poor send, no team id) while this click event
    # carries a team id — that mismatch must not swallow a legitimate
    # interaction (mirrors _handle_approval_action's dual-key lookup).
    marker = self._workspace_message_marker(team_id, msg_ts)
    if msg_ts in self._model_picker_state:
        marker = msg_ts
    state = self._model_picker_state.get(marker)
    if not state:
        logger.debug("[Slack] Model picker state not found for marker=%s", marker)
        # Gateway restarted or the entry aged out of the bounded dict —
        # there is no gateway-side registry to fall back on, so this
        # dict is the picker's only state. Kill the live-looking
        # control visibly instead of silently swallowing clicks
        # (mirrors the clarify handler's expiry notice).
        await self._update_picker_message(
            channel_id, team_id, msg_ts, _model_picker_expired_notice()
        )
        return

    providers = state.get("providers", [])

    # Cancel → dismiss.
    if action_id == _MODEL_PICKER_CANCEL_ACTION:
        self._model_picker_state.pop(marker, None)
        await self._update_picker_message(
            channel_id, team_id, msg_ts, t("platform.slack.picker.cancelled")
        )
        return

    # Provider selected → advance to model stage. The option value is a
    # list index into the stored providers slice (never the raw slug —
    # custom slugs can exceed Slack's 75-char option value cap).
    if action_id == _MODEL_PICKER_PROVIDER_ACTION:
        selected = action.get("selected_option", {})
        idx_token = selected.get("value", "")
        try:
            idx = int(idx_token)
            provider = providers[idx] if idx >= 0 else None
        except (ValueError, IndexError, TypeError):
            provider = None
        if provider is None:
            # Message and stored state are out of sync (stale payload,
            # re-seeded entry) — the picker can no longer resolve, so
            # kill it visibly like the expiry path.
            logger.warning("[Slack] Invalid provider picker index token: %r", idx_token)
            self._model_picker_state.pop(marker, None)
            await self._update_picker_message(
                channel_id, team_id, msg_ts, _model_picker_expired_notice()
            )
            return
        provider_slug = provider.get("slug", "")
        if not provider.get("models"):
            await self._update_picker_message(
                channel_id, team_id, msg_ts,
                t("platform.slack.picker.no_models_for_provider", provider=provider_slug),
            )
            self._model_picker_state.pop(marker, None)
            return

        state["stage"] = "model"
        state["selected_provider_slug"] = provider_slug
        blocks = self._build_model_picker_model_blocks(providers, provider_slug)
        try:
            await self._get_client(channel_id, team_id=team_id or None).chat_update(
                channel=channel_id,
                ts=msg_ts,
                text=t("platform.slack.picker.fallback_model", provider=provider.get('name', provider_slug)),
                blocks=sanitize_blocks(blocks),
            )
        except Exception:
            logger.exception("[Slack] Failed to update model picker (provider→model)")
        return

    # Back → return to provider stage.
    if action_id == _MODEL_PICKER_BACK_ACTION:
        state["stage"] = "provider"
        state["selected_provider_slug"] = ""
        try:
            from hermes_cli.providers import get_label
            provider_label = get_label(
                state.get("current_provider", "")
            )
        except Exception:
            logger.exception("[Slack] Failed to resolve model picker provider label")
            provider_label = state.get("current_provider", "")
        blocks = self._build_model_picker_provider_blocks(
            providers, state.get("current_model", ""), provider_label
        )
        try:
            await self._get_client(channel_id, team_id=team_id or None).chat_update(
                channel=channel_id,
                ts=msg_ts,
                text=t("platform.slack.picker.fallback_provider"),
                blocks=sanitize_blocks(blocks),
            )
        except Exception:
            logger.exception("[Slack] Failed to update model picker (back)")
        return

    if action_id == _MODEL_PICKER_MODEL_ACTION and state.get("stage") == "model":
        await _handle_model_selection(self, action, state, providers, marker, channel_id, team_id, msg_ts)


async def _handle_model_selection(self, action, state, providers, marker, channel_id, team_id, msg_ts) -> None:
    """Resolve a model option and invoke the selected-model callback once."""
    on_model_selected = state.get("on_model_selected")
    selected = action.get("selected_option", {})
    idx_token = selected.get("value", "")
    provider_slug = state.get("selected_provider_slug", "")
    provider = next((p for p in providers if p["slug"] == provider_slug), None)
    models = (provider or {}).get("models", [])
    try:
        idx = int(idx_token)
        model_id = models[idx] if idx >= 0 else None
    except (ValueError, IndexError, TypeError):
        model_id = None
    if model_id is None:
        # Message and stored state are out of sync — kill the picker
        # visibly instead of leaving a dead control.
        logger.warning("[Slack] Invalid model picker index token: %r", idx_token)
        self._model_picker_state.pop(marker, None)
        await self._update_picker_message(
            channel_id, team_id, msg_ts, _model_picker_expired_notice()
        )
        return

    if not on_model_selected:
        self._model_picker_state.pop(marker, None)
        await self._update_picker_message(
            channel_id, team_id, msg_ts, _model_picker_expired_notice()
        )
        return

    # Pop the state up-front (double-click guard, mirrors approval).
    self._model_picker_state.pop(marker, None)
    await self._update_picker_message(
        channel_id, team_id, msg_ts, t("platform.slack.picker.switching", model=model_id)
    )

    switch_failed = False
    try:
        confirmation = await on_model_selected(
            state["chat_id"], model_id, provider_slug
        )
        # The gateway reports a failed in-place swap as a localized
        # error-prefixed return string, not an exception (#50163).
        # Compare against the same i18n prefix so both failure
        # shapes get the failed header.
        try:
            _error_prefix = t("gateway.model.error_prefix", error="").strip()
        except Exception:
            logger.exception("[Slack] Failed to resolve model picker error prefix")
            _error_prefix = "Error:"
        if _error_prefix and str(confirmation).startswith(_error_prefix):
            switch_failed = True
    except Exception as exc:
        logger.error("[Slack] Model picker callback failed: %s", exc, exc_info=True)
        confirmation = t("platform.slack.picker.switch_failed", error=str(exc))
        switch_failed = True

    header = t("platform.slack.picker.switch_failed_header" if switch_failed
               else "platform.slack.picker.switched_header")
    await self._update_picker_message(
        channel_id, team_id, msg_ts, f"{header}\n\n{confirmation}"
    )
    return
