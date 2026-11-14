"""``/moa``: run one prompt through the default Mixture-of-Agents preset, then restore the prior model.

LOCAL sessions (Desktop, Ink TUI, ACP, ``chat -q``) run their frozen launch route, so the preset is bound
to the admitted turn's execution; every other platform takes the one-shot session override.
"""

from __future__ import annotations

import logging

from agent.i18n import t

logger = logging.getLogger("gateway.run")


class GatewayInboundMoaMixin:
    async def _hm_cmd_moa(self, event, source, _quick_key):
        # /moa is one-shot sugar only: run a single prompt through the default MoA preset, then
        # restore the prior model. To *switch* to a MoA preset for the session, pick it from the
        # model picker (MoA presets surface as a virtual "Mixture of Agents" provider).
        from hermes_cli.moa_config import moa_usage
        from gateway.session_moa_turn import admit_moa_turn, default_preset, moa_override

        moa_payload = event.get_command_args().strip()
        if not moa_payload:
            return True, moa_usage()
        preset = await default_preset()
        from gateway.session_policy import policy_for_source
        policy = policy_for_source(self, source)
        if policy is not None:
            # A local session runs its frozen launch route and never reads the session-state override
            # below, so the preset rides the admitted turn's execution, as the classic CLI's own /moa does.
            from gateway.session_cli_turn import rebind_model_override
            from gateway.session_command_sink import active_sink
            from gateway.session_managed_worker import policy_if_managed
            if active_sink() is not None:
                # slash.exec: the client submits /moa as typed and its turn lands here, unless the
                # session's turns run in a managed worker, which never binds the preset.
                if policy_if_managed(policy) is not None:
                    return True, t("gateway.moa.managed_session")
                return False, None
            if not rebind_model_override(event.message_id, {"model": preset, "provider": "moa"}):
                return True, t("gateway.moa.prepare_failed")
            event.text = moa_payload
            return False, None
        try:
            event.text = moa_payload
            if admit_moa_turn(self, event, source, _quick_key, preset):
                return False, None  # its own admission carries the preset (#146026)
            _moa_state = self._session_state(_quick_key)
            # Same one-shot snapshot `/model --once` uses, so eviction/stop/finalizer settle both alike.
            self._claim_one_turn_restore(_quick_key)
            _moa_state.conversation.model_override = moa_override(preset)
            self._evict_cached_agent(_quick_key)
        except Exception:  # the user gets the prepare-failed notice instead of a turn on the wrong model
            logger.warning("/moa: could not bind the one-shot preset", exc_info=True)
            return True, t("gateway.moa.prepare_failed")
        return False, None
