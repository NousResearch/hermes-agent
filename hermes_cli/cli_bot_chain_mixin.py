"""Interactive CLI bot-chain dispatch and transcript persistence."""

import logging
import os
import time

logger = logging.getLogger(__name__)


class CLIBotChainMixin:
    def _persist_bot_chain_exchange(self, user_message: str, response: str) -> None:
        """Add a completed chain to this CLI session for context and /retry."""
        timestamp = time.time()
        messages = [
            {"role": "user", "content": user_message, "timestamp": timestamp},
            {"role": "assistant", "content": response, "timestamp": timestamp},
        ]
        if self._session_db is not None and self.session_id:
            try:
                # The chain turn never runs this CLI session's own agent, so
                # the lazy session row creation in run_agent never fires.
                # When a $Bot chain is the FIRST turn of a fresh session the
                # messages insert below violates the messages→sessions FK.
                # create_session is an ON CONFLICT upsert, so this is a cheap
                # no-op once the row exists.
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                    model=self.model,
                    model_config={
                        "max_iterations": self.max_turns,
                        "reasoning_config": self.reasoning_config,
                    },
                )
                self._session_db.append_messages_batch(self.session_id, messages)
            except Exception:
                logger.warning("Failed to persist CLI bot-chain exchange", exc_info=True)
        self.conversation_history = [*self.conversation_history, *messages]

    def _try_run_bot_chain(self, message, images: list = None) -> bool:
        """Run a leading ``$Bot`` chain, returning False for ordinary chat."""
        from cli import _cprint, _DIM, _RST, _ACCENT

        if not isinstance(message, str):
            return False

        from agent.bot_chain import (
            BotChainCancelled,
            BotChainControl,
            BotChainError,
            BotChainRunner,
            BotChainSyntaxError,
            format_bot_chain_result,
            format_bot_chain_step,
            parse_bot_chain_message,
        )

        try:
            request = parse_bot_chain_message(message)
        except BotChainSyntaxError as exc:
            _cprint(f"  {_DIM}{exc}{_RST}")
            return True
        if request is None:
            return False
        from tools.bot_mode_probe import _session_source, _internal_or_finite_session

        if _session_source(self.agent) != "cli" or _internal_or_finite_session(self.agent):
            _cprint("  Bot chains require an interactive CLI session.")
            return True
        if images:
            _cprint("  Bot chains currently accept text messages only.")
            return True

        from hermes_cli.bot_profiles import resolve_bot_chain

        rendered_steps: list[str] = []
        response = ""
        try:
            profiles = resolve_bot_chain(request.names)
        except (FileNotFoundError, OSError, ValueError) as exc:
            response = f"Bot chain failed: {exc}"
            _cprint(f"\n{response}")
            self._persist_bot_chain_exchange(message, response)
            return True

        control = BotChainControl(
            on_redirect=lambda payload: self._pending_input.put(payload)
        )
        from hermes_constants import get_hermes_home

        control.source_home = get_hermes_home()
        previous_agent = self.agent
        self.agent = control

        def _show_step(step, index: int, total: int) -> None:
            rendered = format_bot_chain_step(step, final=index == total - 1)
            rendered_steps.append(rendered)
            _cprint(f"\n{_ACCENT}{'─' * 40}{_RST}\n{rendered}")

        try:
            result = BotChainRunner().run(
                profiles,
                request.prompt,
                control=control,
                on_step=_show_step,
            )
            response = format_bot_chain_result(result)
        except BotChainCancelled:
            self._last_turn_interrupted = True
            response = "\n\n".join([*rendered_steps, "Bot chain stopped."])
            _cprint(f"\n{_DIM}Bot chain stopped.{_RST}")
        except BotChainError as exc:
            response = "\n\n".join([*rendered_steps, f"Bot chain failed: {exc}"])
            _cprint(f"\nBot chain failed: {exc}")
        finally:
            if self.agent is control:
                self.agent = previous_agent

        self._persist_bot_chain_exchange(message, response)
        return True

    def _dispatch_chat_turn(
        self,
        message,
        *,
        images: list = None,
        voice_input: bool = False,
        seeded_query: bool = False,
    ) -> None:
        """Dispatch one chat turn after command/file routing is complete.

        ``seeded_query`` documents the literal-input contract at this final
        seam: seeded text bypasses slash/bang/file handlers, but deliberately
        does not bypass the first-class ``$Bot`` chat router.
        """
        from cli import _SeededQueryMessage

        if seeded_query and isinstance(message, _SeededQueryMessage):
            if images is None and message.images:
                images = list(message.images)
            message = message.text
        if self._try_run_bot_chain(message, images=images):
            return
        self.chat(message, images=images, voice_input=voice_input)
