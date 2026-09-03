"""Advisor runtime — turn tracking, state, review triggers."""

import hashlib
import json
import logging
import os
import queue
import threading
from pathlib import Path

from .advisor_prompt import ADVISOR_SYSTEM_PROMPT
from .models import AdvisorState, Advice, Severity, TurnDelta

logger = logging.getLogger(__name__)

# Env var to skip live reviews (keeps /advisor test path for manual testing)
ADVISOR_NO_REVIEW = "ADVISOR_NO_REVIEW"
WATCHDOG_FILENAME = "WATCHDOG.md"
REVIEW_TIMEOUT_SECONDS = 90
SHUTDOWN_GRACE_SECONDS = 1.0


class AdvisorRuntime:
    """Tracks turns and triggers reviews via the advisor model.

    Wired into Hermes plugin hooks:
      - ``post_llm_call`` — fires once per turn at end, carries full history.
        This is our ``turn_end`` equivalent.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        from hermes_constants import get_hermes_home

        self.state_file = get_hermes_home() / "advisor" / "state.json"
        self.sessions_dir = self.state_file.parent / "sessions"
        self.legacy_state_file = Path(__file__).parent / "state.json"
        self._state_lock = threading.RLock()
        self.state = self._load_state()
        self._session_states: dict[str, AdvisorState] = {}

        # Current turn tracking — populated from post_llm_call kwargs
        self.last_turn_key: tuple[str, str] | None = None
        self._review_queue: queue.Queue[TurnDelta | None] = queue.Queue(maxsize=1)
        self._queue_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._idle = threading.Event()
        self._idle.set()

    # ── state persistence ────────────────────────────────────────────────

    def _load_state(self) -> AdvisorState:
        for path in (self.state_file, self.legacy_state_file):
            try:
                state = AdvisorState.deserialize(json.loads(path.read_text()))
            except FileNotFoundError:
                continue
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                logger.warning("Advisor: could not load state from %s: %s", path, exc)
                continue

            if path == self.legacy_state_file:
                self.state = state
                self._save_state()
            return state
        return AdvisorState(enabled=True)

    def _save_state(self):
        from utils import atomic_json_write

        with self._state_lock:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            atomic_json_write(
                self.state_file,
                {
                    **self.state.serialize(),
                    "held_notes": [],
                },
                indent=2,
                mode=0o600,
            )

    def _session_path(self, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
        return self.sessions_dir / f"{digest}.json"

    def _session_state(self, session_id: str) -> AdvisorState:
        key = session_id or "default"
        with self._state_lock:
            existing = self._session_states.get(key)
            if existing is not None:
                return existing
            try:
                data = json.loads(self._session_path(key).read_text())
                state = AdvisorState.deserialize(data)
            except FileNotFoundError:
                state = AdvisorState(enabled=True)
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                logger.warning("Advisor: could not load session state: %s", exc)
                state = AdvisorState(enabled=True)
            self._session_states[key] = state
            return state

    def _save_session_state(self, session_id: str, state: AdvisorState) -> None:
        from utils import atomic_json_write

        key = session_id or "default"
        with self._state_lock:
            path = self._session_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json_write(
                path,
                {
                    "session_id": key,
                    "held_notes": state.held_notes,
                },
                indent=2,
                mode=0o600,
            )

    def _held_count(self) -> int:
        count = 0
        for path in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                count += len(data.get("held_notes") or [])
            except (json.JSONDecodeError, OSError, TypeError) as exc:
                logger.warning(
                    "Advisor: could not count held notes in %s: %s", path, exc
                )
        return count

    def _clear_held_notes(self) -> None:
        from utils import atomic_json_write

        with self._state_lock:
            # Clear in-memory session states and track which disk files we cover
            cleared_digests: set[str] = set()
            for key, state in self._session_states.items():
                state.held_notes = []
                self._save_session_state(key, state)
                cleared_digests.add(self._session_path(key).name)

            # Catch orphaned session files not yet loaded into memory
            for path in self.sessions_dir.glob("*.json"):
                if path.name in cleared_digests:
                    continue  # already handled above
                try:
                    atomic_json_write(
                        path,
                        {"session_id": "(migrated)", "held_notes": []},
                        indent=2,
                        mode=0o600,
                    )
                except (OSError, TypeError) as exc:
                    logger.warning("Advisor: could not clear %s: %s", path, exc)

    # ── hook: end of each agent turn ─────────────────────────────────────

    def on_post_llm_call(
        self,
        *,
        session_id: str = "",
        turn_id: str = "",
        user_message: str = "",
        assistant_response: str = "",
        conversation_history: list | None = None,
        model: str = "",
        **kwargs,
    ):
        """Fired at end of each turn (tool-calling loop complete).

        This is the Hermes equivalent of pi's ``turn_end`` hook.
        """
        with self._state_lock:
            if not self.state.enabled:
                return
            turn_key = (session_id or "default", turn_id)
            if turn_id and turn_key == self.last_turn_key:
                return
            self.last_turn_key = turn_key
        if os.environ.get(ADVISOR_NO_REVIEW):
            return

        if not conversation_history:
            return

        logger.debug(
            "Advisor: turn %s complete, user=%s, model=%s, msgs=%d",
            turn_id,
            (user_message or "")[:60],
            model or "?",
            len(conversation_history),
        )

        self._enqueue_review(
            TurnDelta(
                session_id=session_id or "default",
                turn_id=turn_id or "",
                user_message=user_message or "",
                assistant_response=assistant_response or "",
                conversation_history=list(conversation_history),
                model=model or "",
            )
        )

    def _enqueue_review(self, turn: TurnDelta) -> None:
        """Queue the newest completed turn without blocking hook dispatch."""
        self._ensure_worker()
        with self._queue_lock:
            self._idle.clear()
            try:
                self._review_queue.put_nowait(turn)
                return
            except queue.Full:
                pass

            try:
                dropped = self._review_queue.get_nowait()
                self._review_queue.task_done()
                if dropped is not None:
                    logger.info(
                        "Advisor: dropped stale queued turn %s in favor of %s",
                        dropped.turn_id,
                        turn.turn_id,
                    )
            except queue.Empty:
                pass
            self._review_queue.put_nowait(turn)

    def _ensure_worker(self) -> None:
        with self._state_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="hermes-advisor",
                daemon=True,
            )
            self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            turn = self._review_queue.get()
            try:
                if turn is None:
                    return
                self._review_turn(turn)
            finally:
                with self._queue_lock:
                    self._review_queue.task_done()
                    if self._review_queue.unfinished_tasks == 0:
                        self._idle.set()

    def _review_turn(self, turn: TurnDelta) -> None:
        try:
            advice_list = self._run_review(
                user_message=turn.user_message,
                assistant_response=turn.assistant_response,
                conversation_history=turn.conversation_history,
                model=turn.model,
                turn_id=turn.turn_id,
                session_id=turn.session_id,
            )
        except Exception as exc:
            logger.warning("Advisor review failed for turn %s: %s", turn.turn_id, exc)
            return

        if not advice_list:
            logger.debug("Advisor: nothing to deliver for turn %s", turn.turn_id)
            return
        self._deliver_advice(advice_list)

    def on_session_finalize(self, **_kwargs) -> None:
        """Give an in-flight review a small, fixed shutdown grace period."""
        if not self._idle.wait(SHUTDOWN_GRACE_SECONDS):
            logger.info(
                "Advisor: review still running after %.1fs shutdown grace; "
                "daemon worker will not delay exit",
                SHUTDOWN_GRACE_SECONDS,
            )

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        """Wait for queued work to finish. Intended for tests and diagnostics."""
        return self._idle.wait(timeout)

    # ── run the review ────────────────────────────────────────────────────

    def _run_review(
        self,
        *,
        user_message: str,
        assistant_response: str,
        conversation_history: list,
        model: str,
        turn_id: str,
        session_id: str,
    ) -> list[Advice]:
        """Build the prompt, call the advisor model, parse the result."""

        with self._state_lock:
            if not self.state.enabled:
                return []
            review_state = self._session_state(session_id)
            messages = self._build_review_prompt(
                user_message=user_message,
                assistant_response=assistant_response,
                conversation_history=conversation_history,
                cwd=os.getcwd(),
                review_state=review_state,
            )
            advisor_model = self.state.model
            advisor_provider = self.state.provider

        # Call the advisor model — use configured override or inherit primary
        kwargs: dict[str, object] = {
            "messages": messages,
            "timeout": REVIEW_TIMEOUT_SECONDS,
        }
        if advisor_model:
            kwargs["model"] = advisor_model
        if advisor_provider:
            kwargs["provider"] = advisor_provider

        result = self.ctx.llm.complete(**kwargs)

        logger.debug(
            "Advisor: review complete, provider=%s model=%s tokens=%d",
            result.provider,
            result.model,
            result.usage.total_tokens if result.usage else 0,
        )

        with self._state_lock:
            if not self.state.enabled:
                return []
            advice = review_state.parse_response(result.text)
            self._save_session_state(session_id, review_state)
            return advice

    def _build_review_prompt(
        self,
        *,
        user_message: str,
        assistant_response: str,
        conversation_history: list,
        cwd: str,
        review_state: AdvisorState | None = None,
    ) -> list[dict]:
        """Build the message list for the advisor model."""

        # Base system prompt
        system_prompt = ADVISOR_SYSTEM_PROMPT

        # Append WATCHDOG.md if present
        watchdog_path = Path(cwd) / WATCHDOG_FILENAME
        if watchdog_path.exists():
            try:
                wd_content = watchdog_path.read_text().strip()
                if wd_content:
                    system_prompt += (
                        f"\n\nEspecially pay attention to:\n"
                        f"<attention>\n{wd_content}\n</attention>"
                    )
            except OSError as exc:
                logger.warning("Advisor: could not read %s: %s", watchdog_path, exc)

        # Build the user content: reconfirm preamble + turn transcript
        user_content_parts = []

        # Reconfirm preamble for held notes
        preamble = (review_state or self.state).format_reconfirm_preamble()
        if preamble:
            user_content_parts.append(preamble)

        # Format the conversation history as a readable turn transcript
        transcript = self._format_history(
            user_message=user_message,
            response=assistant_response,
            history=conversation_history,
        )
        if transcript:
            user_content_parts.append(transcript)

        if not user_content_parts:
            return [{"role": "system", "content": system_prompt}]

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_content_parts)},
        ]

    @staticmethod
    def _format_history(*, user_message: str, response: str, history: list) -> str:
        """Format conversation history into a markdown transcript for review.

        Shows the user prompt and the final assistant response.
        Intermediate tool calls/results are formatted from the history —
        restricted to the current turn (everything after the last user
        message), which is the transcript the reviewer is asked to judge.
        """
        parts = []

        # User message
        if user_message and user_message.strip():
            parts.append(f"#### User\n\n{user_message.strip()}")

        # Build tool-call and result summary from the current turn's slice
        # of the history. The hook carries the full session history; the
        # advisor reviews one turn, so older tool activity is dropped.
        last_user = max(
            (
                i
                for i, msg in enumerate(history)
                if isinstance(msg, dict) and msg.get("role") == "user"
            ),
            default=-1,
        )
        turn_history = history[last_user + 1 :] if last_user >= 0 else history

        tool_calls: list[str] = []
        tool_results: list[str] = []

        for msg in turn_history:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant":
                for tool_call in msg.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function") or {}
                    tc_name = function.get("name") or tool_call.get("name") or "?"
                    tc_args = function.get("arguments", tool_call.get("arguments", {}))
                    if isinstance(tc_args, str):
                        try:
                            tc_args = json.loads(tc_args)
                        except json.JSONDecodeError:
                            pass
                    if isinstance(tc_args, (dict, list)):
                        tc_text = json.dumps(tc_args, ensure_ascii=False, indent=1)
                    else:
                        tc_text = str(tc_args)
                    tool_calls.append(f"\u2192 tool `{tc_name}`: {tc_text[:500]}")

                # Retain compatibility with providers that use content blocks.
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") in {"toolCall", "tool_use"}:
                                tc_name = block.get("name", "?")
                                tc_args = block.get("arguments", block.get("input", {}))
                                tc_str = json.dumps(
                                    tc_args, ensure_ascii=False, indent=1
                                )[:500]
                                tool_calls.append(f"\u2192 tool `{tc_name}`: {tc_str}")
            elif role == "tool":
                if isinstance(content, list):
                    text = " ".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                elif isinstance(content, dict):
                    text = json.dumps(content, ensure_ascii=False)
                else:
                    text = str(content)
                if text.strip():
                    tool_name = msg.get("name") or "tool"
                    tool_results.append(
                        f"\u2192 `{tool_name}` result: {text.strip()[:500]}"
                    )

        if tool_calls:
            parts.append("#### Tool calls\n\n" + "\n".join(tool_calls))
        if tool_results:
            parts.append("#### Tool results\n\n" + "\n".join(tool_results))

        # Assistant response
        if response and response.strip():
            parts.append(f"#### Assistant\n\n{response.strip()}")

        return "\n\n".join(parts)

    # ── deliver advice back to the conversation ──────────────────────────

    def _deliver_advice(self, advice_list: list[Advice]):
        """Inject advice into the active conversation.

        Uses ctx.inject_message() (CLI only). In gateway mode, falls back
        to logging so the user can check /advisor status.
        """
        if not advice_list:
            return

        lines = []
        for a in advice_list:
            lines.append(f"{a.tag()} {a.note}")

        advisory_text = "\n".join(lines)
        full_msg = f"\u25c6 Advisor review\n\n{advisory_text}"

        ok = self.ctx.inject_message(full_msg, role="user")
        if ok:
            logger.info(
                "Advisor: injected %d item(s) into conversation", len(advice_list)
            )
        else:
            logger.info(
                "Advisor: %d item(s) not injected because CLI delivery is unavailable.",
                len(advice_list),
            )

    # ── interactive model selector ───────────────────────────────────────

    def _interactive_select(self) -> str | None:
        """Open Hermes' native provider/model modal for the advisor slot."""

        def apply_selection(result) -> str:
            if not result.success:
                return f"Advisor model selection failed: {result.error_message}"
            with self._state_lock:
                self.state.model = result.new_model
                self.state.provider = result.target_provider
                self._save_state()
            return (
                f"Advisor model set to: {result.new_model} "
                f"({result.provider_label or result.target_provider})"
            )

        with self._state_lock:
            current_provider = self.state.provider
            current_model = self.state.model
        opened = self.ctx.request_model_selection(
            apply_selection,
            current_provider=current_provider,
            current_model=current_model,
        )
        if opened:
            return None
        return (
            "Interactive advisor model selection is available in the Hermes CLI.\n"
            "Use /advisor model <name> and /advisor provider <name> here."
        )

    # ── slash command ─────────────────────────────────────────────────────

    def handle_command(self, args: str) -> str | None:
        """Handle /advisor [on|off|status|model|provider|providers|models|test].

        Keywords match case-insensitively; values (model names, provider
        slugs, test notes) keep the case the user typed.
        """
        raw = args.strip()
        tokens = raw.split(None, 1)
        head = tokens[0].lower() if tokens else ""
        value = tokens[1].strip() if len(tokens) > 1 else ""

        # ── status ──
        if head in ("", "status", "config") and not (head == "config" and value):
            return self._format_status()

        # ── on ──
        if head == "on":
            self.state.enabled = True
            self._save_state()
            return "Advisor on."

        # ── off ──
        if head == "off":
            self.state.enabled = False
            self._clear_held_notes()
            self._save_state()
            return "Advisor off."

        # ── model (no args) — open interactive selector ──
        if head == "model" and not value:
            return self._interactive_select()

        # ── model <name> ──
        if head == "model":
            self.state.model = value
            self._save_state()
            return f"Advisor model set to: {value}"

        # Provider is selected as the first stage of /advisor model.
        if head == "provider" and not value:
            return "Provider selection is part of /advisor model."

        # ── provider <name> ──
        if head == "provider":
            self.state.provider = value
            self._save_state()
            return f"Advisor provider set to: {value}"

        # ── config <key> <value> ──
        if head == "config":
            sub_tokens = value.split(None, 1)
            sub = sub_tokens[0].lower() if sub_tokens else ""
            sub_value = sub_tokens[1].strip() if len(sub_tokens) > 1 else ""
            if sub == "model" and sub_value:
                self.state.model = sub_value
                self._save_state()
                return f"Advisor model set to: {sub_value}"
            if sub == "provider" and sub_value:
                self.state.provider = sub_value
                self._save_state()
                return f"Advisor provider set to: {sub_value}"
            return "Usage: /advisor config <model|provider> <value>"

        # ── providers — list available providers ──
        if head in ("providers", "list-providers"):
            return self._list_providers()

        # ── models [provider] — list models for a provider ──
        if head in ("models", "list-models"):
            return self._list_models(value)

        # ── test — inject a test advice message ──
        if head == "test":
            import re

            m = re.match(r"(nit|concern|blocker)\s+([\s\S]+)$", value, re.IGNORECASE)
            if m:
                sev = Severity(m.group(1).lower())
                note = m.group(2).strip()
                self._deliver_advice([Advice(note=note, severity=sev)])
                return f"Advisor: delivered test {sev.value}."
            return "Usage: /advisor test <nit|concern|blocker> <note>"

        return "Usage: /advisor [on|off|status|config|model|provider|providers|models|test]"

    def _format_status(self) -> str:
        state = "enabled" if self.state.enabled else "disabled"
        model = self.state.model or "(inherit primary)"
        provider = self.state.provider or "(inherit primary)"
        held = self._held_count()
        return (
            f"Advisor {state}.\n"
            f"  model:    {model}\n"
            f"  provider: {provider}\n"
            f"  held:     {held}\n"
            f"Usage: /advisor [on|off|status|config|model|provider|providers|models]"
        )

    def _picker_providers(self) -> list[dict]:
        """Provider rows from the same source the native model picker uses."""
        try:
            from hermes_cli.inventory import build_models_payload, load_picker_context

            with self._state_lock:
                provider = self.state.provider
                model = self.state.model
            context = load_picker_context().with_overrides(
                current_provider=provider,
                current_model=model,
            )
            # No live probing: a slash command listing must not stall on
            # slow or offline custom endpoints. Rows still carry their
            # cached/configured models.
            payload = build_models_payload(context, probe_custom_providers=False)
            return [p for p in (payload.get("providers") or []) if isinstance(p, dict)]
        except Exception as exc:
            logger.warning("Advisor: could not load provider list: %s", exc)
            return []

    def _list_providers(self) -> str:
        providers = self._picker_providers()
        if not providers:
            return (
                "No configured providers found. Add one via hermes setup "
                "or model.provider in config.yaml."
            )
        lines = ["Providers available to /advisor model:"]
        for p in providers:
            slug = p.get("slug") or "?"
            label = p.get("name") or slug
            marker = "  (current)" if p.get("is_current") else ""
            lines.append(f"  {slug}  —  {label}{marker}")
        return "\n".join(lines)

    def _list_models(self, provider_arg: str) -> str:
        target = (provider_arg or "").strip()
        with self._state_lock:
            target = target or self.state.provider or ""
        providers = self._picker_providers()
        if not target:
            if not providers:
                return (
                    "No configured providers found. Add one via hermes setup "
                    "or model.provider in config.yaml."
                )
            return "Usage: /advisor models <provider>\n(run /advisor providers first)"
        row = next(
            (
                p
                for p in providers
                if str(p.get("slug") or "").lower() == target.lower()
            ),
            None,
        )
        if row is None:
            return (
                f"No provider '{target}' found. Run /advisor providers for valid names."
            )
        models = [m for m in (row.get("models") or []) if isinstance(m, str)]
        total = row.get("total_models")
        if not models:
            return (
                f"No models listed for '{target}'. Open /advisor model to pick "
                "interactively, or refresh the model catalog."
            )
        lines = [f"Models for {target}:"]
        shown = sorted(models)[:30]
        for m in shown:
            lines.append(f"  {m}")
        hidden = len(models) - len(shown)
        capped = (total - len(models)) if isinstance(total, int) else 0
        if hidden or capped > 0:
            lines.append(f"  ... and {hidden + max(capped, 0)} more")
        return "\n".join(lines)
