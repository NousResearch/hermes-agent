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
                logger.warning("Advisor: could not count held notes in %s: %s", path, exc)
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
        self, *,
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
        kwargs = {"messages": messages, "timeout": REVIEW_TIMEOUT_SECONDS}
        if advisor_model:
            kwargs["model"] = advisor_model
        if advisor_provider:
            kwargs["provider"] = advisor_provider

        result = self.ctx.llm.complete(**kwargs)

        logger.debug(
            "Advisor: review complete, provider=%s model=%s tokens=%d",
            result.provider, result.model, result.usage.total_tokens if result.usage else 0,
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
    def _format_history(
        *, user_message: str, response: str, history: list
    ) -> str:
        """Format conversation history into a markdown transcript for review.

        Shows the user prompt and the final assistant response.
        Intermediate tool calls/results are formatted from the history.
        """
        parts = []

        # User message
        if user_message and user_message.strip():
            parts.append(f"#### User\n\n{user_message.strip()}")

        # Build tool-call and result summary from history
        tool_calls: list[str] = []
        tool_results: list[str] = []

        for msg in history:
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
                        b.get("text", "") for b in content
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
            logger.info("Advisor: injected %d item(s) into conversation", len(advice_list))
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
        """Handle /advisor [on|off|status|model|provider|config]."""
        arg = args.strip().lower()

        # ── status ──
        if arg in ("", "status", "config"):
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

        # ── on ──
        if arg == "on":
            self.state.enabled = True
            self._save_state()
            return "Advisor on."

        # ── off ──
        if arg == "off":
            self.state.enabled = False
            self._clear_held_notes()
            self._save_state()
            return "Advisor off."

        # ── model (no args) — open interactive selector ──
        if arg == "model":
            return self._interactive_select()

        # ── model <name> ──
        if arg.startswith("model "):
            model_name = arg[6:].strip()
            if not model_name:
                return "Usage: /advisor model <model-name>"
            self.state.model = model_name
            self._save_state()
            return f"Advisor model set to: {model_name}"

        # Provider is selected as the first stage of /advisor model.
        if arg == "provider":
            return "Provider selection is part of /advisor model."

        # ── provider <name> ──
        if arg.startswith("provider "):
            prov_name = arg[9:].strip()
            if not prov_name:
                return "Usage: /advisor provider <provider-name>"
            self.state.provider = prov_name
            self._save_state()
            return f"Advisor provider set to: {prov_name}"

        # ── config <key> <value> ──
        if arg.startswith("config "):
            # Parse "config model <name>" or "config provider <name>"
            parts = arg[7:].strip().split(None, 1)
            if len(parts) != 2:
                return "Usage: /advisor config <model|provider> <value>"
            subkey, value = parts
            if subkey == "model":
                self.state.model = value
                self._save_state()
                return f"Advisor model set to: {value}"
            elif subkey == "provider":
                self.state.provider = value
                self._save_state()
                return f"Advisor provider set to: {value}"
            return "Usage: /advisor config <model|provider> <value>"

        # ── providers — list available providers ──
        if arg in ("providers", "list-providers"):
            lines = ["Configured providers:"]
            # Read custom providers from config
            try:
                from hermes_cli.config import load_config as _load_cfg
                cfg = _load_cfg()
                custom = cfg.get("custom_providers", [])
                for cp in custom:
                    name = cp.get("name", "?")
                    url = cp.get("base_url", "")
                    lines.append(f"  custom:{name}  ({url})")
            except Exception:
                pass
            # Read model catalog for known providers
            try:
                from pathlib import Path
                from hermes_constants import get_hermes_home
                cat_path = get_hermes_home() / "cache" / "model_catalog.json"
                if cat_path.exists():
                    import json
                    cat = json.loads(cat_path.read_text())
                    providers_dict = cat.get("providers", {}) or {}
                    if providers_dict:
                        lines.append("")
                        lines.append("Catalog providers:")
                        for pname in sorted(providers_dict.keys())[:20]:
                            lines.append(f"  {pname}")
                        if len(providers_dict) > 20:
                            lines.append(f"  ... and {len(providers_dict)-20} more")
            except Exception:
                pass
            return "\n".join(lines)

        # ── models [provider] — list models for a provider ──
        if arg.startswith("models") or arg.startswith("list-models"):
            # Parse optional provider argument
            parts = arg.split(None, 1)
            target_provider = (parts[1].strip() if len(parts) > 1
                               else self.state.provider or "")
            lines = []
            try:
                from pathlib import Path
                from hermes_constants import get_hermes_home
                import json
                cat_path = get_hermes_home() / "cache" / "model_catalog.json"
                if cat_path.exists():
                    cat = json.loads(cat_path.read_text())
                    providers_dict = cat.get("providers", {}) or {}
                    if not target_provider:
                        lines.append("Usage: /advisor models <provider>")
                        lines.append("(run /advisor providers first)")
                    else:
                        # Normalize: strip custom: prefix for lookup
                        lookup = target_provider.replace("custom:", "", 1)
                        models = providers_dict.get(lookup, [])
                        if not models:
                            lines.append(f"No models found for '{target_provider}'.")
                            lines.append(f"Check /advisor providers for valid names.")
                        else:
                            lines.append(f"Models for {target_provider}:")
                            for m in sorted(models)[:30]:
                                lines.append(f"  {m}")
                            if len(models) > 30:
                                lines.append(f"  ... and {len(models)-30} more")
                else:
                    lines.append("No model catalog found (run hermes once to populate).")
                    lines.append("Common models for opencode-go:")
                    lines.append("  mimo-v2.5, mimo-v2.5-pro, minimax-m3")
                    lines.append("  deepseek-v4-pro, deepseek-v4-flash")
                    lines.append("  glm-5, glm-5.1, glm-5.2")
                    lines.append("  kimi-k2.6, kimi-k2.7-code")
                    lines.append("  qwen3.6-plus, qwen3.7-max")
            except Exception as e:
                lines.append(f"Error reading model catalog: {e}")
            return "\n".join(lines)

        # ── test — inject a test advice message ──
        if arg.startswith("test"):
            import re
            m = re.match(r"^test\s+(nit|concern|blocker)\s+([\s\S]+)$", arg, re.IGNORECASE)
            if m:
                sev = Severity(m.group(1).lower())
                note = m.group(2).strip()
                self._deliver_advice([Advice(note=note, severity=sev)])
                return f"Advisor: delivered test {sev.value}."
            return "Usage: /advisor test <nit|concern|blocker> <note>"

        return "Usage: /advisor [on|off|status|config|model|provider|providers|models|test]"
