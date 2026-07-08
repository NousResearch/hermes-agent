"""Advisor runtime — turn tracking, state, review triggers."""

import json
import logging
import os
import pty
import sys
import termios
from pathlib import Path

from .advisor_prompt import ADVISOR_SYSTEM_PROMPT
from .models import AdvisorState, Advice, Severity

logger = logging.getLogger(__name__)

# Env var to skip live reviews (keeps /advisor test path for manual testing)
ADVISOR_NO_REVIEW = "ADVISOR_NO_REVIEW"
WATCHDOG_FILENAME = "WATCHDOG.md"


class AdvisorRuntime:
    """Tracks turns and triggers reviews via the advisor model.

    Wired into Hermes plugin hooks:
      - ``post_llm_call`` — fires once per turn at end, carries full history.
        This is our ``turn_end`` equivalent.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.state_file = Path(__file__).parent / "state.json"
        self.state = self._load_state()

        # Current turn tracking — populated from post_llm_call kwargs
        self.last_turn_id: str | None = None

    # ── state persistence ────────────────────────────────────────────────

    def _load_state(self) -> AdvisorState:
        try:
            data = json.loads(self.state_file.read_text())
            return AdvisorState.deserialize(data)
        except (FileNotFoundError, json.JSONDecodeError):
            return AdvisorState(enabled=True)

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(self.state.serialize(), indent=2)
        )

    # ── hook: end of each agent turn ─────────────────────────────────────

    def on_post_llm_call(
        self, *,
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
        if not self.state.enabled:
            return
        if os.environ.get(ADVISOR_NO_REVIEW):
            return

        # Avoid re-processing the same turn
        if turn_id and turn_id == self.last_turn_id:
            return
        self.last_turn_id = turn_id

        if not conversation_history:
            return

        logger.debug(
            "Advisor: turn %s complete, user=%s, model=%s, msgs=%d",
            turn_id,
            (user_message or "")[:60],
            model or "?",
            len(conversation_history),
        )

        # Build the review prompt and call the advisor model
        try:
            advice_list = self._run_review(
                user_message=user_message or "",
                assistant_response=assistant_response or "",
                conversation_history=conversation_history or [],
                model=model or "",
                turn_id=turn_id or "",
            )
        except Exception as e:
            logger.warning("Advisor review failed for turn %s: %s", turn_id, e)
            return

        if not advice_list:
            logger.debug("Advisor: nothing to flag for turn %s", turn_id)
            return

        # Deliver advice
        self._deliver_advice(advice_list)

    # ── run the review ────────────────────────────────────────────────────

    def _run_review(
        self,
        *,
        user_message: str,
        assistant_response: str,
        conversation_history: list,
        model: str,
        turn_id: str,
    ) -> list[Advice]:
        """Build the prompt, call the advisor model, parse the result."""

        messages = self._build_review_prompt(
            user_message=user_message,
            assistant_response=assistant_response,
            conversation_history=conversation_history,
            cwd=os.getcwd(),
        )

        # Call the advisor model — use configured override or inherit primary
        kwargs = {"messages": messages, "timeout": 90}
        if self.state.model:
            kwargs["model"] = self.state.model
        if self.state.provider:
            kwargs["provider"] = self.state.provider

        result = self.ctx.llm.complete(**kwargs)

        logger.debug(
            "Advisor: review complete, provider=%s model=%s tokens=%d",
            result.provider, result.model, result.usage.total_tokens if result.usage else 0,
        )

        return self.state.parse_response(result.text)

    def _build_review_prompt(
        self,
        *,
        user_message: str,
        assistant_response: str,
        conversation_history: list,
        cwd: str,
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
            except Exception:
                pass

        # Build the user content: reconfirm preamble + turn transcript
        user_content_parts = []

        # Reconfirm preamble for held notes
        preamble = self.state.format_reconfirm_preamble()
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
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "assistant":
                # Check for tool calls in the message
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "toolCall":
                                tc_name = block.get("name", "?")
                                tc_args = block.get("arguments", {})
                                tc_str = json.dumps(tc_args, indent=1)[:200]
                                tool_calls.append(f"\u2192 tool `{tc_name}`: {tc_str}")
            elif role == "tool":
                if isinstance(content, list):
                    text = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                else:
                    text = str(content)[:300]
                if text.strip():
                    tool_results.append(f"\u2192 result: {text.strip()[:200]}")

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
                "Advisor: %d item(s) \u2014 not in CLI mode, advice stored in state.",
                len(advice_list),
            )

    # ── interactive model/provider selector ──────────────────────────────

    def _interactive_select(self, target: str) -> str:
        """Open the Hermes interactive model/provider selector via
        ``hermes model`` in a subprocess PTY.

        Spawns the real ``hermes model`` command in its own pseudo-terminal
        so its curses UI doesn't conflict with prompt_toolkit's terminal
        state inside the running CLI session.  Captures the user's
        selection from the config diff and applies it to the advisor's
        model/provider override, then restores the primary config.

        ``target`` is ``"model"`` or ``"provider"`` — controls which
        field(s) to write from the picker result.
        """
        if not sys.stdin.isatty():
            return (
                f"Interactive {target} picker requires a terminal.\n"
                f"Use /advisor {target} <name> to set directly."
            )

        from hermes_cli.config import load_config
        from hermes_constants import get_hermes_home

        # Snapshot the original config.yaml so we can restore it after
        config_path = get_hermes_home() / "config.yaml"
        original_config = config_path.read_text() if config_path.exists() else ""

        config_before = load_config()
        old_model = ""
        old_provider = ""
        if isinstance(config_before.get("model"), dict):
            old_model = config_before["model"].get("default", "")
            old_provider = config_before["model"].get("provider", "")

        # Run hermes model in a subprocess with its own PTY.
        # The child process inherits HERMES_HOME so it reads/writes the
        # correct profile config.  pty.spawn() relays terminal I/O
        # transparently — the user sees the curses picker, interacts
        # normally, and when it exits the parent's terminal state is
        # preserved (no prompt_toolkit conflict).
        try:
            exit_code = pty.spawn(["hermes", "model"])
        except FileNotFoundError:
            return "Could not find `hermes` in PATH. Is the venv active?"
        except Exception as e:
            logger.warning("Advisor %s subprocess failed: %s", target, e)
            return f"Error launching model selector: {e}"

        if exit_code != 0:
            logger.info("Advisor model picker exited with code %d", exit_code)
            return "Model selector cancelled or failed."

        # Read what changed in the config
        config_after = load_config()
        new_model = ""
        new_provider = ""
        if isinstance(config_after.get("model"), dict):
            new_model = config_after["model"].get("default", "")
            new_provider = config_after["model"].get("provider", "")

        # Restore the original config so the primary model is unchanged
        if original_config:
            try:
                config_path.write_text(original_config)
            except Exception as e:
                logger.warning("Advisor: failed to restore config: %s", e)

        # Flush stdin buffer — the subprocess's curses may have left
        # stray escape-sequence bytes in the OS input buffer
        try:
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass

        # Flush the output buffer and emit a form feed to prompt_toolkit.
        # The PTY subprocess writes directly to the terminal, so after it
        # exits prompt_toolkit's internal screen state is stale.  \x0c
        # (Ctrl+L / form feed) is the standard signal for prompt_toolkit to
        # redraw the entire application — same thing the user does when they
        # hit Ctrl+L manually, but emitted automatically here.
        try:
            sys.stdout.write('\x0c')
            sys.stdout.flush()
        except Exception:
            pass

        if target == "model":
            if not new_model or new_model == old_model:
                return "Advisor model unchanged."
            self.state.model = new_model
            if new_provider and new_provider != old_provider:
                self.state.provider = new_provider
            self._save_state()
            prov_str = f" ({new_provider})" if new_provider else ""
            return f"Advisor model set to: {new_model}{prov_str}"
        else:
            # target == "provider"
            if not new_provider or new_provider == old_provider:
                return "Advisor provider unchanged."
            self.state.provider = new_provider
            self._save_state()
            return f"Advisor provider set to: {new_provider}"

    # ── slash command ─────────────────────────────────────────────────────

    def handle_command(self, args: str) -> str:
        """Handle /advisor [on|off|status|model|provider|config]."""
        arg = args.strip().lower()

        # ── status ──
        if arg in ("", "status", "config"):
            state = "enabled" if self.state.enabled else "disabled"
            model = self.state.model or "(inherit primary)"
            provider = self.state.provider or "(inherit primary)"
            held = len(self.state.held_notes)
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
            self.state.held_notes = []
            self._save_state()
            return "Advisor off."

        # ── model (no args) — open interactive selector ──
        if arg == "model":
            return self._interactive_select("model")

        # ── model <name> ──
        if arg.startswith("model "):
            model_name = arg[6:].strip()
            if not model_name:
                return "Usage: /advisor model <model-name>"
            self.state.model = model_name
            self._save_state()
            return f"Advisor model set to: {model_name}"

        # ── provider (no args) — open interactive selector ──
        if arg == "provider":
            return self._interactive_select("provider")

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
