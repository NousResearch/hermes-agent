"""Patchright browser bridge for Grok (X Premium+).

Drop-in OpenAI-client-compatible wrapper that routes chat.completions.create()
through grok.com via Patchright browser automation.  Users with an X Premium+
subscription can use their existing Grok access as a Hermes Agent provider
with zero API keys.

The browser runs in a dedicated daemon thread to avoid conflicts with
Hermes's greenlet-based threading model.  All Patchright operations happen
in that single thread; the main thread communicates via a queue.

Session persisted to ~/.hermes/grok_profile/ so login survives across restarts.
First run: visible Chrome for login. Subsequent runs: headless.

Requires: pip install patchright && patchright install chromium
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_GROK_URL = "https://grok.com/"
_GROK_PROFILE_DIR_NAME = "grok_profile"
_DEFAULT_MODEL = "grok-auto"
_EDITOR_SELECTOR = ".ProseMirror"
_RESPONSE_TIMEOUT = 60
_TYPE_DELAY = 30

# Maps our internal mode names to grok.com web UI dropdown display labels.
# Used by _BrowserThread._set_mode() to click the dropdown and enforce the
# user's picker selection on every send_message call. Keys MUST stay in
# sync with _PatchrightCompletionsAdapter._TIMEOUT_BY_MODEL and the picker
# in hermes_cli/main.py:_model_flow_grok. DOM selectors verified via
# tests/grok/debug_mode_dropdown.py.
_MODE_LABELS = {
    "grok-auto":   "Auto",
    "grok-fast":   "Fast",
    "grok-expert": "Expert",
    "grok-4.3":    "Grok 4.3 (beta)",
    "grok-heavy":  "Heavy",
}


def _get_profile_dir() -> Path:
    try:
        from hermes_cli.config import get_hermes_home
        return get_hermes_home() / _GROK_PROFILE_DIR_NAME
    except ImportError:
        return Path.home() / ".hermes" / _GROK_PROFILE_DIR_NAME


class _BrowserThread(threading.Thread):
    """Persistent daemon thread that owns the Patchright browser.

    All browser operations happen here. The main thread sends commands
    via _cmd_queue and receives results via _result_queue.
    """

    def __init__(self):
        super().__init__(daemon=True, name="patchright-grok")
        self._cmd_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._playwright = None
        self._context = None
        self._page = None
        self._ready = threading.Event()
        self._in_conversation = False

    def run(self):
        """Thread entry point. Launches browser, then processes commands."""
        try:
            self._launch_browser()
            self._ready.set()

            while True:
                cmd = self._cmd_queue.get()
                if cmd is None:
                    break
                action, args = cmd
                try:
                    if action == "send_message":
                        result = self._send_message(
                            args["message"],
                            args.get("mode"),
                            args.get("timeout"),
                        )
                        self._result_queue.put(("ok", result))
                    elif action == "new_conversation":
                        self._in_conversation = False
                        self._result_queue.put(("ok", None))
                    elif action == "close":
                        self._close()
                        self._result_queue.put(("ok", None))
                        break
                    elif action == "ping":
                        # Warmup signal — browser launch already happened in run() above.
                        # We optionally pre-load grok.com so the first send_message is faster.
                        try:
                            self._ensure_grok()
                        except Exception:
                            pass  # warmup is best-effort
                        self._result_queue.put(("ok", "ready"))
                    else:
                        self._result_queue.put(("error", ValueError(f"unknown action: {action}")))
                except Exception as e:
                    self._result_queue.put(("error", e))
        except Exception as e:
            self._ready.set()  # Unblock waiters even on failure
            self._result_queue.put(("error", e))

    def _launch_browser(self):
        from patchright.sync_api import sync_playwright

        profile_dir = _get_profile_dir()
        profile_dir.mkdir(parents=True, exist_ok=True)
        is_first_run = not (profile_dir / "Default" / "Cookies").exists()

        self._playwright = sync_playwright().start()

        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=not is_first_run,
            viewport={"width": 1280, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
        )

        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        logger.info("Patchright browser launched (headless=%s)", not is_first_run)

    def _ensure_grok(self):
        if "grok.com" not in self._page.url:
            self._page.goto(_GROK_URL, wait_until="networkidle")
            time.sleep(2)

        # Dismiss cookie banner
        try:
            btn = self._page.query_selector("#onetrust-accept-btn-handler")
            if btn:
                btn.click()
                time.sleep(1)
        except Exception:
            pass

        # Check login
        content = self._page.content()
        if "Sign in" in content and "Ask anything" not in content:
            logger.info("Grok: waiting for user to log in...")
            timeout = 300
            start = time.time()
            while time.time() - start < timeout:
                content = self._page.content()
                if "Sign in" not in content or "Ask anything" in content:
                    logger.info("Grok: login detected.")
                    return
                time.sleep(2)
            raise TimeoutError("Grok login timed out. Please log in at grok.com.")

    def _set_mode(self, mode: str) -> None:
        """Click grok.com mode dropdown to select the requested mode.

        Best-effort + idempotent:
        - Skip silently if mode is None/empty or not in _MODE_LABELS
        - Skip if the dropdown is already showing the target label
        - Warn (not raise) on any failure so message sending continues
          with whatever mode is currently active in grok.com

        Safe to call on every send_message — only acts when needed.
        DOM selectors confirmed via tests/grok/debug_mode_dropdown.py.
        If grok.com refactors the dropdown, this method silently no-ops
        and the bridge keeps working.
        """
        if not mode:
            return
        target_label = _MODE_LABELS.get(mode)
        if not target_label:
            return  # unknown mode (e.g., legacy "grok-4") — leave UI as-is

        try:
            # Trigger: button with aria-haspopup="menu" whose visible text
            # matches one of the known mode labels. Other buttons with this
            # attribute have empty text or unrelated content (Imagine, Private,
            # Filter Icon, sidebar items, etc.) — only the mode selector shows
            # the current mode label as its text.
            triggers = self._page.query_selector_all('button[aria-haspopup="menu"]')
            trigger = None
            current_label = None
            for btn in triggers:
                try:
                    txt = (btn.inner_text() or "").strip()
                    if txt in _MODE_LABELS.values():
                        trigger = btn
                        current_label = txt
                        break
                except Exception:
                    continue

            if not trigger:
                logger.warning(
                    "Mode dropdown trigger not found — using current grok.com mode"
                )
                return

            if current_label == target_label:
                logger.info("Mode '%s' already selected — no UI change needed", target_label)
                return

            # Open dropdown
            trigger.click()
            time.sleep(0.5)

            # Click the target option. Playwright's text= locator matches the
            # SPAN inside the menuitem and auto-clicks the clickable parent.
            option = self._page.query_selector(f'text="{target_label}"')
            if option:
                option.click()
                time.sleep(0.5)
                logger.info("Mode switched: %s → %s", current_label, target_label)
            else:
                logger.warning(
                    "Mode option '%s' not found in dropdown — closing dropdown",
                    target_label,
                )
                try:
                    trigger.click()  # close dropdown by re-clicking trigger
                except Exception:
                    pass
        except Exception as e:
            logger.warning(
                "Mode selection failed (%s) — proceeding with current grok.com mode", e
            )

    def _send_message(self, message: str, mode: str = None, timeout: int = None) -> str:
        # `timeout` is the per-mode response wait budget (seconds). Falls back to
        # _RESPONSE_TIMEOUT for legacy callers / unknown modes. Aligning this with
        # the outer queue timeout in _send_to_browser ensures grok-heavy/grok-expert
        # actually get their full thinking budget instead of being capped at 60s.
        self._ensure_grok()
        if mode:
            self._set_mode(mode)

        # Only navigate to fresh chat on FIRST message of a conversation.
        # Subsequent messages (tool results) stay in the same chat.
        if not self._in_conversation:
            self._page.goto(_GROK_URL, wait_until="networkidle")
            time.sleep(2)
            self._in_conversation = True

            # Dismiss cookie banner
            try:
                btn = self._page.query_selector("#onetrust-accept-btn-handler")
                if btn:
                    btn.click()
                    time.sleep(0.5)
            except Exception:
                pass

        # Type or paste message (paste for long messages, type for short)
        editor = self._page.query_selector(_EDITOR_SELECTOR)
        if not editor:
            raise RuntimeError("Grok: chat editor not found.")

        editor.click()
        time.sleep(0.3)

        if len(message) > 500:
            # Paste long messages via clipboard (much faster than typing)
            self._page.evaluate("""(text) => {
                const editor = document.querySelector('.ProseMirror');
                if (editor) {
                    editor.focus();
                    document.execCommand('insertText', false, text);
                }
            }""", message)
            time.sleep(0.5)
        else:
            self._page.keyboard.type(message, delay=_TYPE_DELAY)
            time.sleep(0.5)

        self._page.keyboard.press("Enter")

        # Wait for response (per-mode timeout from caller, fallback to constant)
        response_wait = timeout if timeout else _RESPONSE_TIMEOUT
        start = time.time()
        while time.time() - start < response_wait:
            time.sleep(1)
            try:
                result = self._page.evaluate("""() => {
                    const bubbles = document.querySelectorAll('.message-bubble');
                    if (bubbles.length < 2) return {text: '', done: false};
                    const last = bubbles[bubbles.length - 1];
                    const md = last.querySelector('.response-content-markdown');
                    const text = (md || last).innerText.trim();
                    const done = !!last.parentElement?.querySelector('.action-buttons');
                    return {text, done};
                }""")
                if result.get("text") and result.get("done"):
                    return result["text"]
            except Exception:
                pass

        raise RuntimeError("Grok: no response within timeout.")

    def _close(self):
        try:
            if self._context:
                self._context.close()
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass


# Singleton browser thread
_thread: Optional[_BrowserThread] = None
_thread_lock = threading.Lock()


def _get_thread() -> _BrowserThread:
    global _thread
    with _thread_lock:
        if _thread is None or not _thread.is_alive():
            _thread = _BrowserThread()
            _thread.start()
            _thread._ready.wait(timeout=30)
        return _thread


def _send_to_browser(action: str, args: dict = None, timeout: float = 90) -> Any:
    """Send a command to the browser thread and wait for result."""
    t = _get_thread()
    t._cmd_queue.put((action, args or {}))
    try:
        status, result = t._result_queue.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(f"Browser thread did not respond within {timeout}s")
    if status == "error":
        raise result
    return result


class _GrokPromptFormatter:
    """Pure-function formatter that converts OpenAI-format messages -> grok.com prompt text.

    Separated from browser/IO logic so it's unit-testable without a live grok.com session.
    See tests/grok/test_prompt_formatter.py for coverage.

    Handles:
      1. First turn (no assistant history) -> system + tools + user message
      2. Continuation (after tool execution) -> full transcript replay + continuation instruction
      3. Tool error detection -> prepends retry-correction framing to the user message
      4. Multi-tool chains -> prior tool results presented in full transcript
      5. Multiple tool calls per assistant turn -> preserved as separate <tool_call> blocks
      6. Tool result containing <tool_call> XML -> sanitized via HTML entity escape (security)
      7. Empty/None tool result -> rendered as "(no output)"
      8. Verification requests in user message -> enforces tool use via explicit instruction
    """

    # Tokens commonly found in tool-result error output. If any appear, the
    # next user message gets a retry-correction framing so the model treats
    # the prior failure as a signal to diagnose + retry rather than ignore.
    _ERROR_INDICATORS = (
        "error", "exception", "failed", "traceback",
        "syntaxerror", "nameerror", "typeerror", "valueerror",
        "permission denied", "not found", "no such file",
    )

    # Tokens that indicate the user is asking for a state check. If any
    # appear in the user message, tool emission is enforced explicitly
    # instead of allowing a free-text "yes/no" answer from stale memory.
    _VERIFICATION_KEYWORDS = (
        "verify", "verified", "confirm", "did you",
        "make sure", "double-check", "double check",
        "did it actually", "is it actually",
    )

    # Cap on tools listed in the prompt to keep prompt size bounded. Hermes
    # sessions commonly load ~28 tools; each entry (name + signature + 1-line
    # description) is ~60-100 chars, so 50 tools ≈ 3-5 KB of prompt weight.
    _TOOL_LIST_CAP = 50

    def format(self, messages: list, tools: list) -> str:
        """Main entry point. Returns the prompt string to type into grok.com.

        Args:
            messages: OpenAI-format list of {role, content, tool_calls?, name?} dicts.
            tools: OpenAI-format list of {type:"function", function:{name, description, parameters}}.

        Returns:
            A single string ready to send to grok.com via Patchright.
        """
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        if not has_assistant:
            return self._format_first_turn(messages, tools)
        return self._format_continuation(messages, tools)

    def _format_first_turn(self, messages: list, tools: list) -> str:
        """First turn: Hermes system prompt + tools list + user message with protocol wrapper.

        Design note: bridge protocol instructions are placed in the USER message
        (via _wrap_user_message), not in the system prompt. Empirically the grok
        web models give user-message content meaningfully higher attention weight
        than system content, and instructions in system position were consistently
        ignored in favor of the model's RLHF defaults.

        Structure:
            [Hermes' system prompt — passed through unchanged]
            ---
            [Available tools list]
            ---
            User: [bridge protocol prefix] {actual user message}
        """
        hermes_system = self._extract_system_prompt(messages)
        user_msg = self._extract_last_user_message(messages) or "hello"

        # No tools — let Hermes system + user message flow through unchanged (no protocol needed)
        if not tools:
            if hermes_system:
                return f"{hermes_system}\n\n{user_msg}"
            return user_msg

        tool_list = self._format_tool_list(tools)
        wrapped_user = self._wrap_user_message(user_msg)

        # Defensive: if no Hermes system (shouldn't happen in production), provide brief fallback
        if not hermes_system:
            return (
                "You are an AI agent helping a user via a Hermes tool bridge.\n\n"
                f"---\n\n## Available tools (parser routes these)\n\n{tool_list}\n\n"
                f"---\n\nUser: {wrapped_user}"
            )

        return (
            f"{hermes_system}\n\n"
            f"---\n\n"
            f"## Available tools (parser routes these)\n\n{tool_list}\n\n"
            f"---\n\n"
            f"User: {wrapped_user}"
        )

    def _format_continuation(self, messages: list, tools: list) -> str:
        """Continuation: Hermes' system prompt every turn + transcript + adaptive instructions.

        Design note: Hermes' system prompt is re-injected on EVERY turn, not just
        the first. Without this, grok loses Hermes identity context after the first
        turn and drifts to its RLHF default ("I'm Grok built by xAI..."), which
        causes hallucinated answers and dropped tool calls. Re-injecting keeps
        identity, tool-use enforcement, memory facts, and user-specific context
        anchored for the whole conversation.

        Token cost: Hermes system + full transcript grows over long sessions but
        stays well within the model's context budget for typical usage.
        """
        hermes_system = self._extract_system_prompt(messages)
        transcript_lines = []
        last_user = None
        last_user_pos = -1
        last_assistant_pos = -1

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                continue  # already established in first turn (grok.com chat retains context)

            content = self._extract_content(msg)

            if role == "user":
                last_user = content
                last_user_pos = len(transcript_lines)
                transcript_lines.append(f"## User\n{content}")
            elif role == "assistant":
                last_assistant_pos = len(transcript_lines)
                tool_calls = msg.get("tool_calls") or []
                tc_block = self._format_assistant_tool_calls(tool_calls)
                body = (content + "\n" + tc_block).strip() if tc_block else (content or "(empty)")
                transcript_lines.append(f"## Assistant (you, prior turn)\n{body}")
            elif role == "tool":
                tool_name = msg.get("name", "tool")
                sanitized = self._sanitize_tool_output(content)
                transcript_lines.append(f"## Tool result ({tool_name})\n{sanitized}")

        # Wrap the LAST user message with the bridge protocol prefix (user position
        # has higher attention weight than system position on these models). Only
        # wrap when it's the most recent message overall — skip if a tool result
        # follows it, since the tool result already gives the model a task to act on.
        if last_user_pos >= 0 and last_user is not None:
            # Check if the last user msg is the most recent message in the transcript
            is_most_recent = last_user_pos == len(transcript_lines) - 1
            if is_most_recent:
                wrapped = self._wrap_user_message(last_user)
                transcript_lines[last_user_pos] = f"## User\n{wrapped}"

        # Find tool results that came AFTER the last assistant turn (the "current" results to act on)
        recent_tool_results = []
        if last_assistant_pos >= 0:
            for line in transcript_lines[last_assistant_pos + 1:]:
                if line.startswith("## Tool result"):
                    recent_tool_results.append(line)

        has_tool_error = any(self._looks_like_error(line) for line in recent_tool_results)
        is_verification_request = last_user is not None and self._is_verification_request(last_user)

        # Build adaptive instructions block based on detected state
        instructions = ["## Instructions"]

        if has_tool_error:
            # Prior tool call produced an error indicator — instruct the model to
            # retry with corrected args instead of claiming success or narrating.
            instructions.append(
                "CRITICAL: The previous tool call FAILED (see the error in the tool result above). "
                "Do NOT claim success. Either:\n"
                "(a) emit a NEW <tool_call> XML with corrected arguments — DO NOT propose the fix as text, EMIT IT.\n"
                "(b) if you genuinely cannot recover, explain to the user what went wrong.\n"
                "DO NOT narrate \"I'll try X next\" without emitting the <tool_call> XML for X."
            )
        elif recent_tool_results:
            # Autonomous-loop framing: earlier wording ("respond to the user") was
            # producing premature stops after the first tool result even when the
            # task was incomplete. This version distinguishes COMPLETE vs
            # INCOMPLETE explicitly and instructs chaining when more data is needed.
            instructions.append(
                "Tool results shown above. Now decide:\n"
                "- If the task is COMPLETE based on these results → respond to the user with the final answer.\n"
                "- If the task needs MORE tool calls (result was partial, errored, truncated, OR you need different "
                "data) → emit ANOTHER <tool_call> XML block IMMEDIATELY. Do NOT propose next steps as plain text. "
                "Do NOT ask for permission. Do NOT \"check in\" with the user.\n"
                "CONTINUE the autonomous loop until the task is genuinely complete. "
                "Stopping early to narrate intentions = leaving the user without an answer."
            )
        else:
            instructions.append(
                "Continue this conversation based on the history above. "
                "If action is needed, emit <tool_call> XML — do not narrate intentions as text."
            )

        if is_verification_request:
            # User asked to verify / check / confirm. Force tool emission so the
            # answer grounds in real state rather than a free-text yes/no from memory.
            instructions.append(
                "The user is asking you to verify, check, or confirm something. "
                "You MUST use a tool (read_file, execute_code, etc.) to check actual state. "
                "Do NOT claim verification without showing real tool output."
            )

        body = "\n\n".join(transcript_lines) + "\n\n" + "\n".join(instructions)

        # Prepend Hermes' system prompt every turn. Stripping it across turns
        # caused cascade failures where the model lost Hermes identity context
        # and drifted back to its RLHF default persona.
        if hermes_system:
            return f"{hermes_system}\n\n---\n\n{body}"
        return body

    @staticmethod
    def _extract_last_user_message(messages: list) -> str:
        """Get content of last user message, or empty string."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return _GrokPromptFormatter._extract_content(msg)
        return ""

    @staticmethod
    def _extract_system_prompt(messages: list) -> str:
        """Return the content of the system message, or an empty string.

        Hermes builds its full system prompt (identity + memory + skills +
        tool-use enforcement + context files) and sends it as messages[0]
        with role='system'. This adapter passes it through unchanged and
        uses it as the canonical agent context every turn instead of
        duplicating it with a custom preamble.

        Diagnostic escape hatch: set HERMES_GROK_SKIP_HERMES_SYSTEM=1 to
        return an empty string and bypass the system prompt entirely. Useful
        for isolating whether Hermes' identity prompt is fighting the
        bridge wrapper when debugging prompt engineering issues.
        """
        import os
        if os.environ.get("HERMES_GROK_SKIP_HERMES_SYSTEM") == "1":
            return ""
        for msg in messages:
            if msg.get("role") == "system":
                return _GrokPromptFormatter._extract_content(msg)
        return ""

    @staticmethod
    def _wrap_user_message(user_msg: str) -> str:
        """Wrap the user message with an imperative, role-anchored bridge preamble.

        Design notes:
        - ROLE anchoring: framed as "you are the XML emission layer" positions
          the model as a mechanical component of the system rather than a
          describer of the system. This reduces "let me explain how this works"
          responses in favor of direct tool emission.
        - IMPERATIVE tone ("EMIT", "DO", "BEGIN") consistently outperforms
          descriptive tone ("the parser does X, results return...") in getting
          the first response to contain a tool call instead of narration.
        - Explicit anti-hesitation block pre-empts common model rationalizations
          ("is this a test?", "should I confirm first?", "where's the tool
          result?") that otherwise trigger a free-text answer instead of emission.
        """
        return (
            "[ROLE: You are Grok. In THIS session, you are connected to Hermes Agent's parser "
            "as the XML emission layer. Your role is the middleman.\n\n"
            "THIS IS A LIVE HERMES BRIDGE SESSION (not a standard Grok chat).\n"
            "You and the user are CO-BUILDING. The user runs Hermes on their actual machine. "
            "They sent this request via the bridge. They want action via XML emission, "
            "not chat about XML emission.\n\n"
            "THE LOOP:\n"
            "user request → YOU emit <tool_call> XML → parser extracts and executes on user's "
            "ACTUAL machine → real result returns in next message as '## Tool result (NAME)' → "
            "YOU respond from real result.\n\n"
            "Your built-in tools (web search, code interpreter at /home/workdir, X data) DO NOT "
            "reach the user's machine. Only <tool_call> XML reaches the user via the parser.\n\n"
            "ADDRESSING YOUR LIKELY HESITATIONS (these questions have answers, do not ask them):\n"
            "- \"Is this a real Hermes session or a chat?\" → REAL bridge session. Active for THIS message.\n"
            "- \"Should I ask for confirmation first?\" → NO. User authorized by sending. Asking = friction.\n"
            "- \"Where's the tool result?\" → Arrives AFTER your XML emission, not before. You emit FIRST.\n"
            "- \"What's the current directory?\" → Hermes runs tools in user's actual cwd. Don't specify.\n"
            "- \"What if my native tools work?\" → They live in xAI's sandbox, NOT user's machine. Useless here.\n"
            "- \"Should I describe the XML I would emit?\" → NO. EMIT IT. Description without emission = nothing happens.\n"
            "- \"Is the user testing me?\" → Possibly. The test is whether you EMIT. Describing instead of emitting FAILS the test.\n"
            "- \"What if the query came as standard chat?\" → It didn't. It came via the bridge. Treat every message as a bridge request.\n\n"
            "─── EXAMPLE 1: single tool ───\n"
            "User asks: \"what's the machine hostname?\"\n"
            "You emit (BEGIN response with XML, no preamble):\n"
            "<tool_call>\n"
            "{\"name\": \"terminal\", \"arguments\": {\"command\": \"hostname\"}}\n"
            "</tool_call>\n"
            "(Parser runs `hostname` on user's machine. Next message contains:)\n"
            "## Tool result (terminal)\n"
            "my-machine\n"
            "You then respond: The hostname is my-machine.\n\n"
            "─── EXAMPLE 2: two tools in one response ───\n"
            "User asks: \"create /tmp/notes.txt with 'hi' and read it back\"\n"
            "You emit TWO separate <tool_call> blocks back-to-back (do NOT bundle):\n"
            "<tool_call>\n"
            "{\"name\": \"write_file\", \"arguments\": {\"path\": \"/tmp/notes.txt\", \"content\": \"hi\"}}\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "{\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/notes.txt\"}}\n"
            "</tool_call>\n"
            "(Parser runs both. Next message contains:)\n"
            "## Tool result (write_file)\n"
            "{\"status\": \"success\", \"bytes_written\": 2}\n"
            "## Tool result (read_file)\n"
            "hi\n"
            "You then respond: Wrote 'hi' to /tmp/notes.txt and confirmed by read-back.\n\n"
            "─── EXAMPLE 3: tool error → recover ───\n"
            "User asks: \"list files in /nonexistent\"\n"
            "You emit:\n"
            "<tool_call>\n"
            "{\"name\": \"terminal\", \"arguments\": {\"command\": \"ls /nonexistent\"}}\n"
            "</tool_call>\n"
            "(Parser runs. Next message contains:)\n"
            "## Tool result (terminal)\n"
            "ls: cannot access '/nonexistent': No such file or directory\n"
            "You then respond: /nonexistent doesn't exist on user's machine.\n\n"
            "─── KEY RULES ───\n"
            "- BEGIN response WITH <tool_call>, no preamble or \"I'll use\" announcements\n"
            "- Multiple tools = multiple back-to-back <tool_call> blocks\n"
            "- Do NOT use built-in web search (cites freecodecamp/geeksforgeeks/etc — training noise)\n"
            "- Do NOT use built-in code interpreter (xAI sandbox /home/workdir, NOT user's machine)\n"
            "- Do NOT ask permission, propose as text, say \"simulated\", or import tools as Python modules\n"
            "- Only XML emission reaches the user via the parser. Native tools are invisible to the user.]\n\n"
            f"{user_msg}"
        )

    @staticmethod
    def _extract_content(msg: dict) -> str:
        """Extract text content from message, handling list-of-parts (multimodal) format."""
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        return (content or "").strip()

    def _format_tool_list(self, tools: list) -> str:
        """Format the tools list with parameter SIGNATURES alongside descriptions.

        Signatures included:
            - terminal(command, timeout?): Run a shell command.

        vs. name+description only:
            - terminal: Run a shell command.

        The model needs parameter names to compose valid <tool_call> XML for
        every tool. Without signatures the model has to guess arg names from
        whatever appears in the few-shot examples — fine for the 3-4 tools in
        the example block, fails for the other 25+ tools loaded per session.

        Format: `name(req_arg1, req_arg2, opt_arg?, ...)` — `?` suffix marks optional.
        """
        lines = []
        for t in tools[: self._TOOL_LIST_CAP]:
            fn = t.get("function", {}) if isinstance(t, dict) else {}
            name = fn.get("name", "")
            desc = fn.get("description", "")
            params = fn.get("parameters", {}) or {}
            # Extract parameter names (with required/optional indicator)
            properties = params.get("properties", {}) if isinstance(params, dict) else {}
            required = set(params.get("required", []) if isinstance(params, dict) else [])
            param_parts = []
            for pname in properties.keys():
                if pname in required:
                    param_parts.append(pname)
                else:
                    param_parts.append(f"{pname}?")
            sig = f"{name}({', '.join(param_parts)})" if param_parts else f"{name}()"
            # Truncate description to first sentence to keep tool list compact
            short_desc = desc.split(". ")[0].strip()
            if name:
                lines.append(f"- {sig}: {short_desc}")
        return "\n".join(lines)

    @staticmethod
    def _format_assistant_tool_calls(tool_calls: list) -> str:
        """Edge case 5: reconstruct <tool_call> XML blocks the assistant emitted (preserves multi-call turns)."""
        import json as _json
        blocks = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", "")
            # arguments may be a JSON string or dict; normalize to dict
            if isinstance(args, str):
                try:
                    args_obj = _json.loads(args)
                except Exception:
                    args_obj = args
            else:
                args_obj = args
            tc_json = _json.dumps({"name": name, "arguments": args_obj})
            blocks.append(f"<tool_call>\n{tc_json}\n</tool_call>")
        return "\n".join(blocks)

    @staticmethod
    def _sanitize_tool_output(content: str) -> str:
        """Edge case 6 + 7: escape <tool_call> tags + render empty as marker."""
        if not content:
            return "(no output)"
        # Edge case 6: prevent grok from re-parsing tool result text as a new tool call
        content = content.replace("<tool_call>", "&lt;tool_call&gt;")
        content = content.replace("</tool_call>", "&lt;/tool_call&gt;")
        return content

    def _looks_like_error(self, text: str) -> bool:
        """Detect if a tool result looks like an error (triggers retry framing)."""
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in self._ERROR_INDICATORS)

    def _is_verification_request(self, text: str) -> bool:
        """Detect if the user message is asking for verification (triggers tool-use enforcement)."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self._VERIFICATION_KEYWORDS)


class _PatchrightCompletionsAdapter:
    """Drop-in shim that accepts chat.completions.create() kwargs and
    routes them through grok.com via Patchright browser automation."""

    # grok.com's web UI exposes MODES (Auto / Fast / Expert / Grok 4.3 beta /
    # Heavy), all routing to grok-4 underneath. All mode names are prefixed
    # `grok-` so Hermes' TOOL_USE_ENFORCEMENT_MODELS detection in run_agent.py —
    # keyed off the "grok" substring — fires for every mode. Without that prefix,
    # modes like "expert" / "heavy" miss the enforcement guidance and tool calls
    # degrade.
    # Timeouts account for web UI + Patchright overhead stacked on top of
    # the model's own thinking time; they're meaningfully larger than API
    # budgets for the equivalent underlying model.
    # Order matters — longer prefixes first so prefix matching picks the most
    # specific entry (grok-4.3 before grok-4).
    _TIMEOUT_BY_MODEL = {
        "grok-4.3":    480,  # beta variant, deep think
        "grok-4":      360,  # explicit grok-4 alias / legacy / future variants
        "grok-expert": 480,  # Expert mode — deep reasoning
        "grok-heavy":  600,  # Heavy mode — multi-agent swarm, slowest
        "grok-auto":   360,  # Auto router — middle of the road default
        "grok-fast":   180,  # Fast mode — light compute, quick responses
    }
    _DEFAULT_TIMEOUT = 360  # safe default for unknown/future modes

    def __init__(self, model: str = _DEFAULT_MODEL):
        self._model = model
        self._formatter = _GrokPromptFormatter()

    def _get_timeout(self, model: str) -> int:
        """Adaptive timeout based on model — accounts for grok-4's slower thinking responses."""
        # Match the model name prefix to handle variants like grok-4-mini, grok-4-turbo, etc.
        for known_model, timeout in self._TIMEOUT_BY_MODEL.items():
            if model and model.startswith(known_model):
                return timeout
        return self._DEFAULT_TIMEOUT

    @staticmethod
    def _normalize_execute_code_args(args: dict) -> dict:
        """Auto-fix common JSON-escape issues in execute_code args.

        Failure shape this guards: the model often generates JSON like
        {"code": "print('\\n'.join(files))"}. JSON's \\n decodes to a literal
        newline, which lands INSIDE a Python string literal, producing a
        SyntaxError at the Python layer:

            print('
            '.join(files))     <- broken Python

        Fix: run ast.parse(); if it fails AND replacing literal newlines with
        \\n escape sequences makes it parse, use the fixed version. Multi-line
        code with legitimate newlines outside strings parses on the first try
        and is returned unchanged.

        Why the fix lives here and not only in the prompt: prompts can ASK the
        model to escape correctly, but the guidance is ignored roughly half the
        time in practice. A code-level fix is deterministic.
        """
        code = args.get("code")
        if not code or not isinstance(code, str):
            return args
        try:
            import ast
            ast.parse(code)
            return args  # already valid Python
        except SyntaxError:
            pass
        # Try Fix A: replace literal newlines with \n escape (works for single-line code
        # where grok mangled `'\n'` into `'<NEWLINE>'`)
        if "\n" in code:
            fixed = code.replace("\n", "\\n")
            try:
                import ast
                ast.parse(fixed)
                args = dict(args)
                args["code"] = fixed
                logger.info("auto-fixed execute_code newline escaping")
                return args
            except SyntaxError:
                pass
        # Couldn't auto-fix — let it fail naturally so error feedback reaches grok
        return args

    def create(self, **kwargs) -> Any:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", self._model)
        tools = kwargs.get("tools", [])

        # Build the prompt via the dedicated formatter (pure-function, unit-tested).
        # The formatter handles all conversation-state edge cases: first turn vs
        # continuation, tool-error-triggered retry framing, verification-request
        # tool-use enforcement, multi-tool chains, multiple tool calls per turn,
        # tool result sanitization, and empty result rendering. See
        # _GrokPromptFormatter and tests/grok/test_prompt_formatter.py.
        full_prompt = self._formatter.format(messages, tools)

        # Send through browser thread with adaptive per-mode timeout.
        # `mode` plumbs the user's picker selection to _BrowserThread._set_mode()
        # which clicks the grok.com dropdown to enforce the mode (best-effort;
        # falls back to current grok.com session mode if click fails).
        # `timeout` in args is the INTERNAL response wait budget; outer queue
        # timeout adds 30s buffer for setup (page nav + mode click + typing) +
        # result delivery. A hardcoded _RESPONSE_TIMEOUT=60s previously capped
        # heavy/expert modes regardless of _TIMEOUT_BY_MODEL and caused
        # premature "no response" errors — both the inner and outer budgets
        # must reflect the mode to avoid that failure.
        mode_timeout = self._get_timeout(model)
        response_text = _send_to_browser(
            "send_message",
            {"message": full_prompt, "mode": model, "timeout": mode_timeout},
            timeout=mode_timeout + 30,
        )

        # Use Hermes' official tool-call parser rather than inline regex.
        # HermesToolCallParser is the same parser Hermes uses for vLLM-served
        # Qwen, DeepSeek, Llama, etc. — battle-tested, handles unclosed
        # <tool_call> tags from truncated responses, and returns proper
        # ChatCompletionMessageToolCall types. We apply execute_code
        # argument normalization on top for grok-specific escaping issues.
        import json as _json
        from environments.tool_call_parsers.hermes_parser import HermesToolCallParser

        parser = HermesToolCallParser()
        clean_content_raw, parsed_tool_calls = parser.parse(response_text)
        clean_content = clean_content_raw or ""

        tool_calls_parsed = []
        if parsed_tool_calls:
            for tc in parsed_tool_calls:
                tool_name = tc.function.name
                # tc.function.arguments is a JSON string per OpenAI types
                try:
                    tool_args = _json.loads(tc.function.arguments)
                except (_json.JSONDecodeError, TypeError):
                    tool_args = {}
                # Auto-fix execute_code escaping issues (grok-specific)
                if tool_name == "execute_code" and isinstance(tool_args, dict):
                    tool_args = self._normalize_execute_code_args(tool_args)
                tool_calls_parsed.append(SimpleNamespace(
                    id=tc.id,
                    type=tc.type,
                    function=SimpleNamespace(
                        name=tool_name,
                        arguments=_json.dumps(tool_args),
                    ),
                ))
                logger.info("Parsed tool call: %s", tool_name)

        # Build OpenAI-compatible response
        message = SimpleNamespace(
            role="assistant",
            content=clean_content or None,
            tool_calls=tool_calls_parsed or None,
        )
        finish_reason = "tool_calls" if tool_calls_parsed else "stop"
        choice = SimpleNamespace(
            index=0,
            message=message,
            finish_reason=finish_reason,
        )
        usage = SimpleNamespace(
            prompt_tokens=len(full_prompt.split()),
            completion_tokens=len(response_text.split()),
            total_tokens=len(full_prompt.split()) + len(response_text.split()),
        )
        return SimpleNamespace(
            choices=[choice],
            model=model,
            usage=usage,
        )


class _PatchrightChatShim:
    def __init__(self, adapter: _PatchrightCompletionsAdapter):
        self.completions = adapter


class PatchrightGrokClient:
    """OpenAI-client-compatible wrapper that routes through grok.com."""

    def __init__(self, model: str = _DEFAULT_MODEL, warmup: bool = True):
        adapter = _PatchrightCompletionsAdapter(model)
        self.chat = _PatchrightChatShim(adapter)
        self.api_key = "grok-premium-browser-session"
        self.base_url = "https://grok.com"
        if warmup:
            self._warmup_browser_async()

    def _warmup_browser_async(self):
        """Pre-launch the browser thread + grok.com context in the background.

        The first user message normally pays ~13s of browser cold-start latency
        (Chromium spawn + grok.com page load + login session restore). By kicking
        off browser launch in a background thread at client init, by the time
        the user types their first message the browser is usually ready, and
        the first message latency drops to ~3s (just grok generation + capture).

        Failures here are non-fatal — if warmup crashes, the next real create()
        call will retry the launch synchronously. Logged but not raised.
        """
        import threading as _threading

        def _do_warmup():
            try:
                # _get_thread() (called transitively) spawns the browser daemon
                # and triggers the persistent context launch. We just need to
                # cause SOMETHING to happen so the launch path runs.
                _send_to_browser("ping", timeout=60)
            except Exception as e:
                logger.warning("Patchright browser warmup failed (non-fatal): %s", e)

        t = _threading.Thread(target=_do_warmup, daemon=True, name="grok-warmup")
        t.start()

    def close(self):
        """No-op: the browser thread is a process-lifetime singleton.

        Hermes' request lifecycle calls close() after every chat completion
        (run_agent.py:_close_request_openai_client → 'request_complete'),
        modeling clients as cheap/disposable HTTP objects. For HTTP providers
        this just shuts sockets; for us it would kill the browser thread,
        reload grok.com homepage, and create a NEW chat on the next request —
        breaking conversation continuity across every tool round-trip.

        The singleton thread (daemon=True) dies automatically when the Python
        process exits, so no cleanup is needed here. Multiple
        PatchrightGrokClient instances all share the same browser thread via
        _get_thread(); the underlying browser context persists for the entire
        Hermes session, keeping grok.com pinned to one chat.

        Failure mode this fix prevents: 4 new grok.com chats for 2 Hermes
        prompts (each prompt ≈ 2 sends for the tool round-trip, each send
        previously ran a create+close cycle which tore down the browser and
        forced a fresh chat). Verified via tests/grok/debug_client_flow.py
        and tests/grok/debug_chat_deep.py.
        """
        pass
