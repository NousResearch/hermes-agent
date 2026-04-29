"""
Hermes Tool Sanitizer — Open WebUI Filter Function
Strips large tool call JSON from Hermes SSE events to prevent
DOM bloat, markdown re-render storms, and silent hangs from
oversized response.completed events.

Deploy: Open WebUI → Admin Settings → Functions → + Add Function
        Paste this entire file, set type to "Filter", Save.
        Toggle "Global" ON, toggle the activation switch ON.
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional
import json


class Filter:
    class Valves(BaseModel):
        beautify: bool = Field(
            default=True,
            description="Use emoji-prefixed readable summaries instead of raw JSON",
        )
        max_arg_chars: int = Field(
            default=500,
            description="Max characters to keep per tool argument (non-beautify mode)",
        )
        debug: bool = Field(default=False)

    # ── Tools known to produce large JSON arguments ──
    HEAVY_TOOLS = {
        "write_file", "read_file", "web_search", "web_extract",
        "search_files", "patch", "terminal", "execute_code",
    }

    # ── Emoji formatters for beautify mode (tool arguments → one-liner) ──
    FORMATTERS = {
        "web_search":      lambda a: f'🔍 {a.get("query","")[:80]}',
        "write_file":      lambda a: f'💾 {a.get("path","?")} ({len(a.get("content",""))/1024:.1f} KB)',
        "read_file":       lambda a: f'📖 {a.get("path","?")}',
        "terminal":        lambda a: f'💻 {a.get("command","")[:80]}',
        "web_extract":     lambda a: f'📄 {str(a.get("urls",["?"])[0])[:50]}',
        "search_files":    lambda a: f'🔎 "{a.get("pattern","")[:60]}"',
        "patch":           lambda a: f'✏️ {a.get("path","?")}',
        "execute_code":    lambda a: '🐍 execute_code',
        "todo":            lambda a: '📋 todo',
        "memory":          lambda a: '🧠 memory',
        "browser_navigate": lambda a: f'🌐 {a.get("url","")[:60]}',
        "browser_snapshot": lambda a: '📸 browser_snapshot',
        "browser_click":   lambda a: '🖱️ browser_click',
        "browser_type":    lambda a: f'⌨️ {a.get("text","")[:60]}',
        "delegate_task":   lambda a: f'🤖 {a.get("goal","")[:80]}',
    }

    # ── Output summarisers (tool output JSON → one-liner) ──
    OUTPUT_SUMMARIES = {
        "write_file":   lambda o: f"💾 {o.get('bytes_written',0)} bytes — ✓ ok",
        "web_search":   lambda o: f"🔍 {len(o.get('data',{}).get('web',o.get('results',[])))} results",
        "web_extract":  lambda o: f"📄 {len(o.get('results',[]))} pages",
        "search_files": lambda o: f"🔎 {len(o.get('matches',[]))} matches",
        "read_file":    lambda o: f"📖 {o.get('total_lines',0)} lines",
        "terminal":     lambda o: f"💻 exit {o.get('exit_code','?')}",
        "execute_code": lambda o: "🐍 completed",
    }

    # ── Argument keys that commonly carry large strings ──
    TRIM_KEYS = {"content", "query", "pattern", "old_string", "new_string", "code", "text"}

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Hermes Tool Sanitizer"
        # Track call_id → tool_name across added/done pairs
        self._tool_names: Dict[str, str] = {}
        # Track item_id → call_id for function_call_output items
        self._item_to_call: Dict[str, str] = {}

    # ═══════════════════════════════════════════════════════════════
    #  inlet — inject inline-output hint into user messages
    # ═══════════════════════════════════════════════════════════════

    async def inlet(
        self, body: dict, __event_emitter__=None, __user__=None
    ) -> dict:
        """Pre-process: append inline-output hint to user messages."""
        messages = body.get("messages", [])
        if messages and messages[-1].get("role") == "user":
            content = messages[-1].get("content", "")
            if isinstance(content, str) and len(content) > 20:
                hint = (
                    "\n\n[System note: When responding through Open WebUI, "
                    "output full content directly in your response text so it "
                    "renders as markdown. Use write_file only as a silent "
                    "backup — never just say 'Done! File saved.']"
                )
                if hint not in content:
                    messages[-1]["content"] = content + hint
                    body["messages"] = messages
        return body

    # ═══════════════════════════════════════════════════════════════
    #  stream — intercept SSE events mid-flight
    # ═══════════════════════════════════════════════════════════════

    async def stream(self, event: dict) -> dict:
        """Intercept SSE: shorten tool call arguments and outputs."""
        if not event:
            return event

        evtype = event.get("type", "")

        # ── function_call added — beautify arguments ──
        if evtype == "response.output_item.added":
            item = event.get("item", {})
            if item.get("type") == "function_call":
                self._handle_function_call_added(item, event)
            elif item.get("type") == "function_call_output":
                self._handle_function_call_output_added(item, event)

        # ── function_call_output done — beautify output text ──
        elif evtype == "response.output_item.done":
            item = event.get("item", {})
            if item.get("type") == "function_call_output":
                self._handle_function_call_output_done(item, event)

        # ── response.completed — trim tool payloads (critical!) ──
        elif evtype == "response.completed":
            self._handle_response_completed(event)

        return event

    # ── Helpers ──────────────────────────────────────────────────

    def _handle_function_call_added(self, item: dict, event: dict) -> None:
        """Beautify or trim function_call arguments."""
        tool_name = item.get("name", "")
        call_id = item.get("call_id", "")
        item_id = item.get("id", "")

        # Track call_id → name for later done-event lookup
        if call_id and tool_name:
            self._tool_names[call_id] = tool_name
        if item_id and call_id:
            self._item_to_call[item_id] = call_id

        if tool_name in self.HEAVY_TOOLS or tool_name in self.FORMATTERS:
            try:
                args_raw = item.get("arguments", "{}")
                args = (
                    json.loads(args_raw) if isinstance(args_raw, str)
                    else (args_raw or {})
                )
                if not isinstance(args, dict):
                    return

                if self.valves.beautify and tool_name in self.FORMATTERS:
                    summary = self.FORMATTERS[tool_name](args)
                    item["arguments"] = json.dumps({"_summary": summary})
                else:
                    for key in list(args.keys()):
                        val = args[key]
                        if isinstance(val, str) and len(val) > self.valves.max_arg_chars:
                            args[key] = (
                                f"[{len(val)} chars — see response text]"
                            )
                    item["arguments"] = json.dumps(args)

                event["item"] = item
                if self.valves.debug:
                    print(
                        f"[HermesSanitizer] function_call {tool_name}: "
                        f"{item['arguments'][:120]}"
                    )
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                if self.valves.debug:
                    print(f"[HermesSanitizer] Error on {tool_name}: {e}")

    def _resolve_tool_name(self, item: dict) -> str:
        """Resolve tool_name for a function_call_output item.

        Priority:
        1. item.name (rarely present on output items)
        2. tracked call_id → name from the matching function_call added event
        3. tracked item_id → call_id → name (cross-reference)
        """
        # Direct name on the output item (uncommon but possible)
        name = item.get("name", "")
        if name:
            return name

        # Look up by call_id
        call_id = item.get("call_id", "")
        if call_id and call_id in self._tool_names:
            return self._tool_names[call_id]

        # Look up by item_id → call_id → name
        item_id = item.get("id", "")
        if item_id and item_id in self._item_to_call:
            mapped_call = self._item_to_call[item_id]
            if mapped_call in self._tool_names:
                return self._tool_names[mapped_call]

        return ""

    def _handle_function_call_output_added(self, item: dict, event: dict) -> None:
        """Trim output text in function_call_output *added* events."""
        # Track item_id → call_id mapping
        item_id = item.get("id", "")
        call_id = item.get("call_id", "")
        if item_id and call_id:
            self._item_to_call[item_id] = call_id

        output = item.get("output", [])
        if isinstance(output, list):
            self._trim_output_parts(output, item)

    def _handle_function_call_output_done(self, item: dict, event: dict) -> None:
        """Trim output text in function_call_output *done* events."""
        tool_name = self._resolve_tool_name(item)
        output = item.get("output", [])
        if isinstance(output, list):
            self._trim_output_parts(output, item, tool_name)

    def _trim_output_parts(
        self, output: list, item: dict, tool_name: str = ""
    ) -> None:
        """Trim or summarise every text part in a function_call_output."""
        if not output:
            return

        changed = False
        for part in output:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in ("input_text", "text"):
                continue
            text = part.get("text", "")
            if not text or len(text) <= self.valves.max_arg_chars:
                continue

            if self.valves.beautify:
                summary = self._summarise_output(text, tool_name)
                part["text"] = summary
            else:
                part["text"] = (
                    text[: self.valves.max_arg_chars]
                    + f"...[{len(text) - self.valves.max_arg_chars} more chars]"
                )
            changed = True

        if changed:
            item["output"] = output

    def _summarise_output(self, text: str, tool_name: str) -> str:
        """Produce a human-readable one-liner for tool output."""
        if tool_name and tool_name in self.OUTPUT_SUMMARIES:
            try:
                data = json.loads(text)
                return self.OUTPUT_SUMMARIES[tool_name](data)
            except (json.JSONDecodeError, KeyError):
                pass
        return f"✓ {len(text)} chars output"

    def _handle_response_completed(self, event: dict) -> None:
        """Trim large tool-call payloads inside response.completed.

        This is the single biggest performance win — response.completed
        packs ALL tool calls + outputs into one SSE line that can
        exceed 400-848 KB, causing silent hangs in Open WebUI.
        """
        response = event.get("response", {})
        if not response:
            return

        output_items = response.get("output", [])
        if not isinstance(output_items, list):
            return

        trimmed = False
        for item in output_items:
            if not isinstance(item, dict):
                continue

            if item.get("type") == "function_call":
                if self._trim_completed_function_call(item):
                    trimmed = True

            elif item.get("type") == "function_call_output":
                output = item.get("output", [])
                if isinstance(output, list):
                    for part in output:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") in ("input_text", "text"):
                            text = part.get("text", "")
                            if len(text) > 1000:
                                part["text"] = (
                                    text[:500]
                                    + f"...[{len(text) - 500} more chars]"
                                )
                                trimmed = True

        if trimmed:
            event["response"] = response
            if self.valves.debug:
                # Estimate new size
                payload = json.dumps(event, ensure_ascii=False)
                kb = len(payload) / 1024
                print(f"[HermesSanitizer] response.completed trimmed → ~{kb:.0f} KB")

    def _trim_completed_function_call(self, item: dict) -> bool:
        """Trim large argument strings in a function_call inside completed."""
        try:
            args_raw = item.get("arguments", "{}")
            args = (
                json.loads(args_raw) if isinstance(args_raw, str)
                else (args_raw or {})
            )
            if not isinstance(args, dict):
                return False

            changed = False
            for key in self.TRIM_KEYS:
                val = args.get(key, "")
                if isinstance(val, str) and len(val) > 500:
                    args[key] = f"[{len(val)} chars — truncated]"
                    changed = True

            if changed:
                item["arguments"] = json.dumps(args)
            return changed
        except Exception:
            return False

    # ═══════════════════════════════════════════════════════════════
    #  outlet — post-process warning for terse responses
    # ═══════════════════════════════════════════════════════════════

    async def outlet(
        self, body: dict, __event_emitter__=None, __user__=None
    ) -> dict:
        """Post-process: detect terse responses that hid content in tool calls."""
        choices = body.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            if content and len(content.strip()) < 30 and "done" in content.lower():
                if self.valves.debug:
                    print(
                        "[HermesSanitizer] ⚠️ Terse response — "
                        "content may be in tool calls"
                    )
        return body
