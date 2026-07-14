"""Hermes Hub assistant layer — the "Jarvis" behind the dashboard.

Two operating modes, selected automatically at runtime:

* **claude** — when the official ``anthropic`` SDK is installed and credentials
  are available (``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile), chat
  runs a real tool-use agent loop against Claude, and summaries/briefings are
  model-generated. The dashboard tools (add task, complete task, add event, …)
  execute in the *browser* against local data; the server just relays the
  Messages API conversation, so no personal data is stored server-side.

* **local** — with no SDK or credentials, a deterministic fallback keeps every
  feature working: a command parser handles "add task…"-style requests, an
  extractive summarizer handles summarize buttons, and a rule-based briefing
  analyzes tasks/calendar/headlines. Responses are labeled so the UI can show
  which engine produced them.

Install the AI mode with:  pip install anthropic  (and set ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import date, datetime

DEFAULT_MODEL = os.environ.get("HERMES_HUB_MODEL", "claude-opus-4-8")

try:  # optional dependency — the dashboard must work without it
    import anthropic

    _HAVE_SDK = True
except ImportError:
    anthropic = None
    _HAVE_SDK = False


SYSTEM_PROMPT = """\
You are the Hermes Hub agent — the built-in assistant of a personal dashboard
(codename HERMES//HUB). You can see the user's dashboard context (task lists,
calendar events, notes, headlines, watchlist) when it is provided, and you can
act on the dashboard through your tools.

Style: concise, calm, competent — a mission-control operator, not a chatbot.
Prefer doing over describing: when the user asks for something a tool can do,
call the tool. Batch independent tool calls together. After acting, confirm in
one short sentence. When asked for advice or a briefing, ground every claim in
the provided context; never invent tasks or events the user does not have.

Research before answering: for questions about news, weather, markets or the
world situation, call the matching get_* tool (and read_article for a specific
story) instead of answering from prior knowledge. For recurring wishes
("every morning…", "alert me if…"), create an automation. When you learn a
durable fact about the user (preferences, routines, people), save it with
remember. For anything you cannot do from inside the dashboard, say so plainly
and suggest the closest thing you can do."""

# Tools execute CLIENT-SIDE against the browser's local store. The server only
# relays the conversation, so schemas here must match public/js/actions.js.
DASHBOARD_TOOLS = [
    {
        "name": "add_task",
        "description": "Add a to-do item to one of the user's lists. Call once per task. Use the list name the user mentioned, or 'Today' when unspecified.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The task text"},
                "list": {"type": "string", "description": "Target list name, e.g. 'Today' or 'Groceries'"},
            },
            "required": ["text", "list"],
            "additionalProperties": False,
        },
    },
    {
        "name": "complete_task",
        "description": "Mark an existing task as done. Match by (partial) task text.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text of the task to mark done (partial match ok)"},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "add_event",
        "description": "Add a calendar event on a specific date.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Event date, YYYY-MM-DD"},
                "title": {"type": "string", "description": "Event title"},
            },
            "required": ["date", "title"],
            "additionalProperties": False,
        },
    },
    {
        "name": "add_note",
        "description": "Create a new note on the dashboard.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Note content; first line becomes the title"},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    {
        "name": "add_app",
        "description": "Add an app/link tile to the launcher.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Display name"},
                "url": {"type": "string", "description": "Full https URL"},
            },
            "required": ["name", "url"],
            "additionalProperties": False,
        },
    },
    {
        "name": "open_url",
        "description": "Open a URL or one of the user's launcher apps inside the dashboard's viewer.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to open"},
                "title": {"type": "string", "description": "Human-readable title for the viewer header"},
            },
            "required": ["url", "title"],
            "additionalProperties": False,
        },
    },
    {
        "name": "switch_news_topic",
        "description": "Switch the news widget to a topic: top, world, tech, business, science, sports or entertainment.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "enum": ["top", "world", "tech", "business", "science", "sports", "entertainment"],
                },
            },
            "required": ["topic"],
            "additionalProperties": False,
        },
    },
    # ---- research tools (read server data; proxied via /api/assistant/tool)
    {
        "name": "get_news",
        "description": "Fetch current headlines for a topic. Use before answering questions about the news.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "enum": ["top", "world", "tech", "business", "science", "sports", "entertainment"],
                },
            },
            "required": ["topic"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_article",
        "description": "Fetch and read the text of an article/web page by URL. Use to answer questions about a specific story.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "http(s) URL"}},
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current weather and 7-day outlook for the user's configured location.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_worldstate",
        "description": "Get the state-of-the-world situation board: per-domain stability scores, levels and headline signals.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_markets",
        "description": "Get the market watchlist: prices and 24h changes.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    # ---- memory
    {
        "name": "remember",
        "description": "Save a durable fact about the user or their preferences to long-term memory (persists across sessions).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"fact": {"type": "string", "description": "One concise fact"}},
            "required": ["fact"],
            "additionalProperties": False,
        },
    },
    # ---- automations
    {
        "name": "create_automation",
        "description": (
            "Create a standing automation. Triggers: daily (needs time HH:MM), "
            "market (needs symbol + percent — fires when |24h change| crosses it), "
            "worldstate (needs level — fires when the global index reaches it). "
            "Actions: briefing (auto-generated) or notify (needs message)."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "trigger_type": {"type": "string", "enum": ["daily", "market", "worldstate"]},
                "time": {"type": ["string", "null"], "description": "HH:MM for daily"},
                "symbol": {"type": ["string", "null"], "description": "e.g. BTC, for market"},
                "percent": {"type": ["number", "null"], "description": "threshold, for market"},
                "level": {
                    "type": ["string", "null"],
                    "enum": ["watch", "elevated", "critical", None],
                    "description": "for worldstate",
                },
                "action_type": {"type": "string", "enum": ["briefing", "notify"]},
                "message": {"type": ["string", "null"], "description": "for notify"},
            },
            "required": ["name", "trigger_type", "time", "symbol", "percent", "level",
                         "action_type", "message"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_automations",
        "description": "List the user's standing automations with their ids.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "delete_automation",
        "description": "Delete an automation by id (find ids with list_automations).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    },
]

# Tools the browser proxies back to the server (they read server-side data or
# mutate server-side state). Everything else executes against the local store.
SERVER_TOOLS = {
    "get_news", "read_article", "get_weather", "get_worldstate", "get_markets",
    "remember", "create_automation", "list_automations", "delete_automation",
}


def _credentials_available() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    # `ant auth login` stores profiles the SDK resolves automatically.
    config_dir = os.environ.get("ANTHROPIC_CONFIG_DIR") or os.path.expanduser("~/.config/anthropic")
    return os.path.isdir(os.path.join(config_dir, "credentials"))


class Assistant:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._client = None
        self.services = None  # set by server.Api — access to feeds/memory/automations

    # -- server-side tools (proxied through /api/assistant/tool) -------------
    def run_server_tool(self, name: str, tool_input: dict) -> str:
        if self.services is None:
            raise ValueError("server tools unavailable")
        handler = getattr(self, f"_tool_{name}", None)
        if name not in SERVER_TOOLS or handler is None:
            raise ValueError(f"unknown server tool {name!r}")
        try:
            return handler(tool_input or {})
        except ValueError:
            raise
        except Exception as exc:  # upstream/API errors become tool errors
            raise ValueError(str(exc)) from None

    def _tool_get_news(self, args: dict) -> str:
        data = self.services.news({"topic": [args.get("topic", "top")], "limit": ["10"]})
        lines = [f"{i + 1}. {item['title']} ({item['source']}) {item['url']}"
                 for i, item in enumerate(data["items"])]
        return f"[{data['source']} data] " + "\n".join(lines)

    def _tool_read_article(self, args: dict) -> str:
        data = self.services.reader({"url": [args.get("url", "")]})
        if not data.get("blocks"):
            return data.get("note") or "No readable text could be extracted."
        text = "\n".join(b["text"] for b in data["blocks"])
        return f"{data.get('title', '')}\n\n{text[:6000]}"

    def _tool_get_weather(self, args: dict) -> str:
        return format_weather(self.services.weather({}))

    def _tool_get_worldstate(self, args: dict) -> str:
        return format_worldstate(self.services.worldstate({}))

    def _tool_get_markets(self, args: dict) -> str:
        return format_markets(self.services.markets({}))

    def _tool_remember(self, args: dict) -> str:
        self.services.memory_append(args.get("fact", ""))
        return "Saved to long-term memory."

    def _tool_create_automation(self, args: dict) -> str:
        trigger = {"type": args.get("trigger_type")}
        if trigger["type"] == "daily":
            trigger["time"] = args.get("time")
        elif trigger["type"] == "market":
            trigger["symbol"] = (args.get("symbol") or "").upper()
            trigger["percent"] = args.get("percent")
        elif trigger["type"] == "worldstate":
            trigger["level"] = args.get("level")
        action = {"type": args.get("action_type")}
        if action["type"] == "notify":
            action["message"] = args.get("message") or ""
        rule = self.services.automations.create_rule(
            {"name": args.get("name"), "trigger": trigger, "action": action})
        return f"Automation #{rule['id']} “{rule['name']}” is armed."

    def _tool_list_automations(self, args: dict) -> str:
        rules = self.services.automations.list_rules()
        if not rules:
            return "No automations yet."
        lines = []
        for rule in rules:
            trigger, action = rule["trigger"], rule["action"]
            if trigger["type"] == "daily":
                when = f"daily at {trigger['time']}"
            elif trigger["type"] == "market":
                when = f"{trigger['symbol']} moves ±{trigger['percent']}%"
            else:
                when = f"world state reaches {trigger['level'].upper()}"
            what = "auto-briefing" if action["type"] == "briefing" else f"notify: {action.get('message', '')}"
            state = "" if rule.get("enabled") else " (paused)"
            lines.append(f"#{rule['id']} {rule['name']} — {when} → {what}{state}")
        return "\n".join(lines)

    def _tool_delete_automation(self, args: dict) -> str:
        if not self.services.automations.delete_rule(int(args.get("id", 0))):
            raise ValueError(f"no automation with id {args.get('id')}")
        return f"Automation #{args.get('id')} deleted."

    # -- mode handling ------------------------------------------------------
    @property
    def mode(self) -> str:
        return "claude" if (_HAVE_SDK and _credentials_available()) else "local"

    def _get_client(self):
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "model": self.model if self.mode == "claude" else None,
            "sdk_installed": _HAVE_SDK,
            "hint": None if self.mode == "claude" else (
                "Local rule-based mode. For full AI: pip install anthropic, "
                "then set ANTHROPIC_API_KEY (or run `ant auth login`) and restart."
            ),
        }

    # -- chat (agent loop step) ---------------------------------------------
    def chat(self, payload: dict) -> dict:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        context = payload.get("context") or {}
        if self.mode == "claude":
            return self._chat_claude(messages, context)
        return self._chat_local(messages, context)

    def _chat_claude(self, messages: list, context: dict) -> dict:
        # Keep the system prompt frozen for caching; per-request dashboard
        # context travels as the final block of the first user turn instead.
        request_messages = list(messages)
        if context and request_messages and request_messages[0]["role"] == "user":
            first = request_messages[0]
            content = first["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            request_messages[0] = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<dashboard-context>\n"
                        + json.dumps(context, ensure_ascii=False)
                        + "\n</dashboard-context>",
                    },
                    *content,
                ],
            }
        system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
        memory = self.services.memory_read() if self.services else ""
        if memory.strip():
            # after the cached block, so editing memory never invalidates it
            system.append({"type": "text", "text": "Long-term memory about the user:\n" + memory[-4000:]})
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=system,
            tools=DASHBOARD_TOOLS,
            messages=request_messages,
        )
        content = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
            elif block.type == "thinking":
                # must be echoed back verbatim on the next loop iteration
                content.append(json.loads(block.to_json()))
        return {"mode": "claude", "stop_reason": response.stop_reason, "content": content}

    # Local command grammar: "add task X [to LIST]", "complete/finish X",
    # "add event on 2026-07-08: standup" / "add event tomorrow: standup",
    # "note: ...", "add app NAME URL", "open NAME/URL", "show tech news"
    def _chat_local(self, messages: list, context: dict) -> dict:
        last = messages[-1]
        text = last["content"] if isinstance(last["content"], str) else " ".join(
            b.get("text", "") for b in last["content"] if isinstance(b, dict) and b.get("type") == "text"
        )
        actions, reply = parse_local_command(text, context)
        if not actions and reply is HELP_TEXT:
            inline = self._local_data_reply(text)
            if inline is not None:
                return {"mode": "local", "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": inline}]}
        content = [{"type": "text", "text": reply}]
        for i, (name, args) in enumerate(actions):
            content.append({"type": "tool_use", "id": f"local_{i}", "name": name, "input": args})
        return {"mode": "local", "stop_reason": "end_turn", "content": content}

    def _local_data_reply(self, text: str) -> str | None:
        """Answer data questions directly in local mode (weather, markets, …)."""
        if self.services is None:
            return None
        lowered = text.lower()
        try:
            if re.search(r"\b(weather|forecast|temperature)\b", lowered):
                return format_weather(self.services.weather({}))
            if re.search(r"\b(state of the world|world ?state|global (?:index|situation))\b", lowered):
                return format_worldstate(self.services.worldstate({}))
            if re.search(r"\b(markets?|prices?|crypto|bitcoin|btc|eth(?:ereum)?)\b", lowered) and "task" not in lowered:
                return format_markets(self.services.markets({}))
            if re.fullmatch(r"(what(?:'s| is) (?:in the |the )?news\??|headlines\??|news\??)", lowered.strip()):
                data = self.services.news({"topic": ["top"], "limit": ["6"]})
                return "\n".join(f"• {item['title']} ({item['source']})" for item in data["items"])
            if re.search(r"\bwhat do you (?:remember|know)(?: about me)?\b|^recall\??$", lowered):
                memory = self.services.memory_read().strip()
                return memory if memory else "Long-term memory is empty. Tell me “remember: …” to add something."
        except Exception:
            return None
        return None

    # -- summarize ------------------------------------------------------------
    def summarize(self, payload: dict) -> dict:
        title = (payload.get("title") or "").strip()
        content = (payload.get("content") or "").strip()
        kind = (payload.get("kind") or "content").strip()
        if not content:
            raise ValueError("missing content to summarize")
        content = content[:60_000]
        if self.mode == "claude":
            response = self._get_client().messages.create(
                model=self.model,
                max_tokens=1024,
                output_config={"effort": "low"},
                system="You summarize dashboard data for a busy user. Reply with 2-4 tight sentences (or up to 5 bullet lines for lists of items). No preamble.",
                messages=[{
                    "role": "user",
                    "content": f"Summarize this {kind}" + (f' titled "{title}"' if title else "") + f":\n\n{content}",
                }],
            )
            text = next((b.text for b in response.content if b.type == "text"), "")
            return {"mode": "claude", "summary": text.strip()}
        return {"mode": "local", "summary": extractive_summary(content)}

    # -- briefing --------------------------------------------------------------
    def briefing(self, payload: dict) -> dict:
        context = payload.get("context") or {}
        if self.mode == "claude":
            response = self._get_client().messages.create(
                model=self.model,
                max_tokens=2048,
                thinking={"type": "adaptive"},
                system=(
                    "You are the Hermes Hub agent preparing a personal briefing. "
                    "Given dashboard context (tasks, calendar, notes, headlines, world state), produce:\n"
                    "1. FOCUS — the 2-3 things that matter most today and why.\n"
                    "2. TASKS — per open task, one short suggestion for completing it faster; "
                    "flag any task that could be automated (recurring, reminder-able, delegable) and how.\n"
                    "3. RADAR — one line on calendar and anything notable in the headlines/world state.\n"
                    "Plain text with those three uppercase section headers. Ground everything in the "
                    "context; do not invent items."
                ),
                messages=[{"role": "user", "content": json.dumps(context, ensure_ascii=False)}],
            )
            text = next((b.text for b in response.content if b.type == "text"), "")
            return {"mode": "claude", "briefing": text.strip()}
        return {"mode": "local", "briefing": local_briefing(context)}


# ---------------------------------------------------------------------------
# Data formatters (shared by server tools and local-mode replies)
# ---------------------------------------------------------------------------
def format_weather(data: dict) -> str:
    current = data["current"]
    lines = [
        f"{data['location']['name']}: {round(current['temp'])}° "
        f"(feels {round(current['feels'])}°), humidity {current['humidity']}%, "
        f"wind {round(current['wind'])} {data['units']['wind']}."
    ]
    for day in data["daily"][:5]:
        lines.append(f"{day['date']}: {round(day['min'])}–{round(day['max'])}°, "
                     f"precip {day.get('precipProb') or 0}%")
    return "\n".join(lines)


def format_worldstate(data: dict) -> str:
    lines = [f"Global index {data['overall']['score']} ({data['overall']['level'].upper()})."]
    for domain in data["domains"]:
        lines.append(f"{domain['name']}: {domain['score']} {domain['level'].upper()} — {domain['explanation']}")
    return "\n".join(lines)


def format_markets(data: dict) -> str:
    return "\n".join(
        f"{a['name']} ({a['symbol']}): ${a['price']:,} · "
        f"{'+' if a['change24h'] >= 0 else ''}{a['change24h']:.2f}% 24h"
        for a in data["assets"]
    )


# ---------------------------------------------------------------------------
# Local fallback engines (deterministic, dependency-free)
# ---------------------------------------------------------------------------
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]+")
_STOPWORDS = frozenset(
    "the a an and or but if then than so of to in on at by for with from as is are was were be been "
    "being this that these those it its their his her your our my i you he she they we not no do does "
    "did done has have had will would can could should may might about into over under after before "
    "more most other some such only own same very just also there here when where which who whom what".split()
)


def extractive_summary(text: str, max_sentences: int = 3) -> str:
    """Frequency-scored extractive summary — keeps original sentence order."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # For list-ish content (tasks, headlines) return the top lines directly.
    if len(lines) >= 4 and sum(len(ln) for ln in lines) / len(lines) < 90:
        head = lines[: max_sentences + 2]
        more = len(lines) - len(head)
        return "• " + "\n• ".join(head) + (f"\n…and {more} more." if more > 0 else "")

    sentences = [s.strip() for s in _SENTENCE_RE.split(" ".join(lines)) if len(s.strip()) > 25]
    if len(sentences) <= max_sentences:
        return " ".join(sentences) if sentences else text[:300]

    freq = Counter(
        w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS
    )
    scored = []
    for idx, sentence in enumerate(sentences):
        words = [w.lower() for w in _WORD_RE.findall(sentence)]
        if not words:
            continue
        score = sum(freq[w] for w in words) / len(words)
        if idx == 0:
            score *= 1.25  # lead sentence usually carries the story
        scored.append((score, idx, sentence))
    top = sorted(scored, reverse=True)[:max_sentences]
    return " ".join(s for _, _, s in sorted(top, key=lambda t: t[1]))


_AUTOMATION_RULES = [
    (re.compile(r"\b(pay|bill|rent|invoice|renew|subscription|insurance)\b", re.I),
     "recurring — add a monthly calendar event so it never sneaks up"),
    (re.compile(r"\b(call|email|message|text|reply|contact|follow.?up)\b", re.I),
     "communication — the agent can draft the message for you to send"),
    (re.compile(r"\b(buy|order|purchase|groceries|shopping)\b", re.I),
     "purchase — keep it on the Groceries list and open the store from the launcher"),
    (re.compile(r"\b(read|watch|review|check)\b", re.I),
     "queueable — open it in the in-app viewer straight from the dashboard"),
    (re.compile(r"\b(book|schedule|appointment|reserve|dentist|doctor)\b", re.I),
     "scheduling — add the confirmed slot to the calendar"),
    (re.compile(r"\b(backup|export|sync|update|upgrade)\b", re.I),
     "maintenance — good candidate for a recurring reminder"),
]


def suggest_automation(task_text: str) -> str | None:
    for pattern, suggestion in _AUTOMATION_RULES:
        if pattern.search(task_text):
            return suggestion
    return None


def local_briefing(context: dict) -> str:
    today = date.today().isoformat()
    lines: list[str] = []

    lists = context.get("tasks") or []
    open_tasks = [
        (lst.get("name", "List"), item.get("text", ""))
        for lst in lists
        for item in lst.get("items", [])
        if not item.get("done")
    ]
    done_count = sum(
        1 for lst in lists for item in lst.get("items", []) if item.get("done")
    )
    events = context.get("events") or []
    todays = [e for e in events if e.get("date") == today]
    upcoming = sorted(
        (e for e in events if (e.get("date") or "") > today), key=lambda e: e["date"]
    )[:3]

    lines.append("FOCUS")
    if open_tasks:
        for list_name, text in open_tasks[:3]:
            lines.append(f"• {text}  ({list_name})")
    else:
        lines.append("• Board is clear — nothing open. Set tomorrow's priorities while it's quiet.")
    if todays:
        lines.append(f"• {len(todays)} event(s) today: " + "; ".join(e["title"] for e in todays))

    lines.append("")
    lines.append("TASKS")
    if open_tasks:
        automatable = 0
        for list_name, text in open_tasks[:8]:
            suggestion = suggest_automation(text)
            if suggestion:
                automatable += 1
                lines.append(f"• {text} → {suggestion}")
            else:
                lines.append(f"• {text} → break it into a 10-minute first step and start there")
        remaining = len(open_tasks) - min(len(open_tasks), 8)
        if remaining > 0:
            lines.append(f"• …and {remaining} more open task(s).")
        summary = f"{len(open_tasks)} open · {done_count} done"
        if automatable:
            summary += f" · {automatable} look automatable"
        lines.append(f"• Status: {summary}.")
    else:
        lines.append(f"• All caught up ({done_count} completed).")

    lines.append("")
    lines.append("RADAR")
    if upcoming:
        lines.append(
            "• Upcoming: " + "; ".join(f"{e['date']} {e['title']}" for e in upcoming)
        )
    else:
        lines.append("• Calendar ahead is clear.")
    headlines = context.get("headlines") or []
    if headlines:
        lines.append(f"• Headlines: {headlines[0]}")
    world = context.get("worldstate")
    if world:
        lines.append(
            f"• World index {world.get('score', '—')} ({str(world.get('level', '')).upper()})"
            + (f" — watch {world['watch']}" if world.get("watch") else "")
        )
    lines.append("")
    lines.append("(Local heuristic briefing — install the anthropic SDK + API key for the full agent.)")
    return "\n".join(lines)


_TASK_CMD = re.compile(r"^(?:add\s+)?(?:task|todo)\s*:?\s+(.+?)(?:\s+to\s+(?:the\s+)?(\w[\w ]*?)(?:\s+list)?)?$", re.I)
_DONE_CMD = re.compile(r"^(?:complete|finish|done|check\s*off|mark\s+done)\s*:?\s+(.+)$", re.I)
_NOTE_CMD = re.compile(r"^(?:add\s+)?note\s*:?\s+(.+)$", re.I | re.S)
_EVENT_CMD = re.compile(
    r"^(?:add\s+)?event\s+(?:on\s+)?(\d{4}-\d{2}-\d{2}|today|tomorrow)\s*:?\s+(.+)$", re.I
)
_APP_CMD = re.compile(r"^add\s+app\s+(.+?)\s+(https?://\S+)$", re.I)
_OPEN_CMD = re.compile(r"^(?:open|launch|go\s*to)\s+(.+)$", re.I)
_NEWS_CMD = re.compile(
    r"^(?:show|switch\s+to)?\s*(top|world|tech|business|science|sports|entertainment)\s+news$", re.I
)


_REMEMBER_CMD = re.compile(r"^remember\s*:?\s+(.+)$", re.I | re.S)
_DAILY_BRIEF_A = re.compile(
    r"^(?:every\s?(?:day|morning)|daily)\s*,?\s*(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*[,:]?\s*"
    r"(?:send (?:me )?(?:a )?brief(?:ing)?|brief(?:ing)?(?:\s+me)?)\.?$", re.I)
_DAILY_BRIEF_B = re.compile(
    r"^brief(?:ing)?(?:\s+me)?\s+(?:every\s?(?:day|morning)|daily)\s*,?\s*(?:at\s+)?"
    r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\.?$", re.I)
_MARKET_ALERT = re.compile(
    r"^alert me (?:if|when)\s+(\w+)\s+(?:moves|changes|drops|falls|rises|gains)"
    r"(?:\s+(?:by|more than|over))?\s+(\d+(?:\.\d+)?)\s*%\.?$", re.I)
_WORLD_ALERT = re.compile(
    r"^alert me (?:if|when)\s+(?:the\s+)?world\s?(?:state)?\s*(?:index\s*)?"
    r"(?:reaches|goes(?:\s+to)?|hits|is)\s+(watch|elevated|critical)\.?$", re.I)
_LIST_AUTOS = re.compile(r"^(?:list|show)(?:\s+my)?\s+automations?\.?$", re.I)
_DELETE_AUTO = re.compile(r"^(?:delete|remove)\s+automation\s+#?(\d+)\.?$", re.I)


def _clock(hour: str, minute: str | None, ampm: str | None) -> str:
    h = int(hour) % 24
    if ampm:
        h = h % 12 + (12 if ampm.lower() == "pm" else 0)
    return f"{h:02d}:{minute or '00'}"


def parse_local_command(text: str, context: dict) -> tuple[list, str]:
    """Map a natural-ish command to dashboard actions. Returns (actions, reply)."""
    text = text.strip()

    match = _REMEMBER_CMD.match(text)
    if match:
        fact = match.group(1).strip()
        return [("remember", {"fact": fact})], "Noted — saved to long-term memory."

    match = _DAILY_BRIEF_A.match(text) or _DAILY_BRIEF_B.match(text)
    if match:
        when = _clock(match.group(1), match.group(2), match.group(3))
        return [("create_automation", {
            "name": f"Daily briefing {when}", "trigger_type": "daily", "time": when,
            "symbol": None, "percent": None, "level": None,
            "action_type": "briefing", "message": None,
        })], f"Standing order: I'll compile your briefing every day at {when}."

    match = _MARKET_ALERT.match(text)
    if match:
        symbol, percent = match.group(1).upper(), float(match.group(2))
        return [("create_automation", {
            "name": f"{symbol} ±{percent:g}% alert", "trigger_type": "market", "time": None,
            "symbol": symbol, "percent": percent, "level": None,
            "action_type": "notify", "message": f"{symbol} moved more than {percent:g}% in 24h.",
        })], f"Watching {symbol}: you'll hear from me when it moves more than {percent:g}% in 24h."

    match = _WORLD_ALERT.match(text)
    if match:
        level = match.group(1).lower()
        return [("create_automation", {
            "name": f"World state {level.upper()} alert", "trigger_type": "worldstate",
            "time": None, "symbol": None, "percent": None, "level": level,
            "action_type": "notify", "message": f"Global situation index reached {level.upper()}.",
        })], f"Armed: I'll alert you when the global index reaches {level.upper()}."

    if _LIST_AUTOS.match(text):
        return [("list_automations", {})], "Here are your automations:"

    match = _DELETE_AUTO.match(text)
    if match:
        return [("delete_automation", {"id": int(match.group(1))})], "Removing that automation."

    match = _DONE_CMD.match(text)
    if match:
        return [("complete_task", {"text": match.group(1).strip()})], f"Marking “{match.group(1).strip()}” as done."

    match = _EVENT_CMD.match(text)
    if match:
        when = match.group(1).lower()
        if when == "today":
            when = date.today().isoformat()
        elif when == "tomorrow":
            when = date.fromordinal(date.today().toordinal() + 1).isoformat()
        title = match.group(2).strip()
        return [("add_event", {"date": when, "title": title})], f"Scheduling “{title}” on {when}."

    match = _NOTE_CMD.match(text)
    if match:
        return [("add_note", {"text": match.group(1).strip()})], "Note saved."

    match = _APP_CMD.match(text)
    if match:
        name, url = match.group(1).strip(), match.group(2).strip()
        return [("add_app", {"name": name, "url": url})], f"Added {name} to your launcher."

    match = _NEWS_CMD.match(text)
    if match:
        topic = match.group(1).lower()
        return [("switch_news_topic", {"topic": topic})], f"Switching news to {topic}."

    match = _OPEN_CMD.match(text)
    if match:
        target = match.group(1).strip().rstrip(".?!")
        links = context.get("apps") or []
        for link in links:
            if link.get("name", "").lower() == target.lower():
                return [("open_url", {"url": link["url"], "title": link["name"]})], f"Opening {link['name']}."
        if re.match(r"^https?://", target, re.I):
            return [("open_url", {"url": target, "title": target})], "Opening in the viewer."
        for link in links:
            if target.lower() in link.get("name", "").lower():
                return [("open_url", {"url": link["url"], "title": link["name"]})], f"Opening {link['name']}."
        return [], f"I don't have “{target}” in your launcher. Add it with: add app {target} https://…"

    match = _TASK_CMD.match(text)
    if match and not text.lower().startswith(("what", "how", "why", "brief")):
        task_text = match.group(1).strip().rstrip(".")
        target_list = (match.group(2) or "Today").strip().title()
        reply = f"Added “{task_text}” to {target_list}."
        suggestion = suggest_automation(task_text)
        if suggestion:
            reply += f" Tip: {suggestion}."
        return [("add_task", {"text": task_text, "list": target_list})], reply

    if re.search(r"\b(brief|briefing|focus|plate|agenda|catch me up)\b", text, re.I):
        return [], local_briefing(context)

    return [], HELP_TEXT


HELP_TEXT = (
    "Local mode understands: “add task … [to LIST]”, “complete …”, “note: …”, "
    "“add event 2026-07-10: title” (or today/tomorrow), “add app NAME URL”, "
    "“open APP”, “show tech news”, “brief me”, “remember: …”, "
    "“every morning at 8 brief me”, “alert me if BTC moves 5%”, "
    "“alert me when the world reaches elevated”, “list automations”, and "
    "questions about the weather, markets, headlines or the state of the world. "
    "For free-form conversation, install the anthropic SDK and set ANTHROPIC_API_KEY."
)
