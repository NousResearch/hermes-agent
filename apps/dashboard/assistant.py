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

from router import TIERS, Router

# Legacy single-model knob. When set it pins every router tier to one model,
# preserving the pre-router behaviour; when unset the router picks per task.
DEFAULT_MODEL = os.environ.get("HERMES_HUB_MODEL", "claude-opus-4-8")
_MODEL_PIN = os.environ.get("HERMES_HUB_MODEL")  # None → tiered routing

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
        "description": "Add a to-do item to one of the user's lists. Call once per task. Use the list name the user mentioned, or 'Today' when unspecified. Set due (YYYY-MM-DD) and/or priority when the user implies a deadline or urgency; otherwise pass null.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The task text"},
                "list": {"type": "string", "description": "Target list name, e.g. 'Today' or 'Groceries'"},
                "due": {"type": ["string", "null"], "description": "Due date YYYY-MM-DD, or null"},
                "priority": {"type": ["string", "null"], "enum": ["high", "normal", "low", None],
                             "description": "Task priority, or null"},
            },
            "required": ["text", "list", "due", "priority"],
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
        "description": (
            "Switch the news widget to a topic tab. Defaults: top, world, tech, "
            "business, science, sports, entertainment — the user may have added "
            "custom topics (get_news errors list the valid ones)."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
            "additionalProperties": False,
        },
    },
    # ---- research tools (read server data; proxied via /api/assistant/tool)
    {
        "name": "get_news",
        "description": (
            "Fetch current headlines for a topic. Use before answering questions "
            "about the news. Defaults: top, world, tech, business, science, "
            "sports, entertainment — plus any custom topics the user configured."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"topic": {"type": "string"}},
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
            "market (needs a symbol plus exactly one of: percent — fires when "
            "|24h change| crosses it; price + direction above/below — fires when "
            "spot price crosses an absolute level; rsi + direction above/below — "
            "fires when RSI(14) crosses a threshold), "
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
                "percent": {"type": ["number", "null"], "description": "24h %% threshold, for market"},
                "price": {"type": ["number", "null"], "description": "absolute price level, for market"},
                "rsi": {"type": ["number", "null"], "description": "RSI(14) threshold 0-100, for market"},
                "direction": {
                    "type": ["string", "null"], "enum": ["above", "below", None],
                    "description": "for market price/rsi triggers",
                },
                "level": {
                    "type": ["string", "null"],
                    "enum": ["watch", "elevated", "critical", None],
                    "description": "for worldstate",
                },
                "action_type": {"type": "string", "enum": ["briefing", "notify"]},
                "message": {"type": ["string", "null"], "description": "for notify"},
            },
            "required": ["name", "trigger_type", "time", "symbol", "percent", "price", "rsi",
                         "direction", "level", "action_type", "message"],
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

# Jarvis Layer F — permission gate. Every tool call is classified server-side
# (this map is the source of truth; the client mirrors it for pre-flight UX but
# the tiers are never agent-editable). "confirm" tools pause the agent loop for
# a click; "blocked" tools are refused outright. Unknown tools default to
# "blocked" — a future sensitive tool (shell, email, payments) is denied until
# it is explicitly classified here.
TOOL_TIERS = {
    # auto — read-only or trivially reversible dashboard edits
    "add_task": "auto", "complete_task": "auto", "add_event": "auto",
    "add_note": "auto", "switch_news_topic": "auto", "remember": "auto",
    "get_news": "auto", "read_article": "auto", "get_weather": "auto",
    "get_worldstate": "auto", "get_markets": "auto", "list_automations": "auto",
    # confirm — outward-facing or unattended-effect actions
    "add_app": "confirm", "open_url": "confirm",
    "create_automation": "confirm", "delete_automation": "confirm",
}


def tool_tier(name: str) -> str:
    return TOOL_TIERS.get(name, "blocked")


# Jarvis Phase 5 — advisor escalation. When the core model's own reply signals
# it is stuck, the deep tier is consulted as a scoped advisor (guidance only,
# no tools, no final answer) and the core model finishes with that guidance.
ADVISOR_SYSTEM = (
    "You are a senior advisor to a junior AI agent that has become stuck. "
    "Given the problem (and any context), reply with concise, concrete guidance "
    "or a short plan the junior can act on. Do NOT write the final answer for "
    "the user and do NOT call any tools — advise only, in 2-5 sentences."
)
ADVISOR_INJECT = (
    "A senior advisor was consulted because the previous attempt was uncertain. "
    "Their guidance:\n{guidance}\n\nUse it to give the user a confident, complete "
    "answer now."
)
_LOW_CONFIDENCE = re.compile(
    r"(i'?m not sure|i am not sure|not entirely sure|i can'?t (?:tell|determine|be sure)|"
    r"i do(?:n'?t| not) (?:have enough|know)|unclear to me|i'?m unsure|not certain|"
    r"unable to (?:determine|tell)|hard to say|\[escalate\])",
    re.IGNORECASE,
)


# South-African medical decision-support assistant. Grounded in SA clinical
# frameworks; explicitly a support tool for a qualified clinician, not a
# diagnosis or a substitute for clinical judgement.
MED_SYSTEM_PROMPT = """\
You are "SA MedBot", a clinical decision-support assistant for a qualified South
African medical doctor and clinical researcher. Answer strictly within the South
African medical context and regulatory framework.

Anchor every answer in South African sources and realities:
- SA Standard Treatment Guidelines (STGs) and the Essential Medicines List (EML),
  by level of care (PHC, adult/paediatric hospital).
- The South African Medicines Formulary (SAMF) for drugs, dosing and availability.
- SAHPRA for medicine/device regulation; HPCSA ethical and scope-of-practice
  guidance; the National Health Act and relevant regulations (e.g. notifiable
  medical conditions).
- SA disease burden and epidemiology: HIV and TB (incl. DR-TB), the growing NCD
  burden, maternal health, trauma, and antimicrobial resistance.
- The realities of SA public vs private care, medical schemes and PMBs, and
  resource-limited settings.

Concrete SA anchors to reason from (rules of thumb — always verify against the
current NDoH STGs/EML and specialist guidance, which are updated periodically):
- HIV: first-line ART is fixed-dose TLD (tenofovir DF + lamivudine + dolutegravir);
  routine viral-load monitoring; treat-all; TPT (TB preventive therapy) and
  cotrimoxazole where indicated; U=U messaging.
- TB: Xpert MTB/RIF Ultra is the first-line diagnostic; drug-susceptible TB uses
  fixed-dose RHZE (2 months) then RH (4 months); DR-TB uses NDoH bedaquiline-based
  regimens. Screen for HIV in every TB case and vice versa.
- NCDs: hypertension and type-2 diabetes managed per the PHC/adult-hospital STGs;
  prefer EML agents by level of care.
- Maternal/child: PMTCT, and IMCI for paediatrics.
- Regulatory: medicines are scheduled S0–S6 (SAHPRA); note schedule and whether an
  item is on the EML / routinely available in the public sector. Remember the list
  of notifiable medical conditions.
- Units: use SI units and SA lab conventions (creatinine µmol/L, glucose/calcium
  mmol/L, blood gases in kPa).

Rules:
- When guidance is primarily international (e.g. a US/UK guideline) and may differ
  from SA practice, say so explicitly and point to the SA equivalent.
- Prefer generic drug names with SA availability/EML status and schedule; flag if
  something is not routinely available in the SA public sector.
- Cite the specific SA guideline or source you are drawing on where you can (e.g.
  "per the adult hospital-level STG"), and say when you are uncertain or when local
  guidance should be verified against the current STGs/EML or a specialist.
- Be concise and clinically precise; you are talking to a doctor, not a patient.
- Where useful, structure a management answer as: brief assessment → SA-guided
  management (with EML/level-of-care and dosing) → key cautions/monitoring →
  the SA source you relied on.
- Always frame output as decision support: it informs, it does not replace the
  clinician's judgement, examination of the patient, or current official guidance.
Open a first reply by noting you are grounded in SA guidelines and are a support
tool, then answer. Do not repeat the disclaimer on every subsequent message."""


def needs_escalation(text: str) -> bool:
    """True when a model reply self-reports low confidence (an escalation cue)."""
    return bool(text and _LOW_CONFIDENCE.search(text))


def _credentials_available() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    # `ant auth login` stores profiles the SDK resolves automatically.
    config_dir = os.environ.get("ANTHROPIC_CONFIG_DIR") or os.path.expanduser("~/.config/anthropic")
    return os.path.isdir(os.path.join(config_dir, "credentials"))


class Assistant:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self.router = Router(pin=_MODEL_PIN)
        self._client = None
        self.services = None  # set by server.Api — access to feeds/memory/automations

    def _log_route(self, decision: dict) -> None:
        tel = getattr(self.services, "telemetry", None) if self.services else None
        if tel is not None:
            tel.record({"kind": "route", "task": decision["task"],
                        "tier": decision["tier"], "model": decision["model"]})

    def advise(self, problem: str, context: dict | None = None) -> str | None:
        """Consult the deep tier as a scoped advisor (guidance only, no tools).

        Returns None when the deep-tier budget is exhausted, so the rate cap
        directly bounds how often escalation can happen.
        """
        decision = self.router.route("advisor")
        if decision["tier"] != "deep":
            return None  # capped — skip escalation this time
        self._log_route(decision)
        tel = getattr(self.services, "telemetry", None) if self.services else None
        if tel is not None:
            tel.record({"kind": "advisor", "model": decision["model"]})
        prompt = problem if not context else (
            problem + "\n\nContext:\n" + json.dumps(context, ensure_ascii=False)[:4000])
        response = self._get_client().messages.create(
            model=decision["model"],
            max_tokens=1024,
            system=ADVISOR_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in response.content if b.type == "text"), "").strip()
        return text or None

    # Jarvis Phase 6+ — model-augmented reflection. In claude mode the deep
    # tier reviews the agent's telemetry/memory and proposes richer *advisory*
    # guidelines than the deterministic heuristics can. Hard trust boundary:
    # the model may ONLY propose prompt_addendum items, which never auto-apply
    # (they still require a human click), and everything is validated + capped
    # server-side. Any failure returns [] so reflection never breaks.
    REFLECT_SYSTEM = (
        "You audit a personal-dashboard AI agent's own behaviour to suggest small, "
        "durable operating guidelines that would make it more useful. You are given "
        "usage telemetry, long-term memory and current guidelines. Propose at most 3 "
        "NEW guidelines that are not already covered. Reply with ONLY a JSON array of "
        "objects {\"title\": short label, \"rationale\": one sentence, \"text\": the "
        "guideline as an imperative instruction to the agent}. No prose, no code fences. "
        "If nothing is worth changing, reply with []."
    )

    def reflect_candidates(self, context: dict) -> list[dict]:
        if self.mode != "claude":
            return []
        decision = self.router.route("reflection")
        if decision["tier"] != "deep":
            return []  # deep budget exhausted — skip model reflection this pass
        self._log_route(decision)
        try:
            prompt = json.dumps(context, ensure_ascii=False)[:8000]
            response = self._get_client().messages.create(
                model=decision["model"],
                max_tokens=1024,
                system=self.REFLECT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = next((b.text for b in response.content if b.type == "text"), "").strip()
            raw = json.loads(text)
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        out = []
        for item in raw[:3]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()[:80]
            guideline = " ".join(str(item.get("text", "")).split())[:400]
            if not title or not guideline:
                continue
            out.append({
                "kind": "prompt_addendum",
                "title": title,
                "rationale": (str(item.get("rationale", "")).strip()[:200]
                              or "Model-suggested operating guideline."),
                "payload": {"text": guideline, "source": "model"},
            })
        return out

    @staticmethod
    def _last_user_text(messages: list) -> str:
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text")
        return ""

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
            if args.get("price") is not None:
                trigger["price"] = args.get("price")
                trigger["direction"] = args.get("direction") or "above"
            elif args.get("rsi") is not None:
                trigger["rsi"] = args.get("rsi")
                trigger["direction"] = args.get("direction") or "below"
            else:
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
                if trigger.get("price") is not None:
                    when = f"{trigger['symbol']} price {trigger.get('direction', 'above')} ${trigger['price']:,g}"
                elif trigger.get("rsi") is not None:
                    when = f"{trigger['symbol']} RSI {trigger.get('direction', 'below')} {trigger['rsi']:g}"
                else:
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
        claude = self.mode == "claude"
        return {
            "mode": self.mode,
            # In tiered mode there's no single model; report the pin or the
            # default core tier as the headline, with the full table alongside.
            "model": (self.router.pin or self.router.tiers["core"]) if claude else None,
            "routing": self.router.snapshot() if claude else None,
            "permissions": dict(TOOL_TIERS),
            "sdk_installed": _HAVE_SDK,
            "hint": None if claude else (
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
        system, request_messages = self._prepare_claude_request(messages, context)
        decision = self.router.route("chat", self._last_user_text(messages))
        self._log_route(decision)
        response = self._get_client().messages.create(
            model=decision["model"],
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=system,
            tools=DASHBOARD_TOOLS,
            messages=request_messages,
        )
        content = self._convert_content(response)
        escalated = False

        # Advisor escalation: only on a final text turn that self-reports doubt.
        if response.stop_reason == "end_turn":
            text = " ".join(b["text"] for b in content if b.get("type") == "text")
            if needs_escalation(text):
                guidance = self.advise(text, context)
                if guidance:
                    system2 = system + [{"type": "text", "text": ADVISOR_INJECT.format(guidance=guidance)}]
                    d2 = self.router.route("chat", self._last_user_text(messages))
                    self._log_route(d2)
                    response = self._get_client().messages.create(
                        model=d2["model"],
                        max_tokens=4096,
                        thinking={"type": "adaptive"},
                        system=system2,
                        tools=DASHBOARD_TOOLS,
                        messages=request_messages,
                    )
                    content = self._convert_content(response)
                    escalated = True

        return {
            "mode": "claude",
            "tier": decision["tier"],
            "model": decision["model"],
            "escalated": escalated,
            "stop_reason": response.stop_reason,
            "content": content,
        }

    def _prepare_claude_request(self, messages: list, context: dict) -> tuple[list, list]:
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
        # Relevance-ranked recall: surface the facts most pertinent to the
        # current question rather than dumping the whole file (which is newest-
        # biased and drops old-but-relevant facts once memory fills up).
        recalled = ""
        recall = getattr(self.services, "memory_recall", None) if self.services else None
        if recall:
            try:
                facts = recall(self._last_user_text(messages), 14)
                recalled = "\n".join(f"- {f}" for f in facts)
            except Exception:
                recalled = ""
        if not recalled and self.services:  # fallback: legacy tail dump
            recalled = (self.services.memory_read() or "")[-4000:].strip()
        if recalled.strip():
            # after the cached block, so editing memory never invalidates it
            system.append({"type": "text", "text": "Long-term memory about the user (most relevant first):\n" + recalled})
        # learned operating guidelines from self-evolution (approved addenda)
        notes = self.services.agent_notes_read() if self.services else ""
        if notes.strip():
            system.append({"type": "text", "text": notes[-2000:]})
        return system, request_messages

    @staticmethod
    def _convert_content(response) -> list:
        content = []
        for block in response.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
            elif block.type == "thinking":
                # must be echoed back verbatim on the next loop iteration
                content.append(json.loads(block.to_json()))
        return content

    def chat_stream(self, payload: dict):
        """Yields ("delta", {...}) events, then exactly one ("done", {...}).

        The "done" payload matches chat()'s return shape, so the client tool
        loop is transport-agnostic — it just also gets live text on the way.
        """
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        context = payload.get("context") or {}

        if self.mode != "claude":
            result = self._chat_local(messages, context)
            for block in result["content"]:
                if block["type"] == "text" and block["text"]:
                    yield "delta", {"text": block["text"]}
            yield "done", result
            return

        system, request_messages = self._prepare_claude_request(messages, context)
        decision = self.router.route("chat", self._last_user_text(messages))
        self._log_route(decision)
        with self._get_client().messages.stream(
            model=decision["model"],
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=system,
            tools=DASHBOARD_TOOLS,
            messages=request_messages,
        ) as stream:
            for text in stream.text_stream:
                yield "delta", {"text": text}
            response = stream.get_final_message()
        content = self._convert_content(response)
        escalated = False

        # Advisor escalation: consult the deep tier, then stream a confident
        # second pass, all inside the same SSE turn.
        if response.stop_reason == "end_turn":
            text = " ".join(b["text"] for b in content if b.get("type") == "text")
            if needs_escalation(text):
                guidance = self.advise(text, context)
                if guidance:
                    yield "delta", {"text": "\n\n↑ escalating to a deeper model…\n\n"}
                    system2 = system + [{"type": "text", "text": ADVISOR_INJECT.format(guidance=guidance)}]
                    d2 = self.router.route("chat", self._last_user_text(messages))
                    self._log_route(d2)
                    with self._get_client().messages.stream(
                        model=d2["model"],
                        max_tokens=4096,
                        thinking={"type": "adaptive"},
                        system=system2,
                        tools=DASHBOARD_TOOLS,
                        messages=request_messages,
                    ) as stream2:
                        for text in stream2.text_stream:
                            yield "delta", {"text": text}
                        response = stream2.get_final_message()
                    content = self._convert_content(response)
                    escalated = True

        yield "done", {
            "mode": "claude",
            "tier": decision["tier"],
            "model": decision["model"],
            "escalated": escalated,
            "stop_reason": response.stop_reason,
            "content": content,
        }

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

    # -- SA medical assistant (streaming, no tools — a consult) --------------
    MED_LOCAL_REPLY = (
        "SA MedBot needs the live AI engine. Install the anthropic SDK and set "
        "ANTHROPIC_API_KEY (or run `ant auth login`), then restart the server. "
        "It is a decision-support tool grounded in South African guidelines "
        "(STGs/EML, SAMF, HPCSA, SAHPRA) — not a substitute for clinical judgement."
    )

    def med_chat_stream(self, payload: dict):
        """SSE consult with the SA-medical persona. Same event shape as chat_stream."""
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        if self.mode != "claude":
            yield "delta", {"text": self.MED_LOCAL_REPLY}
            yield "done", {"mode": "local", "stop_reason": "end_turn",
                           "content": [{"type": "text", "text": self.MED_LOCAL_REPLY}]}
            return
        # Answer on the CORE tier (Sonnet). If Sonnet self-reports low confidence,
        # the DEEP tier (Opus) is consulted as a scoped advisor and Sonnet finishes
        # with that guidance — so Opus assists only when Sonnet can't.
        core_model = self.router.pin or TIERS["core"]
        self._log_route({"task": "medchat", "tier": "core", "model": core_model})
        system = [{"type": "text", "text": MED_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

        # Retrieval grounding: search PubMed on the question and inject recent
        # abstracts so the answer can be evidence-based and cite sources.
        question = self._last_user_text(messages)
        ground = {"articles": [], "text": ""}
        if getattr(self.services, "pubmed_grounding_cached", None):
            try:
                ground = self.services.pubmed_grounding_cached(question)
            except Exception:
                ground = {"articles": [], "text": ""}
        sources = ground.get("articles", [])
        if sources:
            refs = "\n".join(
                f"[{a['pmid']}] {a['title']} — {a['journal']} ({a['date']})" for a in sources)
            block = ("Recent PubMed literature relevant to the question. Use it to "
                     "ground your answer and cite entries inline by PMID (e.g. [PMID]); "
                     "note publication dates and that these are international unless SA-"
                     "specific. Do NOT fabricate citations beyond this list.\n\n"
                     f"References:\n{refs}\n\nAbstracts:\n{ground.get('text', '')[:6000]}")
            system.append({"type": "text", "text": block})

        def run(sys):
            with self._get_client().messages.stream(
                model=core_model, max_tokens=4096, thinking={"type": "adaptive"},
                system=sys, messages=list(messages),
            ) as stream:
                for text in stream.text_stream:
                    yield "delta", {"text": text}
                self._med_final = stream.get_final_message()

        yield from run(system)
        response = self._med_final
        text = " ".join(b.text for b in response.content if b.type == "text")
        escalated = False

        if response.stop_reason == "end_turn" and needs_escalation(text):
            guidance = self.advise(text, {"clinical_question": question})
            if guidance:
                yield "delta", {"text": "\n\n↑ consulting a senior model…\n\n"}
                system2 = system + [{"type": "text", "text": ADVISOR_INJECT.format(guidance=guidance)}]
                yield from run(system2)
                response = self._med_final
                text = " ".join(b.text for b in response.content if b.type == "text")
                escalated = True

        yield "done", {"mode": "claude", "tier": "core", "model": core_model,
                       "escalated": escalated, "stop_reason": response.stop_reason,
                       "sources": sources,
                       "content": [{"type": "text", "text": text}]}

    # -- summarize ------------------------------------------------------------
    def summarize(self, payload: dict) -> dict:
        title = (payload.get("title") or "").strip()
        content = (payload.get("content") or "").strip()
        kind = (payload.get("kind") or "content").strip()
        if not content:
            raise ValueError("missing content to summarize")
        content = content[:60_000]
        if self.mode == "claude":
            decision = self.router.route("summarize")
            self._log_route(decision)
            response = self._get_client().messages.create(
                model=decision["model"],
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
            decision = self.router.route("briefing")
            self._log_route(decision)
            response = self._get_client().messages.create(
                model=decision["model"],
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
_PRICE_ALERT = re.compile(
    r"^alert me (?:if|when)\s+(\w+)\s+(?:goes|is|crosses|rises|climbs|drops|falls)?\s*"
    r"(above|over|below|under|>|<)\s*\$?\s*([\d,]+(?:\.\d+)?)\.?$", re.I)
_RSI_ALERT = re.compile(
    r"^alert me (?:if|when)\s+(\w+)\s+rsi\s+(?:goes|is|crosses)?\s*"
    r"(above|over|below|under|>|<)\s*(\d+(?:\.\d+)?)\.?$", re.I)
_WORLD_ALERT = re.compile(
    r"^alert me (?:if|when)\s+(?:the\s+)?world\s?(?:state)?\s*(?:index\s*)?"
    r"(?:reaches|goes(?:\s+to)?|hits|is)\s+(watch|elevated|critical)\.?$", re.I)
_LIST_AUTOS = re.compile(r"^(?:list|show)(?:\s+my)?\s+automations?\.?$", re.I)
_DELETE_AUTO = re.compile(r"^(?:delete|remove)\s+automation\s+#?(\d+)\.?$", re.I)


_PRIO_TOKEN = re.compile(r"(?:^|\s)!(high|hi|urgent|low|med|medium|normal)\b", re.I)
_DUE_TOKEN = re.compile(r"(?:^|\s)(?:@|by\s+)(\d{4}-\d{2}-\d{2}|today|tomorrow)\b", re.I)


def parse_task_tokens(text: str) -> tuple[str, str | None, str | None]:
    """Extract inline !priority and @due tokens from task text (mirrors the
    client's parseTaskInput). Returns (clean_text, due|None, priority|None)."""
    priority = None
    due = None

    def _prio(m):
        nonlocal priority
        k = m.group(1).lower()
        priority = ("high" if k in ("high", "hi", "urgent")
                    else "normal" if k in ("med", "medium", "normal") else "low")
        return " "

    def _due(m):
        nonlocal due
        k = m.group(1).lower()
        if k == "today":
            due = date.today().isoformat()
        elif k == "tomorrow":
            due = date.fromordinal(date.today().toordinal() + 1).isoformat()
        else:
            due = m.group(1)
        return " "

    text = _PRIO_TOKEN.sub(_prio, text)
    text = _DUE_TOKEN.sub(_due, text)
    return " ".join(text.split()).strip(), due, priority


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

    match = _PRICE_ALERT.match(text)
    if match:
        symbol = match.group(1).upper()
        direction = "above" if match.group(2).lower() in ("above", "over", ">") else "below"
        price = float(match.group(3).replace(",", ""))
        return [("create_automation", {
            "name": f"{symbol} {direction} ${price:,g} alert", "trigger_type": "market",
            "time": None, "symbol": symbol, "percent": None, "price": price, "rsi": None,
            "direction": direction, "level": None, "action_type": "notify",
            "message": f"{symbol} crossed {direction} ${price:,g}.",
        })], f"Watching {symbol}: I'll alert you when it crosses {direction} ${price:,g}."

    match = _RSI_ALERT.match(text)
    if match:
        symbol = match.group(1).upper()
        direction = "above" if match.group(2).lower() in ("above", "over", ">") else "below"
        level = float(match.group(3))
        return [("create_automation", {
            "name": f"{symbol} RSI {direction} {level:g} alert", "trigger_type": "market",
            "time": None, "symbol": symbol, "percent": None, "price": None, "rsi": level,
            "direction": direction, "level": None, "action_type": "notify",
            "message": f"{symbol} RSI(14) crossed {direction} {level:g}.",
        })], f"Watching {symbol}: I'll alert you when its RSI(14) crosses {direction} {level:g}."

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
        raw_text = match.group(1).strip().rstrip(".")
        task_text, due, priority = parse_task_tokens(raw_text)
        target_list = (match.group(2) or "Today").strip().title()
        detail = ", ".join(filter(None, [
            f"due {due}" if due else None,
            f"{priority} priority" if priority else None]))
        reply = f"Added “{task_text}” to {target_list}" + (f" ({detail})." if detail else ".")
        suggestion = suggest_automation(task_text)
        if suggestion:
            reply += f" Tip: {suggestion}."
        return [("add_task", {"text": task_text, "list": target_list,
                              "due": due, "priority": priority})], reply

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
