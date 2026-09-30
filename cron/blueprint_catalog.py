"""Automation Blueprints — one slot-schema definition per automation that every surface renders
natively: dashboard form (``blueprint_form_schema``), pre-filled ``/blueprint`` slash command
(``blueprint_slash_command``), agent seed prompt, docs deep-link. ``fill_blueprint`` validates
values into ``cron.jobs.create_job`` kwargs — there is no second job engine. Users never type raw
cron: ``schedule_template`` fixes the recurrence; only human parts (time, weekdays) are slots.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "BlueprintSlot",
    "AutomationBlueprint",
    "CATALOG",
    "get_blueprint",
    "blueprint_form_schema",
    "blueprint_slash_command",
    "blueprint_deeplink",
    "blueprint_catalog_entry",
    "fill_blueprint",
    "BlueprintFillError",
    "WEEKDAY_PRESETS",
]


class BlueprintFillError(ValueError):
    """Raised when supplied slot values fail validation."""


# Slot types the renderers understand.
_SLOT_TYPES = frozenset({"time", "enum", "text", "weekdays"})

# Named weekday recurrences -> cron day-of-week field.
WEEKDAY_PRESETS: Dict[str, str] = {
    "everyday": "*",
    "weekdays": "1-5",
    "weekends": "0,6",
}


@dataclass(frozen=True)
class BlueprintSlot:
    """A single fillable field on a blueprint."""

    name: str
    type: str
    label: str
    default: Any = None
    options: tuple = ()       # for type="enum": allowed values
    optional: bool = False
    help: str = ""
    # When False, ``options`` are suggestions rather than a closed set — any value is accepted (e.g.
    # the deliver slot, where the real set of valid platforms depends on the user's configured
    # gateways and is validated downstream by the cron scheduler).
    strict: bool = True

    def __post_init__(self) -> None:
        if self.type not in _SLOT_TYPES:
            raise ValueError(f"unknown slot type {self.type!r} (slot {self.name})")


@dataclass(frozen=True)
class AutomationBlueprint:
    """A parameterized automation blueprint."""

    key: str
    title: str
    description: str
    category: str
    # Cron expression with ``{slot}`` placeholders, e.g. "{minute} {hour} * * {dow}".
    # Placeholders are filled from resolved slot values (time -> minute/hour,
    # weekdays -> dow). A literal cron string with no placeholders = fixed schedule.
    schedule_template: str
    # Seed instruction for the agent / the cron job prompt; may contain {slot}s.
    prompt_template: str
    slots: List[BlueprintSlot] = field(default_factory=list)
    deliver_default: str = "origin"
    skills: tuple = ()        # skills the job loads before running
    tags: tuple = ()


# ---------------------------------------------------------------------------
# Curated in-repo catalog
# ---------------------------------------------------------------------------

_TIME = lambda default="08:00": BlueprintSlot(  # noqa: E731 - concise factory
    name="time", type="time", label="What time?", default=default,
    help="24h local time, e.g. 08:00",
)
_DELIVER = BlueprintSlot(
    name="deliver", type="enum", label="Where to deliver?",
    default="origin", options=("origin", "local", "telegram", "discord", "email"),
    optional=False, strict=False,
    help="origin = the chat you set this up from (or your configured home "
    "channel when created from the dashboard); local = save only, no message; "
    "or any connected platform name",
)


CATALOG: List[AutomationBlueprint] = [
    AutomationBlueprint(
        key="morning-brief",
        title="Morning briefing",
        description="A short daily briefing: today's calendar, weather, and "
        "anything urgent waiting on you.",
        category="daily",
        schedule_template="{minute} {hour} * * *",
        prompt_template=(
            "Produce a concise morning briefing for the user: today's calendar "
            "events, the local weather, and any urgent items. When Gmail/Google "
            "Calendar are connected, follow the google-workspace skill's "
            "references/daily-brief.md procedure (exact day window, conflict "
            "detection, meeting prep, mail-to-meeting links). Keep it short and "
            "scannable. If no data sources are connected, give a brief "
            "good-morning with the date and offer to connect calendar/email."
        ),
        slots=[_TIME("08:00"), _DELIVER],
        skills=("google-workspace",),
        tags=("daily", "briefing"),
    ),
    AutomationBlueprint(
        key="important-mail",
        title="Important-mail monitor",
        description="Check your inbox periodically and ping you ONLY about mail "
        "that actually needs attention.",
        category="email",
        schedule_template="*/{interval_min} * * * *",
        prompt_template=(
            "Check the user's inbox for new messages since the last run. Surface "
            "ONLY mail matching: {criteria}. Score candidates with the urgency "
            "classifier and deliver only what clears the bar; if nothing does, "
            "respond with [SILENT]. Requires a connected mail source; if none is "
            "configured, explain how to connect one and stop."
        ),
        slots=[
            BlueprintSlot(
                name="interval_min", type="enum", label="How often?",
                default="30", options=("15", "30", "60"),
                help="minutes between checks",
            ),
            BlueprintSlot(
                name="criteria", type="text",
                label="Only notify me if the mail…",
                default="needs a reply today, is from my manager or family, "
                "or mentions a deadline",
            ),
            _DELIVER,
        ],
        skills=("email-inbox-triage",),
        tags=("email", "monitor"),
    ),
    AutomationBlueprint(
        key="weekly-review",
        title="Weekly review",
        description="A weekly recap: what got done, what's still open, and "
        "what's coming up.",
        category="weekly",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Run the weekly-review-planning skill's procedure for the user: "
            "review the completed week and coming 1-2 weeks across connected "
            "calendar, tasks, notes, and email; surface commitments, stalled "
            "projects, and waiting items; build a capacity-aware plan for next "
            "week. Recommendations and drafts only — no mutations without "
            "approval. Keep the output in the skill's seven-section shape."
        ),
        slots=[
            _TIME("18:00"),
            BlueprintSlot(
                name="day", type="enum", label="Which day?",
                default="sunday",
                options=("sunday", "monday", "friday", "saturday"),
            ),
            _DELIVER,
        ],
        skills=("weekly-review-planning",),
        tags=("weekly", "review"),
    ),
    AutomationBlueprint(
        key="workday-start",
        title="Workday start reminder",
        description="A weekday nudge with your agenda and top priorities.",
        category="daily",
        schedule_template="{minute} {hour} * * 1-5",
        prompt_template=(
            "Give the user a brief weekday start-of-day nudge: today's calendar "
            "and the 1-3 highest-priority things to focus on, inferred from "
            "recent context and any task tools. Encouraging, short, one message."
        ),
        slots=[_TIME("09:00"), _DELIVER],
        tags=("daily", "focus"),
    ),
    AutomationBlueprint(
        key="custom-reminder",
        title="Custom reminder",
        description="A recurring reminder in your own words, on your schedule.",
        category="general",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template="Remind the user: {what}",
        slots=[
            BlueprintSlot(name="what", type="text", label="Remind me to…",
                       default="take a break and stretch"),
            _TIME("14:00"),
            BlueprintSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="everyday",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            _DELIVER,
        ],
        tags=("reminder",),
    ),
    AutomationBlueprint(
        key="evening-winddown",
        title="Evening wind-down",
        description="An end-of-day check-in: tomorrow's calendar at a glance "
        "and anything you should prep tonight.",
        category="daily",
        schedule_template="{minute} {hour} * * *",
        prompt_template=(
            "Give the user a short evening wind-down: tomorrow's calendar, any "
            "early commitments to prep for, and one gentle nudge to wrap up "
            "loose ends from today. Keep it calm and brief — one message. If no "
            "calendar is connected, just offer a friendly sign-off and the "
            "weather for tomorrow."
        ),
        slots=[_TIME("21:00"), _DELIVER],
        tags=("daily", "evening"),
    ),
    AutomationBlueprint(
        key="news-digest",
        title="Topic news digest",
        description="A recurring digest on a topic you care about — deduped "
        "against what was already sent, so only genuinely new items land.",
        category="general",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Search the web for new and noteworthy items about: {topic}. "
            "Dedupe against what you sent in previous runs — only include "
            "genuinely new developments. Deliver a tight digest of at most "
            "{count} bullets, each one line with a link. If nothing new since "
            "last run, respond with [SILENT]."
        ),
        slots=[
            BlueprintSlot(
                name="topic", type="text", label="What topic?",
                default="AI and technology",
                help="a subject, product, person, or search phrase",
            ),
            _TIME("18:00"),
            BlueprintSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="weekdays",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            BlueprintSlot(
                name="count", type="enum", label="How many bullets?",
                default="5", options=("3", "5", "8"),
            ),
            _DELIVER,
        ],
        tags=("digest", "research"),
    ),
    AutomationBlueprint(
        key="bill-renewal-watch",
        title="Bills & renewals reminder",
        description="A heads-up before a recurring payment, subscription "
        "renewal, or due date — so nothing auto-charges by surprise.",
        category="general",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Remind the user about an upcoming payment or renewal: {what}. "
            "Phrase it as an actionable heads-up (e.g. 'review or cancel before "
            "it renews'), not just a notification. One short message."
        ),
        slots=[
            BlueprintSlot(
                name="what", type="text", label="What's due?",
                default="my streaming subscription renews soon",
            ),
            _TIME("10:00"),
            BlueprintSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="everyday",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            _DELIVER,
        ],
        tags=("reminder", "finance"),
    ),
    AutomationBlueprint(
        key="price-watch",
        title="Price & availability watch",
        description="Watch an exact product, flight, hotel, or listing and "
        "alert when your price or availability condition is met.",
        category="general",
        schedule_template="0 */{interval_h} * * *",
        prompt_template=(
            "Load the product-price-monitor skill and run the tick for this "
            "watch: {item}. Alert condition: {condition}. Compare the "
            "normalized all-in price/availability against stored state, "
            "suppress duplicate alerts, and never overwrite last-known-good "
            "state with a failed fetch. If no condition is met, respond with "
            "[SILENT]. On the first run, execute the skill's setup phase "
            "first: pin the exact item, verify one live fetch, and write the "
            "watch contract state file."
        ),
        slots=[
            BlueprintSlot(
                name="item", type="text", label="What exactly to watch?",
                default="a product URL or exact flight/hotel/listing description",
                help="URL or precise description — variant, dates, seller",
            ),
            BlueprintSlot(
                name="condition", type="text", label="Alert me when…",
                default="the all-in price drops below my target",
                help="threshold price (state the currency), availability, or terms change",
            ),
            BlueprintSlot(
                name="interval_h", type="enum", label="How often?",
                default="6", options=("1", "3", "6", "12", "24"),
                help="hours between checks — be gentle with rate limits",
            ),
            _DELIVER,
        ],
        skills=("product-price-monitor",),
        tags=("prices", "shopping", "travel", "monitor"),
    ),
    AutomationBlueprint(
        key="competitor-watch",
        title="Competitor news watch",
        description="Track named companies for material news — launches, "
        "pricing, funding, filings — with a cited digest.",
        category="general",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Load the competitor-news-monitor skill and run the tick for this "
            "watch: companies {companies}; event categories {categories}. "
            "Collect incrementally from the last cutoff, deduplicate by "
            "underlying event, score materiality against the watch contract, "
            "and deliver a cited digest of material events only. If there are "
            "no material events, respond with [SILENT]. On the first run, "
            "execute the skill's setup phase first: freeze the watchlist, "
            "build source coverage, and write the watch contract state file."
        ),
        slots=[
            BlueprintSlot(
                name="companies", type="text", label="Which companies?",
                default="two or three competitors, by canonical name",
                help="canonical names and domains; aliases help dedup",
            ),
            BlueprintSlot(
                name="categories", type="text", label="Which events matter?",
                default="product launches, pricing changes, funding, "
                "partnerships, executive moves, incidents",
            ),
            _TIME("09:00"),
            BlueprintSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="monday",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            _DELIVER,
        ],
        skills=("competitor-news-monitor",),
        tags=("competitors", "news", "monitor", "research"),
    ),
    AutomationBlueprint(
        key="habit-checkin",
        title="Habit check-in",
        description="A recurring nudge to keep a habit on track and reflect "
        "on whether you did it.",
        category="general",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Nudge the user about their habit: {habit}. Ask whether they did it "
            "today, keep it warm and non-judgmental, and offer a one-line word "
            "of encouragement. One short message."
        ),
        slots=[
            BlueprintSlot(
                name="habit", type="text", label="Which habit?",
                default="20 minutes of reading",
            ),
            _TIME("20:00"),
            BlueprintSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="everyday",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            _DELIVER,
        ],
        tags=("habit", "wellbeing"),
    ),
    AutomationBlueprint(
        key="hydration-move",
        title="Hydration & movement nudge",
        description="A periodic nudge during the day to drink water, stand up, "
        "and stretch.",
        category="general",
        # NOTE: cron minute-field steps (*/90) wrap per hour — */90 and */120 both degrade to
        # hourly. Use an hour-field step instead so the chosen cadence is what actually fires.
        schedule_template="0 {start_hour}-{end_hour}/{interval_hours} * * 1-5",
        prompt_template=(
            "Send the user a brief, friendly nudge to drink some water, stand "
            "up, and stretch for a moment. Vary the wording each time so it "
            "doesn't feel robotic. One short line."
        ),
        slots=[
            BlueprintSlot(
                name="interval_hours", type="enum", label="How often?",
                default="1", options=("1", "2", "3"),
                help="hours between nudges",
            ),
            BlueprintSlot(
                name="start_hour", type="enum", label="Start hour",
                default="9", options=("7", "8", "9", "10"),
                help="first hour of the active window (24h)",
            ),
            BlueprintSlot(
                name="end_hour", type="enum", label="End hour",
                default="17", options=("16", "17", "18", "19"),
                help="last hour of the active window (24h)",
            ),
            _DELIVER,
        ],
        tags=("wellbeing", "focus"),
    ),
    AutomationBlueprint(
        key="meal-plan",
        title="Weekly meal plan",
        description="A weekly meal plan plus a consolidated grocery list, "
        "tuned to your diet and how much time you have to cook.",
        category="weekly",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Build the user a meal plan for the coming week: {meals} per day, "
            "suited to a {diet} diet and roughly {effort} cooking effort. "
            "Include a consolidated grocery list grouped by aisle. Keep blueprints "
            "simple and skimmable."
        ),
        slots=[
            BlueprintSlot(
                name="diet", type="enum", label="Diet?",
                default="no restrictions",
                options=("no restrictions", "vegetarian", "vegan",
                         "high-protein", "low-carb"),
            ),
            BlueprintSlot(
                name="meals", type="enum", label="Meals per day?",
                default="dinner only",
                options=("dinner only", "lunch and dinner", "all three"),
            ),
            BlueprintSlot(
                name="effort", type="enum", label="Cooking effort?",
                default="quick", options=("quick", "medium", "ambitious"),
            ),
            _TIME("17:00"),
            BlueprintSlot(
                name="day", type="enum", label="Which day?",
                default="sunday",
                options=("sunday", "monday", "friday", "saturday"),
            ),
            _DELIVER,
        ],
        tags=("weekly", "food"),
    ),
    AutomationBlueprint(
        key="learn-daily",
        title="Daily learning drip",
        description="One bite-sized lesson a day on a topic you want to learn, "
        "building progressively over time.",
        category="daily",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Teach the user one bite-sized lesson about: {topic}. Build on "
            "earlier lessons so it progresses rather than repeating. Keep it to "
            "a couple of short paragraphs with one concrete example, and end "
            "with a single question to check understanding."
        ),
        slots=[
            BlueprintSlot(
                name="topic", type="text", label="Learn about…",
                default="Spanish vocabulary",
            ),
            _TIME("08:30"),
            BlueprintSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="weekdays",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            _DELIVER,
        ],
        tags=("learning", "daily"),
    ),
    AutomationBlueprint(
        key="gratitude-journal",
        title="Gratitude & reflection prompt",
        description="A gentle evening prompt to reflect on the day and note "
        "what went well.",
        category="general",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Send the user a short, warm reflection prompt for the end of the "
            "day — invite them to note one thing that went well, one thing they "
            "are grateful for, and one small win. If they reply, acknowledge it "
            "kindly. One message."
        ),
        slots=[
            _TIME("21:30"),
            BlueprintSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="everyday",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            _DELIVER,
        ],
        tags=("wellbeing", "reflection"),
    ),
    AutomationBlueprint(
        key="on-this-day",
        title="On-this-day discovery",
        description="A daily dose of curiosity: a notable historical event, "
        "fact, or word for the day.",
        category="daily",
        schedule_template="{minute} {hour} * * *",
        prompt_template=(
            "Give the user one interesting '{flavor}' item for today — keep it "
            "short, surprising, and genuinely interesting. One or two sentences, "
            "no filler."
        ),
        slots=[
            BlueprintSlot(
                name="flavor", type="enum", label="What kind?",
                default="on this day in history",
                options=("on this day in history", "word of the day",
                         "science fact", "quote of the day"),
            ),
            _TIME("07:30"),
            _DELIVER,
        ],
        tags=("daily", "curiosity"),
    ),
]

_CATALOG_BY_KEY = {r.key: r for r in CATALOG}


def get_blueprint(key: str) -> Optional[AutomationBlueprint]:
    return _CATALOG_BY_KEY.get(key)


def _slot(
    blueprint: AutomationBlueprint, *names: str, type: Optional[str] = None
) -> Optional[BlueprintSlot]:
    """First slot matching any of *names* (or *type*), else None."""
    return next((s for s in blueprint.slots if s.name in names or (type and s.type == type)), None)


def _slot_default(blueprint: AutomationBlueprint, *names: str) -> Any:
    slot = _slot(blueprint, *names)
    return slot.default if slot else None


# --- Renderers --------------------------------------------------------------------------------

def blueprint_form_schema(blueprint: AutomationBlueprint) -> Dict[str, Any]:
    """Emit the JSON a form renderer (dashboard / GUI) needs for this blueprint."""
    return {
        "key": blueprint.key,
        "title": blueprint.title,
        "description": blueprint.description,
        "category": blueprint.category,
        "tags": list(blueprint.tags),
        "fields": [
            {
                "name": s.name,
                "type": s.type,
                "label": s.label,
                "default": s.default,
                "options": list(s.options),
                "optional": s.optional,
                "strict": s.strict,
                "help": s.help,
            }
            for s in blueprint.slots
        ],
    }


def blueprint_slash_command(blueprint: AutomationBlueprint, values: Optional[Dict[str, Any]] = None) -> str:
    """Build the flattened ``/blueprint <key> slot=val …`` command string. Uses each slot's default
    when ``values`` is omitted (ready-to-paste for docs/dashboard). Free-text slots are quoted."""
    values = values or {}
    parts = [f"/blueprint {blueprint.key}"]
    for s in blueprint.slots:
        val = values.get(s.name, s.default)
        if val is None or val == "":
            if s.optional:
                continue
            val = ""
        sval = str(val)
        if s.type == "text" or " " in sval:
            sval = '"' + sval.replace('"', '\\"') + '"'
        parts.append(f"{s.name}={sval}")
    return " ".join(parts)


def blueprint_deeplink(blueprint: AutomationBlueprint, values: Optional[Dict[str, Any]] = None) -> str:
    """Build the ``hermes://blueprint/<key>?slot=val`` deep-link URL."""
    from urllib.parse import quote, urlencode

    values = values or {}
    query = {
        s.name: str(values.get(s.name, s.default))
        for s in blueprint.slots
        if values.get(s.name, s.default) not in (None, "")
    }
    qs = ("?" + urlencode(query)) if query else ""
    return f"hermes://blueprint/{quote(blueprint.key)}{qs}"


def _humanize_schedule(blueprint: AutomationBlueprint) -> str:
    """A short human-readable description of when a blueprint runs (defaults)."""
    sched = blueprint.schedule_template
    if sched.startswith("*/"):
        every = _slot_default(blueprint, "interval_min") or sched.split("/")[1].split()[0]
        return f"every {every} minutes"
    if "{interval_hours}" in sched:
        every = str(_slot_default(blueprint, "interval_hours") or "1")
        scope = "weekdays, " if "* * 1-5" in sched else ""
        return f"{scope}every hour" if every == "1" else f"{scope}every {every} hours"
    time_slot = _slot(blueprint, type="time")
    when = time_slot.default if time_slot else None
    if "* * 1-5" in sched:
        return f"weekdays at {when}" if when else "every weekday"
    if "{dow}" in sched:
        scope = _slot_default(blueprint, "day", "recurrence") or ""
        if scope and when:
            return f"{scope} at {when}"
        return f"at {when}" if when else "on a schedule"
    return f"daily at {when}" if when else "on a schedule"


def blueprint_catalog_entry(blueprint: AutomationBlueprint) -> Dict[str, Any]:
    """Unified serializable shape (docs generator + dashboard API): form schema + ready-to-paste
    slash command + deep-link URL + human-readable schedule."""
    return {
        **blueprint_form_schema(blueprint),
        "schedule": blueprint.schedule_template,
        "scheduleHuman": _humanize_schedule(blueprint),
        "command": blueprint_slash_command(blueprint),
        "appUrl": blueprint_deeplink(blueprint),
    }


# --- Fill + validate + translate to a create_job spec -----------------------------------------

_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_DAY_TO_DOW = {
    "sunday": "0", "monday": "1", "tuesday": "2", "wednesday": "3",
    "thursday": "4", "friday": "5", "saturday": "6",
}


def _resolve_schedule(blueprint: AutomationBlueprint, values: Dict[str, Any]) -> str:
    """Fill the schedule_template placeholders from resolved slot values."""
    sched = blueprint.schedule_template

    # A free-text `schedule` slot passes through verbatim (full flexibility).
    if values.get("schedule"):
        return str(values["schedule"])

    repl: Dict[str, str] = {}

    if "{minute}" in sched or "{hour}" in sched:
        time_val = values.get("time")
        if not time_val:
            raise BlueprintFillError("a time is required")
        m = _TIME_RE.match(str(time_val).strip())
        if not m:
            raise BlueprintFillError(f"invalid time {time_val!r} — use HH:MM (24h)")
        repl["hour"] = str(int(m.group(1)))
        repl["minute"] = str(int(m.group(2)))

    if "{dow}" in sched:
        if "recurrence" in values:
            preset = str(values.get("recurrence", "everyday")).lower()
            if preset not in WEEKDAY_PRESETS:
                raise BlueprintFillError(
                    f"unknown recurrence {preset!r} — one of {', '.join(WEEKDAY_PRESETS)}"
                )
            repl["dow"] = WEEKDAY_PRESETS[preset]
        elif "day" in values:
            day = str(values.get("day", "")).lower()
            if day not in _DAY_TO_DOW:
                raise BlueprintFillError(f"unknown day {day!r}")
            repl["dow"] = _DAY_TO_DOW[day]
        else:
            repl["dow"] = "*"

    if "{interval_min}" in sched:
        iv = str(values.get("interval_min", "")).strip()
        if not iv.isdigit() or int(iv) <= 0:
            raise BlueprintFillError(f"invalid interval {iv!r} — minutes as a positive integer")
        repl["interval_min"] = iv

    # Remaining {slot} placeholders are filled verbatim from validated enum/text slot values (enum
    # options were already checked in fill_blueprint, so they are safe to interpolate).
    for name in re.findall(r"\{(\w+)\}", sched):
        if name not in repl and name in values:
            repl[name] = str(values[name])

    try:
        return sched.format(**repl)
    except KeyError as e:  # pragma: no cover - template/slot mismatch is a dev error
        raise BlueprintFillError(f"schedule template missing value for {e}") from e


def fill_blueprint(
    blueprint: AutomationBlueprint, values: Dict[str, Any], *,
    origin: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate ``values`` and return ``cron.jobs.create_job`` kwargs.

    Missing required slots raise BlueprintFillError naming the slot (forms show field errors, the
    agent knows what to ask). Unknown slot names are rejected (a typo'd ``tiem=07:15`` must not
    silently create a job with the default time). Strict enums are checked against their options.
    """
    known = {s.name for s in blueprint.slots}
    unknown = sorted(set(values) - known)
    if unknown:
        raise BlueprintFillError(
            f"unknown slot{'s' if len(unknown) > 1 else ''}: "
            f"{', '.join(unknown)} — valid: {', '.join(s.name for s in blueprint.slots)}"
        )
    resolved: Dict[str, Any] = {}
    for s in blueprint.slots:
        raw = values.get(s.name, s.default)
        if raw in (None, ""):
            if s.optional:
                continue
            raise BlueprintFillError(f"missing required value: {s.name} ({s.label})")
        if s.type == "enum" and s.strict and s.options and str(raw) not in {str(o) for o in s.options}:
            raise BlueprintFillError(
                f"{s.name}={raw!r} not allowed — one of {', '.join(map(str, s.options))}"
            )
        resolved[s.name] = raw

    schedule = _resolve_schedule(blueprint, resolved)

    try:
        prompt = blueprint.prompt_template.format(**resolved)
    except KeyError as e:
        raise BlueprintFillError(f"blueprint prompt missing value for {e}") from e

    spec: Dict[str, Any] = {
        "prompt": prompt,
        "schedule": schedule,
        "name": blueprint.title,
        "deliver": resolved.get("deliver", blueprint.deliver_default),
    }
    if blueprint.skills:
        spec["skills"] = list(blueprint.skills)
    if origin is not None:
        spec["origin"] = origin
    return spec
