"""Cron Recipes — parameterized automation templates with typed slots.

A *recipe* is a one-place definition of an automation that every surface
renders natively:

  * Dashboard / GUI app  -> a form (one field per slot)
  * CLI / TUI / messenger -> a pre-filled ``/cron-recipe`` slash command
  * Agent                 -> a seed prompt; it asks for any blank/ambiguous slot
  * Docs catalog          -> a copy-paste command + a ``hermes://`` deep-link

The single source of truth is the slot schema below. ``recipe_form_schema``
emits what a form renderer needs; ``recipe_slash_command`` emits the flattened
one-line command; ``fill_recipe`` validates user-supplied values and turns a
recipe into a ``cron.jobs.create_job`` kwargs dict (so there is no second job
engine). The form-where-there's-a-screen / agent-fills-where-there's-a-chat
split both consume this same module.

Design choice: users never type raw cron. A recipe carries a fixed recurrence
in ``schedule_template`` and parameterizes only the human-friendly parts
(time-of-day, weekday set). Recipes needing full flexibility expose a ``text``
slot named ``schedule`` that passes through verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "RecipeSlot",
    "CronRecipe",
    "CATALOG",
    "get_recipe",
    "recipe_form_schema",
    "recipe_slash_command",
    "recipe_deeplink",
    "recipe_catalog_entry",
    "fill_recipe",
    "RecipeFillError",
    "WEEKDAY_PRESETS",
]


class RecipeFillError(ValueError):
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
class RecipeSlot:
    """A single fillable field on a recipe."""

    name: str
    type: str
    label: str
    default: Any = None
    options: tuple = ()       # for type="enum": allowed values
    optional: bool = False
    help: str = ""
    # When False, ``options`` are suggestions rather than a closed set —
    # any value is accepted (e.g. the deliver slot, where the real set of
    # valid platforms depends on the user's configured gateways and is
    # validated downstream by the cron scheduler).
    strict: bool = True

    def __post_init__(self) -> None:
        if self.type not in _SLOT_TYPES:
            raise ValueError(f"unknown slot type {self.type!r} (slot {self.name})")


@dataclass(frozen=True)
class CronRecipe:
    """A parameterized automation template."""

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
    slots: List[RecipeSlot] = field(default_factory=list)
    deliver_default: str = "origin"
    skills: tuple = ()        # skills the job loads before running
    tags: tuple = ()


# ---------------------------------------------------------------------------
# Curated in-repo catalog
# ---------------------------------------------------------------------------

_TIME = lambda default="08:00": RecipeSlot(  # noqa: E731 - concise factory
    name="time", type="time", label="What time?", default=default,
    help="24h local time, e.g. 08:00",
)
_DELIVER = RecipeSlot(
    name="deliver", type="enum", label="Where to deliver?",
    default="origin", options=("origin", "local", "telegram", "discord", "email"),
    optional=False, strict=False,
    help="origin = the chat you set this up from (or your configured home "
    "channel when created from the dashboard); local = save only, no message; "
    "or any connected platform name",
)


CATALOG: List[CronRecipe] = [
    CronRecipe(
        key="morning-brief",
        title="Morning briefing",
        description="A short daily briefing: today's calendar, weather, and "
        "anything urgent waiting on you.",
        category="daily",
        schedule_template="{minute} {hour} * * *",
        prompt_template=(
            "Produce a concise morning briefing for the user: today's calendar "
            "events, the local weather, and any urgent items. Keep it short and "
            "scannable. If no data sources are connected, give a brief "
            "good-morning with the date and offer to connect calendar/email."
        ),
        slots=[_TIME("08:00"), _DELIVER],
        tags=("daily", "briefing"),
    ),
    CronRecipe(
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
            RecipeSlot(
                name="interval_min", type="enum", label="How often?",
                default="30", options=("15", "30", "60"),
                help="minutes between checks",
            ),
            RecipeSlot(
                name="criteria", type="text",
                label="Only notify me if the mail…",
                default="needs a reply today, is from my manager or family, "
                "or mentions a deadline",
            ),
            _DELIVER,
        ],
        tags=("email", "monitor"),
    ),
    CronRecipe(
        key="weekly-review",
        title="Weekly review",
        description="A weekly recap: what got done, what's still open, and "
        "what's coming up.",
        category="weekly",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Produce a weekly review for the user: what was accomplished this "
            "week, still-open items, and next week's calendar. Pull from "
            "connected sources. Keep it tight."
        ),
        slots=[
            _TIME("18:00"),
            RecipeSlot(
                name="day", type="enum", label="Which day?",
                default="sunday",
                options=("sunday", "monday", "friday", "saturday"),
            ),
            _DELIVER,
        ],
        tags=("weekly", "review"),
    ),
    CronRecipe(
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
    CronRecipe(
        key="custom-reminder",
        title="Custom reminder",
        description="A recurring reminder in your own words, on your schedule.",
        category="general",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template="Remind the user: {what}",
        slots=[
            RecipeSlot(name="what", type="text", label="Remind me to…",
                       default="take a break and stretch"),
            _TIME("14:00"),
            RecipeSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="everyday",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            _DELIVER,
        ],
        tags=("reminder",),
    ),
    CronRecipe(
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
    CronRecipe(
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
            RecipeSlot(
                name="topic", type="text", label="What topic?",
                default="AI and technology",
                help="a subject, product, person, or search phrase",
            ),
            _TIME("18:00"),
            RecipeSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="weekdays",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            RecipeSlot(
                name="count", type="enum", label="How many bullets?",
                default="5", options=("3", "5", "8"),
            ),
            _DELIVER,
        ],
        tags=("digest", "research"),
    ),
    CronRecipe(
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
            RecipeSlot(
                name="what", type="text", label="What's due?",
                default="my streaming subscription renews soon",
            ),
            _TIME("10:00"),
            RecipeSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="everyday",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            _DELIVER,
        ],
        tags=("reminder", "finance"),
    ),
    CronRecipe(
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
            RecipeSlot(
                name="habit", type="text", label="Which habit?",
                default="20 minutes of reading",
            ),
            _TIME("20:00"),
            RecipeSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="everyday",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            _DELIVER,
        ],
        tags=("habit", "wellbeing"),
    ),
    CronRecipe(
        key="hydration-move",
        title="Hydration & movement nudge",
        description="A periodic nudge during the day to drink water, stand up, "
        "and stretch.",
        category="general",
        # NOTE: cron minute-field steps (*/90) wrap per hour — */90 and */120
        # both degrade to hourly. Use an hour-field step instead so the chosen
        # cadence is what actually fires.
        schedule_template="0 {start_hour}-{end_hour}/{interval_hours} * * 1-5",
        prompt_template=(
            "Send the user a brief, friendly nudge to drink some water, stand "
            "up, and stretch for a moment. Vary the wording each time so it "
            "doesn't feel robotic. One short line."
        ),
        slots=[
            RecipeSlot(
                name="interval_hours", type="enum", label="How often?",
                default="1", options=("1", "2", "3"),
                help="hours between nudges",
            ),
            RecipeSlot(
                name="start_hour", type="enum", label="Start hour",
                default="9", options=("7", "8", "9", "10"),
                help="first hour of the active window (24h)",
            ),
            RecipeSlot(
                name="end_hour", type="enum", label="End hour",
                default="17", options=("16", "17", "18", "19"),
                help="last hour of the active window (24h)",
            ),
            _DELIVER,
        ],
        tags=("wellbeing", "focus"),
    ),
    CronRecipe(
        key="meal-plan",
        title="Weekly meal plan",
        description="A weekly meal plan plus a consolidated grocery list, "
        "tuned to your diet and how much time you have to cook.",
        category="weekly",
        schedule_template="{minute} {hour} * * {dow}",
        prompt_template=(
            "Build the user a meal plan for the coming week: {meals} per day, "
            "suited to a {diet} diet and roughly {effort} cooking effort. "
            "Include a consolidated grocery list grouped by aisle. Keep recipes "
            "simple and skimmable."
        ),
        slots=[
            RecipeSlot(
                name="diet", type="enum", label="Diet?",
                default="no restrictions",
                options=("no restrictions", "vegetarian", "vegan",
                         "high-protein", "low-carb"),
            ),
            RecipeSlot(
                name="meals", type="enum", label="Meals per day?",
                default="dinner only",
                options=("dinner only", "lunch and dinner", "all three"),
            ),
            RecipeSlot(
                name="effort", type="enum", label="Cooking effort?",
                default="quick", options=("quick", "medium", "ambitious"),
            ),
            _TIME("17:00"),
            RecipeSlot(
                name="day", type="enum", label="Which day?",
                default="sunday",
                options=("sunday", "monday", "friday", "saturday"),
            ),
            _DELIVER,
        ],
        tags=("weekly", "food"),
    ),
    CronRecipe(
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
            RecipeSlot(
                name="topic", type="text", label="Learn about…",
                default="Spanish vocabulary",
            ),
            _TIME("08:30"),
            RecipeSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="weekdays",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            _DELIVER,
        ],
        tags=("learning", "daily"),
    ),
    CronRecipe(
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
            RecipeSlot(
                name="recurrence", type="weekdays", label="Repeat on",
                default="everyday",
                options=tuple(WEEKDAY_PRESETS.keys()),
            ),
            _DELIVER,
        ],
        tags=("wellbeing", "reflection"),
    ),
    CronRecipe(
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
            RecipeSlot(
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


def get_recipe(key: str) -> Optional[CronRecipe]:
    return _CATALOG_BY_KEY.get(key)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def recipe_form_schema(recipe: CronRecipe) -> Dict[str, Any]:
    """Emit the JSON a form renderer (dashboard / GUI) needs for this recipe."""
    return {
        "key": recipe.key,
        "title": recipe.title,
        "description": recipe.description,
        "category": recipe.category,
        "tags": list(recipe.tags),
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
            for s in recipe.slots
        ],
    }


def recipe_slash_command(recipe: CronRecipe, values: Optional[Dict[str, Any]] = None) -> str:
    """Build the flattened ``/cron-recipe <key> slot=val …`` command string.

    Uses each slot's default when ``values`` is omitted, so the docs/dashboard
    can show a ready-to-paste command. Free-text slots are quoted.
    """
    values = values or {}
    parts = [f"/cron-recipe {recipe.key}"]
    for s in recipe.slots:
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


def recipe_deeplink(recipe: CronRecipe, values: Optional[Dict[str, Any]] = None) -> str:
    """Build the ``hermes://cron-recipe/<key>?slot=val`` deep-link URL."""
    from urllib.parse import quote, urlencode

    values = values or {}
    query = {}
    for s in recipe.slots:
        val = values.get(s.name, s.default)
        if val not in (None, ""):
            query[s.name] = str(val)
    qs = ("?" + urlencode(query)) if query else ""
    return f"hermes://cron-recipe/{quote(recipe.key)}{qs}"


def _humanize_schedule(recipe: CronRecipe) -> str:
    """A short human-readable description of when a recipe runs (defaults)."""
    sched = recipe.schedule_template
    if sched.startswith("*/"):
        iv = next((s for s in recipe.slots if s.name == "interval_min"), None)
        every = (iv.default if iv else None) or sched.split("/")[1].split()[0]
        return f"every {every} minutes"
    if "{interval_hours}" in sched:
        iv = next((s for s in recipe.slots if s.name == "interval_hours"), None)
        every = str((iv.default if iv else None) or "1")
        scope = "weekdays, " if "* * 1-5" in sched else ""
        return f"{scope}every hour" if every == "1" else f"{scope}every {every} hours"
    time_slot = next((s for s in recipe.slots if s.type == "time"), None)
    when = time_slot.default if time_slot else None
    if "* * 1-5" in sched:
        return f"weekdays at {when}" if when else "every weekday"
    if "{dow}" in sched:
        day_slot = next((s for s in recipe.slots if s.name in ("day", "recurrence")), None)
        scope = (day_slot.default if day_slot else "") or ""
        if scope and when:
            return f"{scope} at {when}"
        return f"at {when}" if when else "on a schedule"
    if when:
        return f"daily at {when}"
    return "on a schedule"


def recipe_catalog_entry(recipe: CronRecipe) -> Dict[str, Any]:
    """Unified serializable shape for a recipe — used by the docs generator
    and the dashboard API. Combines the form schema, the ready-to-paste slash
    command, the deep-link URL, and a human-readable schedule.
    """
    return {
        **recipe_form_schema(recipe),
        "schedule": recipe.schedule_template,
        "scheduleHuman": _humanize_schedule(recipe),
        "command": recipe_slash_command(recipe),
        "appUrl": recipe_deeplink(recipe),
    }


# ---------------------------------------------------------------------------
# Fill + validate + translate to a create_job spec
# ---------------------------------------------------------------------------

_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
_DAY_TO_DOW = {
    "sunday": "0", "monday": "1", "tuesday": "2", "wednesday": "3",
    "thursday": "4", "friday": "5", "saturday": "6",
}


def _resolve_schedule(recipe: CronRecipe, values: Dict[str, Any]) -> str:
    """Fill the schedule_template placeholders from resolved slot values."""
    sched = recipe.schedule_template

    # A free-text `schedule` slot passes through verbatim (full flexibility).
    if "schedule" in values and values["schedule"]:
        return str(values["schedule"])

    repl: Dict[str, str] = {}

    # time -> minute/hour
    time_val = values.get("time")
    if "{minute}" in sched or "{hour}" in sched:
        if not time_val:
            raise RecipeFillError("a time is required")
        m = _TIME_RE.match(str(time_val).strip())
        if not m:
            raise RecipeFillError(f"invalid time {time_val!r} — use HH:MM (24h)")
        repl["hour"] = str(int(m.group(1)))
        repl["minute"] = str(int(m.group(2)))

    # weekday set -> dow
    if "{dow}" in sched:
        if "recurrence" in values:
            preset = str(values.get("recurrence", "everyday")).lower()
            if preset not in WEEKDAY_PRESETS:
                raise RecipeFillError(
                    f"unknown recurrence {preset!r} — one of {', '.join(WEEKDAY_PRESETS)}"
                )
            repl["dow"] = WEEKDAY_PRESETS[preset]
        elif "day" in values:
            day = str(values.get("day", "")).lower()
            if day not in _DAY_TO_DOW:
                raise RecipeFillError(f"unknown day {day!r}")
            repl["dow"] = _DAY_TO_DOW[day]
        else:
            repl["dow"] = "*"

    # interval (minutes) for */N schedules
    if "{interval_min}" in sched:
        iv = str(values.get("interval_min", "")).strip()
        if not iv.isdigit() or int(iv) <= 0:
            raise RecipeFillError(f"invalid interval {iv!r} — minutes as a positive integer")
        repl["interval_min"] = iv

    # Any remaining {slot} placeholders are filled verbatim from validated
    # enum/text slot values (e.g. an hour-range window). Enum options have
    # already been checked in fill_recipe, so these are safe to interpolate.
    for name in re.findall(r"\{(\w+)\}", sched):
        if name not in repl and name in values:
            repl[name] = str(values[name])

    try:
        return sched.format(**repl)
    except KeyError as e:  # pragma: no cover - template/slot mismatch is a dev error
        raise RecipeFillError(f"schedule template missing value for {e}") from e


def fill_recipe(
    recipe: CronRecipe,
    values: Dict[str, Any],
    *,
    origin: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate ``values`` and return ``cron.jobs.create_job`` kwargs.

    Missing required (non-optional) slots raise RecipeFillError naming the
    slot, so a form can show field errors and the agent knows what to ask.
    Unknown slot names are rejected (a typo'd ``tiem=07:15`` must not silently
    create a job with the default time). Enum values are checked against their
    options. The result is passed straight to ``create_job`` — no second schema.
    """
    known = {s.name for s in recipe.slots}
    unknown = sorted(set(values) - known)
    if unknown:
        raise RecipeFillError(
            f"unknown slot{'s' if len(unknown) > 1 else ''}: "
            f"{', '.join(unknown)} — valid: {', '.join(s.name for s in recipe.slots)}"
        )
    resolved: Dict[str, Any] = {}
    for s in recipe.slots:
        raw = values.get(s.name, s.default)
        if raw in (None, ""):
            if s.optional:
                continue
            raise RecipeFillError(f"missing required value: {s.name} ({s.label})")
        if s.type == "enum" and s.strict and s.options and str(raw) not in {str(o) for o in s.options}:
            raise RecipeFillError(
                f"{s.name}={raw!r} not allowed — one of {', '.join(map(str, s.options))}"
            )
        resolved[s.name] = raw

    schedule = _resolve_schedule(recipe, resolved)

    # Render the prompt with whatever slots it references.
    try:
        prompt = recipe.prompt_template.format(**resolved)
    except KeyError as e:
        raise RecipeFillError(f"recipe prompt missing value for {e}") from e

    spec: Dict[str, Any] = {
        "prompt": prompt,
        "schedule": schedule,
        "name": recipe.title,
        "deliver": resolved.get("deliver", recipe.deliver_default),
    }
    if recipe.skills:
        spec["skills"] = list(recipe.skills)
    if origin is not None:
        spec["origin"] = origin
    return spec
