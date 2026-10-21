"""The first-task catalog of onboarding's build beat: what the model may put on the build card.

Pure. Gates read only hard facts (this OS, plugins installed this run, apps connected this run);
whether a row fits the person is the model's call. ``agent/onboarding_flow.py`` checks the model's
picks against ``eligible_tasks``; ``agent/onboarding_text.py`` renders each row's reminder.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.onboarding_flow import Facts

MAX_PICKS = 4
MAX_LABEL = 32
# The one fixed row: always last on the card, never one of the model's picks.
FIXED_ID = "hermes"
# The value a task the user names in words records; it is never on the card.
OWN_ID = "own"


@dataclass(frozen=True)
class FirstTask:
    """``skills`` load first, before the work; an entry ``a|b`` means one of them, whichever fits.
    A plugin skill is written ``<catalog plugin>:<skill>``; :func:`task_skills` names it by the name it
    registered under on this install. ``goal`` finishes the sentence "Your job: ..."."""

    id: str
    label: str
    gate: Callable[[Facts, Mapping[str, Any]], bool]
    skills: tuple[str, ...]
    goal: str
    steps: tuple[str, ...]
    output: str
    quality: str
    done: str


def _always(facts: Facts, picks: Mapping[str, Any]) -> bool:
    return True


def _plugin(name: str) -> Callable[[Facts, Mapping[str, Any]], bool]:
    # The plugins card answers with every onboarding plugin installed, including ones from before this run.
    return lambda facts, picks: name in (picks.get("installed") or ())


def _connected(facts: Facts, picks: Mapping[str, Any]) -> bool:
    return bool(picks.get("connected"))


def _macos(facts: Facts, picks: Mapping[str, Any]) -> bool:
    return facts.os == "macos"


FIRST_TASKS: tuple[FirstTask, ...] = (
    FirstTask(
        "hermes", "Explain how Hermes works", lambda facts, picks: False, ("hermes-agent", "claude-design", "p5js"),
        goal="build a living page that shows how Hermes works, from the real docs and this install",
        steps=(
            "Read https://hermes-agent.nousresearch.com/docs/llms.txt once with web_extract. Open a page only to "
            "check a fact you show.",
            "Read this install: the model, how many skills are installed, the connected apps, and what this chat did "
            "(tools called, skills loaded, their name and picks).",
            "Within a minute, open it in the preview pane with the title, those facts and three screen headings. "
            "Then fill one screen at a time.",
            "Screen 1, the agent loop: a pulse runs model, tool, result, back to the model, labelled with the tools "
            "this chat called.",
            "Screen 2, learning: a procedure becomes a skill that loads next time; memory keeps facts across chats. "
            "Use the skills this chat loaded.",
            "Screen 3, out in the world: cron jobs, connectors and messaging, one example each built on their apps.",
        ),
        output="One HTML file in ~/hermes-first-task/, three screens that each fill the pane with one animated "
               "graphic; open in the preview pane.",
        quality="Every name and number comes from the docs or this install. A few short lines per screen.",
        done="All three screens move and show real facts, and you said where the file is."),
    FirstTask(
        "games", "Tune my games for this PC", _plugin("nvidia-app"), ("nvidia-app:nvidia-app",),
        goal="optimise the games on this PC with NVIDIA App and show what changed",
        steps=(
            "Its tools are mcp__nvidia_app__nvapp_*. If none are listed, check without changing anything whether "
            "NVIDIA App is installed and running. Not installed: open https://www.nvidia.com/en-us/software/nvidia-app/ "
            "in the preview pane and say how to install it. Not running: say how to start it. Check again each time "
            "they say done.",
            "List the installed games and show them at once. If the driver status shows a newer Game Ready "
            "driver, say so in one line.",
            "Ask in one clarify which preset to apply (Performance, Visual or Recommended), naming the games.",
            "Apply it, then read each game back.",
        ),
        output="A before and after table in chat: one row per game, what changed.",
        quality="Only settings NVIDIA App reports and really changed. Never guess a value.",
        done="The changes are read back and in the table. Next-step idea: launch one of the games."),
    FirstTask(
        "broadcast", "Make my mic and camera pro", _plugin("nvidia-broadcast"), ("nvidia-broadcast:nvidia-broadcast",),
        goal="set up NVIDIA Broadcast so their mic and camera are ready for a call",
        steps=(
            "Its tools are mcp__nvidia_broadcast__*. If none are listed, check without changing anything whether "
            "NVIDIA Broadcast is installed and running. Not installed or too old: open "
            "https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-app/ in the preview pane and offer the "
            "install as the skill says. Not running: say how to start it. Check again each time they say done.",
            "Read the devices and the effects this machine offers, then ask in one clarify what it is for: a "
            "meeting, a stream or a noisy room.",
            "Turn the effects on one at a time. After each, say what to look at or listen for in the Broadcast "
            "window.",
        ),
        output="A short list in chat: each effect, its device, before and after.",
        quality="Only effects Broadcast reports on this machine.",
        done="The effects are on and read back, and they know how to turn each one off."),
    FirstTask(
        "blender", "Build a 3D scene in Blender", _plugin("blender"), ("blender:blender",),
        goal="build a small 3D scene live in their open Blender, one visible step at a time",
        steps=(
            "Drive the open Blender with its mcp__blender__* tools. Never run Blender from the terminal.",
            "If Blender is not connected (no tools, or get_objects_summary fails), this is a first run: open "
            "https://www.blender.org/lab/mcp-server/ in the preview pane and walk them through it in a short list: "
            "Blender 5.1 or newer open; drag the page's Drag and Drop into Blender button into Blender twice (the "
            "first adds the Blender Lab repository, the second installs the add-on); allow Online access; start the "
            "MCP bridge on port 9876. Check again each time they say done.",
            "Offer three scene ideas in one clarify: two that fit them, one that shows off this machine.",
            "First capture their Blender window with get_screenshot_of_window_as_image, copy the image it saved "
            "over preview.png and open that in the preview pane, so they watch Hermes work. Then build it in a new "
            "scene named Hermes, in camera view, in four calls they can watch: ground and camera, main objects, "
            "materials and light, framing. After each, capture the window over preview.png again; the pane "
            "reloads it.",
            "Render one 1280x720 EEVEE still with render_viewport_to_path, copy the filepath it returns to "
            "render.png and open it in the pane. Look at it before you call it done: lit, not black, framed. Fix "
            "and re-render once at most.",
            "Save only the new scene with bpy.data.libraries.write as scene.blend, and the code as scene.py.",
        ),
        output="scene.blend, scene.py, preview.png and render.png in ~/hermes-first-task/; the render open in the "
               "preview pane.",
        quality="Made in their open Blender, never headless, never a placeholder.",
        done="The final still is shown and you named the files. Next-step idea: a turntable video or a GPU Cycles "
             "render."),
    FirstTask(
        "brief", "Brief me on today", _connected, ("claude-design",),
        goal="write today's brief from their connected apps",
        steps=(
            "Check the connected apps with manage_connections(action=\"status\").",
            "Read today from them: mail, calendar, messages, tasks. Read only.",
            "Show a first version early, then fill it in: what needs attention, what changed, what is next.",
            "Offer a morning cron job that sends the brief. Schedule it only on yes.",
        ),
        output="One designed HTML page in ~/hermes-first-task/, open in the preview pane.",
        quality="Every item comes from a connected app and says which one. Nothing invented.",
        done="The brief is shown and the morning job is set up or declined."),
    FirstTask(
        "week", "Recap my week on one page", _connected, ("weekly-review-planning", "claude-design"),
        goal="make one page that recaps their past seven days from the connected apps",
        steps=(
            "Check the connected apps with manage_connections(action=\"status\").",
            "Read the past seven days from them. Read only.",
            "Show a first version early, then fill it in: meetings, threads, work shipped, open loops.",
            "End with a short plan for next week built from the open loops.",
        ),
        output="One designed HTML page in ~/hermes-first-task/, open in the preview pane.",
        quality="Every item comes from a connected app and says which one. Nothing invented.",
        done="The page is shown with all five parts filled from real data."),
    FirstTask(
        "mac", "Set up my Mac helpers", _macos, ("apple-reminders", "apple-notes", "hermes-agent"),
        goal="set up their Mac helpers: one real reminder, one real note and a morning routine",
        steps=(
            "Ask in one clarify what the reminder and the note should be, with three concrete suggestions.",
            "Make the reminder in Reminders and the note in Notes with the skills.",
            "Propose a morning routine as a cron job: what it does and when. Schedule it only on yes.",
        ),
        output="The reminder and note in their apps and the cron job, each named in chat.",
        quality="Real items they asked for. No test items.",
        done="The reminder and note exist, and the morning job is set up or declined."),
    FirstTask(
        "notes", "Turn my notes into a wiki", _always, ("obsidian", "llm-wiki"),
        goal="turn their notes into a small linked wiki",
        steps=(
            "Ask which folder holds their notes. Never search the disk for it.",
            "Read the notes there and find the three main topics.",
            "Write an index page and three linked topic pages, each citing the notes it came from.",
            "Show the index page.",
        ),
        output="Four Markdown pages in ~/hermes-first-task/wiki/, unless they ask for their notes folder.",
        quality="Only what their notes say. Every link resolves.",
        done="The four pages exist, the links work and the index is shown."),
    FirstTask(
        "explain", "Explain a topic I care about", _always,
        ("grounded-citations", "claude-design", "p5js", "blocked-page-recovery"),
        goal="explain a topic they care about on one page built around a figure they can play with",
        steps=(
            "If the label they clicked names a topic, use it. Otherwise offer three guesses from what you know about "
            "them in one clarify.",
            "Within a minute, open it in the preview pane with the title and the figure already moving: the one "
            "mechanism that makes the topic click, which they can drag or tune.",
            "Research it with citations (load arxiv for a technical topic), then write short sections around the "
            "figure. End on the most surprising thing you found.",
        ),
        output="One HTML file in ~/hermes-first-task/, open in the preview pane, sources at the end.",
        quality="Every claim and every number in the figure is cited. Under 400 words of text.",
        done="The figure responds to input and every source is listed."),
    FirstTask(
        "youtube", "Turn a video into a page", _always, ("youtube-content", "claude-design", "p5js"),
        goal="turn one video into a page that lays its ideas out visually",
        steps=(
            "Ask for the video link.",
            "Get the transcript with the skill.",
            "Pull out the main ideas, each with a timestamp and a quote.",
            "Show a first version early, then lay the ideas out visually, with one figure where it helps.",
        ),
        output="One self-contained HTML file in ~/hermes-first-task/, open in the preview pane; each "
               "timestamp links to that moment in the video.",
        quality="Designed with claude-design. Quotes are exact words from the transcript.",
        done="The page is shown and every idea has its timestamp and quote."),
    FirstTask(
        "watch", "Watch something for me daily", _always,
        ("competitor-news-monitor|product-price-monitor", "grounded-citations"),
        goal="watch something for them every day and report here",
        steps=(
            "Ask what to watch in one clarify, with three concrete options from what you know about them.",
            "Run the first check now and report it with sources.",
            "Propose a daily cron job: when it runs and what it reports. Schedule it only on yes.",
        ),
        output="The first report in chat, and the daily cron job.",
        quality="Every finding has a source link. Nothing invented.",
        done="The first report is shown and the daily job is set up or declined."),
    FirstTask(
        "site", "Design a page for me", _always, ("claude-design", "popular-web-designs", "design-md", "p5js"),
        goal="design one web page for them",
        steps=(
            "If you do not know what the page is for, ask in one line.",
            "Offer three design directions in one clarify.",
            "Show a first version early with a p5js hero, then refine the same file.",
            "Offer one variation.",
        ),
        output="One self-contained HTML file in ~/hermes-first-task/, open in the preview pane.",
        quality="Designed with the skills. Real content they gave you, never filler text.",
        done="The page is shown and a variation is offered."),
    FirstTask(
        "repo", "Map one of my repos", _always, ("codebase-inspection", "architecture-diagram", "github"),
        goal="map one of their code repositories on an architecture page",
        steps=(
            "Ask which repo, as a path or a URL, and ask before reading it.",
            "Show its language breakdown.",
            "Read its entry points and main modules.",
            "Draw the architecture page: the modules, how they connect, where to start reading.",
        ),
        output="One self-contained HTML file in ~/hermes-first-task/, open in the preview pane.",
        quality="Only modules that are really in the repo, named by their paths.",
        done="The page is shown and every box on it maps to a real path."),
)
# A task the user names in words: the same prompt shape, without a catalog row.
OWN_TASK = FirstTask(
    OWN_ID, "", lambda facts, picks: False, ("hermes-agent",),
    goal="build the task they named in their own words",
    steps=(
        "Also load the skills that fit what they asked.",
        "If you know what it is, say the first slice in one line and start. If not, ask one short question "
        "(a clarify with three concrete options is fine).",
        "Cut a big ask to a first slice and say so in one line.",
        "Show a first version early, then refine it.",
    ),
    output="Whatever fits the task; a page is one self-contained HTML file, open in the preview pane.",
    quality="Real data only, designed with the skills when it is a page.",
    done="The first slice is shown and you said where its files are.")
TASK_BY_ID = {task.id: task for task in FIRST_TASKS}
EVERY_TASK_SKILL = "humanizer"


def eligible_tasks(facts: Facts, picks: Mapping[str, Any]) -> tuple[FirstTask, ...]:
    """The rows the model may pick from on this machine and run, in catalog order."""
    return tuple(task for task in FIRST_TASKS if task.id != FIXED_ID and task.gate(facts, picks))


def card_options(chosen: Any) -> list[dict]:
    """The build card: the model's picks as given, then the fixed row."""
    return [*(dict(option) for option in chosen or ()), {"id": FIXED_ID, "label": TASK_BY_ID[FIXED_ID].label}]


def _option_error(option: Any, eligible: set[str], seen: set[str]) -> str | None:
    if not isinstance(option, Mapping) or not isinstance(option.get("id"), str):
        return 'each option is {"id": ..., "label": ...}'
    if option["id"] == FIXED_ID:
        return f"{FIXED_ID} is added last by itself; do not pick it"
    if option["id"] not in eligible:
        return f"{option['id']!r} is not an eligible id"
    if option["id"] in seen:
        return f"{option['id']!r} is picked twice"
    label = option.get("label")
    if not isinstance(label, str) or not label.strip():
        return f"the label for {option['id']} is empty"
    if len(label) > MAX_LABEL:
        return f'"{label}" is {len(label)} characters; the limit is {MAX_LABEL}. Shorten it'
    return None


def pick_error(options: Any, facts: Facts, picks: Mapping[str, Any]) -> str | None:
    """Why the model's build picks are refused, naming the eligible ids; None when they are valid."""
    eligible = [task.id for task in eligible_tasks(facts, picks)]
    problems: list[str] = []
    if not isinstance(options, list) or not options:
        problems.append("options is required")
    elif len(options) > MAX_PICKS:
        problems.append(f"pick at most {MAX_PICKS}")
    else:
        # Every bad option is named at once, so one retry can fix them all.
        seen: set[str] = set()
        for option in options:
            if problem := _option_error(option, set(eligible), seen):
                problems.append(problem)
            elif isinstance(option, Mapping):
                seen.add(option["id"])
    if not problems:
        return None
    problem = "; ".join(problems)
    return (f'{problem}. Call setup_beat(beat="build", options=[{{"id": ..., "label": ...}}]) with up to '
            f"{MAX_PICKS} of these ids: {', '.join(eligible)}.")


def task_skills(option: str, resolve_skill: Callable[[str], str | None]) -> list[str]:
    """The skills a task loads first, humanizer last, each by the name ``skill_view`` loads it with here:
    ``resolve_skill`` maps a written name to that name (a plugin skill's registered name differs from
    its ``<plugin>:<skill>`` form), or to None for a skill missing here, which is dropped."""
    names = [*TASK_BY_ID.get(option, OWN_TASK).skills, EVERY_TASK_SKILL]
    kept = []
    for entry in names:
        found = [resolved for name in entry.split("|") if (resolved := resolve_skill(name))]
        if found:
            kept.append(" or ".join(found))
    return kept
