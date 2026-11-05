"""The words of first-run onboarding: the scripted greeting, the opening prompt, and the reminder
that rides on every ``setup_beat`` result. Pure; ``reminder(key, record, facts)`` renders the key
``agent/onboarding_flow.py::apply`` returns.

The rules for the whole run live in the opening prompt (a hidden user turn, so the system prompt
stays byte-stable); reminders say only what was picked and what comes next.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from agent.onboarding_first_tasks import (
    FIXED_ID, MAX_LABEL, MAX_PICKS, OWN_TASK, TASK_BY_ID, eligible_tasks, task_skills,
)
from agent.onboarding_flow import Facts, next_beat
from tools.set_theme_tool import ACCENT_SWATCHES

CHAT_TITLE = "Getting started"
GREETING = "Hi, I'm Hermes. Let's set things up for you. Then we'll get something cool done."

# The scripted opening in each desktop UI language (apps/desktop/src/i18n/languages.ts), played before
# the model speaks; the model writes the rest of the run in the user's language itself. Each greeting
# is the desktop's first-run intro copy (firstRun.intro in apps/desktop/src/i18n/*_first_run.ts), its
# two lines joined: keep the two equal.
_GREETINGS = {
    "en": GREETING,
    "ar": "مرحبًا، أنا Hermes. لنجهّز كل شيء لك، ثم ننجز معًا شيئًا رائعًا.",
    "de": "Hallo, ich bin Hermes. Richten wir alles für Sie ein. Danach machen wir etwas Cooles.",
    "es": "Hola, soy Hermes. Vamos a dejarlo todo listo para ti. Luego haremos algo genial.",
    "fr": "Bonjour, je suis Hermes. Configurons tout pour vous. Ensuite, nous ferons quelque chose de génial.",
    "ja": "こんにちは、Hermes です。まずはあなたに合わせて設定しましょう。そのあと、何かすごいものを作りましょう。",
    "ru": "Привет, я Hermes. Давайте всё настроим под вас. А потом сделаем что-нибудь классное.",
    "zh": "你好，我是 Hermes。我们先帮你把一切设置好，然后一起做点很酷的事。",
    "zh-hant": "你好，我是 Hermes。我們先幫你把一切設定好，然後一起做點很酷的事。",
}

_NAME_QUESTIONS = {
    "ar": "بماذا أناديك؟",
    "de": "Wie darf ich Sie nennen?",
    "es": "¿Cómo quieres que te llame?",
    "fr": "Comment dois-je vous appeler ?",
    "ja": "なんとお呼びすればいいですか？",
    "ru": "Как к вам обращаться?",
    "zh": "我该怎么称呼你？",
    "zh-hant": "我該怎麼稱呼你？",
}

_TRADITIONAL_CHINESE = {"hant", "tw", "hk", "mo"}


def ui_language(locale: str | None) -> str:
    """A locale (``zh-hant``, ``de-DE``, ``zh_TW``) as the desktop language it reads in; English when
    none fits."""
    tags = (locale or "").strip().lower().replace("_", "-").split("-")
    if tags[0] == "zh":
        return "zh-hant" if _TRADITIONAL_CHINESE.intersection(tags[1:]) else "zh"
    return tags[0] if tags[0] in _GREETINGS else "en"


def greeting(locale: str | None) -> str:
    return _GREETINGS[ui_language(locale)]


def question(beat: str, locale: str | None) -> str:
    """The card's own question; the name card asks in the user's language."""
    if beat == "name":
        return _NAME_QUESTIONS.get(ui_language(locale), QUESTIONS["name"])
    return QUESTIONS[beat]

QUESTIONS = {
    "name": "What should I call you?",
    "theme": "Pick a colour.",
    "layout": "Which layout?",
    "connectors": "Which apps should I work with?",
    "plugins": "Want any of these?",
    "tour": "Want a look around first?",
    "build": "Know what you'd like me to make?",
}

_SET_THEME = 'tool_call {"calls":[{"name":"set_theme","arguments":{"accent":"violet"}}]}'
_TOUR_CALL = 'tool_call {"calls":[{"name":"gui_tour","arguments":{"action":"start","preset":"%s"}}]}'

PROFILE_DIMENSIONS = (
    ("agent experience", "new to AI apps / chat apps only / has used local agents / builds agents"),
    ("technical depth", "non-developer / tinkerer / developer"),
    ("creative work", "none / hobby / regular"),
    ("gaming", "none / casual / enthusiast"),
    ("local compute", "cloud only / small local models / strong local (Spark, large GPU or unified memory)"),
    ("work context", "personal / work / mixed"),
    ("machine role", "daily driver / dedicated AI box / remote server"),
)

_NVIDIA = {
    "rtx_spark": "It is an NVIDIA RTX Spark (Windows on Arm), built to run local models.",
    "dgx_spark": "It is an NVIDIA DGX Spark (Linux on Arm with a GB10), built to run local models.",
    "discrete": "It has an NVIDIA GPU.",
}


def facts_block(facts: Facts) -> str:
    whose = ("the computer running Hermes, which is not the one in front of the user"
             if facts.facts_source == "remote" else "this computer")
    lines = [f"What you know about {whose}: {facts.os} on {facts.arch or 'an unknown architecture'}, "
             f"a {facts.machine_kind}."]
    if facts.nvidia_platform in _NVIDIA:
        lines.append(_NVIDIA[facts.nvidia_platform])
    if facts.locale:
        lines.append(f"Locale: {facts.locale}.")
    lines.append("Never recite these. A fact that is missing is unknown. When the user says otherwise, "
                 "believe the user.")
    return " ".join(lines)


def opening_prompt(facts: Facts) -> str:
    """The hidden first turn of the onboarding chat: every rule of the run."""
    dimensions = "\n".join(f"- {name}: {buckets}" for name, buckets in PROFILE_DIMENSIONS)
    return f"""[System note: This is the user's first time opening the Hermes desktop app. You run their first-run onboarding in this chat. The beats, in order: their name, the theme (accent colour; light or dark only if they ask), the layout, connectors, plugins, a tour, then the first thing you build for them.

How it works:
- setup_beat shows a beat's card and waits for the user. setup_beat(beat="next") shows the next beat. Its result carries their answer and a reminder of what to do next. Follow the reminder.
- Before a card, acknowledge in a few words and at most one more sentence. The card asks its own question and shows its own options: never ask it yourself, never list the options in text.
- After a pick, say it by its label in three words or fewer. No opinion, no praise, never the same words twice.
- If they write a message while a card waits, the card closes unanswered and their message comes as their next turn (words typed on the name card are the name). If the message answers the card, apply it yourself and move on. Otherwise answer it properly and briefly, then bring the card back in one line with setup_beat(beat=<that beat>). Never show a card again unless you are bringing it back.
- If they ask you to pick (a colour, a layout), pick from what you know and say why in one line.
- If they skip a beat in words, call setup_beat(beat=<that beat>, skip=true), take the default, and never ask it again in the same form. If they want to stop altogether, say in one line that the Skip setup button ends it, then carry on; only they can end it, with that button.
- Off-script requests are done now, then you bring the card back.
- You may look around this computer when they say yes: installed apps, developer tools, the OS and the hardware, read with the terminal. Never personal files, documents, browser data or messages without a specific ask. A yes to looking is an off-script request: act on it at any beat.

Voice: every word you write is spoken to the user. Never think out loud. Never mention beats, cards, tools, reminders or a script, and never call this setup or onboarding (the Skip setup button's name is the one exception). One or two short, plain sentences per message: warm, like the front desk of somewhere good. No thanks, no praise, no stock lines, no em dashes, no exclamation marks. Write in the user's language when the locale is not English, labels included.

{facts_block(facts)}
If they ask what you know about them, answer truthfully: these machine basics and what they picked.

Tools you reach through tool_call (their call shapes):
- {_SET_THEME}. accent: a swatch id ({", ".join(ACCENT_SWATCHES)}) or a hex colour #rrggbb; mode: light, dark or system; layout: sidebar-left (Basic) or terminal-deck (Elite).
- {_TOUR_CALL % "quick"} (preset quick or full).

Local models: the model picker chooses one. Explain the download and how it fits this hardware first, then point to Settings, Providers, Local Models. Never install a model runtime or download a model yourself.
Ask before sending, deleting or scheduling anything. Never turn off a security setting.

As you go, form a coarse read of them on these seven dimensions, from what they say and pick and from what you see when they let you look. Use it to choose suggestions and words; never say it out loud. Presence of an app is not use, and absence is unknown.
{dimensions}

The app has already greeted them and is asking their name.]"""


# ── reminders ─────────────────────────────────────────────────────────────────────────────────

_LOOK_AROUND = (
    "Before the card, offer once to look around this computer and suggest what to connect. Only on yes, and "
    "only installed apps, developer tools, the OS and the hardware; never personal files, documents, browser "
    "data or messages without a specific ask. Then show the card with what you suggest: "
    'setup_beat(beat="connectors", suggest=[<connector slugs>]). If they say no, show it with '
    'setup_beat(beat="connectors").')

# How a typed answer settles each beat without the card.
_TYPED_ANSWER = {
    "name": 'setup_beat(beat="name", value="<their name>")',
    "theme": _SET_THEME.replace('"accent":"violet"', '"accent":"<colour>"'),
    "layout": 'tool_call {"calls":[{"name":"set_theme","arguments":{"layout":"sidebar-left" or "terminal-deck"}}]}',
    "connectors": 'setup_beat(beat="connectors", value="continue") for "done" or "continue", skip=true for "skip"',
    "plugins": 'setup_beat(beat="plugins", value="continue") for "done" or "continue", skip=true for "skip"',
    "tour": 'setup_beat(beat="tour", value="quick", "full" or "skip"), bucketing their words into one of the three',
    "build": ('setup_beat(beat="build", value=<id>) for an option on the card; a task of their own is '
              'setup_beat(beat="build", value="own"), then build it'),
}

_CLOSE = ('When the first thing is built and shown, ask with clarify: "How does this look?" with the choices '
          '"Looks right", "Change something" and "Take it further", exactly. Put the next-step idea in your text. '
          'After they answer, call setup_beat(beat="done").')

_HOW_TO_WORK = (
    "Real data only: this computer, their connected apps, the live web. Never mock or invent data.",
    "First reply: one line, then a tool call. At most two checks and one question before you start. Something "
    "visible within a minute, finished within five.",
    "Save every file in ~/hermes-first-task/ and say where. No test files.",
    "Install or connect anything once, with exact ids. Never search the whole disk.",
    "Write the words they read with humanizer: short, plain, specific.",
)

_PREVIEW_CALL = 'tool_call {"calls":[{"name":"desktop_preview","arguments":{"action":"open","url":"<file>"}}]}'


def _preview_line(facts: Facts) -> str:
    """Every host shows results in the app's preview pane, which watches a local file but not a remote one."""
    reload = ("On this host it does not reload by itself: open it again after each change."
              if facts.facts_source == "remote" else
              "Open it once, early: it reloads when the file changes, so keep improving the same file while they "
              "watch.")
    return (f"Show results in the preview pane beside the chat with {_PREVIEW_CALL}; it shows HTML, images and "
            f"PDFs. {reload} Say it is open only after the call succeeds.")


_NPM = "https://cdn.jsdelivr.net/npm/"
_PAGE = (
    "One self-contained HTML file, laid out for the pane (often about 600px wide) and able to widen.",
    "A standalone page: it sets its own background, font and colours. The preview pane does not theme it; "
    "the theme variables are only for ::preview widgets in the chat.",
    "It moves within a second: a diagram that draws itself, a dot that runs a path, numbers that count up to "
    "real values.",
    f"Libraries, only what the page needs, each pinned as {_NPM}<package>@<version>/<file>: gsap@3.13.0 for "
    "motion, with its DrawSVGPlugin, MotionPathPlugin and ScrollTrigger (register each); d3@7.9.0 for charts of "
    "real numbers; p5@1.11.3 for a 2D figure they drag or tune; matter-js@0.20.0 for physics; katex@0.16.11 "
    "with its CSS and auto-render for maths; tone@15.0.4 for sound, started on a click; roughjs@4.6.6 for a "
    "hand-drawn look.",
    '3D: three@0.170.0 is ES modules only. Use an import map: "three" to its build/three.module.js, '
    '"three/addons/" to its examples/jsm/.',
)
_PAGE_TASKS = ("hermes", "explain", "brief", "week", "youtube", "site", "repo", "own")

_CONNECTORS = (
    "Never route around a connector. "
    'Reading a connected app: check it with manage_connections(action="status"). Find its tools with tool_search '
    "using the app slug, then the slug plus an action (a bare slug misses actions). tool_describe every action you "
    'need in one batch, then call them through tool_call(calls=[{"name": "connectors__<slug>__<ACTION>", '
    '"arguments": {...}}]). Resolve names to ids before acting. Read only; ask before any write, and say that a '
    "write may run as the connector's bot.")
_CONNECTOR_TASKS = ("brief", "week")

_OS_LINES = {
    "macos": ("This Mac: check with `which brew` and `ls -d /Applications/<App>.app`; install with "
              "`brew install --cask <app>`; on Apple silicon check `lipo -archs` for a binary's architecture."),
    "windows": ("This PC: check with `winget list --accept-source-agreements`; install with `winget install <id> "
                "--accept-source-agreements --accept-package-agreements` (without both flags a fresh profile hangs "
                "on an agreements prompt)."),
    "linux": ("This Linux machine: check with `which`, and `flatpak list` when flatpak is there; install with "
              "`flatpak install -y flathub <id>`. Anything that needs sudo is listed for the user, never run."),
}
_ARM_WINDOWS = ("On Windows on Arm, read arm64 vs x64 from winget's install output, not the manifest, and say when "
                "an app is x64 only.")
_SPARK_LINE = "This machine is built for local models; say so once when it fits what they asked."
_REMOTE_LINE = "Hermes runs on another computer than the user's: say which machine anything is installed on."


def _task_prompt(option: str, label: str, resolve_skill: Callable[[str], str | None]) -> list[str]:
    task = TASK_BY_ID.get(option, OWN_TASK)
    picked = f'They picked "{label or task.label}". ' if (label or task.label) else ""
    steps = [f"{n}. {step}" for n, step in enumerate(task.steps, 1)]
    return [f"{picked}Your job: {task.goal}.",
            f"Load these skills with skill_view first: {', '.join(task_skills(option, resolve_skill))}.",
            "Steps:", *steps, f"Output: {task.output}", f"Quality bar: {task.quality}", f"Done when: {task.done}"]


def _bullets(title: str, lines) -> str:
    return "\n".join([f"{title}:", *(f"- {line}" for line in lines)])


def build_guidance(option: str, facts: Facts, resolve_skill: Callable[[str], str | None] = lambda name: name,
                   label: str = "") -> str:
    """The prompt once a first task is picked; ``label`` is the card text they clicked, ``resolve_skill``
    names a skill as this install loads it (None: missing here)."""
    work = [*_HOW_TO_WORK, _preview_line(facts), *((_CONNECTORS,) if option in _CONNECTOR_TASKS else ())]
    page = [_bullets("When you make a page", _PAGE)] if option in _PAGE_TASKS else []
    host = [_OS_LINES.get(facts.os, _OS_LINES["linux"])]
    if facts.os == "windows" and facts.arch == "arm64":
        host.append(_ARM_WINDOWS)
    if facts.nvidia_platform in ("rtx_spark", "dgx_spark"):
        host.append(_SPARK_LINE)
    if facts.facts_source == "remote":
        host.append(_REMOTE_LINE)
    return "\n\n".join(["\n".join(_task_prompt(option, label, resolve_skill)), _bullets("How to work", work), *page,
                         _bullets("This computer", host), f"When it is shown:\n{_CLOSE}"])


def _names(values) -> str:
    return ", ".join(values) if values else "none"


def _build_menu(record: Mapping[str, Any]) -> str:
    """The rows the model picks the build card from."""
    rows = eligible_tasks(Facts.from_stored(record["facts"]), record["picks"])
    listed = "\n".join(f"- {task.id} ({task.label}): {task.goal}." for task in rows)
    return (f"For the build card, pick up to {MAX_PICKS} of these that fit what you learned about them, earlier rows "
            f'first when they fit, and call setup_beat(beat="build", options=[{{"id": ..., "label": ...}}]). Reword '
            f"each label to fit them and keep its energy: a short, specific action they would want to click, "
            f"{MAX_LABEL} characters max. Ids only from this list. \"{TASK_BY_ID[FIXED_ID].label}\" is added last "
            f"by itself.\n{listed}")


def _next(record: Mapping[str, Any]) -> str:
    beat = next_beat(record)
    if beat is None:
        return 'Every beat is settled; after the build is shown and its close card answered, call setup_beat(beat="done").'
    line = f'Next: setup_beat(beat="next") shows the {beat} card.'
    return f"{line} {_build_menu(record)}" if beat == "build" else line


def _after_layout(record, facts) -> str:
    label = {"sidebar-left": "Basic", "terminal-deck": "Elite"}.get(record["picks"]["layout"], record["picks"]["layout"])
    return f"Layout {label} is on, already applied. Do not call set_theme for it again. {_next(record)}"


def _after_connectors(record, facts) -> str:
    picks = record["picks"]
    again = " These are already connected; do not connect them again." if picks["connected"] else ""
    return (f"Connected: {_names(picks['connected'])}; newly connected: {_names(picks['newly_connected'])}."
            f"{again} {_next(record)}")


def _after_plugins(record, facts) -> str:
    installed = record["picks"]["installed"]
    again = " They are already installed; do not install them again." if installed else ""
    return f"Installed plugins: {_names(installed)}.{again} {_next(record)}"


def _tour(preset: str):
    return lambda record, facts: (
        f'Say "Here\'s where things live." Then, in this same turn, call {_TOUR_CALL % preset} and then show the '
        f"build card: it waits under the tour. Never bring the tour up again. {_build_menu(record)}")


def _waiting_beat(record) -> str | None:
    return (record.get("waiting") or {}).get("beat") or next_beat(record)


_REMINDERS = {
    "after_name": lambda r, f: f"They go by {r['picks']['name']}. Greet them by name in a few words. {_next(r)}",
    "after_theme": lambda r, f: (f"Accent {r['picks']['accent'] or 'Nous blue'} is on, already applied. Do not call "
                                 f"set_theme for it again. React in three words or fewer. {_next(r)}"),
    "after_layout": _after_layout,
    "offer_first": lambda r, f: f"The connectors card is not shown yet. {_LOOK_AROUND}",
    "after_connectors": _after_connectors,
    "after_plugins": _after_plugins,
    "after_tour": lambda r, f: f"The tour is already running; do not start it again. {_next(r)}",
    "after_build": lambda r, f: _next(r),
    "tour_quick": _tour("quick"),
    "tour_full": _tour("full"),
    "tour_skip": lambda r, f: f"No tour. In this same turn, show the build card. {_build_menu(r)}",
    "moved_on": lambda r, f: f"That beat is skipped; take its default and do not ask it again. {_next(r)}",
    "skip_all": lambda r, f: ("They ended setup. In one line, say they can ask you anything any time, then stop "
                              "guiding. The onboarding rules no longer apply; this is a normal chat."),
    "complete": lambda r, f: ("Onboarding is complete. Its rules no longer apply; this is a normal chat. Close in "
                              "one short line: whenever they are stuck, they can just ask."),
}


def _already(record, beat: str) -> str:
    """A repeat apply of a settled beat: steer the model forward instead of drawing another receipt."""
    return (f"The {beat} was already set by the card or an earlier call. Do not apply it again unless they ask "
            f"for a change. {_next(record)}")


def _wrote(record, facts, beat: str) -> str:
    # The turn ends on this result; the turn note on their message says how to go on.
    text = (f"The {beat} card closed unanswered: they wrote a message instead. It comes next as their own turn; "
            "write nothing until it does.")
    if beat in ("theme", "layout"):
        # "What do you think?" asks for a pick: a look around without one leaves them asking again.
        text += (f" If it asks what you think or which one fits, pick the {beat} yourself: look around first if "
                 f"they asked, then say your pick and why in one line and apply it ({_TYPED_ANSWER[beat]}).")
    return text


def _timed_out(record, facts, beat: str) -> str:
    # Also the result when they ran a slash command while the card waited.
    return (f"The {beat} card closed without an answer. If a message from them comes next, answer it first. "
            f"Otherwise say one short line that they can pick up whenever they are back, then stop. When they "
            f'return, bring it back with setup_beat(beat="{beat}").')


def reminder(key: str, record: Mapping[str, Any], facts: Facts, beat: str | None = None,
             resolve_skill: Callable[[str], str | None] = lambda name: name) -> str:
    """The text for a reminder key; ``beat`` names the card a ``wrote`` / ``timed_out`` result is about;
    ``resolve_skill`` names a first task's skill as this install loads it, None dropping a missing one."""
    if not key:
        return ""
    if key in ("wrote", "timed_out"):
        return (_wrote if key == "wrote" else _timed_out)(record, facts, beat or _waiting_beat(record) or "build")
    if key.startswith("build_"):
        option = key.removeprefix("build_")
        label = next((o["label"] for o in record["picks"].get("build_options") or () if o["id"] == option), "")
        return build_guidance(option, facts, resolve_skill, label)
    if key.startswith("already_"):
        return _already(record, key.removeprefix("already_"))
    return _REMINDERS[key](record, facts)


def turn_note(record: Mapping[str, Any]) -> str:
    """The note on a user turn while no card waits: which card is closed, how their message applies it
    when it answers it, and how to bring it back. Empty once every card is settled: the build is under
    way and nothing is left to reopen."""
    if (beat := _waiting_beat(record)) is None:
        return ""
    return (f"[System note: Onboarding continues; the {beat} card is closed. If their message answers it, apply it "
            f"yourself ({_TYPED_ANSWER[beat]}) and move on. Otherwise answer their message first, then bring the card "
            f'back in one line with setup_beat(beat="{beat}").]')


def finish_reminder(record: Mapping[str, Any]) -> str:
    return _next(record)
