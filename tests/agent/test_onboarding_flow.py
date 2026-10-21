"""``agent/onboarding_flow.py``: the beats, the run record and every event, across machine fixtures.

Sequences named after lab runs replay what those runs did; the lab bugs they showed are the
assertions (``.scratch/onboarding-plan/NOTES-onboarding-flow.md``).
"""

from __future__ import annotations

import pytest

from agent import onboarding_flow as flow
from agent.onboarding_flow import (
    DONE, PENDING, SHOWN, SKIPPED, Answered, Applied, Done, Facts, End, Recorded, Shown, Skipped,
    TimedOut, Wrote, apply, new_record, next_beat,
)
from agent.onboarding_first_tasks import eligible_tasks
from agent.onboarding_text import _LOOK_AROUND, opening_prompt, reminder, turn_note

NOTES = [{"id": "notes", "label": "Notes to wiki"}]

MACHINES = {
    "mac": Facts(os="macos", arch="arm64", machine_kind="Mac", plugins=("blender",)),
    "windows_rtx": Facts(os="windows", arch="arm64", machine_kind="Spark", nvidia_platform="rtx_spark",
                         plugins=("nvidia-app", "nvidia-broadcast", "blender")),
    "spark": Facts(os="linux", arch="arm64", machine_kind="Spark", nvidia_platform="dgx_spark", plugins=("blender",)),
    "linux": Facts(os="linux", arch="amd64", machine_kind="computer", plugins=()),
    "windows_laptop": Facts(os="windows", arch="amd64", machine_kind="PC", nvidia_platform="discrete",
                            plugins=("nvidia-app", "nvidia-broadcast", "blender")),
    "remote": Facts(os="linux", arch="amd64", machine_kind="computer", facts_source="remote", tours=False),
}


def run(facts: Facts, *events):
    """Apply events in order from a fresh record; returns every step."""
    record, steps = new_record(facts, now=0), []
    for clock, event in enumerate(events, start=1):
        step = apply(record, event, facts, now=float(clock))
        record = step.record
        steps.append(step)
    return steps


def answer_all(facts: Facts) -> list:
    """The happy path for a machine: every beat it gets, shown and answered on its card."""
    answers = {"name": {"name": "Sid"}, "theme": {"accent": "Ultraviolet"}, "layout": {"layout": "terminal-deck"},
               "connectors": {"connected": ["slack"], "newly_connected": ["slack"]},
               "plugins": {"installed": list(facts.plugins[:1])}, "tour": {"tour": "quick"}}
    record, events = new_record(facts), []
    while (beat := next_beat(record)) is not None:
        first = eligible_tasks(facts, record["picks"])[0]
        answer = answers.get(beat) or {"build": first.id}
        options = [{"id": first.id, "label": first.label}] if beat == "build" else None
        suggest = ["slack"] if beat == "connectors" else None
        for event in (Shown(beat, f"call_{beat}", options, suggest), Answered(beat, answer)):
            events.append(event)
            record = apply(record, event, facts).record
    return [*events, Done()]


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_happy_path_settles_every_beat_this_machine_gets_and_completes(machine):
    facts = MACHINES[machine]
    steps = run(facts, *answer_all(facts))

    final = steps[-1]
    assert final.result == {"completed": True} and final.reminder == "complete"
    assert final.record["phase"] == flow.COMPLETE and final.record["ended_by"] == "done"
    assert set(final.record["beats"].values()) <= {DONE, SKIPPED}
    assert (final.record["beats"]["plugins"] == SKIPPED) is not bool(facts.plugins)
    assert (final.record["beats"]["tour"] == SKIPPED) is not facts.tours
    # Every answer names the next beat, and every reminder renders.
    for step in steps:
        assert ("next" in step.result) is isinstance(step.result.get("answer"), dict)
        assert isinstance(reminder(step.reminder, step.record, facts), str)
    # Each after_<beat> reminder names the card that really comes next.
    afters = [step for step in steps if step.reminder.startswith("after_")]
    assert afters
    for step in afters:
        upcoming = next_beat(step.record)
        expected = f"shows the {upcoming} card" if upcoming else 'setup_beat(beat="done")'
        assert expected in reminder(step.reminder, step.record, facts), step.reminder


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_the_full_name_never_reaches_the_record_or_the_opening(machine):
    facts = Facts(**{**MACHINES[machine].__dict__, "full_name": "Ada Lovelace"})

    assert "Ada Lovelace" not in repr(new_record(facts))
    assert "Ada Lovelace" not in opening_prompt(facts)


def test_run_hrs0_typed_answer_is_applied_and_never_reshows_the_card():
    """hrs0: "elite sounds like my type ;)" got a second layout picker. A message closes the card and
    comes as the next turn; the model applies it with set_theme and the hook settles the beat."""
    facts = MACHINES["mac"]
    steps = run(facts, Shown("name"), Answered("name", {"name": "Sid"}), Shown("theme"),
                Answered("theme", {"accent": "Ultraviolet"}), Shown("layout"), Wrote("layout"),
                Applied("set_theme", {"layout": "terminal-deck"}, {"success": True, "applied": {"layout": "terminal-deck"}}))

    wrote, applied = steps[-2], steps[-1]
    assert wrote.result == {"beat": "layout", "wrote_instead": True}
    assert wrote.record["waiting"] is None and wrote.record["beats"]["layout"] == SHOWN
    assert "wrote a message instead" in reminder(wrote.reminder, wrote.record, facts, "layout")
    assert "apply it yourself" in turn_note(wrote.record)
    assert applied.record["beats"]["layout"] == DONE and applied.record["picks"]["layout"] == "terminal-deck"
    assert next_beat(applied.record) == "connectors"


def test_set_theme_settling_the_layout_beat_tells_the_model_the_next_beat():
    """Without a reminder the model recorded the applied layout with setup_beat and was refused."""
    facts = MACHINES["mac"]
    steps = run(facts, Shown("name"), Answered("name", {"name": "Sid"}), Shown("theme"),
                Answered("theme", {"accent": "Ultraviolet"}), Shown("layout"), Wrote("layout"),
                Applied("set_theme", {"layout": "sidebar-left"}, {"success": True, "applied": {"layout": "sidebar-left"}}),
                Applied("set_theme", {"layout": "terminal-deck"}, {"success": True, "applied": {"layout": "terminal-deck"}}))

    settled, again = steps[-2], steps[-1]
    assert settled.reminder == "after_layout"
    text = reminder(settled.reminder, settled.record, facts)
    assert text.startswith("Layout Basic is on, already applied.") and "Do not call set_theme" in text
    assert 'setup_beat(beat="next") shows the connectors card' in text
    # A second set_theme on the settled layout steers forward instead of returning nothing.
    assert again.reminder == "already_layout" and again.record["picks"]["layout"] == "terminal-deck"


def test_live_run_a_set_theme_after_the_theme_card_is_told_it_is_already_set():
    """Sid's run: the theme card applied green, then the model called set_theme(accent=green) again and a
    second receipt drew. The card's reminder says it is applied; a repeat call names the next step."""
    facts = MACHINES["mac"]
    steps = run(facts, Shown("name"), Answered("name", {"name": "Sid"}), Shown("theme"),
                Answered("theme", {"accent": "green"}),
                Applied("set_theme", {"accent": "green"}, {"success": True, "applied": {"accent": "green"}}))

    answered, repeat = steps[-2], steps[-1]
    card_text = reminder(answered.reminder, answered.record, facts)
    assert "already applied" in card_text and "Do not call set_theme" in card_text
    assert repeat.reminder == "already_theme"
    text = reminder(repeat.reminder, repeat.record, facts)
    assert "The theme was already set" in text and "Do not apply it again" in text
    assert 'setup_beat(beat="next") shows the layout card' in text


def test_run_hrs0_off_script_questions_keep_one_card_and_continue_settles_it():
    """hrs0: each off-script question re-rendered the apps cards (four open copies), and a typed
    "continue" left the beat shown. Bringing a card back is a new Shown of the same beat; a value
    given in words settles it."""
    facts = MACHINES["mac"]
    events = [Shown("name"), Answered("name", {"name": "Sid"}), Skipped("theme"), Skipped("layout")]
    for question in ("tell me about SpaceX first", "who runs it?"):
        events += [Shown("connectors", question), Wrote("connectors")]
    events += [Shown("connectors"), Wrote("connectors"), Recorded("connectors", "continue")]
    steps = run(facts, *events)

    assert all(step.record["waiting"] is None for step in steps if step.result.get("wrote_instead"))
    assert steps[-1].result == {"recorded": "connectors", "next": "plugins"}
    assert steps[-1].record["beats"]["connectors"] == DONE


LAYOUT_SET = Applied("set_theme", {"layout": "terminal-deck"}, {"success": True, "applied": {"layout": "terminal-deck"}})
TO_LAYOUT = (Shown("name"), Answered("name", {"name": "Sid"}), Shown("theme"), Answered("theme", {"accent": "Ember"}),
             Shown("layout"))


def test_run_3ofn_pick_for_me_applies_layout_and_the_look_around_offer_waits_for_the_connectors_card():
    """3ofn: "look around and pick one for me" ended in Elite through set_theme. The offer no longer
    rides the layout result: it comes back on the first try to show the connectors card."""
    facts = MACHINES["mac"]
    steps = run(facts, *TO_LAYOUT, Wrote("layout"), LAYOUT_SET)

    assert steps[-1].record["beats"]["layout"] == DONE and steps[-1].record["look_around"] == "not_offered"
    assert "look around" not in reminder("after_layout", steps[-1].record, facts)


def test_vesper_batched_set_theme_and_next_gets_the_offer_then_the_second_next_shows_connectors():
    """Vesper run 1: set_theme(layout) and setup_beat(next) in one batch put the connectors card up before
    the offer was made, and the run recorded it as offered."""
    facts = MACHINES["windows_laptop"]
    steps = run(facts, *TO_LAYOUT, LAYOUT_SET, Shown("connectors", "call_a"), Shown("connectors", "call_b"))

    refused, shown = steps[-2], steps[-1]
    assert refused.result == {"error": "offer first"} and "beat" not in refused.result
    assert refused.reminder == "offer_first" and refused.record["look_around"] == "offered"
    assert refused.record["beats"]["connectors"] == PENDING and refused.record["waiting"] is None
    assert _LOOK_AROUND in reminder(refused.reminder, refused.record, facts)
    assert shown.result == {} and shown.record["beats"]["connectors"] == SHOWN
    assert shown.record["waiting"]["tool_call_id"] == "call_b"


def test_showing_connectors_directly_before_the_offer_is_refused():
    facts = MACHINES["mac"]
    step = run(facts, Shown("connectors", "call_1"))[-1]

    assert step.result == {"error": "offer first"} and step.reminder == "offer_first"
    assert step.record["beats"]["connectors"] == PENDING and step.record["waiting"] is None
    assert step.record["look_around"] == "offered"


def test_suggestions_mean_the_model_already_looked_so_the_card_shows_at_once():
    facts = MACHINES["mac"]
    step = run(facts, Shown("connectors", "call_1", suggest=["slack", "linear"]))[-1]

    assert step.result == {} and step.record["beats"]["connectors"] == SHOWN
    assert step.record["look_around"] == "offered"
    # Blank slugs are not suggestions.
    assert run(facts, Shown("connectors", "call_1", suggest=["  "]))[-1].result == {"error": "offer first"}


@pytest.mark.parametrize("state", ["declined", "accepted"])
def test_a_settled_look_around_state_shows_the_card_and_stays(state):
    facts = MACHINES["mac"]
    record = {**new_record(facts), "look_around": state}
    step = apply(record, Shown("connectors", "call_1"), facts)

    assert step.result == {} and step.record["beats"]["connectors"] == SHOWN
    assert step.record["look_around"] == state


def test_the_opening_prompt_grants_the_look_and_bounds_it():
    text = opening_prompt(MACHINES["mac"])

    assert "look around this computer when they say yes" in text
    assert "installed apps, developer tools, the OS and the hardware" in text
    assert "Never personal files" in text and "act on it at any beat" in text


def test_skip_setup_while_a_card_waits_ends_the_run_and_refuses_later_beats():
    facts = MACHINES["windows_laptop"]
    steps = run(facts, Shown("name"), Answered("name", {"name": "Sid"}), Shown("theme"), End(), Shown("theme"))

    skipped, refused = steps[-2], steps[-1]
    assert skipped.result == {"skipped_all": True} and skipped.reminder == "skip_all"
    assert skipped.record["phase"] == flow.PHASE_SKIPPED and skipped.record["waiting"] is None
    assert skipped.record["beats"]["name"] == DONE
    assert all(skipped.record["beats"][beat] == SKIPPED for beat in flow.BEAT_IDS if beat != "name")
    assert "error" in refused.result and refused.record == skipped.record


def test_walks_away_closes_the_card_and_the_turn_note_brings_it_back():
    facts = MACHINES["linux"]
    step = run(facts, Shown("name"), Answered("name", {"name": "Sid"}), Shown("theme"), TimedOut("theme"))[-1]

    assert step.result == {"beat": "theme", "timed_out": True}
    assert step.record["beats"]["theme"] == SHOWN and step.record["waiting"] is None
    assert 'setup_beat(beat="theme")' in turn_note(step.record)


def test_out_of_order_card_leaves_earlier_pending_beats_for_next():
    facts = MACHINES["mac"]
    step = run(facts, Shown("connectors"), Answered("connectors", {"connected": [], "newly_connected": []}))[-1]

    assert step.record["beats"]["name"] == PENDING
    assert step.result["next"] == "name"


def test_a_later_card_settles_an_earlier_card_left_open():
    facts = MACHINES["mac"]
    step = run(facts, Shown("name"), Wrote("name"), Shown("theme"))[-1]

    assert step.record["beats"]["name"] == DONE and step.record["beats"]["theme"] == SHOWN


def test_a_done_beat_shown_again_stays_done_and_records_the_new_pick():
    facts = MACHINES["mac"]
    step = run(facts, Shown("theme"), Answered("theme", {"accent": "Ember"}), Shown("theme"),
               Answered("theme", {"accent": "Ultraviolet"}))[-1]

    assert step.record["beats"]["theme"] == DONE and step.record["picks"]["accent"] == "Ultraviolet"


def test_done_is_refused_before_the_build_card_was_reached():
    step = run(MACHINES["mac"], Shown("name"), Answered("name", {"name": "Sid"}), Done())[-1]

    assert "error" in step.result and step.result["next"] == "theme"
    assert step.record["phase"] == flow.ONBOARDING


def test_a_value_outside_the_cards_options_is_refused():
    step = run(MACHINES["mac"], Recorded("tour", "everything"))[-1]

    assert "quick" in step.result["error"] and step.record["beats"]["tour"] == PENDING


def test_a_beat_this_machine_skips_is_never_carded():
    step = run(MACHINES["linux"], Shown("plugins"))[-1]

    assert step.result["skipped"] is True and step.record["waiting"] is None


@pytest.mark.parametrize(("tool", "args", "result", "beat", "pick"), [
    ("set_theme", {"accent": "violet"}, {"success": True, "applied": {"accent": "custom:#8a2be2"}}, "theme",
     ("accent", "custom:#8a2be2")),
    ("set_theme", {"picker": "accent"}, {"success": True, "picked": {"accent": None, "mode": "dark"}}, "theme",
     ("mode", "dark")),
    ("manage_connections", {"action": "display"}, {"connected": ["gmail"], "newly_connected": [], "suggested": []},
     "connectors", ("connected", ["gmail"])),
    ("gui_tour", {"action": "start", "preset": "quick"}, {"success": True}, "tour", ("tour", "quick")),
])
def test_general_tool_results_settle_their_beat(tool, args, result, beat, pick):
    step = run(MACHINES["mac"], Applied(tool, args, result))[-1]

    assert step.record["beats"][beat] == DONE
    assert step.record["picks"][pick[0]] == pick[1]


@pytest.mark.parametrize(("tool", "args", "result"), [
    ("set_theme", {"accent": "nope"}, {"success": False, "error": "unknown", "available": []}),
    ("set_theme", {"picker": "layout"}, {"said": "hmm"}),
    ("manage_connections", {"action": "display"}, {"timed_out": True}),
    ("manage_connections", {"action": "status"}, {"connected": ["gmail"]}),
    ("gui_tour", {"action": "next"}, {"success": True}),
])
def test_results_that_pick_nothing_settle_nothing(tool, args, result):
    step = run(MACHINES["mac"], Applied(tool, args, result))[-1]

    assert set(step.record["beats"].values()) <= {PENDING, SKIPPED}


@pytest.mark.parametrize("options", [
    None, [], [{"id": "games", "label": "Games"}], [{"id": "scan", "label": "Scan"}],
    [{"id": "hermes", "label": "Explain how Hermes works"}], [{"id": x, "label": x} for x in ("notes", "site", "repo", "watch", "explain")],
    [{"id": "notes", "label": "A"}, {"id": "notes", "label": "B"}], [{"id": "notes", "label": "x" * 33}],
    [{"id": "notes", "label": " "}], ["notes"],
])
def test_build_picks_are_refused_with_the_eligible_ids_and_no_card(options):
    step = run(MACHINES["linux"], Shown("build", "call_b", options))[-1]

    assert "notes, explain, youtube, watch, site, repo" in step.result["error"]
    assert step.record["beats"]["build"] == PENDING and step.record["waiting"] is None


def test_valid_picks_are_the_card_with_how_hermes_works_last_and_any_card_id_answers_it():
    facts = MACHINES["mac"]
    picks = [{"id": "mac", "label": "Your Mac, sorted"}, {"id": "repo", "label": "Map your repo"}]
    shown = run(facts, Shown("build", "call_b", picks))[-1]
    card = flow.BEAT_BY_ID["build"].card(facts, shown.record["picks"])

    assert card == {"options": [*picks, {"id": "hermes", "label": "Explain how Hermes works"}]}
    # Brought back without picks, the same card returns.
    assert apply(shown.record, Shown("build", "call_c"), facts).result == {}
    for value in ("repo", "hermes", "own"):
        assert apply(shown.record, Recorded("build", value), facts).record["picks"]["build"] == value
    assert "error" in apply(shown.record, Recorded("build", "notes"), facts).result


@pytest.mark.parametrize(("machine", "present", "absent"), [
    ("mac", "brew install --cask", "winget"),
    ("windows_laptop", "--accept-package-agreements", "brew"),
    ("windows_rtx", "x64 only", "brew"),
    ("linux", "flatpak install", "winget"),
    ("remote", "another computer", "brew"),
])
def test_build_guidance_carries_only_this_hosts_commands(machine, present, absent):
    text = reminder("build_notes", new_record(MACHINES[machine]), MACHINES[machine])

    assert present in text and absent not in text
    assert 'setup_beat(beat="done")' in text


def test_the_build_answer_completes_the_run_and_a_later_done_or_skip_reports_that_end():
    facts = MACHINES["mac"]
    answered, done, skip = run(facts, Shown("build", "", NOTES), Answered("build", {"build": "notes"}), Done(), End())[1:]

    assert answered.result["completed"] is True and answered.reminder == "build_notes"
    assert answered.record["phase"] == flow.COMPLETE and answered.record["ended_by"] == "done"
    assert done.result == {"completed": True} and done.record == answered.record
    assert skip.result == {"skipped_all": True} and skip.record == answered.record


def test_a_general_tool_that_settles_the_waiting_beat_clears_waiting():
    facts = MACHINES["mac"]
    step = run(facts, Shown("theme", "call_t"), Applied("set_theme", {"accent": "violet"},
                                                         {"success": True, "applied": {"accent": "violet"}}))[-1]

    assert step.record["beats"]["theme"] == DONE and step.record["waiting"] is None


def test_a_card_answer_outside_the_cards_options_is_refused_and_the_beat_stays_open():
    facts = MACHINES["mac"]
    record = run(facts, Shown("build", "call_b", NOTES))[-1].record
    step = apply(record, Answered("build", {"build": "games"}), facts, now=2.0)

    assert "error" in step.result and step.record["beats"]["build"] == SHOWN
    assert step.record["picks"]["build"] is None and step.record["waiting"] is None
    for beat, answer in [("name", {"name": "  "}), ("layout", {"layout": "x"}), ("theme", {"accent": 3}),
                         ("plugins", {"installed": ["nvidia-app"]}), ("connectors", {"installed": ["blender"]})]:
        assert "error" in apply(new_record(facts), Answered(beat, answer), facts, now=1.0).result


def test_a_click_carries_the_option_label_for_the_transcript_and_only_the_id_is_recorded():
    facts = MACHINES["mac"]
    record = run(facts, Shown("tour", "call_t"))[-1].record
    step = apply(record, Answered("tour", {"tour": "quick", "label": "Quick look"}), facts, now=2.0)

    assert step.result["answer"] == {"tour": "quick", "label": "Quick look"} and step.reminder == "tour_quick"
    assert step.record["picks"]["tour"] == "quick" and "label" not in step.record["picks"]
    assert "error" in apply(record, Answered("tour", {"tour": "quick", "label": 3}), facts, now=2.0).result


def test_a_layout_or_theme_given_in_words_is_applied_by_set_theme_not_recorded():
    facts = MACHINES["mac"]
    for beat, value in [("layout", "terminal-deck"), ("theme", "Ultraviolet")]:
        step = apply(new_record(facts), Recorded(beat, value), facts, now=1.0)
        assert "set_theme" in step.result["error"] and step.record["beats"][beat] == PENDING
    record = run(facts, Shown("name"), Answered("name", {"name": "Sid"}), Skipped("theme"), Shown("layout"))[-1].record
    assert '"name":"set_theme"' in turn_note(record)


def test_no_turn_note_once_every_card_is_settled():
    facts = MACHINES["linux"]
    steps = run(facts, *answer_all(facts)[:-1])

    assert next_beat(steps[-1].record) is None and turn_note(steps[-1].record) == ""
