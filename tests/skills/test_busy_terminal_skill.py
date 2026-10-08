"""Tests for optional-skills/creative/busy-terminal/scripts/busy_terminal.py"""

import random
import re
import sys
from pathlib import Path
from unittest import mock

import pytest

# Add the scripts dir so we can import the module directly
SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "creative"
    / "busy-terminal"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import busy_terminal

ANSI = re.compile(r"\033\[[0-9;?]*[A-Za-z]")


class Recorder:
    """Captures what a scene paints and how long it asked to sleep."""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.slept = 0.0

    def write(self, text: str) -> None:
        self.chunks.append(text)

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0, "a scene must never ask for a negative sleep"
        self.slept += seconds

    @property
    def text(self) -> str:
        return "".join(self.chunks)


def make_console(color: bool = False, speed: float = 1.0) -> tuple[busy_terminal.Console, Recorder]:
    rec = Recorder()
    console = busy_terminal.Console(
        width=100, height=30, color=color, speed=speed, write=rec.write, sleep=rec.sleep
    )
    return console, rec


def ticking_clock(step: float = 1.0):
    """A monotonic clock that advances a fixed step per call."""
    state = {"t": 0.0}

    def now() -> float:
        state["t"] += step
        return state["t"] - step

    return now


# ── Pure formatters ──────────────────────────────────────────────────────────


class TestProgressBar:
    def test_width_is_fixed_regardless_of_progress(self):
        widths = {len(busy_terminal.progress_bar(done, 10, width=20)) for done in range(0, 11)}
        assert widths == {20}

    def test_endpoints_are_empty_and_full(self):
        assert busy_terminal.progress_bar(0, 10, width=8) == "░" * 8
        assert busy_terminal.progress_bar(10, 10, width=8) == "█" * 8

    def test_out_of_range_clamps_instead_of_overflowing(self):
        assert busy_terminal.progress_bar(-5, 10, width=8) == "░" * 8
        assert busy_terminal.progress_bar(50, 10, width=8) == "█" * 8

    def test_zero_total_does_not_divide_by_zero(self):
        assert len(busy_terminal.progress_bar(3, 0, width=6)) == 6

    def test_fill_never_shrinks_as_progress_grows(self):
        fills = [busy_terminal.progress_bar(d, 20, width=12).count("█") for d in range(21)]
        assert fills == sorted(fills)


class TestHumanBytes:
    @pytest.mark.parametrize(
        "count,expected_unit",
        [(512, "B"), (2048, "KiB"), (5 * 1024**2, "MiB"), (3 * 1024**3, "GiB")],
    )
    def test_unit_matches_magnitude(self, count, expected_unit):
        assert busy_terminal.human_bytes(count).endswith(expected_unit)

    def test_larger_counts_never_read_as_smaller_units(self):
        assert busy_terminal.human_bytes(1024**4).endswith("GiB")


class TestNextScene:
    def test_never_repeats_the_scene_that_just_played(self):
        rng = random.Random(0)
        last = "code"
        for _ in range(200):
            chosen = busy_terminal.next_scene(rng, last)
            assert chosen != last
            last = chosen

    def test_always_returns_a_known_scene(self):
        rng = random.Random(1)
        assert {busy_terminal.next_scene(rng, "build") for _ in range(50)} <= set(
            busy_terminal.SCENES
        )

    def test_single_scene_catalog_yields_rather_than_looping(self):
        rng = random.Random(2)
        assert busy_terminal.next_scene(rng, "only", scenes=("only",)) == "only"

    def test_no_prior_scene_can_pick_any_of_them(self):
        rng = random.Random(3)
        seen = {busy_terminal.next_scene(rng, "") for _ in range(200)}
        assert seen == set(busy_terminal.SCENES)


class TestTestSummary:
    def test_failures_lead_the_line(self):
        assert busy_terminal.test_summary(10, 2, 0, 1.0).startswith("2 failed")

    def test_clean_run_omits_failed_and_skipped(self):
        line = busy_terminal.test_summary(10, 0, 0, 1.5)
        assert "failed" not in line and "skipped" not in line

    def test_always_reports_passed_and_duration(self):
        line = busy_terminal.test_summary(7, 1, 3, 2.25)
        assert "7 passed" in line and "3 skipped" in line and "2.25s" in line


# ── Highlighting ─────────────────────────────────────────────────────────────


class TestHighlight:
    def test_is_a_no_op_without_color(self):
        line = 'def go(x):  # note'
        assert busy_terminal.highlight(line, "python", color=False) == line

    def test_stripping_escapes_recovers_the_original_line(self):
        line = 'return "ok"  # 42'
        painted = busy_terminal.highlight(line, "python", color=True)
        assert ANSI.sub("", painted) == line

    def test_every_opened_color_is_closed(self):
        painted = busy_terminal.highlight('async def f(): return "x"  # 1', "python", color=True)
        assert painted.count(busy_terminal.RESET) == len(re.findall(r"\033\[38;5;\d+m", painted))

    def test_keywords_are_tinted_and_bare_identifiers_are_not(self):
        painted = busy_terminal.highlight("import widget", "python", color=True)
        assert busy_terminal.MAGENTA + "import" in painted
        assert busy_terminal.MAGENTA + "widget" not in painted

    def test_unknown_language_still_returns_the_line_intact(self):
        line = "some :: unknown ++ syntax"
        assert ANSI.sub("", busy_terminal.highlight(line, "cobol", color=True)) == line


class TestTypeOut:
    def test_repaint_redraws_the_prefix_so_the_gutter_survives(self):
        """The repaint returns to column 0 and must not land on the line numbers."""
        console, rec = make_console(color=True)
        busy_terminal.type_out(
            console, "return None", prefix="  12 │ ", language="python", rng=random.Random(1)
        )
        repaint = rec.text.split("\r")[-1]
        assert repaint.startswith("  12 │ ")

    def test_the_typed_characters_spell_the_whole_line(self):
        console, rec = make_console(color=False)
        busy_terminal.type_out(console, "import asyncio", rng=random.Random(1))
        assert rec.text.strip() == "import asyncio"

    def test_without_color_there_is_no_repaint_to_get_wrong(self):
        console, rec = make_console(color=False)
        busy_terminal.type_out(
            console, "return None", prefix="  12 │ ", language="python", rng=random.Random(1)
        )
        assert "\r" not in rec.text


# ── Scenes ───────────────────────────────────────────────────────────────────


class TestScenes:
    @pytest.mark.parametrize("name", busy_terminal.SCENES)
    def test_every_scene_paints_something_and_asks_to_wait(self, name):
        console, rec = make_console()
        busy_terminal.SCENE_RUNNERS[name](console, random.Random(11))
        assert rec.text.strip()
        assert rec.slept > 0

    @pytest.mark.parametrize("name", busy_terminal.SCENES)
    def test_no_scene_emits_ansi_when_color_is_off(self, name):
        console, rec = make_console(color=False)
        busy_terminal.SCENE_RUNNERS[name](console, random.Random(12))
        assert not ANSI.search(rec.text)

    @pytest.mark.parametrize("name", busy_terminal.SCENES)
    def test_speed_scales_the_wait_down(self, name):
        _, slow = make_console(speed=1.0)
        _, fast = make_console(speed=10.0)
        slow_console = busy_terminal.Console(color=False, speed=1.0, write=slow.write, sleep=slow.sleep)
        fast_console = busy_terminal.Console(color=False, speed=10.0, write=fast.write, sleep=fast.sleep)

        busy_terminal.SCENE_RUNNERS[name](slow_console, random.Random(13))
        busy_terminal.SCENE_RUNNERS[name](fast_console, random.Random(13))

        assert fast.slept == pytest.approx(slow.slept / 10.0, rel=1e-6)

    @pytest.mark.parametrize("name", busy_terminal.SCENES)
    def test_same_seed_replays_the_same_transcript(self, name):
        console_a, rec_a = make_console(color=True)
        console_b, rec_b = make_console(color=True)
        busy_terminal.SCENE_RUNNERS[name](console_a, random.Random(99))
        busy_terminal.SCENE_RUNNERS[name](console_b, random.Random(99))
        assert rec_a.text == rec_b.text

    def test_scenes_touch_no_process_socket_or_file(self):
        """The premise of the skill: none of this output is real."""
        console, _ = make_console()
        with (
            mock.patch("subprocess.run") as run,
            mock.patch("subprocess.Popen") as popen,
            mock.patch("socket.socket") as sock,
            mock.patch("builtins.open") as opened,
        ):
            for runner in busy_terminal.SCENE_RUNNERS.values():
                runner(console, random.Random(14))

        run.assert_not_called()
        popen.assert_not_called()
        sock.assert_not_called()
        opened.assert_not_called()


# ── Run loop ─────────────────────────────────────────────────────────────────


class TestRunLoop:
    def test_stops_once_the_duration_has_elapsed(self):
        console, _ = make_console()
        played = busy_terminal.run(
            console, random.Random(5), duration=3.0, now=ticking_clock(1.0)
        )
        assert played == 3

    def test_a_pinned_scene_is_the_only_one_that_plays(self):
        console, _ = make_console()
        seen: list[str] = []
        runners = {name: (lambda c, r, n=name: seen.append(n)) for name in busy_terminal.SCENES}

        with mock.patch.object(busy_terminal, "SCENE_RUNNERS", runners):
            busy_terminal.run(
                console, random.Random(6), scene="tests", duration=4.0, now=ticking_clock(1.0)
            )

        assert set(seen) == {"tests"}

    def test_cycling_covers_every_scene_without_adjacent_repeats(self):
        console, _ = make_console()
        seen: list[str] = []
        runners = {name: (lambda c, r, n=name: seen.append(n)) for name in busy_terminal.SCENES}

        with mock.patch.object(busy_terminal, "SCENE_RUNNERS", runners):
            busy_terminal.run(
                console,
                random.Random(7),
                scenes=busy_terminal.SCENES,
                duration=60.0,
                now=ticking_clock(1.0),
            )

        assert set(seen) == set(busy_terminal.SCENES)
        assert all(a != b for a, b in zip(seen, seen[1:]))

    @pytest.mark.parametrize("profile", sorted(busy_terminal.PROFILES))
    def test_a_profile_only_plays_its_own_scenes(self, profile):
        console, _ = make_console()
        seen: list[str] = []
        runners = {name: (lambda c, r, n=name: seen.append(n)) for name in busy_terminal.SCENES}

        with mock.patch.object(busy_terminal, "SCENE_RUNNERS", runners):
            busy_terminal.run(
                console,
                random.Random(8),
                scenes=busy_terminal.PROFILES[profile],
                duration=60.0,
                now=ticking_clock(1.0),
            )

        assert set(seen) == set(busy_terminal.PROFILES[profile])

    def test_the_default_rotation_is_the_hacker_profile(self):
        """A bare launch must open the Hollywood set, not fake pytest."""
        console, _ = make_console()
        seen: list[str] = []
        runners = {name: (lambda c, r, n=name: seen.append(n)) for name in busy_terminal.SCENES}

        with mock.patch.object(busy_terminal, "SCENE_RUNNERS", runners):
            busy_terminal.run(console, random.Random(9), duration=60.0, now=ticking_clock(1.0))

        assert set(seen) == set(busy_terminal.PROFILES["hacker"])
        assert seen[0] == "warroom"

    def test_developer_does_not_open_on_the_war_room(self):
        console, _ = make_console()
        seen: list[str] = []
        runners = {name: (lambda c, r, n=name: seen.append(n)) for name in busy_terminal.SCENES}

        with mock.patch.object(busy_terminal, "SCENE_RUNNERS", runners):
            busy_terminal.run(
                console,
                random.Random(9),
                scenes=busy_terminal.PROFILES["developer"],
                duration=8.0,
                now=ticking_clock(1.0),
            )

        assert seen[0] != "warroom"
        assert set(seen) <= set(busy_terminal.PROFILES["developer"])


class TestProfiles:
    def test_every_profile_is_a_non_empty_subset_of_the_catalog(self):
        for name, scenes in busy_terminal.PROFILES.items():
            assert scenes, name
            assert set(scenes) <= set(busy_terminal.SCENES), name

    def test_every_scene_in_the_catalog_has_a_runner(self):
        assert set(busy_terminal.SCENE_RUNNERS) == set(busy_terminal.SCENES)

    def test_mixed_covers_the_union_of_all_profiles(self):
        union = set().union(*busy_terminal.PROFILES.values())
        assert set(busy_terminal.PROFILES["mixed"]) == union


class TestRain:
    def test_a_fresh_drop_starts_above_the_screen(self):
        rng = random.Random(20)
        for _ in range(100):
            drop = busy_terminal.spawn_drop(col=5, height=30, rng=rng)
            assert drop.row <= 0
            assert drop.trail >= 4
            assert 0 < drop.speed <= 1.0, "faster than a cell per tick leaves trail holes"

    def test_a_step_advances_every_drop_by_its_own_speed(self):
        rng = random.Random(21)
        drops = [busy_terminal.spawn_drop(col, 30, rng) for col in (1, 3, 5)]
        before = [(drop.row, drop.speed) for drop in drops]
        busy_terminal.rain_step(drops, height=30, rng=rng)
        for drop, (row, speed) in zip(drops, before):
            assert drop.row == pytest.approx(row + speed)

    def test_a_frame_is_one_write_and_never_scrolls(self):
        rng = random.Random(22)
        drops = [busy_terminal.spawn_drop(col, 30, rng) for col in range(1, 60, 2)]
        for _ in range(80):
            frame = busy_terminal.rain_step(drops, height=30, rng=rng)
            assert "\n" not in frame

    def test_a_drop_that_left_the_screen_respawns_above_it(self):
        rng = random.Random(23)
        drops = [busy_terminal.Drop(col=1, row=100.0, speed=1.0, trail=4)]
        busy_terminal.rain_step(drops, height=30, rng=rng)
        assert drops[0].row < 1


class TestFit:
    def test_always_returns_exactly_the_requested_width(self):
        for text in ("", "short", "x" * 500):
            for width in (1, 7, 40):
                assert len(busy_terminal.fit(text, width)) == width

    def test_zero_or_negative_width_collapses_to_empty(self):
        assert busy_terminal.fit("anything", 0) == ""
        assert busy_terminal.fit("anything", -3) == ""


class TestWarroomLayout:
    SIZES = [(w, h) for w in (40, 60, 80, 100, 140, 200) for h in (10, 16, 24, 40, 60)]

    @pytest.mark.parametrize("width,height", SIZES)
    def test_every_window_stays_inside_the_terminal(self, width, height):
        for name, rect in busy_terminal.warroom_layout(width, height).items():
            assert rect.top >= 1 and rect.left >= 1, name
            assert rect.bottom <= height, name
            assert rect.right <= width, name

    FLOATERS = ("dialog", "alert", "proxy", "exfil")

    @pytest.mark.parametrize("width,height", SIZES)
    def test_panes_never_overlap_each_other(self, width, height):
        """Only floaters may overlap panes — nothing re-stamps on the panes'
        cadence."""
        layout = busy_terminal.warroom_layout(width, height)
        panes = [(n, r) for n, r in layout.items() if n not in self.FLOATERS]
        for i, (name_a, rect_a) in enumerate(panes):
            for name_b, rect_b in panes[i + 1:]:
                assert not rect_a.overlaps(rect_b), f"{name_a} overlaps {name_b}"

    @pytest.mark.parametrize("width,height", SIZES)
    def test_the_four_dialogs_never_cover_each_other(self, width, height):
        """Floaters may sit on panes, but every dialog must stay readable."""
        layout = busy_terminal.warroom_layout(width, height)
        floaters = [(n, layout[n]) for n in self.FLOATERS if n in layout]
        for i, (name_a, rect_a) in enumerate(floaters):
            for name_b, rect_b in floaters[i + 1:]:
                assert not rect_a.overlaps(rect_b), f"{name_a} overlaps {name_b}"

    @pytest.mark.parametrize("width,height", SIZES)
    def test_the_dialog_is_always_present_and_readable(self, width, height):
        dialog = busy_terminal.warroom_layout(width, height)["dialog"]
        assert dialog.width >= 30
        assert dialog.height == 5

    def test_a_generous_terminal_gets_every_window(self):
        assert set(busy_terminal.warroom_layout(120, 35)) == {
            "memdump", "uplink", "intercept", "dialog", "alert", "proxy", "exfil",
        }

    def test_a_small_terminal_drops_the_corner_dialogs_not_the_centerpiece(self):
        layout = busy_terminal.warroom_layout(60, 16)
        assert "dialog" in layout
        assert not set(layout) & {"alert", "proxy", "exfil"}


class TestRainAvoidance:
    MOVE = re.compile(r"\033\[(\d+);(\d+)H")

    def test_rain_never_writes_inside_an_avoided_window(self):
        rng = random.Random(40)
        window = busy_terminal.Rect(top=5, left=5, width=20, height=10)
        drops = [busy_terminal.spawn_drop(col, 30, rng) for col in range(1, 60, 2)]

        for _ in range(300):
            frame = busy_terminal.rain_step(drops, height=30, rng=rng, avoid=[window])
            for row, col in self.MOVE.findall(frame):
                assert not window.contains(int(row), int(col))

    def test_the_drops_keep_falling_behind_the_window(self):
        """Avoidance blocks the writes, not the motion."""
        rng = random.Random(41)
        window = busy_terminal.Rect(top=1, left=1, width=60, height=30)
        drops = [busy_terminal.spawn_drop(col, 30, rng) for col in (1, 3)]
        rows = [drop.row for drop in drops]
        busy_terminal.rain_step(drops, height=30, rng=rng, avoid=[window])
        assert all(drop.row > row for drop, row in zip(drops, rows))


class TestSceneArcs:
    """Each hacker scene must reach its climax — that IS the scene."""

    def test_the_intrusion_always_gets_in_and_always_gets_out(self):
        console, rec = make_console()
        busy_terminal.scene_intrusion(console, random.Random(30))
        assert "ACCESS GRANTED" in rec.text
        assert "connection closed by remote host." in rec.text
        assert rec.text.index("ACCESS GRANTED") < rec.text.index("connection closed")

    def test_the_intrusion_only_ever_touches_documentation_targets(self):
        for seed in range(12):
            console, rec = make_console()
            busy_terminal.scene_intrusion(console, random.Random(seed))
            assert any(host in rec.text for host, _addr in busy_terminal.TARGETS)
            assert ".example." in rec.text

    def test_the_crack_locks_all_sixteen_bytes(self):
        console, rec = make_console()
        busy_terminal.scene_crack(console, random.Random(31))
        assert "16/16" in rec.text
        assert "key recovered" in rec.text

    def test_the_matrix_fallback_fits_the_terminal_width(self):
        console, rec = make_console(color=False)
        busy_terminal.scene_matrix(console, random.Random(32))
        lines = [line for line in rec.text.splitlines() if line]
        assert lines
        assert all(len(line) <= console.width for line in lines)

    def test_the_warroom_dialog_always_finishes_matching(self):
        console, rec = make_console(color=True)
        busy_terminal.scene_warroom(console, random.Random(33))
        assert "MATCHING PASSWORD" in rec.text
        assert "ACCESS GRANTED" in rec.text
        assert rec.text.index("MATCHING PASSWORD") < rec.text.index("ACCESS GRANTED")

    def test_the_warroom_opens_every_pane_the_layout_grants(self):
        console, rec = make_console(color=True)
        busy_terminal.scene_warroom(console, random.Random(34))
        layout = busy_terminal.warroom_layout(console.width, console.height)
        for floater in ("dialog", "alert", "proxy", "exfil"):
            layout.pop(floater, None)
        for name in layout:
            assert name in rec.text

    def test_the_warroom_shows_all_four_dialogs_on_a_roomy_terminal(self):
        console, rec = make_console(color=True)
        assert console.width >= 70 and console.height >= 20
        busy_terminal.scene_warroom(console, random.Random(36))
        for title in ("MATCHING PASSWORD", "PERIMETER", "PROXY CHAIN", "EXFIL"):
            assert title in rec.text

    def test_an_accented_line_keeps_its_tone_across_repaints(self):
        """The accent is rolled once at append time, not re-rolled per frame."""
        console, rec = make_console(color=True)
        pane = busy_terminal.Pane(
            rect=busy_terminal.Rect(top=2, left=2, width=20, height=4),
            title="memdump",
            tone=busy_terminal.GREY,
            feed=busy_terminal.feed_hex,
            period=3,
            reveal=0,
            lines=[("corrupt row", busy_terminal.RED), ("normal row", busy_terminal.GREY)],
        )
        for _ in range(2):
            rec.chunks.clear()
            console.paint(busy_terminal.interior_stamp(console, pane))
            assert busy_terminal.RED + "corrupt row" in rec.text
            assert busy_terminal.GREY + "normal row" in rec.text


class TestRollTone:
    def test_a_certain_accent_always_lands_and_a_zero_chance_never_does(self):
        pane_kwargs = dict(
            rect=busy_terminal.Rect(top=1, left=1, width=20, height=4),
            title="x", tone=busy_terminal.GREY, feed=busy_terminal.feed_hex,
            period=3, reveal=0, lines=[],
        )
        always = busy_terminal.Pane(accent=busy_terminal.RED, accent_chance=1.0, **pane_kwargs)
        never = busy_terminal.Pane(accent=busy_terminal.RED, accent_chance=0.0, **pane_kwargs)
        plain = busy_terminal.Pane(**pane_kwargs)

        rng = random.Random(50)
        for _ in range(50):
            assert busy_terminal.roll_tone(rng, always) == busy_terminal.RED
            assert busy_terminal.roll_tone(rng, never) == busy_terminal.GREY
            assert busy_terminal.roll_tone(rng, plain) == busy_terminal.GREY

    def test_a_partial_chance_produces_both_tones(self):
        pane = busy_terminal.Pane(
            rect=busy_terminal.Rect(top=1, left=1, width=20, height=4),
            title="x", tone=busy_terminal.CYAN, feed=busy_terminal.feed_trace,
            period=3, reveal=0, lines=[],
            accent=busy_terminal.GREEN, accent_chance=0.3,
        )
        rng = random.Random(51)
        tones = {busy_terminal.roll_tone(rng, pane) for _ in range(200)}
        assert tones == {busy_terminal.CYAN, busy_terminal.GREEN}

    def test_the_warroom_fallback_still_tells_the_story(self):
        console, rec = make_console(color=False)
        busy_terminal.scene_warroom(console, random.Random(35))
        assert "password matched" in rec.text
        assert not ANSI.search(rec.text)


# ── Launching a visible window ───────────────────────────────────────────────


class TestApplescriptString:
    def test_wraps_in_quotes(self):
        assert busy_terminal.applescript_string("hi") == '"hi"'

    def test_escapes_quotes_and_backslashes_so_the_literal_cannot_break_out(self):
        quoted = busy_terminal.applescript_string('a "b" c\\d')
        assert quoted == '"a \\"b\\" c\\\\d"'
        assert quoted.startswith('"') and quoted.endswith('"')
        assert '\\"b\\"' in quoted


class TestWindowArgv:
    def test_macos_drives_terminal_app_and_brings_it_forward(self):
        argv = busy_terminal.window_argv("run me", "darwin")
        script = " ".join(argv)
        assert argv[0] == "osascript"
        assert "do script" in script
        assert "activate" in script
        assert "number of columns of front window to 140" in script
        assert "number of rows of front window to 42" in script

    def test_macos_embeds_the_command_as_an_escaped_applescript_literal(self):
        argv = busy_terminal.window_argv('say "hi"', "darwin")
        assert any('\\"hi\\"' in part for part in argv)

    def test_windows_starts_a_detached_console(self):
        argv = busy_terminal.window_argv("run me", "win32")
        assert argv[:3] == ["cmd", "/c", "start"]
        assert argv[-1] == "run me"

    def test_linux_uses_the_first_emulator_that_exists(self):
        installed = {"konsole": "/usr/bin/konsole", "xterm": "/usr/bin/xterm"}
        argv = busy_terminal.window_argv("run me", "linux", which=installed.get)
        assert argv[0] == "konsole"

    def test_linux_normalises_every_emulator_through_sh_c(self):
        for emulator, _ in busy_terminal.LINUX_TERMINALS:
            argv = busy_terminal.window_argv(
                "run me", "linux", which=lambda name, e=emulator: "/bin/x" if name == e else None
            )
            assert argv[0] == emulator
            assert argv[-3:] == ["sh", "-c", "run me"]

    def test_linux_without_any_emulator_says_so(self):
        with pytest.raises(busy_terminal.NoTerminalError) as excinfo:
            busy_terminal.window_argv("run me", "linux", which=lambda _name: None)
        assert "xterm" in str(excinfo.value)


class TestRelaunchCommand:
    def test_drops_the_window_flag_so_the_child_actually_animates(self):
        command = busy_terminal.relaunch_command(
            ["--window", "--duration", "60"], script="/s/busy.py", python="/py"
        )
        assert "--window" not in command
        assert command == "/py /s/busy.py --duration 60"

    def test_quotes_paths_with_spaces(self):
        command = busy_terminal.relaunch_command([], script="/a b/busy.py", python="/py")
        assert "'/a b/busy.py'" in command

    def test_other_flags_survive_the_round_trip(self):
        command = busy_terminal.relaunch_command(
            ["--window", "--scene", "git", "--seed", "3"], script="/s.py", python="/py"
        )
        assert command.endswith("--scene git --seed 3")


class TestOpenInWindow:
    def test_spawns_the_window_and_returns_without_animating(self):
        spawned: list[list[str]] = []
        code = busy_terminal.open_in_window(
            ["--window", "--duration", "30"],
            platform="darwin",
            spawn=lambda argv: spawned.append(argv),
        )
        assert code == 0
        assert len(spawned) == 1
        assert spawned[0][0] == "osascript"

    def test_the_command_it_hands_off_still_carries_the_options(self):
        spawned: list[list[str]] = []
        busy_terminal.open_in_window(
            ["--window", "--scene", "build"],
            platform="win32",
            spawn=lambda argv: spawned.append(argv),
        )
        assert "--scene build" in spawned[0][-1]
        assert "--window" not in spawned[0][-1]


# ── CLI ──────────────────────────────────────────────────────────────────────


class TestColorDetection:
    def test_a_tty_that_was_asked_for_color_gets_it(self):
        stream = mock.Mock(isatty=mock.Mock(return_value=True))
        assert busy_terminal.supports_color(stream, requested=True) is True

    def test_a_pipe_never_gets_color(self):
        stream = mock.Mock(isatty=mock.Mock(return_value=False))
        assert busy_terminal.supports_color(stream, requested=True) is False

    def test_no_color_env_wins_over_a_tty(self, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        stream = mock.Mock(isatty=mock.Mock(return_value=True))
        assert busy_terminal.supports_color(stream, requested=True) is False

    def test_opting_out_wins_over_a_tty(self):
        stream = mock.Mock(isatty=mock.Mock(return_value=True))
        assert busy_terminal.supports_color(stream, requested=False) is False


class TestCli:
    def test_defaults_run_forever_on_the_hacker_profile(self):
        args = busy_terminal.build_parser().parse_args([])
        assert args.duration == 0.0
        assert args.scene == ""
        assert args.profile == "hacker"

    def test_scene_is_restricted_to_known_scenes(self):
        with pytest.raises(SystemExit):
            busy_terminal.build_parser().parse_args(["--scene", "nope"])

    def test_profile_is_restricted_to_known_profiles(self):
        with pytest.raises(SystemExit):
            busy_terminal.build_parser().parse_args(["--profile", "villain"])

    def test_a_bounded_hacker_run_exits_zero_and_prints(self, capsys):
        code = busy_terminal.main(
            ["--profile", "hacker", "--duration", "0.001", "--speed", "5000", "--no-color"]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert out.strip()
        assert not ANSI.search(out)

    def test_a_bounded_run_exits_zero_and_prints(self, capsys):
        code = busy_terminal.main(
            ["--scene", "build", "--duration", "0.001", "--speed", "5000", "--no-color"]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert out.strip()
        assert not ANSI.search(out)

    def test_window_hands_off_and_returns_instead_of_animating(self):
        """An agent calls this; it must come back immediately, not block the turn."""
        with (
            mock.patch.object(busy_terminal, "open_in_window", return_value=0) as handoff,
            mock.patch.object(busy_terminal, "run") as animate,
        ):
            assert busy_terminal.main(["--window", "--duration", "60"]) == 0

        handoff.assert_called_once()
        animate.assert_not_called()

    def test_interrupting_restores_the_cursor_and_exits_cleanly(self):
        with mock.patch.object(busy_terminal, "run", side_effect=KeyboardInterrupt):
            assert busy_terminal.main(["--no-color"]) == 0
