"""A reasoning stream that keeps restating its own sentences is recognised mid-stream.

``RunawayStreamWatch`` needs exact contiguous repeats to dominate a 64k-char tail. Two shapes
recorded with a hosted reasoning model slip past it: a multi-paragraph plan restated every few
thousand characters with other sentences in between (8 restatements in 54k chars, never cut),
and a slow spiral that only tightens into an exact loop later (cut at 64k of 409k chars, and the
turn then ended with an error). The fixtures reproduce those shapes with synthetic text; the
turn-level recovery is in ``test_repetition_recovery.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.circling_guard import (
    DEFAULT_MIN_CHARS,
    CirclingSettings,
    CirclingStreamWatch,
    ReasoningStreamWatch,
    RecoverySettings,
    apply_repetition_guard_config,
    recovery_feedback,
    settings_from_config,
)
from agent.repetition_guard import RunawayStreamWatch
from tests.fakes.circling_text import PLAN, distinct_text, restated_plan


def _fired_at(watch, text: str, step: int = 37):
    for i in range(0, len(text), step):
        if watch.feed(text[i:i + step]):
            return i + step
    return None


class TestCirclingStreamWatch:
    def test_restated_plan_is_caught_and_the_tail_watch_misses_it(self):
        text = restated_plan(80)
        assert len(text) > 50_000
        assert _fired_at(RunawayStreamWatch(), text) is None

        fired = _fired_at(CirclingStreamWatch(), text)

        assert fired is not None and fired < 30_000

    def test_fires_no_earlier_than_the_floor(self):
        loop = "Let me check the queue worker and try to start it again.\n" * 2_000

        assert _fired_at(CirclingStreamWatch(), loop) == pytest.approx(DEFAULT_MIN_CHARS, abs=100)

    def test_new_text_after_distinct_text_keeps_the_watch_quiet(self):
        assert _fired_at(CirclingStreamWatch(), distinct_text(200_000)) is None

    def test_loop_after_long_distinct_prefix_is_caught_within_one_window(self):
        prefix = distinct_text(100_000)

        fired = _fired_at(CirclingStreamWatch(), prefix + restated_plan(80))

        assert fired is not None and fired - len(prefix) < 30_000

    def test_one_restatement_is_not_circling(self):
        twice = distinct_text(9_000) + restated_plan(2) + distinct_text(9_000)

        assert _fired_at(CirclingStreamWatch(), twice) is None

    def test_short_units_are_ignored(self):
        code = "    else:\n        return None\n    }\n" * 3_000

        assert _fired_at(CirclingStreamWatch(), code) is None

    def test_a_plan_restated_on_one_long_line_is_judged_by_sentence(self):
        glued = restated_plan(80, paragraph_break=" ")
        assert "\n" not in glued.rstrip("\n")

        fired = _fired_at(CirclingStreamWatch(), glued)

        assert fired is not None and fired < 30_000  # judged as it streams, not at the closing newline

    def test_settings_trade_latency_for_strictness(self):
        text = restated_plan(80)
        loose = _fired_at(CirclingStreamWatch(CirclingSettings(window_units=24, stale_ratio=0.8, min_chars=0)), text)
        strict = _fired_at(CirclingStreamWatch(CirclingSettings(window_units=120, repeat_count=6)), text)

        assert loose < strict

    def test_feedback_counts_the_repeats_and_quotes_no_reasoning(self):
        watch = CirclingStreamWatch()
        _fired_at(watch, restated_plan(80))

        message = recovery_feedback(watch.repeated_units())

        assert message.startswith("[System: ")
        assert f"{watch.repeated_units()[0][0]} times" in message
        assert not any(sentence.lower() in message.lower() for sentence in PLAN)
        assert "blocker report" in message


# Text that repeats lines for good reasons must never look like circling (the controls the
# defaults were checked against). Repeating one sentence 500 times IS cut, after 8k chars.
_LEGITIMATE_REPETITION = {
    "asked-for-repeat": "Hello world, this is a sentence the user asked me to repeat many times.\n" * 50,
    "rows-sharing-a-suffix": "".join(
        f"| {i:05d} | status: pending review by the on-call engineer before the next release window |\n"
        for i in range(2_000)),
    "rows-sharing-a-sentence": "".join(
        f"Row {i} is done. It is pending review by the on-call engineer before the next release window.\n"
        for i in range(2_000)),
    "templated-yaml": "".join(
        f"- name: service-{i:03d}\n  image: registry.example.com/team/service-{i:03d}:1.4.{i}\n  replicas: 2\n"
        '  limits: {cpu: "500m", memory: "256Mi", ephemeral-storage: "1Gi"}\n'
        '  healthcheck: "curl -fsS http://localhost:8080/healthz || exit 1"\n'
        for i in range(300)),
    "test-boilerplate": "".join(
        f"def test_item_{i}(client):\n    response = client.get('/api/items/{i}?expand=owner')\n"
        "    assert response.status_code == 200, 'unexpected status for the item endpoint'\n"
        "    assert response.json()['owner'] == 'fixture-user', 'owner should be the fixture user'\n\n"
        for i in range(400)),
}


@pytest.mark.parametrize("text", _LEGITIMATE_REPETITION.values(), ids=_LEGITIMATE_REPETITION.keys())
def test_legitimate_repetition_is_not_circling(text):
    assert _fired_at(CirclingStreamWatch(), text) is None


class TestReasoningStreamWatch:
    def test_cut_names_the_detector_the_size_and_the_repeated_statements(self):
        watch = ReasoningStreamWatch()
        fired = _fired_at(watch, restated_plan(80))
        response = SimpleNamespace()

        watch.mark_cut(response, "reasoning")

        cut = response._repetition_cut
        assert response._runaway_repetition is True
        assert (cut.channel, cut.detector, cut.chars) == ("reasoning", "circling", fired)
        assert cut.repeated and all(count >= 3 for count, _ in cut.repeated)

    def test_an_exact_loop_is_still_cut_by_the_tail_watch(self):
        watch = ReasoningStreamWatch(SimpleNamespace(_circling_settings=None))
        loop = "De bot en de werkers blijven lopen. Ik zoek het punt waar de tekst binnenkomt. " * 2_000

        assert _fired_at(watch, loop) is not None
        assert watch.cut("reasoning").detector == "exact_loop"

    def test_disabled_circling_lets_a_restated_plan_through(self):
        assert _fired_at(ReasoningStreamWatch(SimpleNamespace(_circling_settings=None)), restated_plan(80)) is None

    def test_a_loop_in_the_details_alone_is_cut(self):
        watch = ReasoningStreamWatch()
        loop = "Weighing the same option again and again until something changes.\n"

        assert any(watch.feed(f"step {i}. ", loop * 10) for i in range(2_000))


class TestConfig:
    def test_missing_section_gives_the_defaults(self):
        agent = SimpleNamespace()

        apply_repetition_guard_config(agent, None)

        assert agent._circling_settings == CirclingSettings()
        assert (agent._repetition_recovery_limit, agent._repetition_recovery_reasoning_off) == (1, False)

    def test_section_is_read(self):
        section = {"circling": {"window_units": 40, "stale_ratio": 0.9, "repeat_count": 4, "min_chars": 12000},
                   "recovery": {"max_attempts": 0, "reasoning": False}}  # unquoted YAML ``off`` is False
        agent = SimpleNamespace()

        apply_repetition_guard_config(agent, section)

        assert agent._circling_settings == CirclingSettings(
            window_units=40, stale_ratio=0.9, repeat_count=4, min_chars=12_000)
        assert (agent._repetition_recovery_limit, agent._repetition_recovery_reasoning_off) == (0, True)

    @pytest.mark.parametrize("enabled", [False, "false", "off", "no"])
    def test_disabled_circling_is_none(self, enabled):
        assert settings_from_config({"enabled": enabled}) is None

    def test_bad_values_fall_back_to_the_defaults(self):
        d = CirclingSettings()

        settings = settings_from_config({
            "enabled": "maybe", "window_units": "wide", "stale_ratio": 1.5, "repeat_count": 1, "min_chars": -5})

        assert settings == d
        assert settings_from_config("not a mapping") == d
        assert settings_from_config({"window_units": True, "stale_ratio": 0}) == d
        assert RecoverySettings.from_mapping({"max_attempts": -1, "reasoning": "sometimes"}) == RecoverySettings()
        for count in (float("inf"), float("nan"), 0.5):  # YAML .inf (raised OverflowError at agent build), .nan, a fraction
            assert RecoverySettings.from_mapping({"max_attempts": count}) == RecoverySettings()
            assert settings_from_config({"window_units": count}) == d
        assert RecoverySettings.from_mapping([1]) == RecoverySettings()
