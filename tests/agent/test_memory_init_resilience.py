"""A bad memory tuning value must not silently disable memory.

`agent_init` builds the whole memory subsystem under one
`except Exception: pass`. `nudge_interval` — a cosmetic "how often to nudge"
knob — was read with a bare `int()` *before* `MemoryStore` was constructed,
so `nudge_interval: ten`, or a bare `nudge_interval:` (YAML `None`), raised
and took the store, `load_from_disk()`, and the entire feature down with it.
No error, no log line: indistinguishable from memory never having been
enabled.
"""

import logging

import pytest

from agent.agent_init import _memory_int_setting


class TestCoercion:
    @pytest.mark.parametrize("bad", ["ten", "10s", "", None, {"a": 1}, [1], True, False])
    def test_a_bad_value_never_raises(self, bad):
        assert _memory_int_setting({"nudge_interval": bad}, "nudge_interval", 10) == 10

    @pytest.mark.parametrize("good,expected", [(7, 7), ("42", 42), (0, 0), (-1, -1)])
    def test_real_values_pass_through(self, good, expected):
        assert (
            _memory_int_setting({"nudge_interval": good}, "nudge_interval", 10)
            == expected
        )

    def test_an_absent_key_uses_the_default(self):
        assert _memory_int_setting({}, "nudge_interval", 10) == 10

    def test_a_bad_value_is_reported(self, caplog):
        """Silent fallback would just move the invisibility, not remove it."""
        with caplog.at_level(logging.WARNING, logger="run_agent"):
            _memory_int_setting({"nudge_interval": "ten"}, "nudge_interval", 10)
        messages = [r.getMessage() for r in caplog.records]
        assert any("nudge_interval" in m for m in messages), messages
        assert any("ten" in m for m in messages), messages

    def test_true_is_not_taken_as_one(self):
        """bool subclasses int; `nudge_interval: true` is a typo, not 1."""
        assert _memory_int_setting({"nudge_interval": True}, "nudge_interval", 10) == 10


class TestTheOriginalFailure:
    @pytest.mark.parametrize("bad", ["ten", None])
    def test_the_old_bare_int_aborted_on_these(self, bad):
        """Documents the shape this replaced — a raise before the store existed."""
        with pytest.raises((ValueError, TypeError)):
            int({"nudge_interval": bad}.get("nudge_interval", 10))

    def test_the_helper_survives_exactly_those(self):
        for bad in ("ten", None):
            assert _memory_int_setting({"nudge_interval": bad}, "nudge_interval", 10) == 10
