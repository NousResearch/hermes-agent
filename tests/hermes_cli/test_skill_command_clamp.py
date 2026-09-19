"""Skill slash-command registration keeps names that collide with built-ins.

Regression for the silent-drop bug in ``hermes_cli.commands._clamp_command_names``:
a skill whose full name collides with a reserved command name (e.g. a skill
called ``handoff`` colliding with the built-in) must be renamed with the same
numeric-suffix disambiguation the truncation branch uses, not dropped.
"""

import logging

from hermes_cli.commands import _clamp_command_names


def test_full_name_collision_renamed_not_dropped():
    entries = [("handoff", "Hand off to another model", "/handoff")]
    result = _clamp_command_names(entries, {"handoff"})
    assert len(result) == 1
    name, desc, cmd_key = result[0]
    assert name == "handoff0"
    assert cmd_key == "/handoff", "cmd_key must survive the collision rename"


def test_non_colliding_name_untouched():
    entries = [("myplan", "My plan skill", "/myplan")]
    result = _clamp_command_names(entries, {"handoff"})
    assert result == [("myplan", "My plan skill", "/myplan")]


def test_all_digits_exhausted_drops_entry():
    reserved = {"handoff"} | {f"handoff{d}" for d in range(10)}
    result = _clamp_command_names([("handoff", "d", "/handoff")], reserved)
    assert result == []


def test_collision_warning_names_final_command(caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_cli.commands"):
        result = _clamp_command_names([("handoff", "d", "/handoff")], {"handoff"})
    assert result[0][0] == "handoff0"
    assert "handoff0" in caplog.text, "warning must name the final command name"
