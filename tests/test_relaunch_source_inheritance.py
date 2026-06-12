"""Regression test for #45107.

`hermes chat --source <name>` was recorded as source='cli' because the CLI
drops --source when it self-relaunches: the flag was registered with plain
add_argument() instead of _inherited_flag(), so it never entered the relaunch
inheritance table. These tests pin the flag into that table.
"""
from hermes_cli.relaunch import (
    _build_inherited_flag_table,
    _extract_inherited_flags,
)


def test_source_flag_is_in_relaunch_inheritance_table():
    table = _build_inherited_flag_table()
    assert ("--source", True) in table, (
        "--source must be tagged inherit_on_relaunch so it survives a "
        "self-relaunch (regression #45107)"
    )


def test_source_value_survives_relaunch_extraction():
    argv = ["chat", "--source", "distiller", "-q", "hello"]

    preserved = _extract_inherited_flags(argv)

    assert "--source" in preserved
    assert preserved[preserved.index("--source") + 1] == "distiller"


def test_source_equals_form_survives_relaunch_extraction():
    argv = ["chat", "--source=distiller", "-q", "hello"]

    preserved = _extract_inherited_flags(argv)

    assert "--source=distiller" in preserved
