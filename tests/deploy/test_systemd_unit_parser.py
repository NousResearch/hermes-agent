"""The unit-file parser the template-unit tests rely on."""

from __future__ import annotations

import pytest

from tests.deploy._systemd_unit import UnitSyntaxError, parse_unit, space_list


def test_sections_keys_comments_and_repeats():
    unit = parse_unit("""
# comment
; also a comment
[Unit]
Description=slot %i

[Service]
Environment=A=1
Environment=B=2 C=3
ExecStart=/bin/sh -c 'echo "$$x = y"'
""")
    assert unit["Unit"] == {"Description": ["slot %i"]}
    assert unit["Service"]["Environment"] == ["A=1", "B=2 C=3"]
    assert unit["Service"]["ExecStart"] == ["/bin/sh -c 'echo \"$$x = y\"'"]


def test_continuation_lines_join():
    unit = parse_unit("[Service]\nExecStart=/bin/echo one \\\n    two \\\n  three\n")
    assert unit["Service"]["ExecStart"] == ["/bin/echo one two three"]


def test_empty_assignment_resets_a_list():
    unit = parse_unit("[Service]\nInaccessiblePaths=/a\nInaccessiblePaths=\nInaccessiblePaths=/b /c\n")
    assert space_list(unit, "Service", "InaccessiblePaths") == ["/b", "/c"]


@pytest.mark.parametrize("text, message", [
    ("User=root\n", "outside a section"),
    ("[Service]\nnot a directive\n", "not a directive"),
    ("[Service]\nA=1\n[Service]\nB=2\n", "repeated"),
    ("[Service]\nExecStart=/bin/true \\\n", "continuation"),
])
def test_malformed_units_are_refused(text, message):
    with pytest.raises(UnitSyntaxError, match=message):
        parse_unit(text)
