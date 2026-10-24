"""scripts/check_namedtuple_required_fields.py: the rules that catch the 2026-10-24 outage.

Field incident (2026-10-24 02:16Z): ``agent/auxiliary_client.py`` gained a
``defaults=`` argument on the FUNCTIONAL ``NamedTuple`` form. ``typing.NamedTuple``
rejects that at IMPORT time ("Either list of fields or keywords can be provided
to NamedTuple, not both"), and because the cron external worker imports the LIVE
checkout, 17 distinct cron jobs died in the same second with the same opaque
"exited before ownership acknowledgement (exit 1)".

The check turns that into a 5-second error at commit time:

* a REQUIRED field added to an EXISTING NamedTuple class is flagged;
* a functional ``NamedTuple("Name", [...])`` given ``defaults=`` is flagged
  (that form accepts NEITHER ``defaults=`` nor per-field defaults);
* a NEW NamedTuple is not flagged (no existing positional callers to break);
* a deliberate exception can be suppressed with a marker comment.
"""

import ast
import importlib.util
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_namedtuple_required_fields.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_namedtuple_required_fields", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(tmp_path, source: str) -> Path:
    path = tmp_path / "sample.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def _scan(tmp_path, source: str, previous: dict | None = None):
    """Write *source* to a temp file and scan it; return the problem list."""
    mod = _load()
    path = _write(tmp_path, source)
    return mod.scan_file(path, previous if previous is not None else {})


def _previous_with_field(field: str) -> dict:
    """Build a fake 'before' ClassDef for a NamedTuple with one field."""
    mod = _load()
    src = textwrap.dedent(f"""
        from typing import NamedTuple, Optional

        class Route(NamedTuple):
            client: object
            {field}
    """)
    tree = ast.parse(src)
    return mod._namedtuple_types(tree)


# ---------------------------------------------------------------------------
# The functional form with `defaults=` — the incident's exact failure
# ---------------------------------------------------------------------------

def test_functional_namedtuple_with_defaults_is_flagged(tmp_path):
    """The incident's own mistake: `defaults=` on the functional form."""
    problems = _scan(tmp_path, """
        from typing import Any, NamedTuple, Optional

        _LadderRoute = NamedTuple("_LadderRoute", [
            ("client", Any), ("tag", str), ("required_tokens", Optional[int]),
        ], defaults=(None,))
    """)
    assert problems
    assert "defaults=" in problems[0]
    assert "functional NamedTuple" in problems[0]


def test_plain_functional_namedtuple_is_not_flagged(tmp_path):
    """The legal functional form passes."""
    problems = _scan(tmp_path, """
        from typing import NamedTuple

        X = NamedTuple("X", [("a", int), ("b", str)])
    """)
    assert problems == []


# ---------------------------------------------------------------------------
# A REQUIRED field added to an EXISTING NamedTuple class
# ---------------------------------------------------------------------------

def test_required_field_added_to_existing_namedtuple_is_flagged(tmp_path):
    """Appending a field with no default breaks every positional constructor."""
    mod = _load()
    previous = _previous_with_field("tag: str")
    path = _write(tmp_path, """
        from typing import NamedTuple, Optional

        class Route(NamedTuple):
            client: object
            tag: str
            required_tokens: Optional[int]
    """)
    problems = mod.scan_file(path, previous)
    assert problems
    assert "required_tokens" in problems[0]
    assert "REQUIRED field" in problems[0]


def test_a_defaulted_field_added_to_existing_namedtuple_is_not_flagged(tmp_path):
    """The correct fix (a default at definition time) passes."""
    mod = _load()
    previous = _previous_with_field("tag: str")
    path = _write(tmp_path, """
        from typing import NamedTuple, Optional

        class Route(NamedTuple):
            client: object
            tag: str
            required_tokens: Optional[int] = None
    """)
    assert mod.scan_file(path, previous) == []


def test_a_new_namedtuple_with_a_required_field_is_not_flagged(tmp_path):
    """A fresh type has no existing positional constructors to break."""
    problems = _scan(tmp_path, """
        from typing import NamedTuple

        class BrandNew(NamedTuple):
            a: int
            b: str
    """)
    assert problems == []


def test_an_existing_namedtuple_with_only_required_fields_is_not_flagged(tmp_path):
    """Only an ADDED required field breaks; the pre-existing ones are fine."""
    mod = _load()
    previous = _previous_with_field("tag: str")
    path = _write(tmp_path, """
        from typing import NamedTuple

        class Route(NamedTuple):
            client: object
            tag: str
    """)
    assert mod.scan_file(path, previous) == []


# ---------------------------------------------------------------------------
# A REQUIRED field that FOLLOWS a defaulted one — the 06:49Z wave
# ---------------------------------------------------------------------------
#
# 2026-10-24 06:49Z: a transient edit put a required field AFTER a defaulted
# one in `_LadderRoute`. Python rejects the CLASS at import time ("Non-default
# namedtuple field 'x' cannot follow default field 'y'"), and because the cron
# external worker imports the LIVE checkout, 8 jobs across 2 profiles died in
# the same second. The "added a required field" rule cannot see this: the
# rule needs a before/after diff, and the failure is purely lexical.

def test_required_field_after_a_defaulted_field_is_flagged(tmp_path):
    """Python rejects the class at import; the check must reject it too."""
    problems = _scan(tmp_path, """
        from typing import Any, NamedTuple, Optional

        class Route(NamedTuple):
            client: Any
            required_tokens: Optional[int] = None
            brand_new_required_field: Optional[str]
    """)
    assert problems
    assert "brand_new_required_field" in problems[0]
    assert "required_tokens" in problems[0]
    assert "cannot follow default field" in problems[0]


def test_required_field_after_a_defaulted_field_is_flagged_on_a_NEW_type(tmp_path):
    """The ordering rule needs no diff, so a brand-new type is flagged too.

    The "gained a required field" rule deliberately skips a NEW NamedTuple
    (no existing positional callers to break). The ordering failure is
    different: the type never comes into existence at all, so there is no
    caller to protect.
    """
    problems = _scan(tmp_path, """
        from typing import Any, NamedTuple, Optional

        class BrandNew(NamedTuple):
            client: Any
            required_tokens: Optional[int] = None
            later_required: Optional[str]
    """)
    assert problems
    assert "later_required" in problems[0]


def test_required_field_before_a_defaulted_field_is_not_flagged(tmp_path):
    """The legal order passes: required fields first, defaulted ones after."""
    problems = _scan(tmp_path, """
        from typing import Any, NamedTuple, Optional

        class Route(NamedTuple):
            client: Any
            brand_new_required_field: Optional[str]
            required_tokens: Optional[int] = None
    """)
    assert problems == []


def test_all_required_fields_are_not_flagged(tmp_path):
    """No default anywhere: nothing can follow a default."""
    problems = _scan(tmp_path, """
        from typing import Any, NamedTuple

        class Route(NamedTuple):
            client: Any
            tag: str
    """)
    assert problems == []


def test_all_defaulted_fields_are_not_flagged(tmp_path):
    """Every field defaulted: the legal shape."""
    problems = _scan(tmp_path, """
        from typing import Any, NamedTuple, Optional

        class Route(NamedTuple):
            client: Any = None
            required_tokens: Optional[int] = None
    """)
    assert problems == []


def test_the_ordering_finding_names_both_fields(tmp_path):
    """The message must say WHICH field is wrong and WHICH field it follows."""
    problems = _scan(tmp_path, """
        from typing import Any, NamedTuple, Optional

        class Route(NamedTuple):
            client: Any
            first_default: Optional[int] = None
            second_default: Optional[str] = None
            the_bad_one: Optional[bool]
    """)
    assert problems
    # The FIRST violation is the import-time error; the rest never run.
    assert "the_bad_one" in problems[0]
    assert "second_default" in problems[0]


def test_a_suppression_marker_silences_the_ordering_finding(tmp_path):
    """The escape hatch works for the ordering rule too."""
    problems = _scan(tmp_path, """
        from typing import Any, NamedTuple, Optional

        class Route(NamedTuple):
            client: Any
            required_tokens: Optional[int] = None
            brand_new_required_field: Optional[str]  # namedtuple-required-field: ok -- intentional
    """)
    assert problems == []


# ---------------------------------------------------------------------------
# The suppression marker
# ---------------------------------------------------------------------------

def test_a_suppression_marker_silences_the_finding(tmp_path):
    """A deliberate exception is opt-in and must be visible in the source."""
    mod = _load()
    previous = _previous_with_field("tag: str")
    path = _write(tmp_path, """
        from typing import NamedTuple, Optional

        class Route(NamedTuple):
            client: object
            tag: str
            required_tokens: Optional[int]  # namedtuple-required-field: ok -- intentional
    """)
    assert mod.scan_file(path, previous) == []


# ---------------------------------------------------------------------------
# The real tree
# ---------------------------------------------------------------------------

def test_the_live_tree_passes_the_check():
    """The shipped tree has no violation."""
    mod = _load()
    assert mod.main([]) == 0
