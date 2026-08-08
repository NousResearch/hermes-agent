"""A route script only ignores a webhook for the three documented reasons.

The documented contract for route scripts is:

    "JSON stdout replaces the payload before prompt templating; empty stdout,
     [SILENT], or a nonzero exit ignores the webhook."

Stdout that parses as JSON but is not an object cannot *replace* the payload,
and it used to drop the webhook. That made the ignore rule depend on whether
the text happened to be JSON-parseable: ``print("hello")`` kept the event while
``print(json.dumps(items))`` — a list, one of the most natural transform
outputs — silently discarded it. Non-object JSON now rides along in
``script_output`` exactly like non-JSON text.
"""

from __future__ import annotations

import pytest

from gateway.platforms.webhook_filters import WebhookRouteProcessor

PAYLOAD = {"id": 7, "action": "opened"}


@pytest.fixture
def run_script(tmp_path, monkeypatch):
    """Run a route script body under a throwaway HERMES_HOME."""
    (tmp_path / "scripts").mkdir()
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    processor = WebhookRouteProcessor()
    counter = {"n": 0}

    def _run(body: str):
        counter["n"] += 1
        script = tmp_path / "scripts" / f"route_{counter['n']}.py"
        script.write_text(body, encoding="utf-8")
        return processor.run_route_script(script.name, PAYLOAD)

    return _run


@pytest.mark.parametrize(
    "body, expected_output",
    [
        ('import json; print(json.dumps([1, 2, 3]))', "[1, 2, 3]"),
        ('print(42)', "42"),
        ('print("true")', "true"),
        ('print("null")', "null"),
        ('import json; print(json.dumps("done"))', '"done"'),
    ],
)
def test_non_object_json_stdout_keeps_the_webhook(run_script, body, expected_output):
    """The regression: JSON that isn't an object must not drop the event."""
    keep, transformed = run_script(body)

    assert keep is True
    assert transformed is not None
    # Carried through like text, with the original payload preserved.
    assert transformed["script_output"] == expected_output
    assert transformed["id"] == PAYLOAD["id"]


def test_json_object_stdout_replaces_the_payload(run_script):
    keep, transformed = run_script('import json; print(json.dumps({"a": 1}))')

    assert keep is True
    assert transformed == {"a": 1}


def test_plain_text_stdout_rides_along_in_script_output(run_script):
    keep, transformed = run_script('print("hello")')

    assert keep is True
    assert transformed["script_output"] == "hello"
    assert transformed["action"] == PAYLOAD["action"]


@pytest.mark.parametrize(
    "body",
    [
        "pass",                      # empty stdout
        'print("[SILENT]")',         # explicit quiet marker
        "import sys; sys.exit(3)",   # nonzero exit
    ],
)
def test_documented_ignore_conditions_still_ignore(run_script, body):
    """The three documented ignores must keep ignoring."""
    keep, transformed = run_script(body)

    assert keep is False
    assert transformed is None


def test_explicit_ignore_markers_in_json_object_still_ignore(run_script):
    """An object carrying the opt-out keys stays an explicit ignore."""
    keep, _ = run_script('import json; print(json.dumps({"__hermes_ignore__": True}))')

    assert keep is False
