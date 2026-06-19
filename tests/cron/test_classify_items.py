"""``classify_items`` must surface items whose score arrives as a JSON float or
a numeric string, not only as a Python ``int``.

The threshold gate used ``isinstance(score, int)``. LLM classifiers routinely
return the score as a JSON float (``8.0`` -> ``float``) or a numeric string
(``"8"``); both failed the strict ``int`` check, so a genuinely urgent item was
silently dropped -> empty stdout -> the cron monitor treats it as "nothing to
report" and suppresses delivery. That is the silent-swallow failure the
script's own docstring warns against.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cron.scripts import classify_items


def _fake_response(score_payload):
    """Build an object shaped like the auxiliary-client response.

    ``main()`` reads ``resp.choices[0].message.content`` and expects a JSON
    array string of ``{"index", "score", "reason"}`` objects.
    """
    content = json.dumps([{"index": 0, "score": score_payload, "reason": "x"}])
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _run(monkeypatch, capsys, tmp_path, *, score_payload, threshold=7):
    """Drive ``classify_items.main()`` with one item and a stubbed classifier."""
    items_file = tmp_path / "items.json"
    items_file.write_text(json.dumps([{"title": "Server on fire"}]))

    def _fake_call_llm(*args, **kwargs):
        return _fake_response(score_payload)

    # main() lazily does ``from agent.auxiliary_client import call_llm``.
    monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm)
    monkeypatch.setattr(
        "sys.argv",
        [
            "classify_items.py",
            "--criteria",
            "urgent",
            "--threshold",
            str(threshold),
            "--input-file",
            str(items_file),
        ],
    )
    rc = classify_items.main()
    return rc, capsys.readouterr().out


def test_float_score_surfaced(monkeypatch, capsys, tmp_path):
    rc, out = _run(monkeypatch, capsys, tmp_path, score_payload=8.0)
    assert rc == 0
    assert "Server on fire" in out


def test_string_score_surfaced(monkeypatch, capsys, tmp_path):
    rc, out = _run(monkeypatch, capsys, tmp_path, score_payload="8")
    assert rc == 0
    assert "Server on fire" in out


def test_int_score_surfaced(monkeypatch, capsys, tmp_path):
    # Guards against regressing the already-working int path.
    rc, out = _run(monkeypatch, capsys, tmp_path, score_payload=8)
    assert rc == 0
    assert "Server on fire" in out


def test_below_threshold_silent(monkeypatch, capsys, tmp_path):
    # Threshold semantics must still hold: a 3.0 stays silent (empty stdout).
    rc, out = _run(monkeypatch, capsys, tmp_path, score_payload=3.0)
    assert rc == 0
    assert out.strip() == ""


@pytest.mark.parametrize(
    "value,expected",
    [
        (8, 8.0),
        (8.0, 8.0),
        ("8", 8.0),
        (" 8 ", 8.0),
        (True, None),
        (False, None),
        ("not-a-number", None),
        (None, None),
    ],
)
def test_coerce_score(value, expected):
    assert classify_items._coerce_score(value) == expected
