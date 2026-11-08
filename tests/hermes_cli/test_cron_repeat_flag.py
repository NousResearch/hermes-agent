"""``--repeat`` accepts the same spellings on the CLI (and ``/cron``) as the cron API.

``cron.jobs.normalize_repeat_value`` has always coerced the user-facing strings
(``'forever'`` -> no limit, ``'once'`` -> 1, numeric strings -> int), and ``create_job`` /
``update_job`` both route through it. The ``hermes cron create/edit`` argparse declaration was
``type=int``, so the spellings never reached it and died as ``invalid int value`` (#150986).

``forever`` must travel as an EXPLICIT value, not ``None``: the update path reads ``None`` as
"leave this field alone" (``tools/cronjob_tools.py::_update_run_fields``), so a collapsed ``None``
would make ``cron edit <id> --repeat forever`` a silent no-op. ``0`` is the API's own no-limit
number and keeps that distinction.
"""

from __future__ import annotations

import argparse

import pytest

from hermes_cli.subcommands.cron import build_cron_parser


def _parse(argv):
    parser = argparse.ArgumentParser(prog="hermes")
    build_cron_parser(parser.add_subparsers(dest="command"), cmd_cron=lambda args: None)
    return parser.parse_args(argv)


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [("3", 3), ("once", 1), ("1x", 1), ("forever", 0), ("infinite", 0), ("none", 0)],
)
def test_create_and_edit_accept_the_api_repeat_spellings(spelling, expected):
    created = _parse(["cron", "create", "in 1h", "x", "--repeat", spelling])
    assert created.repeat == expected
    edited = _parse(["cron", "edit", "j1", "--repeat", spelling])
    assert edited.repeat == expected


def test_repeat_absent_stays_distinguishable_from_forever():
    # "not given" is what keeps an unrelated `cron edit` from rewriting the job's budget.
    assert _parse(["cron", "edit", "j1"]).repeat is None
    assert _parse(["cron", "create", "in 1h", "x"]).repeat is None


def test_repeat_rejects_an_unknown_spelling_at_parse_time(capsys):
    with pytest.raises(SystemExit) as exc:
        _parse(["cron", "create", "in 1h", "x", "--repeat", "always"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "forever" in err and "once" in err


def test_cron_edit_forwards_forever_as_an_explicit_value(monkeypatch):
    from cron import jobs
    from hermes_cli import cron as cron_cli

    monkeypatch.setattr(jobs, "resolve_job_ref", lambda ref: {"id": "j1", "skills": []})
    seen: dict = {}

    def fake_api(**kwargs):
        seen.update(kwargs)
        return {"success": True, "job": {"job_id": "j1", "name": "n", "schedule": "0 9 * * *"}}

    monkeypatch.setattr(cron_cli, "_cron_api", fake_api)
    assert cron_cli.cron_edit(_parse(["cron", "edit", "j1", "--repeat", "forever"])) == 0
    assert seen["repeat"] == 0, "a collapsed None would read as 'leave the repeat budget alone'"


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [("3", 3), ("once", 1), ("forever", None), ("infinite", None)],
)
def test_slash_cron_repeat_accepts_the_same_spellings(spelling, expected):
    from hermes_cli.cli_cron_lines import _parse_cron_flags

    opts = _parse_cron_flags(["in 1h", "x", "--repeat", spelling])
    assert opts is not None, "a valid spelling must not be read as a bad flag line"
    assert opts["repeat"] == expected


def test_slash_cron_repeat_still_rejects_garbage():
    from hermes_cli.cli_cron_lines import _parse_cron_flags

    assert _parse_cron_flags(["in 1h", "x", "--repeat", "always"]) is None
