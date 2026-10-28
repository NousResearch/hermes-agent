"""Tests for the cronjob skills repr fix (issue #132867).

A model caller can pass ``skills="['x', 'y']"`` — a *string repr* of a list instead of a
list. Before the fix every normalization step str()-ed that string into ONE bogus skill
literally named ``['x', 'y']``: the job stored it, the fire-time loader silently skipped
it (jobs ran without their skills), and ``cron doctor`` reported the record clean.

``cron/skill_lists.py`` owns the one definition of "repr-shaped"; these tests hold the callers
to it. Four paths normalize skills, and issue #132867 lists all four:

- Layer 0 — the table of the four call sites, over both shapes the reporter showed;
- Layer 1 — tools/cronjob_job_args.py::_canonical_skills, the create/update write path;
- Layer 2 — the storage read path (``cron.jobs._apply_skill_fields``), which heals pre-fix
  records through the same leaf;
- Layer 3 — hermes_cli/cron.py doctor, which flags a *stored* repr-shaped field.

The doctor check is the subtle one: the read path heals the field, so a doctor that only looks
at normalized records can never see the corruption — the defect Enough1122 found on the first
revision of this PR. Its coverage therefore drives the real entry point (``cron_doctor()`` over
a real jobs.json) instead of hand-built dicts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture()
def cron_store(tmp_path):
    """Route cron storage at a temp home, so these tests never touch the real jobs.json."""
    from cron.jobs import use_cron_store
    with use_cron_store(tmp_path):
        yield tmp_path / "cron"


def _create_stored_job(**raw_skill_fields):
    """Persist a real job through the public API, then rewrite its *stored* skills field the way
    a pre-fix release left it. Returns the file's parsed payload for post-run assertions."""
    from cron.jobs import create_job
    job = create_job(prompt="buy milk", schedule="every 1h", name="digest")
    path = create_job.__globals__["_current_cron_store"]().jobs_file
    data = json.loads(path.read_text(encoding="utf-8"))
    record = next(r for r in data["jobs"] if r["id"] == job["id"])
    record.pop("skills", None)
    record.pop("skill", None)
    record.update(raw_skill_fields)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, job["id"]


# --- Layer 0: every path the reporter listed heals the same input -------------

# The reporter's own repro, in both shapes a store can hold: the bare string a model caller
# sends, and the one-entry list a pre-fix normalizer wrote when it wrapped that string.
CORRUPT_BARE = "['x', 'y']"
CORRUPT_WRAPPED = ["['x', 'y']"]
HEALED = ["x", "y"]


def _store_read_path(value):
    """The read path a scheduled job actually takes: normalize the record's skill fields."""
    from cron.jobs import _apply_skill_fields
    return _apply_skill_fields({"skills": value})["skills"]


def _write_path(value):
    from tools.cronjob_job_args import _canonical_skills
    return _canonical_skills(skills=value)


def _prompt_path(value):
    """The fire-time assembler: an unrepaired field here means the run silently loses skills."""
    from cron.scheduler_prompt import _job_skill_names
    return _job_skill_names({"skills": value})


def _cli_path(value):
    from hermes_cli.cron import _normalize_skills
    return _normalize_skills(skills=value)


@pytest.mark.parametrize("path", [_store_read_path, _write_path, _prompt_path, _cli_path])
@pytest.mark.parametrize("value", [CORRUPT_BARE, CORRUPT_WRAPPED], ids=["bare", "wrapped"])
def test_every_normalization_path_heals_a_repr_skill_list(path, value):
    assert list(path(value)) == HEALED


# --- Layer 1: tool-side create/update normalization --------------------------

def test_canonical_skills_unwraps_repr_string():
    from tools.cronjob_job_args import _canonical_skills
    assert _canonical_skills(skills="['x', 'y']") == ["x", "y"]
    assert _canonical_skills(skills="['x']") == ["x"]


def test_canonical_skills_unwraps_repr_single_skill_param():
    from tools.cronjob_job_args import _canonical_skills
    assert _canonical_skills(skill="['a','b']") == ["a", "b"]


def test_canonical_skills_unwraps_tuple_repr():
    from tools.cronjob_job_args import _canonical_skills
    assert _canonical_skills(skills="('t1', 't2')") == ["t1", "t2"]


def test_canonical_skills_leaves_plain_names_alone():
    from tools.cronjob_job_args import _canonical_skills
    assert _canonical_skills(skills=["x", "y"]) == ["x", "y"]
    assert _canonical_skills(skills="plain-skill") == ["plain-skill"]
    assert _canonical_skills(skill="my skill (v2)") == ["my skill (v2)"]
    assert _canonical_skills(skills=None, skill="one") == ["one"]


def test_canonical_skills_leaves_non_repr_brackets_alone():
    from tools.cronjob_job_args import _canonical_skills
    assert _canonical_skills(skills="not [ a list") == ["not [ a list"]
    assert _canonical_skills(skills="[unclosed") == ["[unclosed"]
    assert _canonical_skills(skills="[]") == []


# --- Layer 2: storage read path heals pre-fix records -------------------------

def test_normalize_skill_names_heals_repr_stored():
    from cron.skill_lists import normalize_skill_names
    assert normalize_skill_names(skills="['x', 'y']") == ["x", "y"]
    assert normalize_skill_names(skill="['a','b']") == ["a", "b"]
    assert normalize_skill_names(skills=("t1", "t2")) == ["t1", "t2"]


def test_normalize_skill_names_plain_paths_unchanged():
    from cron.skill_lists import normalize_skill_names
    assert normalize_skill_names(skills=["x", "x", "y"]) == ["x", "y"]
    assert normalize_skill_names(skill="one") == ["one"]
    assert normalize_skill_names() == []
    assert normalize_skill_names(skills="plain-skill") == ["plain-skill"]
    assert normalize_skill_names(skills="[unclosed") == ["[unclosed"]


def test_apply_skill_fields_heals_corrupted_record():
    from cron.jobs import _apply_skill_fields
    healed = _apply_skill_fields({"skills": "['news-digest', 'summarize']", "id": "j1"})
    assert healed["skills"] == ["news-digest", "summarize"]
    assert healed["skill"] == "news-digest"


def test_list_jobs_heals_a_repr_field_read_from_disk(cron_store):
    """The heal is load-bearing: a job stored before the fix still gets its skills back."""
    from cron.jobs import list_jobs
    _create_stored_job(skills="['news-digest', 'summarize']")
    assert list_jobs()[0]["skills"] == ["news-digest", "summarize"]
    assert list_jobs()[0]["skill"] == "news-digest"


# --- Layer 3: doctor detection, through the real store + entry point ----------

def test_doctor_reports_a_repr_skills_field_stored_on_disk(cron_store, capsys):
    """The regression this guards: the check used to look only at normalized records, which the
    read path had already healed — so it could not fire at all, while the tests stayed green."""
    from hermes_cli.cron import cron_doctor
    _create_stored_job(skills="['news-digest', 'summarize']")
    assert cron_doctor() == 1
    out = capsys.readouterr().out
    assert "string repr" in out
    assert "news-digest" in out


def test_doctor_reports_a_repr_legacy_skill_field_stored_on_disk(cron_store, capsys):
    from hermes_cli.cron import cron_doctor
    _create_stored_job(skills=[], skill="['a', 'b']")
    assert cron_doctor() == 1
    assert "string repr" in capsys.readouterr().out


def test_doctor_reports_a_repr_field_on_a_record_without_an_id(cron_store, capsys):
    """A record with no id is listed under the read path's "unknown" id, so its stored copy has
    to be filed under that same id as well: keying the store by "" would put the two copies in
    different buckets and go quiet, the same silence this revision exists to remove."""
    from hermes_cli.cron import cron_doctor
    path, _ = _create_stored_job(skills="['news-digest']")
    data = json.loads(path.read_text(encoding="utf-8"))
    for record in data["jobs"]:
        record.pop("id", None)
    path.write_text(json.dumps(data), encoding="utf-8")
    assert cron_doctor() == 1
    assert "string repr" in capsys.readouterr().out


def test_doctor_quiet_when_the_store_is_clean(cron_store, capsys):
    from hermes_cli.cron import cron_doctor
    _create_stored_job(skills=["news-digest"])
    assert cron_doctor() == 0
    assert "string repr" not in capsys.readouterr().out


def test_doctor_quiet_on_a_real_bracket_named_skill(cron_store, capsys):
    """A skill legitimately named ``[weird-name]`` is a plain string, not a repr, and must not
    trip the check."""
    from hermes_cli.cron import cron_doctor
    _create_stored_job(skills=["[weird-name]"])
    assert cron_doctor() == 0
    assert "string repr" not in capsys.readouterr().out


def test_doctor_reports_the_wrapped_shape_a_pre_fix_release_wrote(cron_store, capsys):
    """The shape the bug actually left on disk: the pre-fix normalizer str()-ed the caller's
    string into a ONE-entry list holding the repr, so a store can hold either shape."""
    from hermes_cli.cron import cron_doctor
    _create_stored_job(skills=["['news-digest', 'summarize']"])
    assert cron_doctor() == 1
    assert "string repr" in capsys.readouterr().out


def test_doctor_scope_stops_at_the_active_jobs_it_reports(cron_store, capsys):
    """Scope decision, not an oversight: doctor diagnoses the jobs it lists. A disabled record
    is not one of them — and the read path heals it the moment the job runs again."""
    from hermes_cli.cron import cron_doctor
    _create_stored_job(skills="['news-digest']", enabled=False)
    assert cron_doctor() == 0
    assert "string repr" not in capsys.readouterr().out


def test_doctor_flags_the_record_without_rewriting_it(cron_store):
    """Doctor is read-only: the finding is a re-save hint, so the file keeps its shape."""
    from hermes_cli.cron import cron_doctor
    path, job_id = _create_stored_job(skills="['news-digest']")
    cron_doctor()
    stored = next(r for r in json.loads(path.read_text(encoding="utf-8"))["jobs"]
                  if r["id"] == job_id)
    assert stored["skills"] == "['news-digest']"


def test_issues_for_job_judges_the_stored_record_when_it_gets_one():
    """Direct-call contract: the normalized record reads clean, so the stored copy is what
    decides — and without one, the argument the caller handed over is judged as-is."""
    from hermes_cli.cron import _cron_doctor_issues_for_job
    job = {"id": "job-1", "name": "digest", "enabled": True, "state": "active",
           "next_run_at": "2099-01-01T00:00:00", "skills": ["news-digest"]}
    assert not any("string repr" in i for i in _cron_doctor_issues_for_job(job))
    flagged = _cron_doctor_issues_for_job(job, stored={**job, "skills": "['news-digest']"})
    assert any("string repr" in i for i in flagged)
    assert any("string repr" in i
               for i in _cron_doctor_issues_for_job({**job, "skills": "['news-digest']"}))
