"""Regression tests for #97959: curator consolidation can never complete.

The skill_manage handler accepts ``absorbed_into`` on delete (flat and batch
sole-op shapes) and the fail-closed curator delete guard REFUSES deletes that
omit it — but the advertised schema never mentions the parameter. A
schema-strict review model therefore cannot emit it, every consolidation
delete is refused, and the run still reports success.

Layers covered (see LAYERS.md at the branch root):
1. The curator review fork must advertise ``absorbed_into`` in ITS OWN
   ``tools[]`` (fork-local; the shared schema stays lean for every other
   session).
2. (Prompt wording is covered by competitor PR #88892 — excluded here.)
3. A run whose deletes were all refused must surface that (run.json counts +
   REPORT.md + status summary), not report silent success.
4. The shared-schema comment must point at the fork-local advertisement.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Layer 1 — fork-local schema advertisement
# ---------------------------------------------------------------------------

def test_curator_fork_advertises_absorbed_into(tmp_path, monkeypatch):
    """L1 RED: the curator review fork's skill_manage schema must include an
    ``absorbed_into`` property — the guard refuses deletes without it, so a
    schema that omits it makes consolidation structurally unable to complete."""
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import agent.curator as curator
    importlib.reload(curator)

    monkeypatch.setattr(curator, "_resolve_review_provider", lambda: ({}, "", "", {}))

    tools = curator._advertise_absorbed_into(["skills"])
    sm = next(t for t in tools if t["function"]["name"] == "skill_manage")
    props = sm["function"]["parameters"]["properties"]["operations"]["items"]["properties"]
    assert "absorbed_into" in props, (
        "curator fork's skill_manage schema must advertise absorbed_into: the "
        "consolidation delete guard refuses deletes that omit it (#97959)")


def test_shared_schema_stays_lean(tmp_path, monkeypatch):
    """L1 guard: the shared SKILL_MANAGE_SCHEMA must NOT advertise
    ``absorbed_into`` — the omission is deliberate token-saving for every
    non-curator session; only the fork augments."""
    from tools.skill_manager_tool import SKILL_MANAGE_SCHEMA
    props = SKILL_MANAGE_SCHEMA["parameters"]["properties"]["operations"]["items"]["properties"]
    assert "absorbed_into" not in props


def test_fork_advertisement_does_not_poison_shared_cache(tmp_path, monkeypatch):
    """L1 guard: the fork's augmented schema must not leak into the memoized
    get_tool_definitions cache other sessions share."""
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import agent.curator as curator
    importlib.reload(curator)

    fork_tools = curator._advertise_absorbed_into(["skills"])
    fork_entry = next(t for t in fork_tools if t["function"]["name"] == "skill_manage")
    fork_props = fork_entry["function"]["parameters"]["properties"]["operations"]["items"]["properties"]
    assert "absorbed_into" in fork_props

    from model_tools import get_tool_definitions
    plain = get_tool_definitions(enabled_toolsets=["skills"], quiet_mode=True)
    plain_entry = next(t for t in plain if t["function"]["name"] == "skill_manage")
    plain_props = plain_entry["function"]["parameters"]["properties"]["operations"]["items"]["properties"]
    assert "absorbed_into" not in plain_props, (
        "fork-local augmentation leaked into the shared memoized schema cache")


# ---------------------------------------------------------------------------
# Layer 3 — refused-deletes observability
# ---------------------------------------------------------------------------

@pytest.fixture
def curator_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import tools.skill_usage as usage
    importlib.reload(usage)
    import agent.curator as curator
    importlib.reload(curator)

    monkeypatch.setattr(curator, "_run_llm_review", lambda prompt: {
        "final": "", "summary": "stub", "model": "", "provider": "",
        "tool_calls": [], "error": None})
    monkeypatch.setattr(curator, "_load_config", lambda: {"consolidate": True})
    yield {"home": home, "curator": curator, "usage": usage}


def _write_agent_skill(curator_env, name):
    usage = curator_env["usage"]
    skills_dir = curator_env["home"] / "skills"
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: x\n---\n", encoding="utf-8")
    usage.mark_agent_created(name)
    return d


def test_refused_consolidation_deletes_counted(tmp_path, monkeypatch, curator_env):
    """L3 RED: deletes the guard refused (skill still present after the run)
    must be counted, not silently reported as success."""
    curator = curator_env["curator"]
    _write_agent_skill(curator_env, "alpha")
    _write_agent_skill(curator_env, "beta")

    before = {"alpha", "beta"}
    after = {"alpha", "beta"}  # guard refused every delete
    tool_calls = [
        {"name": "skill_manage", "arguments": json.dumps({"action": "delete", "name": "alpha"})},
        {"name": "skill_manage", "arguments": json.dumps({"action": "delete", "name": "beta"})},
    ]
    refused = curator._refused_consolidation_deletes(before, after, tool_calls)
    assert refused == {"alpha", "beta"}


def test_refused_consolidation_deletes_zero_when_deletes_landed(tmp_path, monkeypatch, curator_env):
    """L3 companion: successful consolidations must not be flagged."""
    curator = curator_env["curator"]
    _write_agent_skill(curator_env, "alpha")
    _write_agent_skill(curator_env, "umbrella")

    before = {"alpha", "umbrella"}
    after = {"umbrella"}  # alpha absorbed
    tool_calls = [
        {"name": "skill_manage", "arguments": json.dumps(
            {"action": "delete", "name": "alpha", "absorbed_into": "umbrella"})},
    ]
    assert curator._refused_consolidation_deletes(before, after, tool_calls) == set()


def test_refused_consolidation_deletes_robust_to_truncated_args(curator_env):
    """L3 edge: truncated/garbage tool-call arguments must be skipped, never
    crash or overcount."""
    curator = curator_env["curator"]
    before, after = {"alpha"}, {"alpha"}
    tool_calls = [
        {"name": "skill_manage", "arguments": '{"action": "delete", "name": "alp'},  # truncated
        {"name": "skill_manage", "arguments": None},  # missing
        {"name": "skill_manage", "arguments": json.dumps({"action": "patch", "name": "alpha"})},  # not a delete
        {"name": "skills_list", "arguments": "{}"},  # other tool
    ]
    assert curator._refused_consolidation_deletes(before, after, tool_calls) == set()


def test_refused_consolidation_deletes_counts_distinct_not_retries(curator_env):
    """L3 edge: model retries on the same refused skill count once."""
    curator = curator_env["curator"]
    before, after = {"alpha", "beta"}, {"alpha", "beta"}
    tool_calls = [
        {"name": "skill_manage", "arguments": json.dumps({"action": "delete", "name": "alpha"})},
        {"name": "skill_manage", "arguments": json.dumps({"action": "delete", "name": "alpha"})},  # retry
        {"name": "skill_manage", "arguments": json.dumps({"action": "delete", "name": "beta"})},
    ]
    assert curator._refused_consolidation_deletes(before, after, tool_calls) == {"alpha", "beta"}


def test_refused_consolidation_deletes_batch_shape(curator_env):
    """L3 edge: a delete inside operations[] (sole-op batch) is detected."""
    curator = curator_env["curator"]
    before, after = {"alpha"}, {"alpha"}
    tool_calls = [
        {"name": "skill_manage", "arguments": json.dumps(
            {"operations": [{"action": "delete", "name": "alpha"}]})},
    ]
    assert curator._refused_consolidation_deletes(before, after, tool_calls) == {"alpha"}


def test_refused_consolidation_deletes_ignores_unknown_targets(curator_env):
    """L3 edge: a delete aimed at a skill absent from before_names (already
    gone, hallucinated) is not counted as refused."""
    curator = curator_env["curator"]
    before, after = {"alpha"}, {"alpha"}
    tool_calls = [
        {"name": "skill_manage", "arguments": json.dumps({"action": "delete", "name": "ghost"})},
    ]
    assert curator._refused_consolidation_deletes(before, after, tool_calls) == set()


# ---------------------------------------------------------------------------
# Layer 3 integration — run.json / REPORT.md / status summary
# ---------------------------------------------------------------------------

def test_run_report_surfaces_refused_deletes(curator_env):
    """L3 RED→GREEN: a refused-deletes run must carry the count in run.json
    counts and a warning line in REPORT.md — not report silent success."""
    import json as _json
    curator = curator_env["curator"]
    _write_agent_skill(curator_env, "alpha")
    _write_agent_skill(curator_env, "umbrella")

    from datetime import datetime, timezone as _tz
    llm_meta = {
        "final": "", "summary": "completed the pass", "model": "m", "provider": "p",
        "tool_calls": [
            {"name": "skill_manage", "arguments": _json.dumps({"action": "delete", "name": "alpha"})},
        ], "error": None}
    report_path = curator._write_run_report(
        started_at=datetime.now(_tz.utc), elapsed_seconds=1.0,
        auto_counts={"checked": 2, "marked_stale": 0, "archived": 0, "reactivated": 0},
        auto_summary="no changes",
        before_report=curator_env["usage"].curated_report(),
        before_names={"alpha", "umbrella"},
        after_report=curator_env["usage"].curated_report(),  # alpha still present
        llm_meta=llm_meta)
    assert report_path is not None
    run_json = _json.loads((report_path / "run.json").read_text())
    assert run_json["counts"]["refused_consolidation_deletes"] == 1
    assert run_json["refused_consolidation_delete_names"] == ["alpha"]
    report_md = (report_path / "REPORT.md").read_text()
    assert "Refused consolidation deletes" in report_md


def test_consolidation_pass_summary_flags_refused(curator_env, monkeypatch):
    """L3 RED→GREEN: last_run_summary (what `hermes curator status` shows) must
    flag refused consolidation deletes instead of the LLM's self-reported
    success."""
    import json as _json
    curator = curator_env["curator"]
    _write_agent_skill(curator_env, "alpha")
    _write_agent_skill(curator_env, "umbrella")

    monkeypatch.setattr(curator, "_run_llm_review", lambda prompt: {
        "final": "", "summary": "Completed the pass and verified the catalog.",
        "model": "m", "provider": "p",
        "tool_calls": [
            {"name": "skill_manage", "arguments": _json.dumps({"action": "delete", "name": "alpha"})},
        ], "error": None})

    final_summary, llm_meta = curator._consolidation_pass("auto: ", "no changes", False, {"alpha", "umbrella"})
    assert "refused" in final_summary
    assert "alpha" in final_summary
