"""Curator LLM pass profile scope: the review thread must inherit the caller's contextvars.

On a multiplexed gateway the weekly tick runs inside ``profile_scoped_chore`` →
``_profile_runtime_scope``, which installs ``set_hermes_home_override`` (and the profile
secret scope) as CONTEXTVARS. ``run_curator_review()`` fires its LLM half on a daemon
thread; a bare ``threading.Thread`` starts with an empty context, so the whole pass used
to run unscoped: provider resolution hit ``UnscopedSecretError`` and every home lookup
(snapshot, run.json/REPORT.md, ``.curator_state``) fell back to the process home — the
ROOT home's skills were read, reported and overwritten under another profile's run
(#125032).

The fix copies the caller's context into the thread (``copy_context().run``), the same
shape the gateway uses for executor work.
"""

from __future__ import annotations

import importlib
import json
import threading
from pathlib import Path

import pytest


@pytest.fixture
def curator_env(tmp_path, monkeypatch):
    """Root home + a served profile home; HERMES_HOME pinned to the ROOT (process home)."""
    root = tmp_path / "hermes_home"
    profile = root / "profiles" / "served"
    for home in (root, profile):
        (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))

    import tools.skill_usage as usage
    import agent.curator as curator
    for m in (usage, curator):
        importlib.reload(m)

    # Offline LLM: record the candidate list the fork actually saw (home-sensitive!).
    seen: dict = {}

    def _fake_review(prompt: str):
        seen["prompt"] = prompt
        return {"final": "", "summary": "stubbed", "model": "m", "provider": "p",
                "tool_calls": [], "error": None}

    monkeypatch.setattr(curator, "_run_llm_review", _fake_review)
    monkeypatch.setattr(curator, "_load_config", lambda: {})
    yield {"root": root, "profile": profile, "curator": curator, "usage": usage, "seen": seen}
    for t in threading.enumerate():
        if t.name == "curator-review" and t.is_alive():
            t.join(timeout=10.0)


def _write_managed_skill(home: Path, name: str):
    """A curator-manageable skill: on-disk dir + ``created_by: agent`` usage record."""
    d = home / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: x\n---\n", encoding="utf-8")
    usage_file = home / "skills" / ".usage.json"
    data = json.loads(usage_file.read_text(encoding="utf-8-sig")) if usage_file.exists() else {}
    data[name] = {"created_by": "agent"}
    usage_file.write_text(json.dumps(data), encoding="utf-8")
    return d


def _join_review_thread(timeout: float = 10.0):
    for t in threading.enumerate():
        if t.name == "curator-review" and t.is_alive():
            t.join(timeout=timeout)


def test_threaded_llm_pass_stays_in_the_callers_profile_scope(curator_env, monkeypatch):
    curator, root, profile, seen = (
        curator_env["curator"], curator_env["root"], curator_env["profile"], curator_env["seen"])
    _write_managed_skill(profile, "profile-skill")   # candidate ONLY under the profile home
    _write_managed_skill(root, "root-skill")          # must never appear in the profile's run

    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from contextvars import copy_context

    def _tick():
        # What profile_scoped_chore installs around the chore (home override as a contextvar;
        # the secret scope is the same mechanism and fails the same way when lost).
        token = set_hermes_home_override(str(profile))
        try:
            curator.run_curator_review(synchronous=False, consolidate=True)  # default: daemon thread
        finally:
            reset_hermes_home_override(token)

    # Run the tick the way the gateway does: scope installed in THIS context, review fired
    # from inside it. The unpatched thread boundary is exactly what drops the scope.
    tick = threading.Thread(target=copy_context().run, args=(_tick,), daemon=True)
    tick.start()
    tick.join(timeout=15.0)          # let the synchronous half finish and fire the review
    _join_review_thread()            # then the review daemon itself

    # 1. The fork saw the PROFILE's candidate list, not the root home's.
    assert "profile-skill" in seen.get("prompt", ""), "LLM pass reviewed the profile's skills"
    assert "root-skill" not in seen.get("prompt", ""), "root home's skills leaked into the pass"

    # 2. Report + final state landed under the PROFILE home, not the process home.
    assert list((profile / "logs" / "curator").glob("*/run.json")), "report under profile home"
    assert not (root / "logs" / "curator").exists(), "report must not land in the root home"

    profile_state = json.loads((profile / "skills" / ".curator_state").read_text(encoding="utf-8-sig"))
    assert profile_state["last_run_summary"], "profile state records the run summary"
    assert profile_state["last_report_path"] and str(profile) in profile_state["last_report_path"], (
        "last_report_path points INSIDE the profile home")
    assert not (root / "skills" / ".curator_state").exists(), "root state must not be touched"
