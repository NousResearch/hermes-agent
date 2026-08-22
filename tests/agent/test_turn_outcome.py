"""Tests for agent/turn_outcome.py — end-of-turn outcome evaluation (Layer 0).

Pins the Layer 0 decisions from patch.md to behavior:

  - signal-gated trigger: no signal (all used skills verified clean, no
    residue, no ``run: always``) ⇒ no aux call, no global verdict; a clean
    verifier PASS is still recorded as per-skill evidence
  - down-only override: a mechanical verifier FAIL wins over an eval that
    claims success
  - pass-is-not-success: a verifier PASS never confirms success; the eval's
    semantic failure is still recorded, and an eval-blamed skill gets the fail,
    not the pass (never double-recorded)
  - pass needs per-skill evidence: only a mechanical verifier PASS banks a
    success; an unverified skill on a confident eval success records a NEUTRAL
    (None) outcome — a sample, never a pass
  - weak pass: eval success at low confidence over unverified residue is not
    recorded (must not clear ``needs_review`` on its own)
  - dumb-recorder attribution: mechanical FAILs always land on their skill;
    empty ``failure_points`` writes nothing
  - reason corpus: verifier reason and eval reason both surface
  - best-effort: a broken aux call never breaks the turn
  - enumerated-evidence guard: the judge's ``failure_points`` is now
    ``[{"skill": ..., "evidence": [IDs]}]`` citing a numbered evidence catalog;
    a verifier-FAIL citation is hard (no confidence gate), tool-error/file-
    mutation or uncited legacy blame is confidence-gated, and fabricated/PASS/
    wrong-skill citations are rejected to NEUTRAL

The verifier path is the REAL one — real SKILL.md frontmatter, real subprocess
against a temp skill dir. Only the aux model call is injected (a seam); there
is no live network anywhere.
"""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def turn_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a clean skills/ dir per test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    import tools.skill_usage as mod

    importlib.reload(mod)
    monkeypatch.setattr(mod, "_prune_builtins_enabled", lambda: False)
    return home


def _verify_script(success: bool, reason: str) -> str:
    """Body of a verifier script that prints valid structured JSON on stdout."""
    payload = json.dumps({"success": success, "reason": reason})
    return "print(" + repr(payload) + ")\n"


def _write_skill_with_verify(skills_dir: Path, name: str, script_body: str) -> Path:
    d = skills_dir / name
    (d / "scripts").mkdir(parents=True, exist_ok=True)
    (d / "scripts" / "verify.py").write_text(script_body, encoding="utf-8")
    (d / "SKILL.md").write_text(
        f"""---
name: {name}
description: test skill
metadata:
  hermes:
    verify:
      run: scripts/verify.py
---

# body
""",
        encoding="utf-8",
    )
    return d


def _write_plain_skill(skills_dir: Path, name: str) -> Path:
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"""---
name: {name}
description: test skill
---

# body
""",
        encoding="utf-8",
    )
    return d


def _eval(**kwargs):
    """Shortcut for the injected aux-eval seam."""
    return lambda _prompt: kwargs


def _raise_if_called(called: list):
    """Aux-eval seam that fails loudly if invoked.

    Used where the test's property is "the signal gate returns before the
    aux seam is invoked": if the gate regresses and the seam runs, it
    raises instead of silently yielding a function object that
    ``evaluate_turn_outcome`` would treat as "no verdict".
    """

    def _seam(_prompt: str) -> dict:
        called.append(_prompt)
        raise AssertionError("aux eval invoked although the signal gate should have returned early")

    return _seam


def test_no_signal_skips_aux_but_records_verifier_pass(turn_env):
    """All used skills verified clean, no residue, run=auto ⇒ no aux call.

    The signal gate stays: without a failure or residue the judge does not run.
    But the mechanical PASS is still recorded — it is per-skill evidence, and
    skipping it would leave a flagged skill unable to recover on its own
    verifier's testimony (previously these passes were discarded entirely).
    """
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record, set_verify_enabled

    d = _write_skill_with_verify(
        turn_env / "skills", "golden", _verify_script(True, "ok")
    )
    set_verify_enabled("golden", True)

    called = []
    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"golden": d},
        outcome_config={"enabled": True, "run": "auto"},
        _aux_eval=_raise_if_called(called),
    )
    assert outcome is None
    assert called == []
    assert get_record("golden")["recent_outcomes"] == [True]


def test_down_only_verifier_fail_wins_over_llm_success(turn_env):
    """A mechanical FAIL is recorded even when the eval claims success."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record, set_verify_enabled

    d = _write_skill_with_verify(
        turn_env / "skills",
        "bad",
        _verify_script(False, "commit message 'fix stuff' has no type prefix"),
    )
    set_verify_enabled("bad", True)

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"bad": d},
        outcome_config={"enabled": True},
        _aux_eval=_eval(
            task_succeeded=True, confidence=0.95, failure_points=[], reason="looks fine"
        ),
    )
    assert outcome is not None
    assert outcome.task_succeeded is False
    assert outcome.confidence == 1.0
    assert outcome.failure_points == ["bad"]
    assert get_record("bad")["recent_outcomes"] == [False]
    # The mechanical FAIL's verifier reason must reach the sidecar so the
    # curator review pass has something actionable, not just a boolean.
    assert get_record("bad")["recent_outcome_reasons"] == [
        "commit message 'fix stuff' has no type prefix"
    ]


def test_down_only_blocks_eval_blaming_unverified_sibling(turn_env):
    """Down-only covers attribution too: a mechanical FAIL on skill A forecloses
    the turn, so the eval's ``failure_points`` must not pin blame on an
    unrelated, unverified skill B that also ran. Only A gets bump_outcome(False);
    B's record stays untouched even though the eval named it."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record, set_verify_enabled

    da = _write_skill_with_verify(
        turn_env / "skills", "mechfail", _verify_script(False, "verifier says no")
    )
    set_verify_enabled("mechfail", True)
    db = _write_plain_skill(turn_env / "skills", "unverified_sibling")

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"mechfail": da, "unverified_sibling": db},
        outcome_config={"enabled": True},
        _aux_eval=_eval(
            task_succeeded=False,
            confidence=0.9,
            failure_points=["mechfail", "unverified_sibling"],
            reason="wrong change committed",
        ),
    )
    assert outcome is not None
    assert outcome.task_succeeded is False
    assert outcome.failure_points == ["mechfail"]
    assert get_record("mechfail")["recent_outcomes"] == [False]
    assert get_record("unverified_sibling").get("recent_outcomes") == []


def test_low_confidence_eval_blame_never_lands_false(turn_env):
    """The misattribution guard: judge-only blame below the confidence floor
    must never land a hard False on a used skill.

    Two unverified skills ran and the work failed; the judge — working from a
    summary, not real diffs — names the wrong one at low confidence. The named
    skill is recorded NEUTRAL (a suspicion carrying the judge's reason), never
    a False; the unnamed skill records nothing. A coincidental name cannot
    corrupt either skill's outcome history.
    """
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record

    da = _write_plain_skill(turn_env / "skills", "real_cause")
    db = _write_plain_skill(turn_env / "skills", "coincidence")

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"real_cause": da, "coincidence": db},
        outcome_config={"enabled": True},
        _aux_eval=_eval(
            task_succeeded=False,
            confidence=0.4,
            failure_points=["coincidence"],  # wrong name, low confidence
            reason="the deployed change was wrong",
        ),
    )
    assert outcome is not None
    assert outcome.task_succeeded is False
    # Not confidently attributable — nobody is blamed.
    assert outcome.failure_points == []
    # The coincidentally-named skill records NEUTRAL, not False, and carries
    # the judge's reason for curator review.
    assert get_record("coincidence")["recent_outcomes"] == [None]
    assert get_record("coincidence")["recent_outcome_reasons"] == [
        "the deployed change was wrong"
    ]
    assert get_record("real_cause").get("recent_outcomes") == []
    assert get_record("coincidence").get("needs_review") is not True


def test_confident_eval_blame_still_lands_false(turn_env):
    """At or above the confidence floor, judge-only blame is a hard False."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record

    d = _write_plain_skill(turn_env / "skills", "sure")

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"sure": d},
        outcome_config={"enabled": True},
        _aux_eval=_eval(
            task_succeeded=False,
            confidence=0.9,
            failure_points=["sure"],
            reason="confident it broke",
        ),
    )
    assert outcome is not None
    assert outcome.task_succeeded is False
    assert outcome.failure_points == ["sure"]
    assert get_record("sure")["recent_outcomes"] == [False]


def test_pass_is_not_success_when_eval_flags_semantics(turn_env):
    """Verifier PASS never confirms success; the eval's semantic fail is recorded.

    ``run: always`` here because under ``run: auto`` a clean verifier-backed
    turn has no residue to trigger the eval — the semantic-fail-over-pass
    case is exactly when the eval must still run.
    """
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record, set_verify_enabled

    d = _write_skill_with_verify(
        turn_env / "skills", "rel", _verify_script(True, "ok")
    )
    set_verify_enabled("rel", True)

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"rel": d},
        outcome_config={"enabled": True, "run": "always"},
        _aux_eval=_eval(
            task_succeeded=False,
            confidence=0.8,
            failure_points=["rel"],
            reason="commit describes the wrong change",
        ),
    )
    assert outcome.task_succeeded is False
    assert outcome.failure_points == ["rel"]
    assert get_record("rel")["recent_outcomes"] == [False]


def test_weak_pass_low_confidence_not_recorded(turn_env):
    """Unverified residue + low-confidence eval success ⇒ nothing written."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record

    d = _write_plain_skill(turn_env / "skills", "open")

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"open": d},
        outcome_config={"enabled": True},
        _aux_eval=_eval(
            task_succeeded=True, confidence=0.4, failure_points=[], reason="probably fine"
        ),
    )
    assert outcome is not None
    assert outcome.task_succeeded is True
    assert outcome.confidence == 0.4
    assert get_record("open").get("recent_outcomes") == []


def test_empty_failure_points_no_sidecar_write(turn_env):
    """A turn-level failure with no attributable skill writes nothing."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record

    d = _write_plain_skill(turn_env / "skills", "mystery")

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"mystery": d},
        outcome_config={"enabled": True},
        _aux_eval=_eval(
            task_succeeded=False,
            confidence=0.7,
            failure_points=[],
            reason="turn failed but no skill to blame",
        ),
    )
    assert outcome is not None
    assert outcome.task_succeeded is False
    assert outcome.failure_points == []
    assert get_record("mystery").get("recent_outcomes") == []


def test_reason_corpus_merges_verifier_and_eval(turn_env):
    """Both the mechanical reason and the semantic reason surface together."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import set_verify_enabled

    d = _write_skill_with_verify(
        turn_env / "skills",
        "cc",
        _verify_script(False, "commit message 'fix stuff' has no type prefix"),
    )
    set_verify_enabled("cc", True)

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"cc": d},
        outcome_config={"enabled": True},
        _aux_eval=_eval(
            task_succeeded=False,
            confidence=0.9,
            failure_points=[],
            reason="message also describes the wrong change",
        ),
    )
    assert "verifier (cc)" in outcome.reason
    assert "no type prefix" in outcome.reason
    assert "wrong change" in outcome.reason


def test_aux_raise_falls_back_to_mechanical(turn_env):
    """A broken aux call never breaks the turn; mechanical verdict still lands."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record, set_verify_enabled

    d = _write_skill_with_verify(
        turn_env / "skills", "cc", _verify_script(False, "verifier said no")
    )
    set_verify_enabled("cc", True)

    def _boom(_prompt):
        raise RuntimeError("aux provider down")

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"cc": d},
        outcome_config={"enabled": True},
        _aux_eval=_boom,
    )
    assert outcome is not None
    assert outcome.task_succeeded is False
    assert outcome.failure_points == ["cc"]
    assert get_record("cc")["recent_outcomes"] == [False]


def test_file_mutation_failure_forces_fail(turn_env):
    """The existing per-turn file-mutation state forces a fail down-only."""
    from agent.turn_outcome import evaluate_turn_outcome

    d = _write_plain_skill(turn_env / "skills", "open")
    fm = {"src/foo.py": {"tool": "write_file", "error_preview": "permission denied"}}

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"open": d},
        outcome_config={"enabled": True},
        file_mutation_state=fm,
        _aux_eval=_eval(
            task_succeeded=True, confidence=0.95, failure_points=[], reason="all good"
        ),
    )
    assert outcome is not None
    assert outcome.task_succeeded is False
    assert "file-mutation" in outcome.reason


def test_disabled_config_is_inert(turn_env):
    """With the feature disabled the verifier never even runs."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record, set_verify_enabled

    d = _write_skill_with_verify(
        turn_env / "skills", "cc", _verify_script(False, "verifier said no")
    )
    set_verify_enabled("cc", True)

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"cc": d},
        outcome_config={"enabled": False},
        _aux_eval=_eval(task_succeeded=False, confidence=0.9, failure_points=["cc"], reason="x"),
    )
    assert outcome is None
    assert get_record("cc").get("recent_outcomes") == []


def test_high_confidence_eval_success_records_pass(turn_env):
    """A confirmed success (run=always) is recorded, so recovery is possible."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import bump_outcome, get_record, set_verify_enabled

    d = _write_skill_with_verify(
        turn_env / "skills", "golden", _verify_script(True, "ok")
    )
    set_verify_enabled("golden", True)
    for _ in range(3):
        bump_outcome("golden", False)

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"golden": d},
        outcome_config={"enabled": True, "run": "always"},
        _aux_eval=_eval(
            task_succeeded=True, confidence=0.9, failure_points=[], reason="held up"
        ),
    )
    assert outcome is not None
    assert outcome.task_succeeded is True
    assert get_record("golden")["recent_outcomes"][-1] is True


def test_unverified_skill_on_confident_eval_success_records_neutral(turn_env):
    """A skill that ran unverified on a confident eval success gets a NEUTRAL
    outcome, never a pass — the core of the pass-inflation fix. The eval's
    global success must not mint per-skill wins for skills the turn never
    mechanically checked."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record

    d = _write_plain_skill(turn_env / "skills", "open")

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"open": d},
        outcome_config={"enabled": True},
        _aux_eval=_eval(
            task_succeeded=True, confidence=0.9, failure_points=[], reason="held up"
        ),
    )
    assert outcome is not None
    assert outcome.task_succeeded is True
    # Stored raw — None, not a coerced False, and definitely not a pass.
    assert get_record("open")["recent_outcomes"] == [None]
    assert get_record("open")["needs_review"] is False


def test_stringified_false_verdict_is_a_failure_not_a_pass(turn_env):
    """The judge LLM frequently returns ``"task_succeeded": "false"`` as a
    string. ``bool("false")`` is True — the classic Python trap — so the
    coercion must handle string booleans explicitly, or a judged FAIL would
    be recorded as a PASS and the needs-review signal would never fire."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import bump_outcome, get_record

    d = _write_plain_skill(turn_env / "skills", "strflag")

    # Seed prior failures so the string "false" verdict crosses the
    # needs-review sample floor (_OUTCOME_MIN_SAMPLES = 4).
    for _ in range(3):
        bump_outcome("strflag", False)

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"strflag": d},
        outcome_config={"enabled": True},
        _aux_eval=_eval(
            task_succeeded="false", confidence=0.9, failure_points=["strflag"],
            reason="verifier: schema violated",
        ),
    )
    assert outcome is not None
    assert outcome.task_succeeded is False
    assert get_record("strflag")["recent_outcomes"][-1] is False
    assert get_record("strflag")["needs_review"] is True


def test_stringified_true_verdict_is_a_success(turn_env):
    """A string ``"true"`` from the judge must also coerce correctly — the
    coercion is symmetric, not just the failure arm."""
    from agent.turn_outcome import evaluate_turn_outcome

    d = _write_plain_skill(turn_env / "skills", "strtrue")

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"strtrue": d},
        outcome_config={"enabled": True},
        _aux_eval=_eval(
            task_succeeded="true", confidence=0.9, failure_points=[], reason="ok"
        ),
    )
    assert outcome is not None
    assert outcome.task_succeeded is True


def test_verifier_pass_survives_eval_success_alongside_unverified_neutral(turn_env):
    """On the same eval success: the verifier-backed skill gets the pass, the
    unverified sibling gets a neutral — per-skill evidence decides, not the
    judge's global verdict."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record, set_verify_enabled

    d_golden = _write_skill_with_verify(
        turn_env / "skills", "golden", _verify_script(True, "ok")
    )
    d_open = _write_plain_skill(turn_env / "skills", "open")
    set_verify_enabled("golden", True)

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"golden": d_golden, "open": d_open},
        outcome_config={"enabled": True, "run": "always"},
        _aux_eval=_eval(
            task_succeeded=True, confidence=0.9, failure_points=[], reason="fine"
        ),
    )
    assert outcome is not None
    assert outcome.task_succeeded is True
    assert get_record("golden")["recent_outcomes"] == [True]
    assert get_record("open")["recent_outcomes"] == [None]


def test_mechanical_pass_recovers_a_flagged_skill_without_eval(turn_env):
    """A needs-review skill whose verifier keeps passing must recover on its own
    evidence — clean turns (no eval) still bank the passes, so the flag clears
    instead of being stuck until a judge happens to fire."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import bump_outcome, get_record, set_verify_enabled

    d = _write_skill_with_verify(
        turn_env / "skills", "recover", _verify_script(True, "ok")
    )
    set_verify_enabled("recover", True)
    for _ in range(4):
        bump_outcome("recover", False)
    assert get_record("recover")["needs_review"] is True

    # Five clean verifier-pass turns — each recorded even though no eval runs.
    # 4 fails + 5 passes = 4/9 ≈ 0.44 < 0.5 threshold → flag clears.
    # The aux seam must never be invoked: a gate regression raises loudly
    # instead of silently yielding a callable treated as "no verdict".
    called = []
    for _ in range(5):
        evaluate_turn_outcome(
            skills_used_this_turn={"recover": d},
            outcome_config={"enabled": True, "run": "auto"},
            _aux_eval=_raise_if_called(called),
        )
    assert called == []
    rec = get_record("recover")
    assert rec["recent_outcomes"][-5:] == [True, True, True, True, True]
    assert rec["needs_review"] is False
    assert rec["needs_review_since"] is None


def test_interrupted_turn_is_not_a_work_failure(turn_env):
    """User-stopped turns produce no outcome and no writes."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record

    d = _write_plain_skill(turn_env / "skills", "open")

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"open": d},
        outcome_config={"enabled": True},
        interrupted=True,
        _aux_eval=_eval(task_succeeded=False, confidence=0.9, failure_points=["open"], reason="x"),
    )
    assert outcome is None
    assert get_record("open").get("recent_outcomes") == []


def test_infra_failure_reports_outcome_without_blaming_a_skill(turn_env):
    """An infra-failed turn yields an outcome but no sidecar attribution."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record

    d = _write_plain_skill(turn_env / "skills", "open")

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"open": d},
        outcome_config={"enabled": True},
        failed=True,
        exit_reason="session_persistence_failed",
    )
    assert outcome is not None
    assert outcome.task_succeeded is False
    assert outcome.failure_points == []
    assert "session_persistence_failed" in outcome.reason
    assert get_record("open").get("recent_outcomes") == []


def test_verifier_runs_in_agent_cwd_not_process_cwd(turn_env):
    """Verifiers run against the agent's working directory — the same resolver
    the system prompt advertises — not the backend process's cwd. A gateway
    session pinned to its worktree must verify that tree, or a passing check
    certifies the wrong directory."""
    from agent.runtime_cwd import clear_session_cwd, set_session_cwd
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import set_verify_enabled

    session_cwd = turn_env.parent / "session-cwd"
    session_cwd.mkdir()
    d = _write_skill_with_verify(
        turn_env / "skills", "cwdcheck", _verify_script(True, "ok")
    )
    set_verify_enabled("cwdcheck", True)
    # Overwrite the script to drop a sentinel into whatever cwd it runs in.
    (d / "scripts" / "verify.py").write_text(
        "from pathlib import Path\n"
        "Path('ran-here').write_text('ran')\n"
        + _verify_script(True, "ok"),
        encoding="utf-8",
    )
    set_session_cwd(str(session_cwd))
    try:
        evaluate_turn_outcome(
            skills_used_this_turn={"cwdcheck": d},
            outcome_config={"enabled": True},
            _aux_eval=_eval(task_succeeded=True, confidence=0.9, failure_points=[], reason="ok"),
        )
    finally:
        clear_session_cwd()
    assert (session_cwd / "ran-here").exists()
    assert not (Path.cwd() / "ran-here").exists()


def test_verify_budget_exhausted_skips_remaining_verifiers(turn_env):
    """An aggregate verify budget caps the mechanical layer: once elapsed time
    passes ``total_verify_budget_seconds``, remaining skills record ``skip``
    instead of launching more subprocesses."""
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record, set_verify_enabled

    skills = turn_env / "skills"
    slow = _write_skill_with_verify(skills, "slow", _verify_script(True, "ok"))
    # The fast skill's verifier drops a sentinel if it ever runs.
    sentinel = turn_env / "fast-ran"
    fast = _write_skill_with_verify(
        skills,
        "fast",
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('ran')\n"
        + _verify_script(True, "ok"),
    )
    set_verify_enabled("slow", True)
    set_verify_enabled("fast", True)
    # Give the first verifier real work to burn the budget; the second must
    # never launch.
    (slow / "scripts" / "verify.py").write_text(
        "import time\ntime.sleep(0.3)\n" + _verify_script(True, "ok"),
        encoding="utf-8",
    )

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"slow": slow, "fast": fast},
        outcome_config={
            "enabled": True,
            "run": "always",
            "total_verify_budget_seconds": 0.05,
        },
        _aux_eval=_eval(task_succeeded=True, confidence=0.9, failure_points=[], reason="ok"),
    )
    assert outcome is not None
    assert not sentinel.exists(), "budget-exhausted skill's verifier must not run"
    # The fast skill was skipped at the mechanical layer — its verifier never
    # ran, so it can't bank a pass; on the confident eval success it records
    # a NEUTRAL sample (unverified residue), never a pass.
    assert get_record("fast")["recent_outcomes"] == [None]


def test_eval_attribution_respects_curation_eligibility(turn_env):
    """Eval-attributed failure points must not flip needs_review on skills the
    curator can't manage.

    The mechanical path already refuses non-eligible skills (a bundled/hub/
    external skill's verifier never runs, so it can never FAIL). But the judge
    may name ANY skill — including a hub-installed one that also ran this turn.
    Recording that verdict would flip ``needs_review`` on a skill the curator
    never surfaces, leaving a permanent orphan reason in the sidecar. The
    attribution recorder must apply the same eligibility gate.
    """
    import json as _json

    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record, load_usage, _read_hub_installed_names

    # The skill used this turn lives under skills/ but is hub-installed:
    # a local record exists (findable), yet it is NOT curator-managed.
    d = _write_plain_skill(turn_env / "skills", "hubskill")
    hub_dir = turn_env / "skills" / ".hub"
    hub_dir.mkdir()
    (hub_dir / "lock.json").write_text(
        _json.dumps({"installed": {"hubskill": {"install_path": "hubskill"}}}),
        encoding="utf-8",
    )
    assert "hubskill" in _read_hub_installed_names()

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"hubskill": d},
        outcome_config={"enabled": True, "run": "always"},
        _aux_eval=_eval(
            task_succeeded=False, confidence=0.9, failure_points=["hubskill"], reason="x"
        ),
    )
    assert outcome is not None
    # The verdict itself still names the skill — but nothing reaches the sidecar.
    assert outcome.failure_points == ["hubskill"]
    assert get_record("hubskill").get("recent_outcomes") == []
    assert "hubskill" not in load_usage()


def test_eval_cannot_blame_a_skill_not_used_this_turn(turn_env):
    """The judge must not pin blame on a skill the turn never touched.

    The judge only sees a summarized prompt and can hallucinate a skill name
    ("summarization", ...) that never ran. Attribution is intersected with the
    actually-used skills, so a phantom name never reaches the sidecar, never
    flips needs_review, and never even surfaces in failure_points. The outcome
    still records the turn failed — with no skill to blame, like an infra
    failure.
    """
    from agent.turn_outcome import evaluate_turn_outcome
    from tools.skill_usage import get_record, load_usage

    d = _write_plain_skill(turn_env / "skills", "real")

    outcome = evaluate_turn_outcome(
        skills_used_this_turn={"real": d},
        outcome_config={"enabled": True, "run": "always"},
        _aux_eval=_eval(
            task_succeeded=False,
            confidence=0.9,
            failure_points=["summarization"],  # hallucinated, never used
            reason="claimed summary but provided no content",
        ),
    )
    assert outcome is not None
    assert outcome.task_succeeded is False
    # The phantom name is not attributed.
    assert outcome.failure_points == []
    assert "summarization" not in load_usage()
    # The used skill is not blamed either — the judge gave no reason to.
    assert get_record("real").get("recent_outcomes") == []


class TestEnumeratedEvidenceGuard:
    """Recorder-side rejection: blame must cite real, matching evidence.

    The judge's ``failure_points`` is now ``[{"skill": ..., "evidence":
    [IDs]}]`` pointing at a numbered catalog of this turn's evidence (verifier
    verdicts, tool errors, file mutations). The recorder refuses to write a
    hard False a citation can't back:

      - verifier-FAIL citation of the exact skill → hard, lands regardless of
        confidence (the citation IS the evidence)
      - tool-error / file-mutation citation, or NO citation (legacy bare name)
        → gated: a hard False only when the judge's confidence clears the
        floor
      - fabricated ID, cited PASS, or another skill's FAIL → soft NEUTRAL,
        never a hard False

    These exercise the real verifier path (real SKILL.md, real subprocess)
    with the aux seam, exactly like the rest of the suite.
    """

    def test_cited_verifier_fail_lands_regardless_of_confidence(self, turn_env):
        """A verifier FAIL is foreclosed: the skill lands a hard False even when
        the judge's confidence is below the floor. The mechanical item — not the
        judge's self-assessment — is what's being recorded; the eval's citation
        to it is consistent with the down-only outcome, not the source of it."""
        from agent.turn_outcome import evaluate_turn_outcome
        from tools.skill_usage import get_record, set_verify_enabled

        d = _write_skill_with_verify(
            turn_env / "skills", "vfail", _verify_script(False, "verifier says no")
        )
        set_verify_enabled("vfail", True)

        outcome = evaluate_turn_outcome(
            skills_used_this_turn={"vfail": d},
            outcome_config={"enabled": True, "run": "always"},
            _aux_eval=_eval(
                task_succeeded=False,
                confidence=0.1,  # below _BLAME_CONFIDENCE_THRESHOLD
                failure_points=[{"skill": "vfail", "evidence": [1]}],  # [1] is its FAIL
                reason="the verifier is right",
            ),
        )
        assert outcome is not None
        assert outcome.failure_points == ["vfail"]
        assert get_record("vfail")["recent_outcomes"] == [False]

    def test_fabricated_evidence_id_rejected(self, turn_env):
        """A citation that doesn't exist in the catalog is a suspicion, not a
        fact: the judge can't invent evidence to justify a blame."""
        from agent.turn_outcome import evaluate_turn_outcome
        from tools.skill_usage import get_record

        d = _write_plain_skill(turn_env / "skills", "suspect")

        outcome = evaluate_turn_outcome(
            skills_used_this_turn={"suspect": d},
            outcome_config={"enabled": True},
            _aux_eval=_eval(
                task_succeeded=False,
                confidence=0.9,  # confident, but the citation is fabricated
                failure_points=[{"skill": "suspect", "evidence": [99]}],
                reason="judge invented a reason",
            ),
        )
        assert outcome is not None
        assert outcome.task_succeeded is False
        # No hard blame — the fabricated citation is rejected.
        assert outcome.failure_points == []
        # Recorded NEUTRAL with the reason preserved for curator review.
        assert get_record("suspect")["recent_outcomes"] == [None]
        assert get_record("suspect")["recent_outcome_reasons"] == [
            "judge invented a reason"
        ]

    def test_cited_verifier_pass_is_not_failure_evidence(self, turn_env):
        """Citing a mechanical PASS as evidence of failure must be rejected —
        a verifier that PASSED is the opposite of evidence the skill broke."""
        from agent.turn_outcome import evaluate_turn_outcome
        from tools.skill_usage import get_record, set_verify_enabled

        d = _write_skill_with_verify(
            turn_env / "skills", "passer", _verify_script(True, "ok")
        )
        set_verify_enabled("passer", True)

        outcome = evaluate_turn_outcome(
            skills_used_this_turn={"passer": d},
            outcome_config={"enabled": True, "run": "always"},
            _aux_eval=_eval(
                task_succeeded=False,
                confidence=0.9,
                failure_points=[{"skill": "passer", "evidence": [1]}],  # [1] is its PASS
                reason="judge cites a pass as failure evidence",
            ),
        )
        assert outcome is not None
        assert outcome.task_succeeded is False
        # The PASS-cited blame is rejected — soft NEUTRAL, and the skill keeps
        # its mechanical PASS (per-skill evidence is not suppressed by a
        # rejected citation).
        assert outcome.failure_points == []
        assert get_record("passer")["recent_outcomes"] == [True]

    def test_mechanical_fail_forecloses_eval_blame_on_siblings(self, turn_env):
        """Down-only covers the whole turn: with a mechanical FAIL on the table,
        the judge's blame on an unrelated sibling is dropped entirely — not even
        soft. A cited verifier-FAIL of the mechanically-failing skill is the
        only thing that lands. (The "cite another skill's FAIL" rejection logic
        itself is pinned at the unit level in test_validate_eval_blame_tiers —
        through the real flow, any verifier FAIL triggers down-only and clears
        all eval attribution.)"""
        from agent.turn_outcome import evaluate_turn_outcome
        from tools.skill_usage import get_record, set_verify_enabled

        da = _write_plain_skill(turn_env / "skills", "innocent")
        db = _write_skill_with_verify(
            turn_env / "skills", "guilty", _verify_script(False, "b broke")
        )
        set_verify_enabled("guilty", True)

        outcome = evaluate_turn_outcome(
            skills_used_this_turn={"innocent": da, "guilty": db},
            outcome_config={"enabled": True, "run": "always"},
            _aux_eval=_eval(
                task_succeeded=False,
                confidence=0.9,
                failure_points=[
                    # innocent cites guilty's FAIL (id 2) as its own evidence.
                    {"skill": "innocent", "evidence": [2]},
                    {"skill": "guilty", "evidence": [2]},
                ],
                reason="one of them broke it",
            ),
        )
        assert outcome is not None
        # Down-only: the mechanical FAIL explains the turn; the judge's extra
        # attribution (even soft) is dropped. Only guilty lands.
        assert outcome.failure_points == ["guilty"]
        assert get_record("innocent").get("recent_outcomes") == []
        assert get_record("guilty")["recent_outcomes"] == [False]

    def test_tool_error_citation_gated_by_confidence(self, turn_env):
        """A tool-error citation is existence-checked but skill-attributed by
        the judge — it lands as a hard False only above the confidence floor."""
        from agent.turn_outcome import evaluate_turn_outcome
        from tools.skill_usage import get_record

        d = _write_plain_skill(turn_env / "skills", "open")

        outcome = evaluate_turn_outcome(
            skills_used_this_turn={"open": d},
            outcome_config={"enabled": True},
            tool_error_evidence=[{"tool": "terminal", "error": "command not found"}],
            _aux_eval=_eval(
                task_succeeded=False,
                confidence=0.9,
                failure_points=[{"skill": "open", "evidence": [1]}],  # [1] is the error
                reason="the command was never there",
            ),
        )
        assert outcome is not None
        assert outcome.failure_points == ["open"]
        assert get_record("open")["recent_outcomes"] == [False]

    def test_tool_error_citation_below_floor_is_neutral(self, turn_env):
        """Same tool-error citation, but low confidence: gated → NEUTRAL."""
        from agent.turn_outcome import evaluate_turn_outcome
        from tools.skill_usage import get_record

        d = _write_plain_skill(turn_env / "skills", "open")

        outcome = evaluate_turn_outcome(
            skills_used_this_turn={"open": d},
            outcome_config={"enabled": True},
            tool_error_evidence=[{"tool": "terminal", "error": "command not found"}],
            _aux_eval=_eval(
                task_succeeded=False,
                confidence=0.2,
                failure_points=[{"skill": "open", "evidence": [1]}],
                reason="maybe the command was missing",
            ),
        )
        assert outcome is not None
        assert outcome.failure_points == []
        assert get_record("open")["recent_outcomes"] == [None]

    def test_legacy_bare_name_still_gated(self, turn_env):
        """Back-compat: a bare-name failure_points (no evidence) is gated by
        the confidence floor exactly as before — confident → hard, low →
        NEUTRAL. The new guard must not silently weaken legacy callers."""
        from agent.turn_outcome import evaluate_turn_outcome
        from tools.skill_usage import get_record

        d = _write_plain_skill(turn_env / "skills", "legacy")

        outcome = evaluate_turn_outcome(
            skills_used_this_turn={"legacy": d},
            outcome_config={"enabled": True},
            _aux_eval=_eval(
                task_succeeded=False,
                confidence=0.9,
                failure_points=["legacy"],
                reason="confident it broke",
            ),
        )
        assert outcome is not None
        assert outcome.failure_points == ["legacy"]
        assert get_record("legacy")["recent_outcomes"] == [False]

    def test_prompt_renders_catalog_and_citation_instruction(self, turn_env):
        """The judge's prompt must expose the numbered catalog and demand
        cite-by-ID failure points — that's what makes the recorder-side
        rejection meaningful (it can only reject citations it asked for)."""
        from agent.turn_outcome import _build_prompt, _render_evidence_catalog

        catalog = [
            {"eid": 1, "kind": "verifier", "skill": "golden", "subject": "golden",
             "verdict": True, "text": "PASS — ok"},
            {"eid": 2, "kind": "tool_error", "skill": "", "subject": "terminal",
             "verdict": None, "text": "command not found"},
            {"eid": 3, "kind": "file_mutation", "skill": "", "subject": "marker.txt",
             "verdict": None, "text": "permission denied"},
        ]
        prompt = _build_prompt(
            "do the thing",
            "done",
            "  - golden: pass (ok)",
            "",
            1,
            evidence_catalog=_render_evidence_catalog(catalog),
        )
        assert "[1] verifier(golden) PASS — ok" in prompt
        assert "[2] tool_error(terminal) command not found" in prompt
        assert "[3] file_mutation(marker.txt) permission denied" in prompt
        assert '"evidence": [<IDs>]' in prompt
        assert "cite at least one ID from the evidence catalog" in prompt

    def test_validate_eval_blame_tiers(self):
        """Unit-level pin of the tier split (used_names filtering included)."""
        from agent.turn_outcome import _validate_eval_blame

        catalog = [
            {"eid": 1, "kind": "verifier", "skill": "v", "verdict": False,
             "text": "FAIL"},
            {"eid": 2, "kind": "verifier", "skill": "p", "verdict": True,
             "text": "PASS"},
            {"eid": 3, "kind": "tool_error", "skill": "", "verdict": None,
             "text": "(terminal): boom"},
        ]
        used = {"v", "p", "u"}
        hard, gated, soft = _validate_eval_blame(
            [
                {"skill": "v", "evidence": [1]},    # verifier FAIL of v → hard
                {"skill": "p", "evidence": [1]},    # v's FAIL cited for p → soft
                {"skill": "p", "evidence": [2]},    # p's own PASS cited → soft
                {"skill": "u", "evidence": [3]},    # tool error → gated
                {"skill": "u", "evidence": []},     # uncited → gated
                {"skill": "p", "evidence": [99]},   # fabricated ID → soft
                {"skill": "ghost", "evidence": [1]},  # not used this turn → dropped
            ],
            catalog,
            used,
        )
        assert hard == ["v"]
        assert gated == ["u"]
        assert soft == ["p"]


class TestParseJudgeJson:
    """The judge's verdict parser must survive real-world model output shapes.

    The aux LLM's raw response is what lands on ``json.loads`` in
    ``_default_aux_eval``; models wrap answers in ```json fences, lead with a
    sentence of prose, or truncate mid-JSON at ``max_tokens``. A bare strict
    parse returns None for all three and silently records nothing — the eval's
    verdict is the whole feature. These pin the tolerant path.
    """

    def test_exact_json_object(self):
        from agent.turn_outcome import _parse_judge_json

        verdict = _parse_judge_json(
            '{"task_succeeded": false, "confidence": 0.8, '
            '"failure_points": ["web_extract"], "reason": "timeout"}'
        )
        assert verdict is not None
        assert verdict["task_succeeded"] is False
        assert verdict["confidence"] == 0.8
        assert verdict["failure_points"] == ["web_extract"]

    def test_fenced_json_block(self):
        from agent.turn_outcome import _parse_judge_json

        verdict = _parse_judge_json(
            '```json\n{"task_succeeded": true, "confidence": 0.9, '
            '"failure_points": [], "reason": "looked good"}\n```'
        )
        assert verdict is not None
        assert verdict["task_succeeded"] is True
        assert verdict["confidence"] == 0.9

    def test_prose_before_json(self):
        from agent.turn_outcome import _parse_judge_json

        verdict = _parse_judge_json(
            'The turn did not meet the goal. '
            '{"task_succeeded": false, "confidence": 0.6, "failure_points": ["x"], "reason": "n"}'
        )
        assert verdict is not None
        assert verdict["task_succeeded"] is False
        assert verdict["failure_points"] == ["x"]

    def test_json_then_trailing_prose(self):
        from agent.turn_outcome import _parse_judge_json

        verdict = _parse_judge_json(
            '{"task_succeeded": true, "confidence": 0.7, "failure_points": [], "reason": "ok"} '
            "Everything completed as expected."
        )
        assert verdict is not None
        assert verdict["task_succeeded"] is True

    def test_trailing_prose_with_braces(self):
        from agent.turn_outcome import _parse_judge_json

        # Trailing prose that re-opens a brace must not swallow the verdict —
        # the object ends at its own balanced closing brace, not the last one.
        verdict = _parse_judge_json(
            '{"task_succeeded": false, "confidence": 0.9, "failure_points": ["x"], '
            '"reason": "wrong"} Then I checked the output. {Re-checked, same result.}'
        )
        assert verdict is not None
        assert verdict["task_succeeded"] is False
        assert verdict["failure_points"] == ["x"]

    def test_second_json_fragment_ignored(self):
        from agent.turn_outcome import _parse_judge_json

        # A second object later in the response must not shadow the first.
        verdict = _parse_judge_json(
            '{"task_succeeded": true, "confidence": 0.8, "failure_points": [], "reason": "ok"} '
            '{"task_succeeded": false, "confidence": 0.2}'
        )
        assert verdict is not None
        assert verdict["task_succeeded"] is True

    def test_braces_inside_string_value(self):
        from agent.turn_outcome import _parse_judge_json

        # Braces inside a quoted string are not object delimiters.
        verdict = _parse_judge_json(
            '{"task_succeeded": true, "confidence": 0.6, "failure_points": [], '
            '"reason": "formatted {like this}"}'
        )
        assert verdict is not None
        assert verdict["task_succeeded"] is True

    def test_truncated_object_returns_none(self):
        from agent.turn_outcome import _parse_judge_json

        # Unbalanced braces at max_tokens cut-off — must not half-parse.
        assert _parse_judge_json('{"task_succeeded": false, "confid') is None

    def test_non_json_prose_returns_none(self):
        from agent.turn_outcome import _parse_judge_json

        assert _parse_judge_json("The task failed because of a network issue.") is None

    def test_empty_and_none_inputs_return_none(self):
        from agent.turn_outcome import _parse_judge_json

        assert _parse_judge_json("") is None
        assert _parse_judge_json("   ") is None
        assert _parse_judge_json(None) is None

    def test_non_dict_json_returns_none(self):
        from agent.turn_outcome import _parse_judge_json

        assert _parse_judge_json("[1, 2, 3]") is None
        assert _parse_judge_json('"just a string"') is None


class TestDefaultAuxEvalRealResolution:
    """``_default_aux_eval`` must survive the real resolution chain + transport.

    Unlike the injected ``_aux_eval`` seam (used by every decision-logic
    test), these exercise the production default: the eval routes through the
    real ``call_llm`` auxiliary wrapper (which resolves task provider/model,
    honors ``auxiliary.outcome.timeout``/``extra_body``/``reasoning_effort``,
    and translates ``max_tokens`` where the provider requires it), then parses
    the raw-output JSON. Covers the two seams the rest of the suite stubs:
    client construction and the raw-output parse.
    """

    def _fake_response(self, content):
        class _FakeMessage:
            pass

        class _FakeChoices:
            pass

        class _FakeResp:
            pass

        msg = _FakeMessage()
        msg.content = content
        ch = _FakeChoices()
        ch.message = msg
        resp = _FakeResp()
        resp.choices = [ch]
        return resp

    def test_fenced_verdict_through_real_call_llm(self, monkeypatch):
        from agent.turn_outcome import _default_aux_eval

        captured = {}

        def _fake_call_llm(task=None, **kw):
            captured["task"] = task
            captured["max_tokens"] = kw.get("max_tokens")
            captured["messages"] = kw.get("messages")
            return self._fake_response(
                '```json\n{"task_succeeded": false, "confidence": 0.8, '
                '"failure_points": ["web_extract"], "reason": "curl timed out"}\n```'
            )

        monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm)
        # Outcome enabled so the default config reader resolves max_tokens.
        monkeypatch.setattr(
            "agent.turn_outcome._default_outcome_config",
            lambda: {"enabled": True, "max_tokens": 1000},
        )

        verdict = _default_aux_eval("judge this turn")
        assert verdict is not None
        assert verdict["task_succeeded"] is False
        assert verdict["failure_points"] == ["web_extract"]
        assert "curl timed out" in verdict["reason"]
        # Must route the outcome task through call_llm and apply the
        # config-driven budget — the fixed 200-token hardcode silently
        # no-ops on reasoning models.
        assert captured["task"] == "outcome"
        assert captured["max_tokens"] == 1000
        assert captured["messages"][0]["role"] == "user"

    def test_custom_max_tokens_from_config(self, monkeypatch):
        from agent.turn_outcome import _default_aux_eval

        captured = {}

        def _fake_call_llm(task=None, **kw):
            captured["max_tokens"] = kw.get("max_tokens")
            return self._fake_response(
                '{"task_succeeded": true, "confidence": 0.9, '
                '"failure_points": [], "reason": "ok"}'
            )

        monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm)
        monkeypatch.setattr(
            "agent.turn_outcome._default_outcome_config",
            lambda: {"enabled": True, "max_tokens": 2048},
        )

        verdict = _default_aux_eval("judge this turn")
        assert verdict is not None
        assert verdict["task_succeeded"] is True
        assert captured["max_tokens"] == 2048

    def test_empty_content_from_reasoning_model_returns_none(self, monkeypatch):
        """A reasoning model that burns its budget on thinking returns empty
        content — must not half-parse, must not raise."""
        from agent.turn_outcome import _default_aux_eval

        def _fake_call_llm(task=None, **kw):
            return self._fake_response("")

        monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm)
        monkeypatch.setattr(
            "agent.turn_outcome._default_outcome_config",
            lambda: {"enabled": True, "max_tokens": 200},
        )

        assert _default_aux_eval("judge this turn") is None

    def test_empty_choices_returns_none(self, monkeypatch):
        """A provider response with an empty/missing ``choices`` list must not
        IndexError into the response shape — it records no verdict."""
        from agent.turn_outcome import _default_aux_eval

        class _FakeResp:
            choices = []

        def _fake_call_llm(task=None, **kw):
            return _FakeResp()

        monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm)
        monkeypatch.setattr(
            "agent.turn_outcome._default_outcome_config",
            lambda: {"enabled": True, "max_tokens": 1000},
        )

        assert _default_aux_eval("judge this turn") is None

    def test_missing_choices_attribute_returns_none(self, monkeypatch):
        """A response that carries no ``choices`` at all must also be handled."""
        from agent.turn_outcome import _default_aux_eval

        class _FakeResp:
            pass

        def _fake_call_llm(task=None, **kw):
            return _FakeResp()

        monkeypatch.setattr("agent.auxiliary_client.call_llm", _fake_call_llm)
        monkeypatch.setattr(
            "agent.turn_outcome._default_outcome_config",
            lambda: {"enabled": True, "max_tokens": 1000},
        )

        assert _default_aux_eval("judge this turn") is None
