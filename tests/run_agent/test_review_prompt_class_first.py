"""Behavior contracts for the operative background skill-review prompts."""

from types import SimpleNamespace

from agent.background_review import spawn_background_review_thread


def _assembled_prompt(*, review_memory: bool) -> str:
    """Exercise the prompt-selection path used by the background review thread."""
    _target, prompt = spawn_background_review_thread(
        SimpleNamespace(), [], review_memory=review_memory, review_skills=True, task_cfg={},
    )
    return prompt


# ---------------------------------------------------------------------------
# _SKILL_REVIEW_PROMPT
# ---------------------------------------------------------------------------

def _assert_selective_skill_policy(prompt: str, label: str) -> None:
    lower = prompt.lower()
    assert "taking no skill action is explicitly valid" in lower, label
    assert "all five" in lower, label
    for requirement in (
        "narrow loading trigger",
        "reusable procedural value",
        "evidence the workflow recurs",
        "no better home",
        "safe selective loading",
    ):
        assert requirement in lower, f"{label}: missing creation requirement {requirement!r}"
    for better_home in (
        "config", "user.md", "memory.md", "project instructions", "code",
        "session/git/issue/pr", "existing skill",
    ):
        assert better_home in lower, f"{label}: missing better home {better_home!r}"
    assert "unrelated conversations must work correctly without loading" in lower, label
    assert "prefer extending an existing matching skill" in lower, label
    for rejected in (
        "global preferences", "standard agent behavior", "one-off task state", "raw logs",
        "generic advice",
    ):
        assert rejected in lower, f"{label}: must reject {rejected}"
    assert "standing user preferences" not in lower, (
        f"{label}: must not route global preferences into selectively loaded skills"
    )


def test_skill_review_prompt_requires_selective_creation():
    _assert_selective_skill_policy(_assembled_prompt(review_memory=False), "skill-only review")
















# ---------------------------------------------------------------------------
# _COMBINED_REVIEW_PROMPT
# ---------------------------------------------------------------------------

def test_combined_review_prompt_has_memory_section():
    """Memory half must still cover user facts and preferences."""
    prompt = _assembled_prompt(review_memory=True)
    assert "**Memory**" in prompt
    assert "memory tool" in prompt


def test_combined_review_prompt_requires_selective_creation():
    prompt = _assembled_prompt(review_memory=True)
    _assert_selective_skill_policy(prompt, "combined review")
    assert "memory action does not require skill action" in prompt.lower()














# ---------------------------------------------------------------------------
# Anti-pattern guidance — see issue #6051. The reviewer was learning transient
# environment failures (e.g. "browser tools do not work" from a fresh-install
# Playwright miss) as durable skill rules, then citing them against itself for
# weeks after the environment was fixed. Both review prompts must explicitly
# tell the reviewer not to capture environment-dependent or negative-framing
# content as skills.
# ---------------------------------------------------------------------------


def _assert_anti_pattern_guidance(prompt: str, label: str) -> None:
    """Both review prompts must carry the same anti-pattern section."""
    lower = prompt.lower()
    assert "do not capture" in lower, (
        f"{label}: must have an explicit 'Do NOT capture' section"
    )
    # Environment-dependent failures (the #6051 root cause)
    assert any(k in lower for k in ("missing binar", "command not found", "uninstalled", "fresh-install")), (
        f"{label}: must call out environment/setup failures as not-skill-worthy"
    )
    # Negative-framing avoidance
    assert any(k in lower for k in ("negative claim", "do not work", "is broken")), (
        f"{label}: must call out negative-claim phrasings as the failure mode"
    )
    # Positive reframing — "capture the fix, not the failure"
    assert "capture the fix" in lower or "capture the fix " in lower, (
        f"{label}: must redirect tool-failure capture toward the fix, not the constraint"
    )
    # One-off task narratives (#12812 family)
    assert "one-off" in lower, (
        f"{label}: must call out one-off task narratives as not-skill-worthy"
    )


def _assert_unresolved_failure_guidance(prompt: str, label: str) -> None:
    """Unresolved task attempts must not become persistent skill guidance."""
    lower = prompt.lower()
    assert "unresolved failures" in lower, f"{label}: must identify unresolved failures"
    assert "working method" in lower, f"{label}: must require a working method"
    assert "told the user to check manually" in lower, (
        f"{label}: must recognize an explicitly unresolved session"
    )
    assert "never the dead ends" in lower, f"{label}: must exclude failed attempts"
    assert "independently confident" in lower, (
        f"{label}: must limit exceptions to verified alternatives"
    )


def test_skill_review_prompt_rejects_unresolved_failures():
    _assert_unresolved_failure_guidance(_assembled_prompt(review_memory=False), "skill-only review")


def test_combined_review_prompt_rejects_unresolved_failures():
    _assert_unresolved_failure_guidance(_assembled_prompt(review_memory=True), "combined review")


def _assert_read_before_write_guidance(prompt: str, label: str) -> None:
    """Both review prompts must teach the enforced read-before-write handshake.

    The skill_manage guard refuses patch/edit of an existing SKILL.md (and
    overwrite/remove of an existing support file) unless the exact target was
    loaded via skill_view during the review. Without prompt guidance the model
    walks into the refusal and burns iterations retrying (#62397).
    """
    lower = prompt.lower()
    assert "read-before-write" in lower, f"{label}: must name the read-before-write rule"
    assert "skill_view(name)" in prompt, (
        f"{label}: must give the exact SKILL.md pre-read call"
    )
    assert "file_path=..." in prompt, (
        f"{label}: must give the support-file pre-read form"
    )
    # Scope: only EXISTING targets need a pre-read; new creations are exempt.
    assert "new" in lower and "no prior read" in lower, (
        f"{label}: must exempt new skills / new support files from the pre-read"
    )
    # Transcript quotes must not be treated as satisfying the guard.
    assert "does not count" in lower or "does NOT count" in prompt or "not satisfy" in lower, (
        f"{label}: must say transcript-quoted content doesn't satisfy the guard"
    )
    # Bounded recovery: one view + one retry, never a loop.
    assert "do not loop" in lower, (
        f"{label}: must bound refusal recovery to a single retry"
    )


def test_skill_review_prompt_teaches_read_before_write():
    _assert_read_before_write_guidance(_assembled_prompt(review_memory=False), "skill-only review")


def test_combined_review_prompt_teaches_read_before_write():
    _assert_read_before_write_guidance(_assembled_prompt(review_memory=True), "combined review")






# ---------------------------------------------------------------------------
# _MEMORY_REVIEW_PROMPT — unchanged, still memory-focused
# ---------------------------------------------------------------------------


def _assert_lesson_layer_guidance(prompt: str, label: str) -> None:
    """Skill writes must be lessons (rule + why), not incident logs or per-session reference files."""
    lower = prompt.lower()
    assert "specifications" in lower and "procedure" in lower, (
        f"{label}: must state the primary purpose — how to do the task, to the user's specifications")
    assert "why" in lower and "rule" in lower, f"{label}: must ask for rule + why"
    assert "pr/issue numbers" in lower or "pr numbers" in lower, f"{label}: must ban PR/issue numbers as content"
    assert "one rule" in lower, f"{label}: must collapse repeated lessons into one rule"
    assert "agents.md" in lower, f"{label}: must forbid duplicating always-loaded context"
    assert "per-session" in lower or "per-incident" in lower, f"{label}: must forbid per-session reference files"


def test_skill_review_prompt_teaches_lesson_layer():
    _assert_lesson_layer_guidance(_assembled_prompt(review_memory=False), "skill-only review")


def test_combined_review_prompt_teaches_lesson_layer():
    _assert_lesson_layer_guidance(_assembled_prompt(review_memory=True), "combined review")


def test_curator_prompt_consolidates_by_distilling():
    from agent.curator import CURATOR_REVIEW_PROMPT
    lower = CURATOR_REVIEW_PROMPT.lower()
    assert "distill" in lower, "curator must distill absorbed content, not file it"
    assert "verbatim" in lower and "per-incident" in lower, "curator must not copy siblings verbatim into references/"
