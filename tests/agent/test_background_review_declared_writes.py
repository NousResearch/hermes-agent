"""The review fork writes only the surfaces the deployment declares.

``auxiliary.background_review.writes`` (``memory`` | ``skills`` | ``memory+skills``) narrows
the dispatch-side whitelist the review fork runs under. It can only remove: an undeclared
profile keeps the historical memory+skills arm byte-for-byte, and a declaration can never
widen it back — neither a typo nor an ``extra_tools`` entry may re-admit ``skill_manage`` to
a review whose deployment declared memory writes only.

Why it exists: a deployment whose declared surface is ``memory`` had its review fork armed
with ``skill_manage`` on every turn; the platform's own skill-tree guard then refused each
attempt, so the declared surface held as a refusal storm rather than as the arming.
"""

from types import SimpleNamespace

import agent.background_review as bg


def _review_agent(memory_enabled=True, user_profile_enabled=False) -> SimpleNamespace:
    """The whitelist reads only the profile's memory flags off the fork."""
    return SimpleNamespace(_memory_enabled=memory_enabled, _user_profile_enabled=user_profile_enabled)


def _whitelist(task_cfg, *, review_memory=True):
    whitelist, configured_extra = bg._review_tool_whitelist(_review_agent(), task_cfg, review_memory)
    return whitelist, configured_extra


def test_a_memory_only_declaration_arms_no_skill_writer():
    whitelist, _extra = _whitelist({"writes": "memory"})
    assert "skill_manage" not in whitelist, "a memory-only deployment must not arm skill writes"
    assert "memory" in whitelist
    # Reading skills stays: the review may read what it is not allowed to write.
    assert "skill_view" in whitelist
    assert "skills_list" in whitelist


def test_memory_plus_skills_keeps_both_writers():
    whitelist, _extra = _whitelist({"writes": "memory+skills"})
    assert "skill_manage" in whitelist
    assert "memory" in whitelist


def test_an_undeclared_surface_keeps_todays_arm():
    whitelist, _extra = _whitelist({})
    assert "skill_manage" in whitelist
    assert "memory" in whitelist


def test_a_typo_narrows_instead_of_widening():
    # ``memry`` is not a surface: the declaration is unreadable, so the review is held to the
    # narrower store rather than silently keeping the skill writer the operator meant to drop.
    whitelist, _extra = _whitelist({"writes": "memry"})
    assert "skill_manage" not in whitelist
    assert "memory" in whitelist


def test_extra_tools_cannot_readmit_a_narrowed_writer():
    whitelist, extra = _whitelist({"writes": "memory", "extra_tools": ["skill_manage"]})
    assert "skill_manage" not in whitelist, "extra_tools must not widen a declared surface"
    assert "skill_manage" not in extra, "the deny message must not advertise a denied writer"


def test_a_skills_only_declaration_drops_the_memory_writer():
    whitelist, _extra = _whitelist({"writes": "skills"})
    assert "memory" not in whitelist
    assert "skill_manage" in whitelist


def test_a_narrowed_review_is_not_told_skill_management_is_available():
    phrases = getattr(bg, "_review_surface_phrases", None)
    assert phrases is not None, "the review's deny/prompt wording is not derived from its whitelist"
    narrowed = phrases(_whitelist({"writes": "memory"})[0])
    deny_memory, deny_skill, prompt_subject = narrowed
    assert "skill_manage" not in deny_skill
    assert "memory for notes" in deny_memory
    assert prompt_subject == "memory "
    both = phrases(_whitelist({"writes": "memory+skills"})[0])
    assert "skill_manage" in both[1]
    assert both[2] == "memory and skill "
