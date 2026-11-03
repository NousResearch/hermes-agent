"""Every background-review prompt carries the system-note disclaimer.

The review fork sends the chosen prompt as its ``user_message``, so the
instruction text itself reads like a user turn; without the banner the
reviewer stores the prompt's own guidance ("identify user preferences",
"be ACTIVE") as user preferences. These tests pin the shared disclaimer on
all three default prompts and the selector routing in
``spawn_background_review_thread``.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import agent.background_review as bg


class TestReviewPromptDisclaimer:
    def test_all_default_prompts_open_with_disclaimer(self):
        for name in ("_MEMORY_REVIEW_PROMPT", "_SKILL_REVIEW_PROMPT", "_COMBINED_REVIEW_PROMPT"):
            prompt = getattr(bg, name)
            assert prompt.startswith(bg._REVIEW_DISCLAIMER), name

    def test_memory_only_review_selects_disclaimed_memory_prompt(self):
        _target, prompt = bg.spawn_background_review_thread(
            SimpleNamespace(), [], review_memory=True, review_skills=False)
        assert prompt == bg._MEMORY_REVIEW_PROMPT
        assert prompt.startswith(bg._REVIEW_DISCLAIMER)

    def test_skills_only_review_selects_disclaimed_skill_prompt(self):
        _target, prompt = bg.spawn_background_review_thread(
            SimpleNamespace(), [], review_memory=False, review_skills=True)
        assert prompt == bg._SKILL_REVIEW_PROMPT
        assert prompt.startswith(bg._REVIEW_DISCLAIMER)

    def test_combined_review_selects_disclaimed_combined_prompt(self):
        _target, prompt = bg.spawn_background_review_thread(
            SimpleNamespace(), [], review_memory=True, review_skills=True)
        assert prompt == bg._COMBINED_REVIEW_PROMPT
        assert prompt.startswith(bg._REVIEW_DISCLAIMER)

    def test_default_scope_falls_back_to_disclaimed_skill_prompt(self):
        _target, prompt = bg.spawn_background_review_thread(SimpleNamespace(), [])
        assert prompt == bg._SKILL_REVIEW_PROMPT
        assert prompt.startswith(bg._REVIEW_DISCLAIMER)
