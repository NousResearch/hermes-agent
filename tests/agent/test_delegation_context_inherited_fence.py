"""The delegate-child fence's inherited (``os.environ``) half can be masked for in-process
execution that is not a descendant of anything (cron runs, gateway respawns), while a live
``delegated_child_context`` is never maskable. See #136081."""
from __future__ import annotations

import pytest

from agent import delegation_context as dc


@pytest.fixture()
def inherited_marker(monkeypatch, tmp_path):
    """A HERMES_DELEGATED_CHILD_CONTEXT inherited verbatim from a fenced parent shell."""
    monkeypatch.setenv(dc.DELEGATED_CHILD_ENV_MARKER, str(tmp_path / "board"))


class TestInheritedFenceIsContamination:
    def test_marker_without_live_context_is_contamination(self, inherited_marker):
        assert dc.inherited_child_fence_is_contamination() is True

    def test_live_child_context_is_not_contamination(self, inherited_marker):
        with dc.delegated_child_context():
            assert dc.inherited_child_fence_is_contamination() is False

    def test_clean_process_is_not_contamination(self, monkeypatch):
        monkeypatch.delenv(dc.DELEGATED_CHILD_ENV_MARKER, raising=False)
        assert dc.inherited_child_fence_is_contamination() is False


class TestPredicateMasking:
    def test_env_marker_alone_still_fences(self, inherited_marker):
        assert dc.is_delegated_child_process_context() is True

    def test_mask_suppresses_only_the_inherited_half(self, inherited_marker):
        with dc.suppressed_inherited_fence():
            assert dc.is_delegated_child_process_context() is False
        # The scope exit restores the inherited marker's fence, and os.environ is untouched.
        assert dc.is_delegated_child_process_context() is True

    def test_live_child_context_wins_over_the_mask(self, inherited_marker):
        # The load-bearing property: a genuine delegate child fired from inside a masked
        # run re-arms its own fence — the mask can never unfence a real descendant.
        with dc.suppressed_inherited_fence():
            with dc.delegated_child_context():
                assert dc.is_delegated_child_process_context() is True

    def test_token_form_is_nestable_and_restores_in_order(self, inherited_marker):
        outer = dc.enter_suppressed_inherited_fence()
        inner = dc.enter_suppressed_inherited_fence()
        dc.exit_suppressed_inherited_fence(inner)
        assert dc.is_delegated_child_process_context() is False
        dc.exit_suppressed_inherited_fence(outer)
        assert dc.is_delegated_child_process_context() is True
