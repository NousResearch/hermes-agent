"""Regression: tool-invocation progress bubbles carry ``_interim_send=True``.

Issue #102271 — ``ctx._progress_metadata`` was assembled from
``_thread_metadata_for_source``/``_thread_metadata_for_target`` and passed
through ``_non_conversational_metadata()``, which is a Discord-only no-op
on every other platform. Nothing stamped ``_interim_send`` on the dict, so
every downstream call site (lines 5251, 5259, 5278, 5350, 5482, 5587, 5637,
5653, 5661, 5673) handed the relay adapter an UNMARKED metadata dict. The
relay adapter's ``send_for_platform`` / ``send`` both pop
``_interim_send`` before deciding whether to seal the open native stream
— without the marker, the first progress bubble during a streaming turn
was treated as the turn-final and the real answer posted as a duplicate
while later frames were silently swallowed by the seal tombstone.

The fix lives at the single assembly site (formerly ``gateway/run.py``
31726-31750): thread metadata flows through a new helper
``_build_progress_metadata`` that composes ``_non_conversational_metadata``
+ ``_interim_metadata`` in the right order, with Slack-native-card
recipient stamping as a pure pass-through. These tests pin that helper
so a future "let me just inline this back" refactor breaks loud.

Related: ``tests/gateway/test_interim_send_lanes.py`` already pins
``_interim_metadata`` itself; this file pins the *composition* the
progress path now relies on.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest


# Importing the module triggers a 31k+ line load — slow but real production
# path. The test must exercise the actual helper that lives next to the
# assembly site, not recompute the fix in test body.
import gateway.run as gw_run  # noqa: E402


def _slack_source(scope_id: str = "T123", user_id: str = "U456"):
    """Minimal Source-like namespace matching the Slack-native-card branch.

    The assembly only inspects ``source.scope_id`` and ``source.user_id``;
    everything else is irrelevant. We use ``SimpleNamespace`` so callers
    can pass ``None`` for either field.
    """

    return SimpleNamespace(scope_id=scope_id, user_id=user_id)


class TestBuildProgressMetadataContract:
    def test_marker_is_set_when_thread_metadata_present(self):
        """The plain case: thread metadata resolved, Slack cards OFF.

        Must yield a non-None dict carrying ``_interim_send=True`` so the
        truthiness gate ``if ctx._progress_metadata:`` at gateway/run.py:5426
        still fires AND the relay adapter's seal-skip path is honoured.
        """

        md = gw_run._build_progress_metadata(
            thread_metadata={"thread_id": "t1", "reply_to_message_id": "m1"},
            platform="telegram",
            source=None,
            slack_task_cards_enabled=False,
        )
        assert md is not None
        assert md.get("_interim_send") is True
        # Original thread metadata must survive the wrap.
        assert md.get("thread_id") == "t1"
        assert md.get("reply_to_message_id") == "m1"

    def test_marker_is_set_when_thread_metadata_is_none(self):
        """Edge case: no thread metadata could be resolved.

        Today the dict is sometimes None (the truthiness gate at line 5426
        depends on it). The fix must keep the result non-None so:
        (a) the truthiness gate keeps firing, and
        (b) the relay consumer still strips the marker.
        """

        md = gw_run._build_progress_metadata(
            thread_metadata=None,
            platform="telegram",
            source=None,
            slack_task_cards_enabled=False,
        )
        assert md is not None
        assert md.get("_interim_send") is True

    def test_marker_is_set_on_discord_with_non_conversational_marker(self):
        """Discord path: ``_non_conversational_metadata`` flips a flag, then
        ``_interim_metadata`` adds the seal-skip marker. Both must coexist
        in the final dict — Discord's lifecycle marker does NOT replace the
        interim marker.
        """

        md = gw_run._build_progress_metadata(
            thread_metadata={"thread_id": "tD"},
            platform="discord",
            source=None,
            slack_task_cards_enabled=False,
        )
        assert md is not None
        assert md.get("_interim_send") is True
        assert md.get("non_conversational") is True
        assert md.get("thread_id") == "tD"

    def test_slack_task_cards_keeps_marker(self):
        """Slack native task cards branch must also carry the marker.

        Without it, the first task-card progress update during a streaming
        turn seals the live draft with the task list. Same fix, same
        composition order.
        """

        md = gw_run._build_progress_metadata(
            thread_metadata={"thread_id": "tS"},
            platform="slack",
            source=_slack_source(),
            slack_task_cards_enabled=True,
        )
        assert md is not None
        assert md.get("_interim_send") is True
        # Slack-specific recipient stamping must survive the interim wrap.
        assert md.get("recipient_team_id") == "T123"
        assert md.get("recipient_user_id") == "U456"
        assert md.get("slack_team_id") == "T123"

    def test_slack_task_cards_without_source_recipient_ids(self):
        """Slack branch with no source.recipient info: don't crash, marker still set."""

        md = gw_run._build_progress_metadata(
            thread_metadata=None,
            platform="slack",
            source=_slack_source(scope_id=None, user_id=None),
            slack_task_cards_enabled=True,
        )
        assert md is not None
        assert md.get("_interim_send") is True
        assert "recipient_team_id" not in md
        assert "recipient_user_id" not in md

    def test_no_thread_metadata_slack_still_sets_marker(self):
        """Slack task cards + thread_metadata=None: Slack stamps still applied."""

        md = gw_run._build_progress_metadata(
            thread_metadata=None,
            platform="slack",
            source=_slack_source(),
            slack_task_cards_enabled=True,
        )
        assert md is not None
        assert md.get("_interim_send") is True
        assert md.get("recipient_team_id") == "T123"
        assert md.get("recipient_user_id") == "U456"

    def test_idempotent_marker_application(self):
        """Calling the helper twice on its own output never flips the marker
        off. Belt-and-suspenders for callers that accidentally re-wrap.
        """

        md1 = gw_run._build_progress_metadata(
            thread_metadata={"thread_id": "t1"},
            platform="telegram",
            source=None,
            slack_task_cards_enabled=False,
        )
        md2 = gw_run._interim_metadata(md1)
        assert md2.get("_interim_send") is True
        assert md2.get("thread_id") == "t1"

    def test_does_not_mutate_input_dict(self):
        """Source thread metadata must not be aliased — the helper copies
        before stamping. Defends against caller mutation leaking back into
        the shared ``TurnContext``.
        """

        src: Dict[str, Any] = {"thread_id": "t1"}
        gw_run._build_progress_metadata(
            thread_metadata=src,
            platform="telegram",
            source=None,
            slack_task_cards_enabled=False,
        )
        assert "_interim_send" not in src
        assert src == {"thread_id": "t1"}
