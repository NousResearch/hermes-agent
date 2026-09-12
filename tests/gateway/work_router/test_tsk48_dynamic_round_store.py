import json
import tempfile
import unittest
from pathlib import Path

from gateway.work_router.models import MeetingCandidate, MeetingState
from gateway.work_router.store import RouterStore, StoreError


class DynamicRoundStoreTest(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = RouterStore(Path(self.tempdir.name) / "router.db")
        self.meeting = MeetingState(
            meeting_id="meeting-dynamic-store",
            channel_id="CMEETING",
            thread_ts="1.0",
            status="ready",
            generation=1,
            participants=("Hans", "Mason", "Watson"),
        )
        self.store.create_meeting(self.meeting)

    def tearDown(self):
        self.store.close()
        self.tempdir.cleanup()

    def test_round_subset_is_opened_with_pending_candidates(self):
        updated = self.store.start_dynamic_meeting_round(
            self.meeting.meeting_id,
            round_id=1,
            participants=("Hans", "Watson"),
            packet_json=json.dumps({"participants": ["Hans", "Watson"]}),
            expected_generation=1,
        )

        self.assertEqual(updated.status, "round_open")
        self.assertEqual(updated.current_round, 1)
        self.assertEqual(
            self.store.get_meeting_candidate(
                self.meeting.meeting_id, 1, "Hans"
            ).status,
            "pending",
        )
        self.assertIsNone(
            self.store.get_meeting_candidate(
                self.meeting.meeting_id, 1, "Mason"
            )
        )

    def test_submitted_subset_closes_round_and_allows_next_round(self):
        self.store.start_dynamic_meeting_round(
            self.meeting.meeting_id,
            round_id=1,
            participants=("Hans", "Watson"),
            packet_json="{}",
            expected_generation=1,
        )
        for profile in ("Hans", "Watson"):
            self.store.finalize_dynamic_meeting_candidate(
                MeetingCandidate(
                    meeting_id=self.meeting.meeting_id,
                    round_id=1,
                    generation=1,
                    participant=profile,
                    status="submitted",
                    reason_to_speak=True,
                    reason_class="fact_support",
                    statement=f"{profile} contribution",
                )
            )

        closed = self.store.close_dynamic_meeting_round(
            self.meeting.meeting_id,
            round_id=1,
            expected_generation=1,
        )
        self.assertEqual(closed.status, "awaiting_continue")

        opened = self.store.start_dynamic_meeting_round(
            self.meeting.meeting_id,
            round_id=2,
            participants=("Watson",),
            packet_json="{}",
            expected_generation=1,
        )
        self.assertEqual(opened.current_round, 2)
        self.assertIsNotNone(
            self.store.get_meeting_candidate(
                self.meeting.meeting_id, 2, "Watson"
            )
        )

    def test_round_cannot_close_while_candidate_is_pending(self):
        self.store.start_dynamic_meeting_round(
            self.meeting.meeting_id,
            round_id=1,
            participants=("Hans", "Watson"),
            packet_json="{}",
            expected_generation=1,
        )
        self.store.finalize_dynamic_meeting_candidate(
            MeetingCandidate(
                meeting_id=self.meeting.meeting_id,
                round_id=1,
                generation=1,
                participant="Hans",
                status="submitted",
                reason_to_speak=True,
                reason_class="fact_support",
                statement="Hans contribution",
            )
        )

        with self.assertRaises(StoreError):
            self.store.close_dynamic_meeting_round(
                self.meeting.meeting_id,
                round_id=1,
                expected_generation=1,
            )

    def test_round_rejects_participant_outside_frozen_meeting(self):
        with self.assertRaises(StoreError):
            self.store.start_dynamic_meeting_round(
                self.meeting.meeting_id,
                round_id=1,
                participants=("Tesla",),
                packet_json="{}",
                expected_generation=1,
            )

    def test_durable_pool_can_expand_beyond_one_round_participant_limit(self):
        self.store.start_dynamic_meeting_round(
            self.meeting.meeting_id,
            round_id=1,
            participants=("Hans",),
            packet_json="{}",
            expected_generation=1,
        )
        self.store.finalize_dynamic_meeting_candidate(
            MeetingCandidate(
                meeting_id=self.meeting.meeting_id,
                round_id=1,
                generation=1,
                participant="Hans",
                status="submitted",
                reason_to_speak=True,
                reason_class="fact_support",
                statement="Hans contribution",
            )
        )
        self.store.close_dynamic_meeting_round(
            self.meeting.meeting_id,
            round_id=1,
            expected_generation=1,
        )

        expanded = self.store.extend_dynamic_meeting_participants(
            self.meeting.meeting_id,
            participants=("Tesla", "Turing", "Wendy"),
            expected_generation=1,
        )

        self.assertEqual(
            expanded.participants,
            ("Hans", "Mason", "Watson", "Tesla", "Turing", "Wendy"),
        )


if __name__ == "__main__":
    unittest.main()