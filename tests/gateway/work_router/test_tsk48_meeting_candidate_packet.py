import json
import unittest

from gateway.work_router.meeting_orchestrator import RoundAssignment
from gateway.work_router.meeting_runtime import build_candidate_packet


class MeetingCandidatePacketTest(unittest.TestCase):

    def test_round_one_packet_uses_dynamic_round_and_participant(self):
        assignment = RoundAssignment(
            profile="hans",
            round_id=1,
            context="Independent commercial analysis.",
            independent=True,
        )

        packet = json.loads(
            build_candidate_packet(
                meeting_id="meeting-123",
                generation=4,
                assignment=assignment,
                agenda="Evaluate proposal.",
                facts=["Fact A"],
            )
        )

        self.assertEqual(packet["meeting_id"], "meeting-123")
        self.assertEqual(packet["round_id"], 1)
        self.assertEqual(packet["generation"], 4)
        self.assertEqual(packet["participant"], "hans")
        self.assertTrue(packet["independent"])
        self.assertIn(
            "Independent commercial analysis.",
            packet["round_context"],
        )

    def test_later_round_packet_can_target_subset_participant(self):
        assignment = RoundAssignment(
            profile="watson",
            round_id=2,
            context="Verify the lifecycle evidence.",
            independent=False,
        )

        packet = json.loads(
            build_candidate_packet(
                meeting_id="meeting-123",
                generation=4,
                assignment=assignment,
                agenda="Evaluate proposal.",
                facts=["Fact A"],
            )
        )

        self.assertEqual(packet["round_id"], 2)
        self.assertEqual(packet["participant"], "watson")
        self.assertFalse(packet["independent"])
        self.assertIn(
            "Verify the lifecycle evidence.",
            packet["round_context"],
        )

    def test_packet_preserves_existing_agenda_and_facts(self):
        assignment = RoundAssignment(
            profile="mason",
            round_id=3,
            context="Challenge the remaining assumption.",
            independent=False,
        )

        packet = json.loads(
            build_candidate_packet(
                meeting_id="meeting-xyz",
                generation=2,
                assignment=assignment,
                agenda="Choose deployment strategy.",
                facts=["Budget fixed", "Deadline Q4"],
            )
        )

        self.assertEqual(
            packet["agenda"],
            "Choose deployment strategy.",
        )
        self.assertEqual(
            packet["facts"],
            ["Budget fixed", "Deadline Q4"],
        )

    def test_packet_rejects_empty_meeting_id(self):
        assignment = RoundAssignment(
            profile="hans",
            round_id=1,
            context="Independent analysis.",
            independent=True,
        )

        with self.assertRaises(ValueError):
            build_candidate_packet(
                meeting_id="",
                generation=1,
                assignment=assignment,
                agenda="Evaluate.",
                facts=[],
            )


if __name__ == "__main__":
    unittest.main()