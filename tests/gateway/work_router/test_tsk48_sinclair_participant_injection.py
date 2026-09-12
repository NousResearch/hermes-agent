import unittest

from gateway.work_router.config import BotRegistry
from gateway.work_router.meeting_injection import SinclairParticipantInjection


IDS = {
    "Demian": "UDEMIAN",
    "Hans": "UHANS",
    "Wendy": "UWENDY",
    "Tesla": "UTESLA",
    "Turing": "UTURING",
    "Mason": "UMASON",
    "Watson": "UWATSON",
}


class SinclairParticipantInjectionTest(unittest.TestCase):

    def setUp(self):
        self.injection = SinclairParticipantInjection(
            registry=BotRegistry.from_mapping(IDS),
        )

    def test_registry_profile_can_be_forced_for_next_round(self):
        changed = self.injection.add("Tesla")

        self.assertTrue(changed)
        self.assertEqual(self.injection.participants, ("Tesla",))

    def test_unknown_registry_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            self.injection.add("Unknown")

    def test_demian_is_rejected_as_facilitator(self):
        with self.assertRaises(ValueError):
            self.injection.add("Demian")

    def test_duplicate_add_is_idempotent(self):
        self.assertTrue(self.injection.add("Tesla"))
        self.assertFalse(self.injection.add("Tesla"))
        self.assertEqual(self.injection.participants, ("Tesla",))

    def test_forced_participant_is_merged_even_when_demian_omits_it(self):
        self.injection.add("Tesla")

        merged = self.injection.merge_next_round(("Watson",))

        self.assertEqual(merged, ("Watson", "Tesla"))

    def test_merge_is_deterministic_and_preserves_first_seen_order(self):
        self.injection.add("Tesla")
        self.injection.add("Wendy")

        merged = self.injection.merge_next_round(("Wendy", "Watson"))

        self.assertEqual(merged, ("Wendy", "Watson", "Tesla"))

    def test_current_round_snapshot_is_not_mutated_by_later_injection(self):
        current_round = self.injection.merge_next_round(("Hans", "Mason"))

        self.injection.add("Tesla")

        self.assertEqual(current_round, ("Hans", "Mason"))
        self.assertEqual(
            self.injection.merge_next_round(("Watson",)),
            ("Watson", "Tesla"),
        )

    def test_merge_rejects_demian_selection(self):
        with self.assertRaises(ValueError):
            self.injection.merge_next_round(("Demian",))

    def test_merge_rejects_duplicate_demian_selection(self):
        with self.assertRaises(ValueError):
            self.injection.merge_next_round(("Hans", "Hans"))

    def test_configured_round_limit_fails_closed_without_dropping_forced_staff(self):
        self.injection.add("Tesla")

        with self.assertRaises(ValueError):
            self.injection.merge_next_round(
                ("Hans", "Mason"),
                max_participants=2,
            )


if __name__ == "__main__":
    unittest.main()