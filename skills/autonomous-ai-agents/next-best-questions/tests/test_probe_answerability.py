"""Tests for the offline answerability retro probe."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evals"))

import probe_answerability  # noqa: E402


def _qa(question, answer, revealed):
    return {"question": question, "answer": answer, "revealed": revealed}


def _row(task, arm, qa, frac):
    return {
        "task": task,
        "arm": arm,
        "questions": [entry["question"] for entry in qa],
        "qa": qa,
        "meta": {"q_values": [0.9 - i * 0.1 for i, _ in enumerate(qa)]},
        "frac": frac,
    }


class TestProbeAnswerability(unittest.TestCase):
    def _fixture(self):
        return {
            "rows": [
                _row("pass-answerable", "nbq", [
                    _qa("q0", "Use stable sorting.", True),
                    _qa("q1", "Return a list.", True),
                ], 1.0),
                _row("fail-top-unanswerable", "nbq", [
                    _qa("q0", "The spec doesn't say.", False),
                    _qa("q1", "Use UTF-8.", True),
                ], 0.5),
                _row("pass-later-unanswerable", "nbq", [
                    _qa("q0", "Use ascending order.", True),
                    _qa("q1", "The spec doesn't say.", False),
                ], 1.0),
                _row("fail-top-text-unanswerable", "nbq", [
                    _qa("q0", "The spec doesn't say.", True),
                    _qa("q1", "Round half up.", True),
                ], 0.0),
                _row("ignored-other-arm", "baseline", [
                    _qa("q0", "The spec doesn't say.", False),
                ], 0.0),
            ]
        }

    def test_is_unanswerable_from_revealed_and_answer_text(self):
        self.assertFalse(probe_answerability.is_unanswerable(
            _qa("q", "Use stable sorting.", True)))
        self.assertTrue(probe_answerability.is_unanswerable(
            _qa("q", "Use stable sorting.", False)))
        self.assertTrue(probe_answerability.is_unanswerable(
            _qa("q", "The spec doesn't say.", True)))

    def test_contingency_counts_match_hand_computation(self):
        stats = probe_answerability.assemble_stats(self._fixture())
        self.assertEqual(stats["n"], 4)
        cells = stats["framings"]["top1_unans"]["contingency"]
        self.assertEqual(cells["pred_true_fail_true"], 2)
        self.assertEqual(cells["pred_true_fail_false"], 0)
        self.assertEqual(cells["pred_false_fail_true"], 0)
        self.assertEqual(cells["pred_false_fail_false"], 2)

    def test_point_biserial_sign_matches_fixture(self):
        stats = probe_answerability.assemble_stats(self._fixture())
        self.assertGreater(stats["framings"]["top1_unans"]["r"], 0.0)
        self.assertGreater(stats["framings"]["any_unans"]["r"], 0.0)
        self.assertLess(stats["n_unans_frac_r"], 0.0)

    def test_se_for_r_matches_preregistered_formula(self):
        self.assertAlmostEqual(probe_answerability.se_for_r(0.0, 6), 0.5)

    def test_degenerate_flags_constant_and_varied_predictors(self):
        varied = [task["top1_unans"] for task in probe_answerability.assemble_stats(
            self._fixture())["per_task"]]
        self.assertFalse(probe_answerability.is_degenerate(varied))

        constant = {
            "rows": [
                _row(f"constant-{i}", "nbq", [
                    _qa("q0", "The spec doesn't say.", True),
                ], 0.5)
                for i in range(4)
            ]
        }
        constant_bools = [task["top1_unans"] for task in probe_answerability.assemble_stats(
            constant)["per_task"]]
        self.assertTrue(probe_answerability.is_degenerate(constant_bools))


if __name__ == "__main__":
    unittest.main()
