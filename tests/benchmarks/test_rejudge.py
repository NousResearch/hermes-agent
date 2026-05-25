"""Tests for the rejudge utility."""
import json
import tempfile
from pathlib import Path
from benchmarks.rejudge import rejudge_file, _select_judge


class TestRejudge:
    """Verify the rejudge utility can re-score benchmark results."""

    def test_heuristic_judge_created(self):
        """_select_judge('heuristic') should return a HeuristicJudge."""
        judge = _select_judge("heuristic")
        assert judge is not None
        assert hasattr(judge, "judge_answer")

    def test_rejudge_preserves_metadata(self):
        """Rejudge should preserve benchmark metadata fields."""
        sample_result = {
            "benchmark": "locomo",
            "backend": "baseline-flat",
            "score": 0.474,
            "judge": "heuristic",
            "dated": True,
            "records": [
                {"id": 1, "question": "What is X?", "answer": "correct",
                 "gold_answer": "correct", "context": "X is correct"},
                {"id": 2, "question": "What is Y?", "answer": "wrong",
                 "gold_answer": "right", "context": "Y is right"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "result.json"
            output_path = Path(tmpdir) / "rejudged.json"
            input_path.write_text(json.dumps(sample_result))
            rejudge_file(input_path, output_path, model="heuristic")
            result = json.loads(output_path.read_text())
            assert result["benchmark"] == "locomo"

    def test_rejudge_produces_score(self):
        """Rejudge should produce a score between 0 and 1."""
        sample_result = {
            "benchmark": "test", "backend": "test", "score": 0.0, "judge": "old",
            "records": [
                {"question": "Q1", "answer": "a", "gold_answer": "a", "context": "c"},
                {"question": "Q2", "answer": "a", "gold_answer": "a", "context": "c"},
                {"question": "Q3", "answer": "a", "gold_answer": "b", "context": "c"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "result.json"
            output_path = Path(tmpdir) / "rejudged.json"
            input_path.write_text(json.dumps(sample_result))
            rejudge_file(input_path, output_path, model="heuristic")
            result = json.loads(output_path.read_text())
            assert 0.6 <= result["score"] <= 0.7  # 2/3 correct
