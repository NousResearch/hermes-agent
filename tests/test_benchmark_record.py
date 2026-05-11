import tempfile
import unittest
from pathlib import Path

from benchmark_record import record_benchmark_run
from hermes_state import SessionDB


class BenchmarkRecordTests(unittest.TestCase):
    def test_records_checkpoint_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "state.db"
            artifact = Path(tmpdir) / "sample.json"
            artifact.write_text("[]", encoding="utf-8")

            task_id = record_benchmark_run(
                benchmark_name="csv-json",
                prompt="Convert CSV to JSON",
                validation_command="python3 -m unittest discover -s tests -v",
                artifacts=[artifact],
                task_id="task-123",
                session_id="session-abc",
                db_path=db_path,
                status="passed",
            )

            self.assertEqual(task_id, "task-123")
            db = SessionDB(db_path)
            task = db.get_task("task-123")
            self.assertIsNotNone(task)
            self.assertEqual(task["status"], "passed")
            self.assertEqual(task["session_id"], "session-abc")
            self.assertEqual(task["current_step"], "benchmark:passed")
            self.assertEqual(task["checkpoint_data"]["benchmark_name"], "csv-json")
            self.assertEqual(task["checkpoint_data"]["validation_command"], "python3 -m unittest discover -s tests -v")
            self.assertEqual(task["artifacts"], [str(artifact)])


if __name__ == "__main__":
    unittest.main()
