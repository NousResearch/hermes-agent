import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "operator_eval_gate.py"
spec = importlib.util.spec_from_file_location("operator_eval_gate", MODULE_PATH)
operator_eval_gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = operator_eval_gate
spec.loader.exec_module(operator_eval_gate)


def test_newest_batch_uses_mtime(tmp_path):
    batch_dir = tmp_path / "evals" / "batches"
    batch_dir.mkdir(parents=True)
    older = batch_dir / "older.json"
    newer = batch_dir / "newer.json"
    older.write_text("{}")
    newer.write_text("{}")

    assert operator_eval_gate.newest_batch(tmp_path) == newer


def test_eval_gate_writes_receipts(tmp_path):
    runner = tmp_path / "evals" / "run_relative_trajectory_batch.py"
    runner.parent.mkdir(parents=True)
    runner.write_text(
        "import pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.write_text('report ok')\n"
    )
    spec = tmp_path / "evals" / "batches" / "batch.json"
    spec.parent.mkdir(parents=True)
    spec.write_text("{}")
    report = tmp_path / "evals" / "runs" / "batch.md"

    result = operator_eval_gate.run_eval(tmp_path, spec, report)
    json_path, md_path = operator_eval_gate.write_receipt(tmp_path, spec, report, result)

    assert result["returncode"] == 0
    assert report.read_text() == "report ok"
    assert json_path.exists()
    assert md_path.exists()
    assert "Passed: `True`" in md_path.read_text()
