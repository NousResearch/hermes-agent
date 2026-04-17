from pathlib import Path


def test_scheduler_module_source_compiles():
    scheduler_path = Path(__file__).resolve().parents[2] / "cron" / "scheduler.py"
    source = scheduler_path.read_text(encoding="utf-8")

    compile(source, str(scheduler_path), "exec")
