import importlib.util
import json
import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(ROOT, "bin", "curse_track.py")


def load_tracker():
    spec = importlib.util.spec_from_file_location("curse_track", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def keyword_book(disabled=None):
    return {
        "version": 2,
        "repeat_warn_threshold": 3,
        "targets": {
            "opus": ["fuck you opus"],
            "hermes": ["fuck you hermes"],
        },
        "generic_curse": ["fuck", "badword"],
        "jargon_markers": ["ภาษาคน"],
        "disabled": disabled or [],
    }


def run_tracker(data_dir, *args):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, SCRIPT, *args, "--data", str(data_dir)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    return proc


def read_issues(data_dir):
    return json.loads((data_dir / "issues.json").read_text(encoding="utf-8"))


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_detect_uses_schema_v2_target_generic_and_disabled():
    tracker = load_tracker()

    target = tracker.detect("Fuck you Opus", keyword_book())
    generic = tracker.detect("Fuck ass", keyword_book())
    disabled = tracker.detect("badword", keyword_book(disabled=["badword"]))

    assert target["target"] == "opus"
    assert target["category"] == "target:opus"
    assert target["keywords"] == ["fuck you opus"]
    assert generic["target"] == "-"
    assert generic["category"] == "curse-generic"
    assert generic["keywords"] == ["fuck"]
    assert disabled is None


def test_ingest_imports_log_once_and_does_not_inflate_on_rerun(tmp_path):
    data_dir = tmp_path / "data"
    source = tmp_path / "log.jsonl"
    rows = [
        {"ts": "2026-07-05T10:00:00Z", "host": "mac", "cwd": "/repo/proj-a", "category": "target:opus", "phrase": "fuck you opus", "target": "opus"},
        {"ts": "2026-07-05T10:01:00Z", "host": "mac", "cwd": "/repo/proj-a", "category": "target:opus", "phrase": "fuck you opus", "target": "opus"},
        {"ts": "2026-07-05T10:02:00Z", "host": "mac", "cwd": "/repo/proj-b", "category": "curse-generic", "phrase": "fuck", "target": "-"},
        {"ts": "2026-07-05T10:03:00Z", "host": "mac", "cwd": "/repo/proj-c", "category": "jargon", "phrase": "ภาษาคน", "target": "-"},
        {"ts": "2026-07-05T10:04:00Z", "host": "mac", "cwd": "/repo/proj-b", "category": "hermes-fail", "phrase": "fuck you hermes", "target": "hermes"},
    ]
    write_jsonl(source, rows)

    first = run_tracker(data_dir, "ingest", "--from", str(source))
    second = run_tracker(data_dir, "ingest", "--from", str(source))

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "นำเข้าใหม่=5" in first.stdout
    assert "ข้ามซ้ำ=0" in first.stdout
    assert "นำเข้าใหม่=0" in second.stdout
    assert "ข้ามซ้ำ=5" in second.stdout

    issues = {item["fingerprint"]: item for item in read_issues(data_dir)}
    assert issues["opus"]["count"] == 2
    assert issues["curse-generic"]["count"] == 1
    assert issues["jargon"]["count"] == 1
    assert issues["hermes"]["count"] == 1

    events = (data_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ingested = (data_dir / "ingested.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == 5
    assert len(ingested) == 5


def test_report_flags_issue_over_threshold(tmp_path):
    data_dir = tmp_path / "data"
    for _ in range(3):
        proc = run_tracker(data_dir, "log", "Fuck you Opus")
        assert proc.returncode == 0, proc.stderr

    report = run_tracker(data_dir, "report")

    assert report.returncode == 0, report.stderr
    assert "[เกินเกณฑ์]" in report.stdout


def test_log_promote_then_repeat_reopens_issue(tmp_path):
    data_dir = tmp_path / "data"
    first = run_tracker(data_dir, "log", "Fuck you Opus")
    promote = run_tracker(data_dir, "promote", "ISS-1", "--root", "แก้ root แล้ว")
    repeat = run_tracker(data_dir, "log", "Fuck you Opus")

    assert first.returncode == 0, first.stderr
    assert promote.returncode == 0, promote.stderr
    assert repeat.returncode == 0, repeat.stderr

    issue = read_issues(data_dir)[0]
    assert issue["status"] == "open"
    assert issue["repeat_after_fix"] == 1
    assert issue["fix_count"] == 1
    assert issue["count"] == 2


def test_report_html_writes_summary_and_top_projects(tmp_path):
    data_dir = tmp_path / "data"
    source = tmp_path / "log.jsonl"
    html = tmp_path / "curse-report.html"
    write_jsonl(
        source,
        [
            {"ts": "2026-07-05T10:00:00Z", "host": "mac", "cwd": "/repo/proj-a", "category": "target:opus", "phrase": "fuck you opus", "target": "opus"},
            {"ts": "2026-07-05T10:01:00Z", "host": "mac", "cwd": "/repo/proj-a", "category": "target:opus", "phrase": "fuck you opus", "target": "opus"},
            {"ts": "2026-07-05T10:02:00Z", "host": "mac", "cwd": "/repo/proj-b", "category": "target:opus", "phrase": "fuck you opus", "target": "opus"},
        ],
    )
    ingest = run_tracker(data_dir, "ingest", "--from", str(source))
    report = run_tracker(data_dir, "report", "--html", str(html))

    assert ingest.returncode == 0, ingest.stderr
    assert report.returncode == 0, report.stderr
    assert html.exists()
    content = html.read_text(encoding="utf-8")
    assert "สรุปรวม" in content
    assert "ด่าทั้งหมด 3 ครั้ง" in content
    assert "เรื่องเปิดอยู่ 1 เรื่อง" in content
    assert "เรื่องเกินเกณฑ์ 1 เรื่อง" in content
    assert "โปรเจกต์ที่โดนด่าบ่อยสุด" in content
    assert "proj-a (2)" in content
    assert "proj-b (1)" in content
